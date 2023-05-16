# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import sys
import os
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    """先对原始logits进行softmax，然后根据labels指定的index对最后一维进行查找，
    取出labels每个标签对应的logit
    """
    log_probs = F.log_softmax(logits, dim=-1)

    # out[i,j,k] = input[index[i,j,k]][j][k] #dim=0
    # out[i,j,k] = input[i][index[i,j,k]][k] #dim=1
    # out[i,j,k] = input[i,j,:][index[i,j,k]] #dim=2

    # out[i,j] = input[index[i,j]][j] #dim=0
    # out[i,j] = input[i][index[i,j]] #dim=1
    # log_probs: batch_size * answer_len - 1 * dict_size, index: batch_size * (answer_len - 1) * 1
    # 返回维度为batch_size * (answer_len -1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]

        # Those value can be changed
        self.kl_ctl = 0.02
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

    def _generate_sequence(self, prompts, mask):
        """
        以无梯度的方式生成给定prompts的answer,
        然后删除回复长度小于1的条目
        """
        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        with torch.no_grad():
            seq = self.actor_model.module.generate(prompts,
                                                   attention_mask=mask,
                                                   max_length=max_min_length,
                                                   min_length=max_min_length)

        # Filter out seq with no answers (or very short). 
        # This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        ans = seq[:, prompt_length:]
        self.prompt_length = prompt_length
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        return out_seq

    def generate_experience(self, prompts, mask):
        """
        根据给定的prompts和mask，先调用actor模型生成answer和mask
        然后以非追踪梯度的方式计算answer在actor_model、refernece_model下的logit;
        以非追踪梯度的方式计算answer在critic_model下的最后一个字符全部序列的得分，
        在reward_model模型下chosen序列最后一个实际字符的得分
        返回全部中间及最终结果

        """
        #* 1.首先令actor模型针对给定的prompts,生成answer，无需梯度追踪
        self.eval()
        seq = self._generate_sequence(prompts, mask)
        #* 2. 生成answer的掩码
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        # 与给定的对象比较，返回一个与seq形状一致的bool类型，即确定attention的字符
        # 非padding字符为True, padding字符为False
        attention_mask = seq.not_equal(pad_token_id).long()

        with torch.no_grad():
            #* 3. 根据生成的answer和掩码，调用actor和reference模型生成answer的嵌入矩阵
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            #* 4. 调用reward模型计算答案最后一个非padding token的得分，
            #* 调用critic模型计算答案的最后一个token的原始得分
            # 计算actor返回序列的奖励值，
            reward_score = self.reward_model.forward_value(
                seq, attention_mask,prompt_length=self.prompt_length)['chosen_end_scores'].detach()
            # RewardModel类的只返回得分不返回损失的函数,是最后一层映射的结果 batch_size * seq 
            # 返回的是batch_size * seq的tensor,不取最后一个得分，不考虑pad_token的位置等因素
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1] # batch_size * (answer_len -1)
        #* 5. 计算actor模型和reference模型的logits
        # actor_model和reference_model的forward函数返回的是tf.modeling_outputs.Output类，其包括：
        # loss(optional), logits,hidden_states(optional),attentions(optional),cross_attention(optional),past_key_values(optional)
        logits = output.logits # batch_size * seq_len * dict_size
        logits_ref = output_ref.logits

        return {
            'prompts': prompts,
            #? logits[:,:-1,:] batch_size * answer_len -1 * dict_size, 去掉最后一个token
            #? 因为flip函数
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]), # actor模型的seq的除第一个外每个token概率
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]), # 返回维度为batch_size * (answer_len -1)
            'value': values, # RewardModel中v_head返回的最后个token的得分，不考虑pad_token的位置,batch_size * (answer_len -1)
            'rewards': reward_score, # 即RewardModel返回的chosen_end_scores , batch_size * (answer_len -1)
            'input_ids': seq, # actor_model针对prompts的输出
            "attention_mask": attention_mask
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):
        """
        首先计算actor_model和reference_model针对answer的嵌入的logits的差，乘以-\beta 即KL奖励系数
        然后加上每个answer最后一个非padding token的得分，即文中的公式2.
        """
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate # reward: batch_size * (answer_len - 1)
        #? prompts是否进行padding?->应该是，否则不能调用prompts.shape
        start = prompts.shape[1] - 1 # batch_size * seq_len, 故start=prop_seq_len - 1
        # action_mask = attention_mask[:,1:]
        # action_mask[:,start:] = attention_mask[:,1:][:,prop_seq_len - 1:]=attention_mask[:,prop_seq_len:]
        # 故为每个answer序列超过prompts序列长度的部分未被padding的字符数
        #* 故ends = 每个answer序列最后一个非padding字符的index
        ends = start + action_mask[:, start:].sum(1)
        # 对chosen_end_scores进行裁剪，
        #? reward_score得分为forward_value调用的返回值，返回的是最后一个非padding token的得分
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            # start: ends[j]每个序列promp_seq_len -1 : answer最后一个非padding字符的位置
            # rewards[j,start:ends[j]][-1] answer最后一个字符的kl_divergence
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards # batch_size * (answer_len - 1)

    def train_rlhf(self, inputs):
        """
        0. 前置调用generate_experience方法，令actor,ref,reward,critic模型生成:
        answer_seq, 
        actor针对answer_seq除第一个外所有token的logit:log_probs,
        ref针对answer_seq除第一个外所有token的logit:ref_log_probs,
        reward针对answer_seq最后一个非padding token的得分:rewards，
        critic针对answer_seq最后一个token的原始得分:values, 重命名为old_values
        
        1. 首先根据log_probs和ref_log_probs, reward调用compute_reward计算公式2的PPO目标损失
        2. 然后调用get_advantages_and_returns计算PPO模型更新所需的advantages, returns；
        3. 然后再调用actor模型计算对answer_seq除第一个外所有token的logit:actor_log_probs
        4. 基于log_probs, actor_log_probs, andvantage计算actor模型的损失函数；
        5. 再调用critic模型计算answer_seq除最后一个外所有token的得分values，
        6. 基于values, old_values, return计算critic模型的损失函数；
        7. 更新actor critic模型。

        """
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']
        # start = prompt_seq_len - 1
        start = prompts.size()[-1] - 1
        #? action_mask 去除第一个token,默认第一个为padding token?
        action_mask = attention_mask[:, 1:]

        old_values = values # batch_size * (answer_len -1)
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask) # batch_size * (answer_len -1)
            # # old_values answer最后一个token的得分,old_reward公式2计算的以
            # 最后一个非padding token为基准的得分，start: prompts_seq_len - 1
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        #* 再计算一次actor模型针对 answer_seq的log_prob
        # 计算actor_model针对answer_seq的logit, batch_size * (answer_len ) 
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        # 先softmax，然后取出seq[:,1:]对应的概率，得到一个batch_size *(answer_len -1) 
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        # 计算actor模型损失函数
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        # 更新actor模型
        self.actor_model.backward(actor_loss)
        self.actor_model.step()
        #* 再计算一次critic模型针对answer_seq的得分
        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        # 计算critic模型的得分
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,start:],
                                          returns, action_mask[:, start:])
        # 更新critic模型
        self.critic_model.backward(critic_loss)
        self.critic_model.step()

        return actor_loss, critic_loss

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1] # answer_len - 1
        # start = prompt_len - 1
        # 计算每一个answer字符的last_galelam
        for t in reversed(range(start, length)):
            # 第t+1个token的得分
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            # rewards除最后一个外，都是KL
            # 第t个token的KL+下一个token的score*gamma - token的score
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            # 
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        # list[::-1]反序
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    #? 为什么可以分块训练，分开训练的梯度更新与联合训练的梯度更新一致吗？
    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
