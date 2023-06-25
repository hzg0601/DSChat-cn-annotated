# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn

"""
RewardModel应为transformers的BaseModel,其返回值的一个为hidden_states
输入必须自带正负样本，前一半为正样本，后一半为负样本
将最后一层映射为一个标量，然后将输出按正负样本分开；
如果正样本与负样本对应位置间没有重复，则以num_padding位置或seq_len位置的得分为
chosen和reject得分，以二者的差的sigmoid.mean()为损失
则以(第一个重合的位置) 到 (num_padding的位置或seq_len的较大者)的值为chosen和reject的得分，
以二者差的sigmoid.mean()为损失；
训练时返回loss, chosen_score,reject_score;
预测时返回标量值或(标量值+chosen_scores)


"""
## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):
        loss = None
        # BaseModelOutput的输出：
        # last_hidden_state: Sequence of hidden-states at the output of the last layer of the model.
        # hidden_sates: optional, output of each layer  shape (batch_size, sequence_length, hidden_size)
        # attention: optional shape (batch_size, num_heads, sequence_length, sequence_length)
        # 

        # CausalModelOutput的输出为：
        # loss: if label provided
        # logits: scores for each vocabulary token before SoftMax
        # hidden_states:Tuple, the output of each layer
        #               returned when output_hidden_states=True is passed or when config.output_hidden_states=True)
        #               shape (batch_size, sequence_length, hidden_size)
        # attentions: Tuple,returned when output_attentions=True is passed or when config.output_attentions=True)
        # cross_attention: Tuple, returned when output_attentions=True is passed or when config.output_attentions=True
        #                   shape (batch_size, num_heads, sequence_length, sequence_length) 
        # past_key_values: Tuple,returned when use_cache=True is passed or when config.use_cache=True)
        #                  each tuple containing the cached key, value states of the self-attention and 
        #                   the cross-attention layers if model is used in encoder-decoder setting.
        #                   Contains pre-computed hidden-states (key and values in the attention blocks) 
        #                   that can be used (see past_key_values input) to speed up sequential decoding.
        #? 由于没有传入label,因此其第一个返回值是logits(batch_size, sequence_length, config.vocab_size)
        #? 此处的写法意思应是BaseModelOutput，第一个返回值为hidden_states->RewardModel调用的是AutoModel
        #? 因此是预训练（而监督微调的）基础模型，故返回为BaseMdoelOutput, 
        transformer_outputs = self.rwtranrsformer(
            input_ids, # 就是输入的序列batch_size*seq_len
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)
        #? 基础模型的返回的logits,按论文应该是最后一个embedding吧?
        hidden_states = transformer_outputs[0]
        # 然后输出一个标量
        rewards = self.v_head(hidden_states).squeeze(-1) # 原始返回为3维，加squeeze为 batch_size * seq 
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        # 以前一半为接受集，后一半为拒绝集
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2 # bs=batch_size//2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1 #? 应为bs\tiems seq 
        rejected_ids = input_ids[bs:] #? 如果batch_size不为偶数怎么办
        chosen_rewards = rewards[:bs] # bs x seq 
        rejected_rewards = rewards[bs:] # bs x seq 

        # Compute pairwise loss. Only backprop on the different tokens before padding
        # 
        loss = 0
        for i in range(bs):
            chosen_id = chosen_ids[i] # 应为seq
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i] #? seq 
            rejected_reward = rejected_rewards[i] #? seq 
            # chosen_index,选出第i个句子，然后找出等于self.PAD_ID的元素，
            # #然后调用nonzero取出不为0的轴的index,返回一个z\times n, n为dim, z为非零元的个数
            # ? 其实就是返回非零元的坐标
            # 因此该语句取出了所有chosen_id中包含self.PAD_ID的条目
            # Returns a tensor containing the indices of all non-zero elements of input
            # 返回所有非零元的坐标，故为 num_nonzero \times 2 #? 应为num_numzero的向量
            # 例如，如果有两个padding,则c_inds为2\times 2,如果只有一个padding，则为1\times 2

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            # 从起始token的数量num_padding_at_begining开始数，如果num_padding_at_begining小于句子的padding数
            # c_ind等于num_padding_at_begining位置的值，否则等于句子长度
            #? 按nonzero的返回为num_padding\times 2,则取值为一个2维向量,无法调用item()方法
            #? 按前，c_ind应为num_padding的位置或seq_len
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            # 检查chosen_id, reject_id的重复程度
            check_divergence = (chosen_id != rejected_id).nonzero()
            # 如果chosen_id, reject_id没有重复项，令end_ind=seq,diver_ind=end_ind-1,r_ind=c_ind
            #? end_ind = seq, diver_ind = seq -1 ,  
            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            # 如果chosen_id,reject_id间有重复项，则end_ind为c_ind和r_ind的较大者，
            # diver_ind为chosen_id中和rejected_id中第一个重合的位置
            else:
                # Check if there is any padding otherwise take length of sequence
                # 按前，r_ind应为num_padding的位置或seq_len
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len

                end_ind = max(c_ind, r_ind)
                #? diver_ind为chosen_id中和rejected_id中第一个重合的位置
                divergence_ind = check_divergence[0]
            # divergence_ind应为 seq-1或chosen和reject间第一个重合的位置
            assert divergence_ind > 0
            # 若完全没有重复c_truncated_reward = chosen_reward的最后得分，r_truncated_reward也是最后一个得分
            # 若存在重复，则二者的值为第一个重合的位置到 num_padding的位置或seq_len的较大者的值
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            # num_padding的位置或seq_len位置的值
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores, # bs * 1
            "rejected_mean_scores": rejected_mean_scores, # bs * 1 
        }
    # 定义只返回得分不返回损失的函数
    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):
        """
        forward_value模型仅返回input_ids的得分,因其无需训练，因此无需reject样本;
        若return_value_only=True,则返回所有token的得分,shape: batch_size * input_len;
        若return_value_only=False，则返回最后一个非padding token的得分，shape：batch_size * 1;
        """
        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values # batch_size * input_len 
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            # 
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores), # batch_size * 1
            }
