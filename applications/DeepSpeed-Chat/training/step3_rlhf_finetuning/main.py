#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
"""
准备阶段：
#* 1. 定义分布式训练的device,rank,同步各端通信;
#* 2. 基于actor模型加载tokenizer，构造训练数据;
#* 3. 调用DeepSpeedRLHFEngine构造RLHF的deepspeed engine，其包含actor,ref,critic,reward四个engine;
#* 4. 基于RLHF的engine，调用PPOTrainer(PPOTrainerUnsupervised)类构造ppo(ppp-ptx)的训练实例
#* 5. 定义MiniDataset实例，每次搜集generation_batch_numbers个mini-batch,
    #* 然后将它们统一分割为per_device_mini_train_batch_size大小的micro-batch;
训练阶段：
#* 6. 对unsupervised的数据进行micro数据切分；
#* 7. 调用ppotrainer的generate_experience方法，
    #* 根据给定的prompts和mask，先调用actor模型生成answer和mask,然后计算answer在actor,ref模型下的logit;
    #* 计算answer在critic_model下的最后一个字符全部序列的得分，在reward_model模型下chosen序列最后一个实际字符的得分，
    #* 返回全部中间及最终结果；
#* 8. 调用MiniDataset类，将generate_experience返回的mini-batch继续分割为micro-batch；
    #* **在将mini-batch的个数搜集到max_size个时才开始ppo训练**
#* 9. 根据args启动actor模型的梯度检查点方法；
#* 10. 根据args.定义的ppo训练epochs训练ppo模型：
    #* 11. 对每个micro-batch调用ppotrainer的train_rlhf方法训练模型，并返回actor_loss和critic_loss；
    #* 12.调用ppotrainer的train_unsupervised针对unsupervised数据训练actor模型，并返回unsup_loss；
    #* 13. 根据args.ema参数启动移动平均方法更新模型权重；
#* 14. ppo_epochs训练完毕后，调用all_reduce方法计算全局平均奖励；
#* 15. ppo_epochs训练完毕后，根据args.actor_gradient_checkpointing参数在训练完毕后关闭actor的梯度检查点功能；
#* 16. 根据lora、ema、actor_zero_stage等参数保存actor,critic,actor_ema模型。

问题：
1. 数据组装时为什么flip;
2. 数据gather时，为什么-1;
"""
# DeepSpeed Team
"""

engine = DeepSpeedRLHFEngine(actor_model_name_or_path=actor_model_name_or_path,
                             critic_model_name_or_path=critic_model_name_or_path,
                             tokenizer=tokenizer,
                             args=args)
trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
    out = trainer.generate_experience(prompt_batch)
    actor_loss, critic_loss = trainer.train_rlhf(out)

"""
import argparse
import os
import random
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    default_data_collator,
)

import deepspeed

from ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from rlhf_engine import DeepSpeedRLHFEngine

import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model, load_hf_tokenizer
from utils.module.lora import convert_lora_to_linear_layer


def parse_args():
    parser = argparse.ArgumentParser(
        description="(Step 3) RLHF training arguments")

    parser.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='2,4,4',
        help=
        'Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` '
        'will use 60% of data for phase 1, 20% for phase 2 and 20% for phase 3.'
    )
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    #* RL模型中的PPO-ptx模型需要使用unsupervised数据的分布作为惩罚以保持无监督的表现
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).")

    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library)."
    )
    #? KL奖励系数$\beta$和预训练损失系数$\gamma$控制KL惩罚和预训练部分的强度，对于PPO模型,预训练损失系数为零。
    # $\gamma$控制预训练损失，以保持在公告数据集上（其他非生成任务上）的性能回归。
    parser.add_argument("--unsup_coef",
                        type=float,
                        default=27.8,
                        help='''gamma in Equation 2 from InstructGPT paper''')

    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )
    parser.add_argument(
        "--per_device_mini_train_batch_size",
        type=int,
        default=16,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument("--generation_batch_numbers",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")
    parser.add_argument("--max_prompt_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument("--max_answer_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    # L2惩罚
    parser.add_argument("--actor_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--critic_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # DeepSpeed 
    #? hybrid engine -> 模型同时用于训练和推理，非hybrid模型仅训练
    # 为actor模型启用混合引擎，以通过 DeepSpeed 优化推理和训练 
    parser.add_argument(
        "--enable_hybrid_engine",
        action='store_true',
        help=
        "Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed."
    )
    # 在生成期间取消固定 actor 的参数
    parser.add_argument(
        "--unpin_actor_parameters",
        action='store_true',
        help=
        "Unpin actor's parameters during generation. This makes generation slower but requires less memory."
    )
    # 在推断阶段释放缓存
    parser.add_argument(
        "--release_inference_cache",
        action='store_true',
        help=
        "Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size."
    )
    # 推断优化中张量并行的程度，在hybrid engine中必须开启
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature."
    )
    #? 在混合engine中，张量并行分片里引入到层中的粒度-> 这里应该是将tensor parallelism和model parallelism等价了
    #? 但按Megatron-LM和GPipe,分别称为TensorParallelism和Pipeline Parallelism,均是为模型并行
    #? micro-batch的术语是在Pipeline Parallelism提到的，或者micro-batch也可以用在TensorParallelism?
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help=
        "Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature."
    )
    # ZeRO的offload技巧
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    #? reference model是啥->即原文式2中的\pi^{SFT},即步骤1的sft模型，作为基准模型；log(RL/SFT)作为reward模型的惩罚项
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model')
    # Actor模型的ZeRO优化步骤
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        '--critic_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Critic model (and reward).')
    # 开启HF的梯度检查点优化
    parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument(
        '--critic_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Critic model.')
    # 取消actor模型中的dropout
    parser.add_argument('--disable_actor_dropout',
                        action='store_true',
                        help='Disable the dropout of the actor model.')
    parser.add_argument('--disable_critic_dropout',
                        action='store_true',
                        help='Disable the dropout of the critical model.')
    ## LoRA for efficient training setting
    parser.add_argument("--actor_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    #* 执行LoRA优化的actor模型的模块
    parser.add_argument("--actor_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument("--critic_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--critic_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")

    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    ## Make EMA as an optional feature
    #? EMA检查点？-> 维护一个影子权重，在每次更新模型权重的时候，对影子权重进行EMA。
    #? #评估模型和导出模型的时候都使用影子权重。在checkpoint中，model的权重即是 EMA 后的权重。
    # 如果我们对数据进行二次采样(例如，随机采样再进行一次随机采样)，就有可能产生信息损失，
    # 因为每次采样都只能表征部分数据的分布情况，而采样本身存在一定的不确定性，
    # 可能导致选取的数据样本并不能完全代表原始数据集的统计特征。

    #使用EMA可以避免数据二次采样带来的信息损失风险，这是因为EMA是基于历史权重指数平均得到的，
    # 而不是直接对数据样本求平均值。在这个过程中，算法会记录参数历史的加权平均值，
    # 因此每个数据点的贡献是逐渐变小的。这样，在使用EMA进行优化时，
    # 可以更好地反映出历史信息的影响，使得评估指标更准确、更稳定。
    # 
    parser.add_argument('--enable_ema',
                        action='store_true',
                        help='Enable EMA checkpoint for the model.')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if (args.actor_gradient_checkpointing
            and args.actor_lora_dim > 0) or (args.critic_gradient_checkpointing
                                             and args.critic_lora_dim > 0):
        assert (
            not args.only_optimize_lora
        ), "--{actor,critic}_gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    return args


def create_datasets(args, tokenizer, train_phase=3):
    """
    构造step3的数据集，包括两部分：PPO训练的数据集和unsupervised pretrain训练的数据集,步骤为：
    1. 根据config文件确定是否启动unsupervised_training,如是则根据config获取unsupervised训练数据；
    2. 构造prompt_train数据的Dataset类；
    3. 定义RLHF数据collator，其作用是以给定的pad_token将矩阵填充至最长句子的长度或max_token_len；
       无监督数据仍按默认collator组装数据（主要是padding)默认为torch_default_data_collator,
       即将所有数据统一为tensor,然后将标签的key变为'label'
    4. 调用Sampler和Dataloader组装数据；
    5. 计算每个epoch权重更新的次数和总更新次数；

    """
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    prompt_train_dataset, _ = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_prompt_seq_len)
    # 先判断是否使用无监督训练数据，亦即使用PPO-ptx模式
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    # 根据给定的pad_token填充至最长句子的长度或max_token_len
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len,
                                     args.inference_tp_size)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)
    # PPO训练数据按照DataCollatorRLHF组装数据
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_train_batch_size)
    #* 无监督数据仍按默认collator组装数据（主要是padding)
    # 默认为torch_default_data_collator,即将所有数据统一为tensor,然后将标签的key变为'label'
    # 如果不包含无监督数据，则无监督数据为[None]*监督数据长度
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_train_batch_size)
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader
    # 确定每个epoch权重更新的次数
    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (args.per_device_train_batch_size / args.per_device_mini_train_batch_size) * \
        args.ppo_epochs / args.gradient_accumulation_steps
    # 总更新次数
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters


def main():
    args = parse_args()
    #* 1. 定义分布式训练的device,rank,同步各端通信

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        #? if we enable unsupervised training, we need to double the batch size for actor model
        #? 为什么可以先后训练？
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    # 同步各节点的通信
    torch.distributed.barrier()
    #* 2. 基于actor模型加载tokenizer，构造训练数据
    # create common tokenizer based on actor model
    tokenizer = load_hf_tokenizer(args.actor_model_name_or_path,
                                  fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    # make sure tokenizer is right pad in our logic
    tokenizer.padding_side = 'right'
    prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(
        args=args, tokenizer=tokenizer, train_phase=3)

    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    #* 3. 调用DeepSpeedRLHFEngine构造RLHF的deepspeed engine，其包含actor,ref,critic,reward四个engine
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args)
    #? 自定义 end_of_conversation_token
    args.end_of_conversation_token = "<|endoftext|>"
    #* 4. 基于RLHF的engine，调用PPOTrainer(PPOTrainerUnsupervised)类构造ppo(ppp-ptx)的训练实例
    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)
    #* 流水线并行中的数据并行
    # first number is how many experience-batch to generate,
    #  second number is the training batch size, which is the micro-batch size used
    #* 5. 定义MiniDataset实例，每次搜集generation_batch_numbers个mini-batch,
    #* 然后将它们统一分割为per_device_mini_train_batch_size大小的micro-batch
    # 其作用是先搜集max_size个mini-batch,
    #然后将每个mini-batch分割为small_batch_size个较小的micro-batch，
    #组成一个list返回
    exp_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                   args.per_device_mini_train_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                     args.per_device_mini_train_batch_size)

    #* Train!
    print_rank_0("***** Running training *****", args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
            args.global_rank)
        for step, (batch_prompt, batch_unsupervised) in enumerate(
                zip(prompt_train_dataloader, unsupervised_train_dataloader)):
            #* 6. 对unsupervised的数据进行micro数据切分
            batch_prompt = to_device(batch_prompt, device)
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_train_batch_size])
            # prompts = batch_prompt['prompt']
            # length = prompts.size(-1)
            # if length > args.max_prompt_seq_len:
            #     prompts = prompts[:, length - args.max_prompt_seq_len:]
            #     raise ValueError("Prompt length is too long")

            #  根据给定的prompts和mask，先调用actor模型生成answer和mask
            # 计算answer在actor_model、refernece_model下的logit;
            # 计算answer在critic_model下的最后一个字符全部序列的得分，
            # 在reward_model模型下计算序列最后一个实际字符的得分
            # 返回全部中间及最终结果
            #* 7. 调用ppotrainer的generate_experience方法，
            #* 根据给定的prompts和mask，先调用actor模型生成answer和mask,然后计算answer在actor,ref模型下的logit;
            #* 计算answer在critic_model下的最后一个字符全部序列的得分，在reward_model模型下chosen序列最后一个实际字符的得分，
            #* 返回全部中间及最终结果
            out = trainer.generate_experience(batch_prompt['prompt'],
                                              batch_prompt['prompt_att_mask'])
            #* 8. 调用MiniDataset类，将generate_experience返回的mini-batch继续分割为micro-batch
            #* 在将mini-batch的个数搜集到max_size个时才开始ppo训练
            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                average_reward = 0
                #* 9. 根据args启动actor模型的梯度检查点方法
                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()
                #* 10. 根据args.定义的ppo训练epochs训练ppo模型
                for ppo_ep in range(args.ppo_epochs):
                    #* 11. 对每个micro-batch调用ppotrainer的train_rlhf方法训练模型，并返回actor_loss和critic_loss
                    for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)):
                        # 调用train_rlhf,训练actor,critic模型 
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                        actor_loss_sum += actor_loss.item()
                        critic_loss_sum += critic_loss.item()
                        average_reward += exp_data["rewards"].mean()
                        #* 12.调用ppotrainer的train_unsupervised针对unsupervised数据训练actor模型，并返回unsup_loss
                        # PPO-ptx
                        if unsupervised_training_enabled:
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, args.unsup_coef)
                            unsup_loss_sum += unsup_loss.item()

                        inner_iter += 1
                        #* 13. 根据args.ema参数启动移动平均方法更新模型权重
                        if args.enable_ema:
                            # 调用moving_average更新actor模型权重
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=args.actor_zero_stage)

                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)

                print_rank_0(
                    f'epoch: {epoch}|step: {step}|ppo_ep: {ppo_ep+1}|act_loss: {actor_loss_sum/inner_iter}|cri_loss: {critic_loss_sum/inner_iter}|unsuper_loss: {unsup_loss_sum/inner_iter}',
                    args.global_rank)
                # 分布式训练一般分为同步训练和异步训练，
                # 同步训练中所有的worker读取mini-batch的不同部分，同步计算损失函数的gradient，最后将每个worker的gradient整合之后更新模型。
                # 异步训练中每个worker独立读取训练数据，异步更新模型参数。
                # 通常同步训练利用AllReduce来整合不同worker计算的gradient，异步训练则是基于参数服务器架构（parameter server）

                # AllReduce其实是一类算法，目标是高效得将不同机器中的数据整合（reduce）之后再把结果分发给各个机器。
                # 在深度学习应用中，数据往往是一个向量或者矩阵，通常用的整合则有Sum、Max、Min等。
                # AllReduce具体实现的方法有很多种，最单纯的实现方式就是每个worker将自己的数据发给其他的所有worker，然而这种方式存在大量的浪费。
                # 一个略优的实现是利用主从式架构，将一个worker设为master，其余所有worker把数据发送给master之后，
                # 由master进行整合元算，完成之后再分发给其余worker。不过这种实现master往往会成为整个网络的瓶颈。
                # Ring AllReduce算法分为两个阶段。第一阶段，将N个worker分布在一个环上，并且把每个worker的数据分成N份。
                # 第k个worker会把第k份数据发给下一个worker，同时从前一个worker收到第k-1份数据。
                # 之后worker会把收到的第k-1份数据和自己的第k-1份数据整合，再将整合的数据发送给下一个worker。
                # 以此循环N次之后，每一个worker都会包含最终整合结果的一份。
                # 第二阶段，每个worker将整合好的部分发送给下一个worker。worker在收到数据之后更新自身数据对应的部分即可。
                # 假设每个worker的数据是一个长度为S的向量，那么个Ring AllReduce里，每个worker发送的数据量是O(S)，和worker的数量N无关。
                #* 14. ppo_epochs训练完毕后，调用all_reduce方法计算全局平均奖励
                average_reward = get_all_reduce_mean(average_reward).item()

                print_rank_0(f"average reward score: {average_reward/inner_iter}",args.global_rank)
                print_rank_0("-----------------------------------------------------",args.global_rank)
            #* 15. 在ppo_epochs训练完毕，根据args.actor_gradient_checkpointing参数在训练完毕后关闭actor的梯度检查点功能。
            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()
    #* 16. 根据lora、ema、actor_zero_stage等参数保存actor,critic,actor_ema模型
    if args.output_dir is not None:
        print_rank_0('saving model ...')
        rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
        rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)
        if args.enable_ema:
            rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                rlhf_engine.actor_ema)

        if torch.distributed.get_rank() == 0:
            save_hf_format(rlhf_engine.actor,
                           tokenizer,
                           args,
                           sub_folder='actor')
            save_hf_format(rlhf_engine.critic,
                           tokenizer,
                           args,
                           sub_folder='critic')
            if args.enable_ema:
                save_hf_format(rlhf_engine.actor_ema,
                               tokenizer,
                               args,
                               sub_folder='actor_ema')

        if args.actor_zero_stage == 3:
            save_zero_three_model(rlhf_engine.actor,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'actor'),
                                  zero_stage=args.actor_zero_stage)
            if args.enable_ema:
                save_zero_three_model(rlhf_engine.actor_ema,
                                      global_rank=args.global_rank,
                                      save_dir=os.path.join(
                                          args.output_dir, 'actor_ema'),
                                      zero_stage=args.actor_zero_stage)
        if args.critic_zero_stage == 3:
            save_zero_three_model(rlhf_engine.critic,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'critic'),
                                  zero_stage=args.critic_zero_stage)


if __name__ == "__main__":
    main()
