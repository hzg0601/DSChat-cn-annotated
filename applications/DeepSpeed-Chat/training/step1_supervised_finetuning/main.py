#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
训练框架是以pytorch的分布式训练为基础，

pytorch的数据并行DP训练框架如下：
1. master(一般是GPU0)从磁盘或者合页内存中取数据。
2. master将数据分到其他GPU上
3. master将模型复制到其他GPU上
4. 每块GPU单独进行前向计算，得到输出
5. master收集每块GPU上的输出，计算损失
6. master将损失分到其他卡上，每块卡单独进行反向传播，计算梯度
7. master收集每块GPU上的梯度，汇总以后，进行reduce操作，结果分发到每块卡上。
8. 每块GPU根据梯度，单独更新模型

可以看出DP模型比较多的操作是在0号卡上进行的。分数据、将模型的副本传到其他模型，
收集每块卡的输出、计算loss,将loss传到每块卡上，在每块卡上进行反向传播得到梯度后，
收集每块卡上的梯度，进行reduce上操作后，传到其他卡上。
这样使得0号卡所占的内存比较多，使得内存使用不均衡，而且会经常出现其他卡等待0好卡计算的情形。

pytorch的分布式数据并行DDP训练框架如下：
1. 从master(一般是GPU0)从磁盘或者合页内存中取数据。
2. 所有GPU同时去取数据，不需要GPU0去分
3. 每块GPU单独进行前向计算
4. 每块GPU单计算loss
5. 每块GPU单独进行反向过程，计算出参数的梯度，并进行参数之间的传递(计算和参数传递间存在交叉过程)
6. 在GPU0上进行梯度的allreduce操作，然后将梯度传递到其他GPU上。
7. 每个GPU单独地进行参数更新

可以看出DDP只有在梯度收集的时候，存在GPU间的通信，其余的操作都是在各自的GPU上进行的，这样可以均衡负载，也可以节省时间。

rank：用于表示进程的编号/序号（在一些结构图中rank指的是软节点，rank可以看成一个计算单位），每一个进程对应了一个rank的进程，整个分布式由许多rank完成。
node：物理节点，可以是一台机器也可以是一个容器，节点内部可以有多个GPU。
rank与local_rank： rank是指在整个分布式任务中进程的序号；local_rank是指在一个node上进程的相对序号，local_rank在node之间相互独立。（注意：在代码中，会使用local_rank来指定GPU，并且local_rank和实际的gpu编号存在映射关系，比如，指定gpu 4,5进行训练，local_rank仍然是0,1，但前提是要先设置os.environ['CUDA_VISIBLE_DEVICES'] = "4,5"）。
nnodes、node_rank与nproc_per_node： nnodes是指物理节点数量，node_rank是物理节点的序号；nproc_per_node是指每个物理节点上面进程的数量。
world size ： 全局（一个分布式任务）中，rank的数量。

单机多卡 python -m torch.distributed.run --nproc_per_node 2 --master_port 2393 ddp_train.py
多机多卡
在机器0上运行python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.0.0.1" --master_port=1234 ddp.py
在机器1上与运行python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="192.0.0.1" --master_port=1234 ddp.py
不同机器上的node_rank不同，其余相同

本脚本的各步骤为：
1. 根据local_rank参数，设定本进程的执行设备。
    如果local_rank==-1,则设定进行执行设备为当前设备；
    否则，根据参数指定设备，并启动deepspeed初始化。
    而后获取本进程的全局rank.
2. 获取deepspeed的配置信息
    # 其中train_micro_batch_size_per_gpu根据args.per_device_train_batch_size重设
    # train_batch_size根据args.per_device_train_batch_size，总进程数world_size、梯度更新步的乘积重设
    # 故ds_config的train_batch_size是每次模型全局更新梯度的batch_size
3. 调用torch.distributed.barrier进行同步通信
4. 加载hf的tokenizer和model
5.根据lora相关参数确定将模型修改为lora形式，及是否只更新lora的参数
6. 分割prompt数据为训练集和测试集
7. 根据local_rank参数将train、eval数据组装为DataLoader
8. 定义评估函数，以损失的exp为评估指标
9. 定义待更新参数，优化器（DeepSpeedCPUAdam 或 FusedAdam），scheduler
10. 调用deepspeed.initialize将模型修改为deepspeed的模型\优化器\scheduler
    注意deepspeed.initialize会将optimizer,scheduler等作为model的方法，
    因此在训练中，optimizer,scheduler不会再使用，所有操作都由model
11. 在训练模型前，先评估一次perplexity
12. 以标准pytorch模式训练模型
13. 每次训练后都评估模型的perplexity
14. 保存模型
"""
import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    #? weight decay是啥-> weight decay (L2 penalty) (default: 0)
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    #? 执行反向/更新前累计更新步
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
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
    #? lr scheduler的warmup步
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    # Pytorch并行主要有两种方式，DataParallel（DP）和DistributedDataParallel（DDP）。
    # DP方式较为简单，但是多线程训练，并且主卡显存占用比其他卡会多很多。
    # DDP是多进程，将模型复制到多块卡上计算，数据分配较均衡。
    # local_rank参数赋值给一个分布式进程组组内的每个进程的唯一识别，
    # 例如机器一上有0,1,2,3,4,5,6,7，机器二上也有0,1,2,3,4,5,6,7。local_rank在node之间相互独立。
    #? local_rank =-1即只有一个GPU或没有GPU

    # 用于表示进程的序号，用于进程间通信。每一个进程对应了一个rank。
    # rank=0的进程就是master进程。单机多卡时，rank就等于local_rank

    # world size 全局的并行数。全局（一个分布式任务）中，rank的数量。
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args


def main():
    args = parse_args()
    #* 1. 设置当前进程的执行设备
    # 如果local_rank==-1，则设定执行设备为当前设备，否则设定执行设备为指定设备
    if args.local_rank == -1:
        device = torch.device("cuda") # 设定device为当前设备
    else:
        # 如果local_rank不为-1，则设置当前进程的执行设备，并启动deepspeed的分布式初始化
        #? 如果执行设备与当前设备一致，则无需再启动deepspeed分布式初始化
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    # 根据torch.distributed.get_rank()设定全局rank
    args.global_rank = torch.distributed.get_rank()

    #* 2. 获取deepspeed的配置信息
    # 其中train_micro_batch_size_per_gpu根据args.per_device_train_batch_size重设
    # train_batch_size根据args.per_device_train_batch_size，总进程数world_size、梯度更新步的乘积重设
    # 故ds_config的train_batch_size是每次模型全局更新梯度的batch_size
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    #  Pytorch在分布式训练过程中，对于数据的读取是采用主进程预读取并缓存，
    # 然后其它进程从缓存中读取，不同进程之间的同步通信需要通过torch.distributed.barrier()实现
    # 主要就是通过对其他进程进行阻塞来等所有的进程的计算都完毕之后在进行后续的计算。
    #* 3. 调用torch.distributed.barrier进行同步通信
    torch.distributed.barrier()
    #* 4. 加载hf的tokenizer和model
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # 指定填充标记(pad_token)使用结束标记(eos_token)。pad_token 是 tokenizer 中用于补足输入序列长度的填充标记,默认是 [PAD]。
    # eos_token 是 tokenizer 中用于表示序列结束的标记,默认是 [SEP]。
    # 所以,这个设置就是指定我们使用 [SEP] 标记来进行补充填充,而不是默认的 [PAD] 标记。
    tokenizer.pad_token = tokenizer.eos_token
    # make sure tokenizer is right pad in our logic
    tokenizer.padding_side = 'right'
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            disable_dropout=args.disable_dropout)
    #* 5.根据lora相关参数确定将模型修改为lora形式，及是否只更新lora的参数
    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    # Prepare the data
    #* 6. 分割prompt数据
    train_phase = 1
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
        sft_only_data_path=args.sft_only_data_path)
    #* 7. 根据local_rank参数将train、eval数据组装为DataLoader
    # DataLoaders creation:
    # 如果执行设备为当前设备，则调用RandomSampler采样训练数据，调用SequentialSampler采样评估数据
    # 如果执行设备不为当前设备，则调用DistributedRandomSampler采样训练数据，调用DistributedSequentialSampler采样评估数据
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    #* 8. 定义评估函数，以损失的exp为评估指标
    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            # 定义损失的指数为perplexity，最终返回也是perplexity
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity
    #* 9. 定义待更新参数，优化器，scheduler
    # Split weights in two groups, one with weight decay and the other not.
    # 返回一个两个字典的列表，每个字典都是存在两个字段params,weight_decay
    # params都是需要更新梯度的参数
    # 第一weight_decay字段的值为args的参数，第二个weight_decay的值为0.
    # weight decay (L2 penalty) (default: 0)
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
    # 根据梯度累积步计算每个epoch的更新步
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    # 根据更新步和训练epoch设定scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    #* 10.调用deepspeed.initialize将模型修改为deepspeed的模型\优化器\scheduler
    #? 但优化器\scheduler在后续未曾调用-> deepspeed.initialize会将optimizer,train_dataloader,model_prams,lr_scheduler
    #? 作为model的方法,因此optimizer、lr_scheduler都在模型内执行
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    # 是否启动保存梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    #* 11. 在训练模型前，先评估一次perplexity
    # 只在global_rank=0,-1的设备上打印信息
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    perplexity = evaluation(model, eval_dataloader)
    print_rank_0(f"ppl: {perplexity}", args.global_rank)
    #* 12. 以标准pytorch模式训练模型
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            model.step()

        # Evaluate perplexity on the validation set.
        #* 13. 每次训练后都评估模型的perplexity
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        perplexity = evaluation(model, eval_dataloader)
        print_rank_0(f"ppl: {perplexity}", args.global_rank)
        # 每次计算和评估后更新模型更新的次数
        model.tput_timer.update_epoch_count()
    #* 14. 保存模型
    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        #? 如果没使用lora还会有用吗
        model = convert_lora_to_linear_layer(model)
        # 在global_rank==0的机器上保存模型，防止多次保存
        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
