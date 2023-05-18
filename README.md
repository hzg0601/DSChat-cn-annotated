# DeepSpeed Examples

This repository contains various examples including training, inference, compression, benchmarks, and applications that use [DeepSpeed](https://github.com/microsoft/DeepSpeed).

## 1. Applications

This folder contains end-to-end applications that use DeepSpeed to train and use cutting-edge models.

### step1的训练逻辑step1\_supervised\_finetuning/main.py:

本脚本的各步骤为：

1. 根据local\_rank参数，设定本进程的执行设备。

   如果local\_rank==-1,则设定进行执行设备为当前设备；

   否则，根据参数指定设备，并启动deepspeed初始化。

   而后获取本进程的全局rank.
2. 获取deepspeed的配置信息

   其中train\_micro\_batch\_size\_per\_gpu根据args.per\_device\_train\_batch\_size重设

   train\_batch\_size根据args.per\_device\_train\_batch\_size，总进程数world\_size、梯度更新步的乘积重设

   故ds\_config的train\_batch\_size是每次模型全局更新梯度的batch\_size
3. 调用torch.distributed.barrier进行同步通信
4. 加载hf的tokenizer和model

5.根据lora相关参数确定将模型修改为lora形式，及是否只更新lora的参数

1. 分割prompt数据为训练集和测试集
2. 根据local\_rank参数将train、eval数据组装为DataLoader
3. 定义评估函数，以损失的exp为评估指标
4. 定义待更新参数，优化器（DeepSpeedCPUAdam 或 FusedAdam），scheduler
5. 调用deepspeed.initialize将模型修改为deepspeed的模型\优化器\scheduler

   注意deepspeed.initialize会将optimizer,scheduler等作为model的方法，

   因此在训练中，optimizer,scheduler不会再使用，所有操作都由model
6. 在训练模型前，先评估一次perplexity
7. 以标准pytorch模式训练模型
8. 每次训练后都评估模型的perplexity
9. 保存模型

### step2的训练逻辑step2\_reward\_model\_finetuni/main.py:

脚本的流程基本如step1的训练流程，只是更换了模型的输出和评估标准，注释详见step1的main.py
其核心为RewardModel类：
RewardModel应为transformers的BaseModel,其返回值的一个为hidden\_states

输入必须自带正负样本，前一半为正样本，后一半为负样本
将最后一层映射为一个标量，然后将输出按正负样本分开；

（似乎与原始InstructGPT有异：在原始InstructGPT中，
以由标注人员的排序为全部样本集，将任意两个prompt构成一组，
靠前的一个奖励值减去靠后的一个奖励值作为损失函数）

如果正样本与负样本对应位置间没有重复：
则以num\_padding位置或seq\_len位置的得分为
chosen和reject得分，以二者的差的sigmoid.mean()为损失。
如果正负样本对应位置间存在重复：
则以(第一个重合的位置) 到 (num\_padding的位置或seq\_len的较大者)的值为chosen和reject的得分，
以二者差的sigmoid.mean()为损失。
训练时返回loss, chosen\_score,reject\_score;
预测时返回标量值或(标量值+chosen\_scores)

### step3的训练逻辑step3\_flhf\_finetuning:

准备阶段：

1. 定义分布式训练的device,rank,同步各端通信;
2. 基于actor模型加载tokenizer，构造训练数据;
3. 调用DeepSpeedRLHFEngine构造RLHF的deepspeed engine，其包含actor,ref,critic,reward四个engine;
4. 基于RLHF的engine，调用PPOTrainer(PPOTrainerUnsupervised)类构造ppo(ppp-ptx)的训练实例
5. 定义MiniDataset实例，每次搜集generation\_batch\_numbers个mini-batch,
   然后将它们统一分割为per\_device\_mini\_train\_batch\_size大小的micro-batch;

训练阶段：

1. 对unsupervised的数据进行micro数据切分；
2. 调用ppotrainer的generate\_experience方法，
   根据给定的prompts和mask，先调用actor模型生成answer和mask,然后计算answer在actor,ref模型下的logit;
   计算answer在critic\_model下的最后一个字符全部序列的得分，在reward\_model模型下chosen序列最后一个实际字符的得分，
   返回全部中间及最终结果；
3. 调用MiniDataset类，将generate\_experience返回的mini-batch继续分割为micro-batch；
   **在将mini-batch的个数搜集到max\_size个时才开始ppo训练**
4. 根据args启动actor模型的梯度检查点方法；
5. 根据args.定义的ppo训练epochs训练ppo模型：
   11\. 对每个micro-batch调用ppotrainer的train\_rlhf方法训练模型，并返回actor\_loss和critic\_loss；
   12.调用ppotrainer的train\_unsupervised针对unsupervised数据训练actor模型，并返回unsup\_loss；

   1. 根据args.ema参数启动移动平均方法更新模型权重；
6. ppo\_epochs训练完毕后，调用all\_reduce方法计算全局平均奖励；
7. ppo\_epochs训练完毕后，根据args.actor\_gradient\_checkpointing参数在训练完毕后关闭actor的梯度检查点功能；
8. 根据lora、ema、actor\_zero\_stage等参数保存actor,critic,actor\_ema模型。

该脚本的核心是PPO训练类，该类的主方法为train_rlhf,主要流程如下：

    0. 前置调用generate_experience方法，令actor,ref,reward,critic模型生成:
    answer_seq, 
    actor针对answer_seq除第一个外所有token的logit:log_probs,
    ref针对answer_seq除第一个外所有token的logit:ref_log_probs,
    reward针对answer_seq最后一个非padding token的得分:rewards，
    critic针对answer_seq去除最后一个token（bos_token）的原始得分:values, 重命名为old_values
    
    1. 首先根据log_probs和ref_log_probs, reward调用compute_reward计算公式2的PPO目标损失
        首先计算actor_model和reference_model针对answer的嵌入的logits的差，乘以-\beta 即KL奖励系数
        然后加上每个answer最后一个非padding token的得分，即文中的公式2,作为actor模型的奖励
    2. 然后调用get_advantages_and_returns计算PPO模型更新所需的advantages, returns；
        advantage函数衡量的是从某个状态s_t出发，自主选择一个动作比根据策略抽取一个动作所带来的奖励有多大，
        即状态-动作值函数和状态函数的差. critic模型针对answer_seq去除bos_token作为每个原模型的值函数，
        值函数+动作奖励，即得到reward序列
    3. 然后再调用actor模型计算对answer_seq除第一个外所有token的logit:actor_log_probs，作为新模型的输出
    4. 基于log_probs, actor_log_probs, andvantage计算actor模型的损失函数；
    5. 再调用critic模型计算answer_seq除最后一个外所有token的得分values，
    6. 基于values, old_values, return计算critic模型的损失函数；
    7. 更新actor critic模型。

## 2. Training

There are several training and finetuning examples so please see the individual folders for specific instructions.

## 3. Inference

The DeepSpeed Huggingface inference [README](./inference/huggingface/README.md) explains how to get started with running DeepSpeed Huggingface inference examples.

## 4. Compression

Model compression examples.

## 5. Benchmarks

All benchmarks that use the DeepSpeed library are maintained in this folder.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [https://cla.opensource.microsoft.com](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact <opencode@microsoft.com> with any additional questions or comments.
