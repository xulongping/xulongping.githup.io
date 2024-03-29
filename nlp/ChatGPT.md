<center><font size='60'>ChatGPT</font></center>

## 1.特点

## 2.原理

## 3.技术架构

### 3.3 ChatGPT的训练过程

​		ChatGPT的训练过程分为三个阶段：训练监督策略模型、训练奖励模型（Reward Model， RM）、采用PPO（Proximal Policy Optimization，近端策略优化）强化学习来优化策略。

#### 3.3.1 训练监督策略模型

​		GPT3.5本身很难理解人类不同类型指令中蕴含的不同意图，也很难判断生成内容是否高质量的结果。为了让GPT3.5初步具备理解指令的意图，首先会在数据集中随机抽取问题，由人类标注人员，给出高质量答案，然后用这些人工标注好的数据来微调GPT-3.5模型（获得SFT模型，Supervised Fine-Tuning）。

​		此时的SFT模型在遵循指令/对话方面已经优于GPT-3，但不一定符合人类偏好。

#### 3.3.2 训练奖励模型（Reward Model， RM）

​		这个阶段的主要是通过人工标注训练数据（约33K个数据），来训练回报模型。在数据中随机抽取问题，使用第一阶段生成的模型，对于每个问题，生成多个不同的回答。人类标注者对这些结果综合考虑给出排名顺序。这一过程类似于教练或老师辅导。

​		接下来，使用这个排序结果数据来训练奖励模型。对多个排序结果，两两组合，形成多个训练数据对。RM模型接受一个输入，给出评价回答质量的分数。这样，对于一对训练数据，调节参数使得高质量回答的打分比低质量的打分要高。

#### 3.3.3 采用PPO强化学习来优化策略

​		PPO（Proximal Policy Optimization）近端策略优化，PPO的核心思路在于将Policy Gradient中On-policy的训练过程转化为Off-policy，即将在线学习转为离线学习，这个转化过程被称为Importance Sampling。这一阶段利用第二阶段训练好的奖励模型，靠奖励打分来更新新预测模型参数。在数据集中随机抽取问题，使用PPO模型生成回答，并用上一阶段训练好的RM模型给出高质量分数。把回报分数依次传递，由此产生策略梯度，通过强化学习的方式以更新PPO模型参数。

​		如果不断重复第二和第三阶段，通过迭代，会训练出更高质量的ChatGPT模型。

































