# 侠客行一

## 1.推荐

在推荐系统中常使用的方案和算法，实际的工业推荐系统，一般会有四个环节：召回、粗排、精排、重排

### 1.1 用户画像

[User profile](user_profile/用户画像.md)

### 1.2 召回

实现召回层有三个技术方案：简单快速的单策略召回、业界主流的多路召回、深度学习常用的Embedding召回

基于[Embedding](recommend/Embedding召回.md)召回：最核心的召回模式是I2I、U2I

- I2I
  - 文本类型
    - 静态词向量表征
      - [Word2vec](recommend/word2vec模型.html)
      - FastText
    - 动态词向量表征
      - Bert、ELMo、GPT
  - 图片视频
    - CNN、RNN、Transformer
- U2I
  - 矩阵分解
    - ALS
  - 浏览序列
    - [DSSM](recommend/DSSM模型.html)
    - Bert+LSTM
  - 图神经网络
    - DeepWalk
    - GraphSage

### 1.3 排序

[DeepFM](recommend/DeepFM.html)



## 2.NLP

## 3.神经网络

### 3.1 深度学习模型调参方法

[深度学习Batch size大小对训练过程的影响](tuning_parameter/深度学习Batch size调参.html)

[优化器Optimizer](tuning_parameter/优化器Optimizer.html)

## 4.数据结构和算法

[Data Structure && Algorithm](algorithm/DirIndex.md)

## 5.你能撕多久
[问题](interview/interview.html)

<center><big><b>赞赏码</b></big></center>

------

<center><b>写文不易，欢迎各种花式赞赏：</b></center>																					

<center><img src="image/wxzs.jpeg" alt="赞赏" width="50%"/></center>

















