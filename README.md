# 侠客行一

# 1.推荐

在推荐系统中常使用的方案和算法，实际的工业推荐系统，一般会有四个环节：召回、粗排、精排、重排

## 1.1 用户画像

[User profile](user_profile/用户画像.html)

## 1.2 召回

实现召回层有三个技术方案：简单快速的单策略召回、业界主流的多路召回、深度学习常用的Embedding召回

基于[Embedding](recommend/Embedding召回.html)召回：最核心的召回模式是I2I、U2I

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

# 2.NLP

# 3.神经网络

## 									创作不易，来点赞赏

<img src="image/wxzs.jpeg" alt="赞赏" style="zoom:33%;" />

















