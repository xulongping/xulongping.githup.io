# 1. DSSM模型原理

## 1.1 DSSM模型简介

​	DSSM模型的全称是Deep Structured Semantic Model，由微软研究院开发，利用深度神经网络把文本（句子，Query，实体等）表示成向量，应用于文本相似度匹配场景下的算法。

​	DSSM模型在信息检索、文本排序、问答、图片描述、机器翻译等中有广泛的应用。该模型是为了衡量搜索的关键词和被点击的文本标题之间的相关性。

​	DSSM模型的原理比较简单，通过搜索引擎里Query和Document的海量点击曝光日志，用DNN深度网络把Query和Document表达为低纬语义向量，并通过余弦相似度来计算两个语义向量的距离，最终训练出语义相似度模型。该模型既可以用来预测两个句子的语义相似度，又可以获得某句子的低纬语义Embedding向量。



## 1.2 模型结构

![img](https://upload-images.jianshu.io/upload_images/7055779-60b0aa512a9af215.png?imageMogr2/auto-orient/strip|imageView2/2)

​	DSSM模型的整体结构图：Q代表Query信息，D表示Document信息

- Term Vector

  表示文本embedding向量

- Word Hashing

  为了解决Term Vector太大问题，对bag-of-word向量降维

  使用word hashing方法将句子50w的one-hot表示降低到了3w，原理是对句子做letter level的trigrim并累加

  如下图：#boy#会被切成#-b-o, b-o-y, o-y#

  <img src="https://upload-images.jianshu.io/upload_images/7055779-21ac94f454da8873.png?imageMogr2/auto-orient/strip|imageView2/2" alt="img" style="zoom:100%;" />

  选用trigrim而不用bigrim或者unigrim的原因是为了权衡表示能力和冲突，两个单词冲突表示和两个单词编码后的表示完全相同。

- Multi-layer nonlinear projection

  表示深度学习网络的隐藏层

  第二层到第四层是典型的MLP网络，最终得到128维的句子表示

  ​					$l_1=W_1x$

  ​					$l_i = f(W_il_{i-1} + b_i),  i = 2,...,N-1$

  ​					$y = f(W_Nl_{N-1} + b_N)$

  激活函数是tanh

  ​					$f(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$

- Semantic feature

  表示query和document最终的embedding向量

- Relevance measured by cosine similarity

  表示计算query与document之间的余弦相似度

  对正负样本计算cosine距离

  ​					$R(Q,D) = cosine(y_Q, y_D)=\frac{y_Q^Ty_D}{||y_Q||  ||y_D||}$

- Posterior probability computed by sofamax

  表示通过softmax函数把query与正样本document的语义相似性转化为一个后验概率

  ​					$P(D^+|Q)=\frac{exp(\gamma R(Q,D^+))}{\sum_{D` \in D}exp(\gamma R(Q,D^+))}$

  其中：$\gamma$为softmax的平滑因子，$D^+$为query下的正样本，$(D^, - D^+)$为query的随机采取的负样本，D为query下的整个样本空间。

  在训练阶段，通过极大似然估计，最小化损失函数：

  ​					$L\Lambda = -log\prod_{(Q,D_+)}p(D^+|Q)$

























