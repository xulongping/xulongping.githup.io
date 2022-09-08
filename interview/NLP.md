<center><font size='60'>NLP</font></center>

[TOC]



## 2. NLP

### 2.1 理论

### 2.2 模型

####  2.2.1 Transformer

**1.Transformer模型结构**

![transformer网络结构](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/transformer网络结构.jpg)

​	Transformer是一个encoder-decoder结构，由若干个编码器和解码器堆叠形成。左侧部分为编码器，由Multi-Head Attention和一个全连接组成，用于输入语料转化成特征向量。右侧部分是解码器，其输入为编码器的输出及已经预测的结果，由Masked Multi-Head Attention，Multi-Head Attention以及一个全连接组成，用于输出最后结果的条件概率。



2. **attention机制**



#### 2.2.2 Bert（Bidirectional Encoder Representations from Transformer）

**1. Bert模型目标是什么？**

​		从名字中可以看出，BERT模型的目标是**利用大规模无标注预料训练、获得文本的包含丰富语义信息的Representation**，即：文本的语义表示，然后将文本的语义表示在特定NLP任务重作微调，最终应用于该NLP任务。



**2. Bert模型结构**

![Bert网络结构](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/Bert网络结构.jpg)

​		BERT基于**Transformer的双向编码**表示，它是一个预训练模型，模型训练时的两个任务是**预测句子中被掩盖的词**以及**判断输入两个句子是不是上下句**。在预训练好的BERT模型后面根据特定任务加上相应的网络，可以完成NLP的下游任务，比如文本分类、机器翻译等。

​		BERT网络结构是由输入Embedding，多层双向的Transformer block（Transformer 左侧encoder单元）连接：

- Embedding：wordpiece token embedding + segment embedding + position embedding

  - wordpiece embedding：单词本身的向量表示
  - position embedding：单词位置信息编码成特征向量
  - segment embedding：区分两个句子的向量表示

- Transformer endcoder

  一个Transformer的encoder单元由一个multi-head-Attention + Layer Normalization + Feed Forword + Layer Normalization 叠加产生。

- 模型层数

  $BERT_{base}$ ：L=12， H=768，A=12，参数总量110M

  $BERT_{large}$ ：L=24， H=1024，A=16， 参数总量340M

  L表示网络的层数（即Transformer blocks的数量），A表示Multi-Head Attention中self-Attention的数量，H表示词向量的维度（隐藏层数）。

  

**3. Self-Attention出现的原因**

​		a.为了解决RNN、LSTM等常用于处理序列化数据的网络结构无法在GPU中并行加速计算的问题

​		b.由于每个目标词是直接与句子中所有词分别计算相关度（attention）的，所以解决了传统的RNN模型中长距离依赖的问题。通过attention，可以将两个距离较远的词之间的距离拉近为1直接计算词的相关度，而传统的RNN模型，随着距离的增加，词之间的相关度会被削弱。



**4. Bert模型输入**

​		X=(batch_size, max_len, embedding)，假设batch_size=1，输入的句子长度为512，每个词的向量表示的长度为768，那么整个模型的输入就是一个512 * 768的tensor



**5. 单个self-attention的计算过程**

​		![self-attention计算过程](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/self-attention计算过程.jpg)

​		self-attention的计算涉及到三个中间权重矩阵$W_q, W_k, W_v$，他们分别对输入X进行线性变换，生成query、key、value这三个新的tensor，整个计算步骤如下：

​		step1:  输入X分别与$W_q, W_k, W_v$矩阵相乘，得到Q，K，V

​			q：query（to match others），它要去match的

​			k：key（to be matched），用来被q match的

​			v：value（information to be extracted），要被抽取出来的information

​		step2:  拿每个query去对每个key做attention，按比例缩小的点积（Scaled Dot-Product Attention）  $\frac{Q * K^T}{\sqrt{d}}$得到x中各个词之间的相关度（d is the dim of q and k）。

​					$\alpha_{1, i} = \frac{q^1 * k^i }{\sqrt{d}}$

​		attentuon有许多算法，本质是两个向量，输出一个分数，这个分数表明两个向量有多匹配

​		step3:  将step2相关度通过softmax函数归一化，得到归一化后各个词与其他词的相关度。

​					$\hat{\alpha_{1, i} }= \frac{exp(\alpha_{1, i})}{\sum_{j}exp(\alpha_{1, j})}$

​		step4:  将step3相关度矩阵与V相乘，即加权求和，得到每个词新的向量编码

​					$ b^1= \sum_{i}\hat{\alpha_{1, i} } v^i$

​	self-attention输入是一个sequence，输出也是sequence，可以并行计算

​					$Attention(Q, K, V) = Softmax(\frac{QK^T}{\sqrt{d_k}})V$



**6. 矩阵维度看self-attention计算过程**

​		在Bert base模型中，每个head的神经元个数64，12个head总的神经元的个数即为786，也就是H=786。单个$W_q, W_k, W_v$都是768 × 64的矩阵，X(512 × 768)  *   $W_q$(768 × 64) = 512 *×64，那么Q，K，V则都是512 × 64的矩阵，Q(512 × 64) * $K^T$(64 × 512) = 512 × 512，归一化后跟V相乘后的z矩阵的大小则为512 × 64。

​		12个attention则是将12个512 × 64大小的矩阵横向concat，得到一个512 × 768大小的多头输出，这个输出再接一层768的全连接层，最后就是整个multi-head-attentuin的输出。



**7. multi-head attention的计算**

​	Multi-Head Self-Attention将多个不同单头的Self-Attention输出Concat成一条，然后再经过一个全连接层降维输出。

​		例如：一个self-attention计算的输出为output_0 = (batch_size, max_len, w_length)，那么n个attention进行concat之后，输出就为output_sum = (batch_size, max_len, n * w_length)，这个concat的结果再连一层全连接层即为整个multi-head attention的输出。



**8. 为什么选择Layer Normalization而不是Batch Normalization？**

​		此时，我们应该先对我们的数据形状有个直观的认识，当一个batch的数据输入模型的时候，形状是长方体如图所示，大小为(batch_size, max_len, embedding)，其中batch_size为batch的批数，max_len为每一批数据的序列最大长度，embedding则为每一个单词或者字的embedding维度大小。而Batch Normalization是在batch间选择同一个位置的值做归一化，相当于是对batch里相同位置的字或者单词embedding做归一化，**Layer Normalization是在一个Batch里面的每一行做normalization，相当于是对每句话的embedding做归一化**。显然，LN更加符合我们处理文本的直觉。

<img src="/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/ayer normal和batch normal的对比.png" alt="ayer normal和batch normal的对比" style="zoom:50%;" />



**9. masked language model**

​		随机掩盖掉一些单词，然后通过上下文预测该单词。Bert中有15%的wordpiece token会被随机掩盖，这15%的token中80%用[MASK]来代替，10%用随机的一个词来代替，10%保持这个词不变。这种设计使得模型具有捕捉上下文关系的能力，同时能够有利于token-level tasks，例如序列标注。



**10. 为什么选中的15%的wordpiece token不能全部用 [MASK]代替，而要用 10% 的 random token 和 10% 的原 token**

​		[MASK]是以一种显式的方式告诉模型『这个词我不告诉你，你自己从上下文里猜』，从而防止信息泄露。如果[MASK] 以外的部分全部都用原token，模型会学到『如果当前词是[MASK]，就根据其他词的信息推断这个词；如果当前词是一个正常的单词，就直接抄输入』。这样一来，在finetune阶段，所有词都是正常单词，模型就照抄所有词，不提取单词间的依赖关系了。

​		以一定的概率填入random token，就是让模型时刻堤防着，在任意token的位置都需要把当前token的信息和上下文推断出的信息相结合。这样一来，在finetune阶段的正常句子上，模型也会同时提取这两方面的信息，因为它不知道它所看到的『正常单词』到底有没有被动过手脚的。

​		**在后续微调任务中语句中并不会出现[MASK]标记，而且这么做的另一个好处是：预测一个词汇时，模型并不知道输入对应位置的词汇是否为正确的词汇（10%概率），这就迫使模型更多地依赖上下文信息去预测词汇，并且赋予模型一定的纠错能力。**



**11. 最后怎么利用[MASK] token做的预测？**

​		最终的损失函数只计算被mask掉的token的，每个句子里[MASK]的个数是不定的。实际代码实现是每个句子有一个 maximum number of predictions，取所有[MASK]的位置以及一些PADDING位置的向量拿出来做预测（总共凑成 maximum number of predictions这么多个预测，是定长的），然后再用掩码把PADDING盖掉，只计算[MASK]部分的损失。



**12. 模型特点**

​		使用transformer作为算法的主要框架，transformer能更彻底的捕捉语句中的双向关系

​		使用预测句子中被掩盖的词和判断输入两个句子是不是上下句多任务训练目标，是一个自监督的过程，不需要数据标注。

​		使用tpu这种强大的机器训练大规模语料，使NLP的很多任务达到全新的高度



**13. 可优化空间**

​		如何让模型捕捉token序列关系的能力，而不是简单依靠位置嵌入

​		模型太大，太耗机器



**14.残差连接（ResidualConnection）**

​		将模块的输入与输出直接相加，作为最后的输出。这种操作背后的一个基本考虑：修改输入比重构整个输出更容易（“锦上添花”比“雪中送碳”容易多了！）。这样一来，可以使网络更容易训练。



**15.线性转换**

​		对每个字的增强语义向量再做两次线性变换，以增强整个模型的表达能力。变换后向量与原向量保持长度相同。



**16.单文本分类任务**

​		Bert模型在文本前插入一个[CLS]符号，并将该符号对应的输入向量作为整篇文本的语义表示，用于文本分类，如下图，可以理解为：与文本中已有的其它字/词相比，这个无明显语义信息的符号会更“公平”地融合文本中各个字/词的语义信息。

<img src="/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/单文本分类任务.jpeg" alt="单文本分类任务" />



**（17）语句对分类任务**

​		该任务的实际应用场景包括：问答（判断一个问题与一个答案是否匹配）、语句匹配（两句话是否表达同一个意思）等。对于该任务，Bert模型出来添加[CLS]符号并将对应的输出作为文本的语义表示，还对输入的两句话用一个[SEP]符号作为分割，并分别对两句话附加两个不同的文本向量以作区分。

<img src="/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/语句对分类任务.jpeg" alt="语句对分类任务" style="zoom:80%;" />



**（18）序列标注任务**

​		该任务的实际应用场景包括：中文分词&新词发现（标注每个字时词的首字、中间字或末字）、答案抽取（答案的起止位置）等。对于该任务，Bert模型利用文本中每个字对应的输出向量对该字进行标注（分类），如下图（B、I、E分别表示一个词的第一个字、中间字和最后一个字）。

![序列标注任务](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/序列标注任务.jpeg)

（4）Bert、GPT、ELMo的区		

#### 2.2.3 JointBert

1. **为什么选择JointBert**

   ​		意图分类和插槽填充是自然语言的两个基本任务。他们经常受到小规模的人工标签训练数据的影响，导致泛化能力差，尤其是对于低频单词。基于Bert的联合意图分类和插槽填充模型，旨在解决传统NLU模型泛化能力差的问题。实验结果表明，联合Bert模型优于分别建模意图分类和插槽填充的Bert模型，证明了利用这两个任务之间关系的有效性。

   

2. **JointBert模型是如何实现的？**

   先简要描述Bert模型，然后介绍基于Bert的联合模型：

   <img src="/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/JointBert网络结构.png" alt="JointBert网络结构" style="zoom:50%;" />

   （1）Bert

   ​		Bert模型架构是基于Transform model中的多层双向Transformer encoder编码器。输入表示形式是WordPiece embedding + position embedding + segment embedding。特别地，对于单句分类和标记任务，segment embedding没有区别，特殊embedding([CLS])作为第一个token，embedding([SEP])作为最终token。给定输入序列$x= (x_1, ..., x_T)$，Bert的输出为$H = (h_1, ..., h_T)$。

   ​		Bert模型在大型未标记文本上使用两个策略进行预测训练，即**屏蔽语言模型**和**下一句预测**。预训练的Bert模型提供了功能强大的上下文相关语句表示形式，可通过微调过程用于各种目标任务，例如意图分类和空位填充，类似于用于其他NLP任务的方式。

   （2）Joint Intent Classification and Slot Filling

   ​		Bert可以轻松扩展到意图分类和插槽填充联合模型。基于第一个特殊token([CLS])的隐藏状态，其表示为$h_1$，其意图预测为：

   ​								$y^i = softmax(W^ih_1 + b^i)$,                     (1)

   ​		对于插槽填充，我们为其他token的隐藏状态$h_2, ..., h_T$准备了softmax layer，以对插槽填充标签进行分类。为了使该过程与workPiece tokenization兼容，我们将每个标记化的输入词送入WordPiece标记器，并使用与第一个子标记相对应的隐藏状态作为softmax分类器的输入。

   ​								$y_n^s = softmax(W^sh_n + b^s), n \in 1 ... N$         (2)

   ​		$h_n$是与单词$x_n$的第一个子标记相对应的隐藏状态。

   ​		为了联合建模意图分类和插槽填充，目标公式：

   ​								$p(y^i, y^s|x) = p(y^i|x)\prod_{n=1}^Np(y_n^s|x)$                 (3)

   ​		学习目标是最大化条件概率p，通过最小化交叉熵损失来对模型进行端到端微调。

   

3. **JointBert模型效果**

   下图为Snips和ATIS数据集的模型性能，如果插槽填充F1，意图分类准确率和句子级语义框架精度。

   ![JointBert模型效果](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/JointBert模型效果.png)

   ​		联合Bert+CRF用户CRF取代了softmax分类器，它的性能与Bert相当，这可能是由于Transformer中的自注意机制可能已经对标签结构进行了充分建模。

4. JointBert是如何实现的？

5. 

6. 

#### 2.2.4 CRF

**3.CRF和HMM区别**

（1）HMM

​		HMM是有向图，HMM是生成式模型，对P(X, Y)建模

​		HMM做了下面相关假设：

​			一阶马尔科夫假设、隐藏状态假设、转换函数稳定性假设

（2）CRF

​		CRF是无向图，是判别式模型，对P(Y|X)建模

​		CRF能提取到更多上下文信息

#### 2.2.5 Rase Core

1. 



​	自然语言处理技术近几年发展非常快，像BERT、GPT-3、图神经网络、知识图谱等技术被大量应用于项目实践中。经常会被揪着细节一步一步让你解释：“**为什么这么做？效果如何？你如何调整模型，你思考的逻辑是什么**？”

​	“说说自己在项目中具体负责的模块中用到的技术细节，遇到了什么问题？你使用的模型的损失函数、如何优化、怎么训练模型的、用的什么数据集？优化算法的选择做过哪些？为啥这么做？”	

​	我们罗列了一些常见的大厂NLP项目深度考察问题：

- BERT模型太大了，而且效果发现不那么好比如next sentence prediction, 怎么办？
- 文本生成评估指标，BLUE的缺点
- loss设计 triplet loss和交叉熵loss各自的优缺点，怎么选择
- attention机制
- ernie模型
- 介绍一下flat及对于嵌套式语料的融合方式
- 为什么使用lightGBM，比起xgboost的优点是什么
- 样本不均衡问题的解决办法有哪些？具体项目中怎么做的？
- 长文本的处理
- 引入词向量的相似性对于结果有什么不好的影响
- 如何引入知识图谱
- 词向量中很稀疏和出现未登录词，如何处理
- kmeans的k怎么选择
- 新词发现怎么做
- 模型选取、数据增强
- 从数据标注的制定标准，到选取模型，再到改进模型、错误分析
- NER数据中没有实体标注的句子过多解决方式
- 同一句话两个一样字符串如何消岐
- 模型好坏的评估,如何衡量模型的性能
- 方面级情感分析的模型结构
- 模型学习中，正负样本的训练方式不同有什么影响
- 减轻特征工程的手段



























