# 1.Embedding概念

​	推荐系统的召回阶段，需要对每个用户和每个被推荐物品做数学层面的表示，目前比较主流的方法是通过向量，也就是Embedding表示。

​	举例：假设傲海和两个物品的Embedding表示法如下：

​				傲海 = [1, 32, 53, 657, 863]

​				物品1=[32, 53,46, 75, 68]

​				物品2=[2, 32, 53, 657, 863]

​	相比于物品1，傲海的向量距离显然与物品2更小，在推荐系统中就会优先为傲海推荐物品1而不是物品2.这种表示方法就将推荐召回模块抽象成是否可以准确的表示每个人和物品的Embedding，越准确则推荐效果越佳。

## 1.1 什么是Embedding

​	embedding是一种稠密向量的表示形式，在embedding大行其道之前onehot才是最靓的仔。直观上看embedding相当于onehot做了平滑，而onehot相当于是对embedding做了max pooling

<img src="https://pic1.zhimg.com/v2-1ded42011e9dd14893d7872074b808d8_r.jpg" alt="preview" style="zoom:50%;" />

​	比如RGB（三原色， red，green，blue）任何颜色都可以用一个RGB向量来表示，其每一维度都有明确的物理含义（和一个具体的物理量相对应）。RGB比较特殊，每一维度都是事先规定好的，所以解释性很强。而一般意义的==embedding则是神经网络倒数第二层的参数权重==，只具有整体意义和相对意义，不具备局部意义和局对含义，这与embedding的产生过程有关，**任何embedding一开始都是一个随机数，然后随着优化算法，不断迭代更新，最后网络收敛停止迭代的时候，网络各个层的参数就相对固化，得到隐层权重表（此时就相当于得到了我们想要的embedding），然后通过查表可以单独查看每个元素的embedding。**

<img src="https://pic2.zhimg.com/v2-22fd6e8ebecd09a234d17532a268ba6d_r.jpg" alt="preview" style="zoom:50%;" />

## 1.2 Embedding化的意义

embedding作为一种新思想，他的意义包含以下几个方面：

- embedding表示：把自然语言转化为一串数字，从此自然语言可以计算
- embedding替代onehot：极大的降低了特征的维度
- embedding替代协同矩阵：极大地降低了计算复杂堵

# 2.I2I和U2I的召回方案

​	最核心的召回模式是I2I和U2I，U2I主要是通过计算（User）和物品（Item）的距离做召回。跟上文傲海和两个物品的向量距离计算一样。U2I对算法要求是，需要把人和物品的特征同时加入算法进行计算，这样人和物品的向量维度和意义才一致，做人和物品的向量距离计算才有意义。

​	I2I不考虑人的因素，I2I一般应用到的场景：比如一个喜欢买各种X类型的手机，把所有物品Embedding，然后找跟X类型手机距离近的物品推荐给这个人即可。

​	I2I更多考虑如何求物品间的相关性，并表示成Embedding。

![preview](https://pic3.zhimg.com/v2-ac8198da43f7be187787b3fcb579f726_r.jpg)

