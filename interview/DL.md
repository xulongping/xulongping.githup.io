## 3.深度学习

### 3.1 理论

### 3.2 常用模型

#### 3.2.1 全连接网络

#### 3.2.2 CNN

#### 3.2.3 RNN

**1.RNN解决什么问题**

​		在前馈神经网络中，信息传递是单向的，前馈神经网络可以看作一个复杂的函数，每次输入都是独立的，即网络的输出只依赖于当前的输入。**但在很多现实任务中，网络的输入不仅和当前时刻的输入相关，也和过去一段时间的输出相关。**

​		循环神经网络（Recurrent Neural Network，RNN）是一类具有短期记忆能力的神经网络，可以适应不同长度/尺寸的输入数据。

​		RNN的参数学习可以通过随时间反向传播算法实现，即按照时间的逆序将错误信息一步步地往前传递。

​		当输入序列较长时，会存在**梯度爆炸和消失问题**，为解决该问题，对循环神经网络进行了很多的改进，其中最有效的改进方式引进**门控机制**



**2.RNN模型结构**

![展开的RNN模型](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/展开的RNN模型.png)

​		展开的RNN网络结构，在每个时间戳t，网络层接受**当前时间戳的输入$x_t$**和**上一个时间戳的网络状态向量$h_{t-1}$**，经过 

​										$h_t = f_\theta(h_{t-1}, x_t)$

​		变换后得到**当前时间戳的新状态向量$h_t$**，并写入内存状态中，其中$f_\theta$代表了网络的运算逻辑， 𝜃为网络参数集。在每个时间戳上，网络层均有输出产生$o_t, o_t=g_\phi(h_t)$，即将网络的状态向量变换后输出。

​		上述网络结构在时间戳上折叠，网络循环接受序列的每个特征向量$x_t$，并刷新内部状态向量$h_t$，同时形成输出$o_t$：

![折叠的RNN模型](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/折叠的RNN模型.png)

​		如果使用张量$W_{xh}、W_{hh}$和偏置b来参数化$f_\theta$网络，并按照

​								$h_t = \sigma(W_{xh}x_t + W_{hh}h_{t-1} + b)$

​		的方式更新内存状态。在循环神经网络中，激活函数更多采用tanh函数。



**3.RNN为什么有记忆功能**

​		由RNN的结构可知，RNN在每个时间步都会将前一步的激活值传递给当前步。RNN的状态取决于当前输入和先前输入，而先前状态又取决于它的输入和它之前的状态，因此状态可以间接访问序列的所有先前输入，RNN正是以这种方式保存着过去的记忆

**4.RNN的损失函数**

**5.RNN存在梯度消失/爆炸问题的原因是什么**

​		梯度消失：是指在做方向传播，计算损失函数对权重的梯度时，随着越向后传播，梯度变得越来越小，这就意味着在网络的前面一些层的神经元，会比后面的训练要慢很多，甚至不会变化。至使结果不准确，训练时间非常长。

​		![image-20220217161719820](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/RNN梯度推动公式.png)

​		从时间戳i到时间t的梯度$\frac{\partial h_t}{\partial h_i}$包含了$W_{hh}$的连乘运算。当$W_{hh}$ 的最大特征值小于1时，多次连乘运算会使得$\frac{\partial h_t} {\partial h_i}$的元素值接近于零；当$\frac{\partial h_t}{\partial h_i}$的值大于1时，多次连乘运算会使得$\frac{\partial h_t}{\partial h_i}$的元素值爆炸式增长。



#### 3.2.4 LSTM

**1.LSTM可以解决的问题**

​		RNN有一个严重问题就是短时记忆，对于较长范围类的有用信息往往不能够很好的利用起来。LSTM长短时记忆网络（Long Short-Term Mermory）相对于基础的RNN网络来说，记忆能力更强，更擅长处理较长的序列信号数据。

**2.LSTM网络结构**

![image-20220217163751494](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/LSTM网络结构.png)

​		基础的RNN网络结构，上一个时间戳的状态向量$h_{t-1}$与当前时间戳的输入$x_t$经过线性变换后，通过激活函数tanh后得到新的状态向量$h_t$。相对于基础的RNN网络只有一个状态向量$h_t$，**LSTM新增了一个状态向量$C_t$，同时引入了门控（Gate）机制，通过门控单元来控制信息的遗忘和刷新**。

​		在LSTM中，有两个状态向量c和h，其中c做为LSTM的内部状态向量，可以理解为LSTM的内存状态向量Memory，而h表示LSTM的输出向量。

​		**门控机制：**可以理解为控制数据流通的一种手段，类比于水阀门：当水阀门全部打开时，水流畅通无阻地通过；当水阀门全部关闭时，水流完全被隔断。

​		在LSTM中，阀门开和程度利用门控值向量g表示，通过$\sigma(g)$激活函数将门控制压缩到[0,1]之间区间，**当$\sigma(g) = 0$时，门控全部关闭，输出o=0；当$\sigma(g) = 1$时，门控全部打开，输出o=x**。通过门控机制可以较好地控制数据的流量程度。

![image-20220217165524346](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/LSTM 门控机制.png)



遗忘门、输入门、输出门：

**（1）遗忘门**

![image-20220217165913265](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/LSTM 遗忘门.png)

**（2）输入门**

![image-20220217165952578](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/LSTM 输入门.png)

**（3）刷新Memory**

**（4）输出门**

![image-20220217170033309](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/LSTM 输出门.png)

#### 3.2.5 GRU

**1.GRU网络结构**

![image-20220217170542941](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/GRU网络结构.png)

​		GRU把内部状态向量和输出向量合并，统一为状态向量h，门控数量也减少到2个：复位门（Reset Gate）和更新门（Update Gate）

**（1）复位门**

![image-20220217170847446](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/GRU 复位门.png)

**（2）更新门**

![image-20220217170921192](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/GRU 更新门.png)

### 3.3 算法调优

1.过拟合的解决方式



### 3.4 算法评估

1.AUC的意义，ROC的绘制方式，AUC的优势（不平衡数据集的情况）

