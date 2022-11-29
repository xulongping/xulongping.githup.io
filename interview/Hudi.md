## 4.Spark

#### 4.1 spark核心概念和架构

1. **Spark特点是什么？**

   ​		Spark是一个基于内存的，用于大规模数据处理（离线计算、实时计算、快速查询-交互式查询）的统一分析引擎。它内部的组成模块：SparkCore、SparkSQL、SparkStreaming、SparkMlib、SparkGraghX等。

   ​		快：Spark计算速度是MapReduce计算速度的10～100倍

   ​		易用：MR支持1种计算模型，Spark支持更多的计算模型（算法多）

   ​		通用：Spark能够进行离线计算、交换式查询（快速查询）、实时计算、机器学习、图计算

   ​		兼容性：Spark支持大数据中的Yarn调度，支持mesos。可以处理hadoop计算的数据。

   

2. **Spark提交作业的参数**

   使用spark-submit  提交任务，会涉及到几个重要那个的参数：

   num-executors: 启动executors的数量，默认为2

   executor-cores: 每个executor使用的内核数，默认为1，官方建议2—5个，我们使用是4个

   executor-memory: executor内存大小，默认1G

   diver-cores: driver使用内核数，默认为1

   driver-memory: driver内存大小，默认512M

   

3. **Spark on Yarn的作业的提交流程**

   Spark客户端直接连接Yarn，不需要额外构建Spark集群。有yarn-client和yarn-cluster两种模式，主要区别在于：Driver程序的运行节点。

   **Driver**：是spark驱动器节点，主要用于执行spark任务中的main方法，负责实际代码的执行。

   ​	a.将用户程序转化为作业（Job）

   ​	b.在Executor之间调度任务（task）

   ​	c.跟踪Executor的执行情况

   ​	d.通过UI展示查询运行情况

   Executor：Executor并不是一个进程，而是ExecutorBackend的一个进程，Executor是它的执行类，负责spark作业中运行具体的任务，任务之间是彼此相互独立的。Spark应用启动时，Executor节点同时启动，并且始终伴随整个spark应用中的生命周期存在，如果executor节点发生故障或者崩溃，spark会将故障节点的任务调度到其它executor节点上执行。

   ​	a.运行组成spark应用的任务，并将结果返回给驱动器进程

   ​	b.通过自身的块管理器（Block Manager）为用户程序中要求缓存的RDD提供内存式存储。RDD时直接缓存在executor进程内的，因此任务可以在运行时充分利用缓存数据加速计算。

   ![spark通用运行流程图](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/spark通用运行流程图.png)

   

   

   **（1）yarn client运行模式**

   ![spark-yarn-client运行模式](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/spark-yarn-client运行模式.png)

   

   **（2）yarn cluster运行模式**

   ![spark-yarn-cluster-运行模式](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/spark-yarn-cluster-运行模式.png)

   

4. **spark的容错机制**

   一般而言，对于分布式系统，数据集的容错性通常有两种方式：

   ​	a.数据检查点（在Spark中对应的Checkpoint机制）

   ​	b.记录数据的更新（在Spark中对应Lineage血缘机制）

   ​		对于大数据而言，数据检查点操作（一般是将RDD写入持久存储，如HDFS）成本较高，可能涉及大量数据复制操作，消耗 I/O资源。而通过血统机制则不需要存储正在的数据，容错的成本比较低。但是问题在于如果血缘很长（即依赖的关联链路很长），如果失败重算，那代价也是很高的，所以spark提供了checkpoint的API，将恢复代价更小的选择交给用户，进一步控制容错的时间。

   ​		通常在含有宽依赖的容错中，使用Checkpoint机制设置检查点，这样就不至于重新计算父RDD而产生冗余计算了。

   

5. **如何理解Spark中血统（RDD）的概念？它的作用是什么？**

   概念：RDD是弹性分布式数据集，是Spark中最基本的数据抽象，代表一个不可变、可分区、里面的元素可并行计算的集合。

   作用：提供了一个可抽象的数据模型，将具体的应用逻辑表达为一系列转换操作（函数）。另外不同RDD之间的转换操作之间还可以形成依赖关系，进而实现管道化，从而避免了中间结果的存储，大大降低了数据复制、磁盘IO和序列化开销，并且还提供了更多的API（map/reduce/filter/groupBy）。

   RDD在lineage依赖方面分为两种窄依赖和宽依赖，用来解决数据容错时的高效性以及划分任务时候起到重要作用。

   

6. **简述Spark的宽窄依赖，以及Spark如何划分stage，每个stage又根据什么决定task个数**

   窄依赖：父RDD的一个分区只会被子RDD的一个分区依赖

   宽依赖：父RDD的一个分区会被子RDD的多个分区依赖（涉及到shuffle）

   Stage是如何划分的呢？

   根据RDD之间的依赖关系的不同将Job划分成不同的stage，遇到一个宽依赖则划分一个stage

   每个stage又根据什么决定task个数？

   Stage是一个taskSet，将stage根据分区数划分成一个个的task

   ![spark-stage](/Volumes/Computer/Learning/Git/xulongping.githup.io/image/interview/spark-stage.png)

   

7. **列举Spark常用的transformation和action算子，有哪些算子会导致shuffle**

   spark的运算操作有两种类型：transformation和action：

   transformation：代表的是转化操作，就是计算流程，然后是RDD[T]，可以是一个链式的转化，并且是延迟触发的。

   action：代表是一个具体的行为，返回的值非RDD类型，可以是object，或是一个数值，也可以是Unit代表无返回值，并且action会立即触发job的执行。

   Transformation的官方文档方法集合如下：

   ```
   map
   filter
   flatMap
   mapPartitions
   mapPartitionsWithIndex
   sample
   union
   intersection
   distinct
   groupByKey
   reduceByKey
   aggregateByKey
   sortByKey
   join
   cogroup
   cartesian
   pipe
   coalesce
   repartition
   repartitionAndSortWithPartitions
   ```

   Action的官方文档方法集合如下：

   ```
   reduce
   collect
   count
   first
   take
   takeSample
   takeOrdered
   saveAsSequenceFile
   saveAsObject
   ```

   有哪些会引起shuffle过程的算子：

   ```
   reduceByKey
   groupByKey
   ...ByKey
   ```

   

8. **foreachPartition和mapPartitions的区别**

   从官网文档的api可以看出foreachPartition返回值为空，应该属于action运算操作，而mapPartitions是在tranformation中，所以是转化操作。此外在应用场景上区分是mapPartitions可以获取返回值，继续在返回RDD上做其它操作，而foreachPartition因为没有返回值并且是action操作，所以使用它一般都是在程序末尾比如说要落地数据到存储系统中如mysql、es、hbase中。

   

9. **reduceByKey与groupByKey的区别，哪一种更具优势？**

   reduceByKey：按照key进行聚合，在shuffle之前有combine（预聚会操作），返回结果是RDD[k,v]。聚合操作可以通过函数自定义。

   groupByKey：按照key进行分组，直接进行shuffle。对每个key进行操作，但只生成一个sequence，如果需要对sequence进行aggregation操作（注意，groupByKey本身不能自定义操作函数），那么选择reduceByKey/aggregateByKey更好。因为groupByKey不能自定义函数，需要先用groupByKey生成RDD，然后才能对此RDD通过map进行自定义函数操作。

   ```
   val words = Array("one", "two", "two", "three", "three", "three")
   
   val wordPairsRDD = sc.parallelize(words).map(word => (word, 1))
   
   val wordCountsWithReduce = wordPairsRDD.reduceByKey(_ + _)
   
   val wordCountsWithGroup = wordPairsRDD.groupByKey().map(t => (t._1, t._2.sum))
   ```

   对大数据进行复杂计算时，reduceByKey优于groupByKey。

   

10. **Repartition和Coalesce的关系与区别？**

    关系：两者都是用来改变RDD的partition数量的，repartition底层调用的就是coalesce方法：coalesce(numPartitions, shuffle=true)

    区别：repartition一定会发生shuffle，coalesce根据传入的参数来判断是否发生shuffle。一般情况下增大rdd的partition数量使用repartition，减少partition数量时使用coalesce。

    

11. **简述spark中的缓存（cache和persist）与checkpoint机制，并指出两者的区别和联系**

    位置：Persist和Cache将数据保存在内存，Checkpoint将数据保存在HDFS

    生命周期：Presist和Cache程序结束后会被清除或手动调用unpersist方法，Checkpoint永久存储不会被删除。

    RDD依赖关系：Presist和Cache，不会被丢掉RDD间的依赖链关系，CheckPoint会斩断依赖链。

    

12. **spark中共享变量（广播变量和累加器）的基本原理与用途**

    广播变量：广播变量是在每个机器上缓存一份，不可变，只读的，相同的变量，该节点每个任务都能访问，起到节省资源和优化的作用。它通常用来高效分发较大的对象。

    累加器：是spark提供的一种分布式的变量机制，其原理类似于mapreduce，即分布式的改变，然后聚合这些改变。累加器的一个常见用途是在调试时对作业执行过程中的事件进行计数。

    

13. **当spark涉及到数据库的操作时，如何减少spark运行中的数据连接数？**

    使用foreachPartition代替foreach，在foreachPartition内获取数据库的连接

    

14. **能介绍下你所知道和使用过的spark调优吗？**

    **资源参数调优：**

    ​		num-executors: 设置spark作业总共要用多少个executor进程来执行

    ​		executors-memory: 设置每个executor进程的内存

    ​		executors-cores: 设置每个executor进程的cpu core数量

    ​		driver-memory: 设置driver进程的内存

    ​		spark.default.parallelism: 设置每个stage的默认task数量

    **开发调优：**

    ​	避免创建重复的RDD

    ​	尽可能复用同一个RDD

    ​	对多次使用的RDD进行持久化

    ​	尽量避免使用shuffle类算子

    ​	使用map-side预聚合的shuffle操作		

    ​	使用高性能的算子：

    ​		a.使用reduceByKey/aggregateByKey替代groupByKey

    ​		b.使用mapPartition替代普通map

    ​		c.使用foreachPartitions替代foreach

    ​		d.使用filter之后进行coalesce操作

    ​		e.使用repartitionAndSortWithinPartitions替代repartition与sort类操作

    ​	广播大变量：在算子函数中使用到外部变量时，默认情况下，Spark会将该变量复制多个副本，通过网络传输到task中，此时每个task都有一个变量副本。如果变量本身比较大的话（比如100M，甚至1G），那么大量的变量副本在网络中传输的性能开销，以及在各个节点的executor中占用过多内存导致的频繁GC，都会极大影响性能。

    

15. **如何使用spark实现topN的获取**

    （1）按照key对数据进行聚合（groupByKey）

    （2）将value转换为数组，利用scala的sortBy或者sortWith进行排序（mapValues）

    

16. **spark从HDFS读入文件默认是怎样分区的？**

    spark从HDFS读入文件的分区数默认等于HDFS文件的块数（blocks），HDFS中的block是分布式存储的最小单元。如果我们上传一个30GB的非压缩文件到HDFS，HDFS默认的容量大小128MB，因此文件在HDFS上会被分为235块（30GB/128MB）；spark读取SparkContent.textFile()读取该文件，默认分区等于块数即235。

    

17. **spark如何设置合理分区数**

    （1）分区越多越好吗？

    不是的，分区数太多意味着任务数太多，每次调度任务也是很耗时的，所以分区太多会导致总统耗时增多。

    （2）分区数太少又什么影响？

    分区太少的话，会导致一些节点没有分配到任务；另一方面，分区数少则每个分区要处理的数据量就会增大，从而对每个结点的内存要求就会提高；还有分区数不合理，会导致数据倾斜的问题。

    （3）合理的分区数是多少？如何设置？

    总核数=executor-cores * num-executor

    一般合理的分区数设置为总核数的2～3倍

    （4）partition和task的关系

    Task是spark的任务运行单元，通常一个partition对应一个task。有失败时另行恢复。

18. 

19. 

20. 



#### 4.2 spark编程



## 5. Flink

#### 5.1 核心概念和基础考察

1. Flink的特性

   支持高吞吐、低延迟、高性能的流处理

   支持带有事件时间的窗口（Window）操作

   支持有状态计算的Exactly-once语义

   支持高度灵活的窗口（Window）操作，支持基于time、count、session以及data-driven的窗口操作

   支持基于轻量级分布式快照（Snapshot）实现的容错

   一个运行时同时支持Batch on Streaming处理和Streaming处理

   Flink在JVM内部实现了自己的内存管理

   支持迭代计算

   支持程序自动优化：避免特定情况下Shuffle、排序等昂贵操作、中间结果有必要进行缓存

2. 

3. 

#### 5.2 应用架构

1. **怎么提交实时任务的，有多少Job Manager？**

   使用yarn session模式提交任务。每次提交都会创建一个新的Flink集群，为每个job提供一个yarn-session，任务之间相互独立，互不影响，方便管理。任务执行完成之后创建的集群也会消失。

   线上命令脚本如下：

   bin/yarn-session.sh -n 7 -s 8 -jm 3072 -tm 32768 -qu root.\*.* -nm \*-* -d 

   其中申请7个taskManager，每个8核，每个taskmanager有32768M内存。

   集群默认只有一个Job Manager。但为了防止单点故障，我们配置了高可用。我们一般配置一个主Job Manager，两个备用Job Manager，然后结合Zookeeper的使用，来达到高可用。

   

2. **Flink最大并行度是如何确定的**

   Spark：Executor数 * 每个Executor中cup core

   Flink：TaskManager数 * 每个TaskManager中Task Slot

   

3. **集群部署模式类型对比**

   根据以下两种条件将集群部署模式分为三种类型：

   ​	a.集群的生命周期和资源隔离

   ​	b.根据程序main()方法执行在Client还是JobManager

   （1）**Seesion Mode**

   ​		共享JobManager和TaskManager，所有提交的Job都在一个Runtime中运行，JobManager的生命周期不受提交的Job的影响，会长期运行

   ​		优点：资源充分分享，提升资源利用率；Job在Flink Session集群中管理，运维简单。

   ​		缺点：资源隔离相对较差，非Native类型部署，TM不易拓展，Slot计算资源伸缩较差

   ​		提交任务脚本：

   ```shell
   $./bin/yarn-seesion.sh -jm 1024m -tm 4096m
   ```

   （2）**Per-Job Mode**

   ​		独享JobManager与TaskManger，好比为每个Job单独启动一个Runtime；TM中Slot资源根据Job指定；JobManager的生命周期和Job生命周期绑定。

   ​		优点：Job和Job之间资源隔离充分；资源根据Job需要进行申请，TM Slots数量可以不同

   ​		缺点：资源相对比较浪费，JobManager需要消耗资源；Job管理完全交给ClusterManger，管理复杂

   ​		提交任务脚本：

   ```shell
   $./bin/flink run -m yarn-cluster -p 4 -yjm 1024m -ytm 4096m ./examples/batch/WordCount.jar
   ```

   （3）**Application Mode（1.11版本提出）**

   ​		Application的main()运行在Cluster上，而不在客户端；每个Application对应一个Runtime，Application中可以含有多个Job；

   ​		a.每个Application对应一个JobManager，且可以运行多个Job

   ​		b.客户端无需将Dependencies上传到JobManager，仅负责管理Job的提交与管理

   ​		c.main()方法运行JobManager中，将JobGraph的生成放在集群上运行，客户端压力降低

   ​		优点：有效降低带宽消耗和客户端负载；Application实现资源隔离，Application中显现资源共享

   ​		缺点：仅支持Yarn和Kubunetes

   ​		提交任务脚本：

   ```shell
   $./bin/flink run-application -t yarn-application \
   ​			-Djobmanager.memory.process.size=2048m \
   ​			-Dtaskmanager.memory.process.size=4096m \
   ​			-Dyarn.provided.lib.dirs="hfs://node02:8020/flink-training/flink-1.11.1" \
   ​			./MyApplication.jar
   ```

4. 

5. 

#### 5.3 压测和监控

#### 5.4 Flink编程

