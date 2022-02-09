<center><font size='60'>编码总结</font></center>

## 1.数组、链表、哈希表

### 1.1 数组

### 1.2 链表

**1.反转链表 - reverse linked list**

```python
def reverseList(self, head):
  cur, prev = head, None
  while cur:
    cur.next, prev, curr = prev, cur, cur.next
  return prev
```

**2.链表交互相邻元素**

```python
def swapPairs(self, head):
  dummyHead = ListNode(0) # dummy node
  pre, pre.next = dummyHead, head
  while pre.next and pre.next.next:
    a = pre.next
    b = a.next
    pre.next, b.next, a.next = b, a, b.next
    pre = a
  return dummyHead.next
  
```



### 1.3 哈希表HashTable



### 2.堆栈、队列

## 3.树、二叉树

### 3.1 二叉树

**1.二叉树的最近公共祖先**

```python
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
  """
  若root是p，q的最近公共祖先，则只可能为以下情况之一：
  1.p和q在root的子树中，且分列root的异侧（分别在左、右子树中）
  2.p=root，且q在root的左子树或右子树中
  3.q=root，且p在root的左子树或右子树中
  """
  			# 终止条件
        if not root or root == p or root==q: return root
        # 递归处理
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # 返回值
        # left为空，right不为空，p，q都不在root的左子树中，直接返回right
        if not left:
            return right
        elif not right:
            return left
        # left 和right都不为空，p，q在root的两侧，root为最近公共祖先
        elif left and right:
            return root
```



## 4.递归、分治、回溯、贪心

### 4.1 递归

递归编写模版

```python
def recursion(level, param1, param2, ...):
  # recursion terminator递归终止条件
  if level > MAX_LEVEL:
    print_result
    return
  
  # 当前层的处理逻辑 process logic in current level
  process_data(level, data...)
  
  # drill down 下钻
  self.recursion(level+1, p1, ...)
  
  # reverse the current level status if needed
  reverse_state(level)
  
```



## 5.深度优先DFS + 广度优先BFS

### 5.1DFS

1.递归写法模版

```python
visited = set()
def dfs(node, visited):
  visited.add(node)
  # process current node here
  ...
  for next_node in node.children():
    if not next_node in visited:
      dfs(next_node, visited)
```

2.非递归写法模版

```python
def dfs(tree):
  if tree.root is None:
    return []
  
  visited, stack = [], [tree.root]
  
  while stack:
    node = stack.pop()
    visited.add(node)
    
    process(node)
    nodes = generate_related_nodes(node)
    stack.push(nodes)
  # other processing work
  ...
```



### 5.2 BFS

广度优先模版

```python
def bfs(graph, start, end):
  queue = []
  queue.append([start])
  visited.add(start)
  
  while queue:
    node = queue.pop()
    visited.add(node)
    
    process(node)
    nodes = generate_related_nodes(node)
    queue.push(nodes)
  # other processing work
  ...
```



## 6.位运算操作

​		程序中的所有树在计算机内存中都是以二进制的形式存储的，位运算说穿了，就是直接对整数在内存中的二进制位进行操作。不需要转成十机制，因此处理速度非常快。

| 符号 | 描述 | 运算规则                                                     |
| :--: | :--: | ------------------------------------------------------------ |
|  &   |  与  | 两个位都为1时，结果为1                                       |
|  \|  |  或  | 两个位都为0时，结果为0                                       |
|  ^   | 异或 | 两个位相同为0，相异为1                                       |
|  ～  | 取反 | 0变1， 1变0                                                  |
|  <<  | 左移 | 各二进位全部左移若干位，高位丢弃，低位补0                    |
|  >>  | 右移 | 各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移） |

常用位运算：

1. x&1 ==1 or == 0判断奇偶（x % 2 ==1）
2. x = x&(x-1)  => 清零最低位1
3. x & -x  => 得到最低位的1

## 7.动态规划

### 7.1 理论

**1.解决问题思维模式**

- 定义状态

  问题要求什么，要什么，我们dp的因变量就是什么，自变量根据题目要求，为物品和容量

- 状态转移方程

  确定好了状态，就要看看这个父问题如何转化为子问题了，这也是状态方程要解决的

- 初始化

  主要是看有没有要求得到最值的时候，满负载

- 考虑压缩空间

  自变量如果能从物品和容量单纯的变成容量，那自然是好事



**2.对比动态规划 vs 回溯 vs 贪心算法**

- 回溯（递归）：重复计算
- 贪心算法：永远局部最优
- 动态规划：记录局部最优子结构/多种记录值



### 7.2 常见题目

**1.斐波那契数列、爬楼梯问题**

（1）斐波那契数列[LeetCode509](https://leetcode-cn.com/problems/fibonacci-number/)

（2）爬楼梯[LeetCode70](https://leetcode-cn.com/problems/climbing-stairs/)

​	状态转移方程：$f[n] = f[n-1] + f[n-2]$



**2.不同路径问题**

（1）不同路径[LeetCode62](https://leetcode-cn.com/problems/unique-paths/)    [LeetCode63](https://leetcode-cn.com/problems/unique-paths-ii/)    [LeetCode980](https://leetcode-cn.com/problems/unique-paths-iii/)

定义状态：

状态转移方程：$opt[i][j] = opt[i -1][j] + opt[i, j-1]$

```python
if a[i][j] = "空地":
  opt[i][j] = opt[i-1][j] + opt[i][j-1]
else: //石头
  opt[i][j] = 0
```



**3.最长递增子序列**

[LeetCode300](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

```python
def lengthOfLIS(self, nums: List[int]) -> int:
    """
    动态规划：
    1.定义状态：dp[i]表示从头到第i个元素（且包括第i个元素）最长子序列的长度
    2.动态转移方程：dp[i]=max(dp[j])+1,其中0≤j<i且num[j]<num[i]
    3.返回结果：max(dp[0]...dp[n-1])
    """
    if not nums: return 0
    n = len(nums)
    dp = []
    for i in range(n):
        dp.append(1)
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] +1)
    return max(dp)
```



4.最大子数组和

[LeetCode53](https://leetcode-cn.com/problems/maximum-subarray/)

```python
def maxSubArray(self, nums: List[int]) -> int:
  # 动态规划：f(i)=max{f(i−1)+nums[i],nums[i]}
  pre = 0
  maxAns = nums[0]
  for i in nums:
    pre = max(pre+i, i)
    maxAns = max(pre, maxAns)
  return maxAns
```

5.乘积最大子数组

[LeetCode152](https://leetcode-cn.com/problems/maximum-subarray/)

```python
def maxProduct(self, nums: List[int]) -> int:
    """
    动态规划：
    1.状态定义：
       dp[i][0]: 表示第i个元素结尾的乘积的最大子数组的乘积
       dp[i][1]: 表示第i个元素结尾的乘积的负的最大子数组的乘积
       a 表示输入参数nums
    2.动态转移方程：
       dp[i][0] = if a[i] >=0:  dp[i-1][0] * a[i]
                  else:         dp[i-1][1] * a[i]
       dp[i][1] = if a[i] >=0:  dp[i-1][1] * a[i]
                  else:         dp[i-1][0] * a[i]
    """
    if not nums: return 0

    res, curMax, curMin = nums[0], nums[0], nums[0]
    for i in range(1, len(nums)):
        num = nums[i]
        curMax, curMin = curMax * num, curMin * num
        curMin, curMax = min(curMin, curMax, num), max(curMax, curMin, num)

        res = curMax if curMax > res else res
    return res
```

















