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

**1.何时采用动态规划**

​		整个数组或在固定大小的滑动窗口中找到**总和**或者**最大值**或**最小值**问题，可以通过动态规划（DP）在线性时间内解决。



**2.动态规划解题步骤**

​		定义状态、状态转移方程、初始化、输出

- 定义状态

  问题要求什么，要什么，我们dp的因变量就是什么，自变量根据题目要求，为物品和容量

  常见状态定义方法：

  一维动态规划：

  - dp[i]定义为数组前i个元素的最值或者总和
  - dp[i]定义为num[i]作为结尾元素的最值或总和

  二维动态规划：

  - dp\[i][j]定义为数组或字符串从num[i...j]之间的最值
  - dp\[i][j]定义为以num[i]开始并且以num[j]结尾的子数组的最值
  - dp\[i][j]定义为两个数组分别以num[i]和num[j]结尾的最值
  - dp\[i][j]在01背包问题中表示添加前i个数值后剩余的容量为j

- 状态转移方程

  确定好了状态，就要看看这个父问题如何转化为子问题了，这也是状态方程要解决的



**3.对比动态规划 vs 回溯 vs 贪心算法**

- 回溯（递归）：重复计算
- 贪心算法：永远局部最优
- 动态规划：记录局部最优子结构/多种记录值



### 7.2 常见题目

#### 7.2.1 斐波那契数列、爬楼梯问题

**1.斐波那契数列、爬楼梯问题**

（1）斐波那契数列[LeetCode509](https://leetcode-cn.com/problems/fibonacci-number/)

（2）爬楼梯[LeetCode70](https://leetcode-cn.com/problems/climbing-stairs/)

​	状态转移方程：$f[n] = f[n-1] + f[n-2]$



#### 7.2.2 不同路径问题

**1.不同路径问题**

（1）不同路径[LeetCode62](https://leetcode-cn.com/problems/unique-paths/)    [LeetCode63](https://leetcode-cn.com/problems/unique-paths-ii/)    [LeetCode980](https://leetcode-cn.com/problems/unique-paths-iii/)

定义状态：

状态转移方程：$opt[i][j] = opt[i -1][j] + opt[i, j-1]$

```python
if a[i][j] = "空地":
  opt[i][j] = opt[i-1][j] + opt[i][j-1]
else: //石头
  opt[i][j] = 0
```



#### 7.2.3 背包问题

**1.0-1背包问题**

​		从数组中选出一些数值，使其满足特定的容量，从而求其最大值。

​		比如有n个物品，它们有各自的体积w和价值v，现有给定容量的背包bagV，如何让背包里装入的物品有最大的价值总和？（0/1背包问题就是n个物品中某个物品选还是不选，分别表示1和0）

​		状态转移方程：$dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]]+v(i))$

​		初始值：i=1， j=1

**总结：如果是0-1背包问题，数组中的元素不可重复使用，nums放在外循环，target在内循环，且内循环倒序：**

```python
for num in nums:
  for i in range(target, num-1, -1):
    
```

[0-1背包]()

问题描述：存在一个容量为C的背包，和N类物品。这些物品分别有两个属性，重量w和价值v，每个物品的重量为w[i]，价值为v[i]，每个物品只有一个。在不超过背包容量的情况下能装入最大的价值为多少？（这个背包可以不装满）

```python
def knapstack(self, w: List[int], v: List[int], bagV: int) -> int:
  """
  动态规划：
  1.定义状态：dp[i][j]表示前i件物品装进容量为j的背包可以获得的最大价值
  2.状态转移方程：dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]] + v[i])
        不装入第i件物品：dp[i][j] = dp[i-1][j]
        装入第i件物品：dp[i][j] = dp[i-1][j-w[i]] + v[i] (j > w[i]背包容量大于w[i]物体体积)
  3.初始化：dp[0][:] = 0 第0个物品时不存在的，价值为0
  4.输出：dp[len(w)][bagV]
  """
  n = len(w)
  dp = [[0 for _ in range(c+1)] for _ in range(bagV+1)]
  # 初始化
  for j in range(c+1):
    dp[0][j] = 0
  
  for i in range(1, n+1):
    for j in range(bagV, i-1, -1):
      dp[i][j] = dp[i-1][j]
      if j >= w[i]:
        dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]] + v[i])
  return dp[n][bagV]

```

[LeetCode 416.分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)：数组nums分割成两个子集，使得两个子集元素和相等

```python
def canPartition(self, nums: List[int]) -> bool:
    """
    动态规划：
    1.定义状态：dp[i][j]表示从前i个元素中挑选子序列是否可以计算出和j；当j=sum/2时，dp[i][j]是否为true
    2.状态转移方程：dp[i][j] = dp[i-1][j]          至少是这个答案
                            true                 nums[i] = j
                            dp[i-1][j-num[i]]    nums[i] < j
      dp[i-1][j]已经为true，已经可以由前i-1个元素中挑选子序列计算出和j，那么d[i][j]自然为true
      dp[i-1][i-num[j]]为ture，前i-1个元素中挑选子序列计算出和j-nums[i]，那么加上nums[i]刚好可以完成
    3.初始化: d[0][0] = fase
    4.输出：dp[nums.len][degV]

    优化方法：用nums中数字凑出sum(nums) // 2的问题
    """
    target = sum(nums)
    if target % 2 != 0: return False
    target = target // 2

    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] =  dp[i] or dp[i - num]
    return dp[-1]
```



**2.无限背包问题**

​		与01背包不同之处在于，**数组中的元素可以重复选择**。比如：硬币找零问题、切割钢条、剪绳子等。

​		状态转移方程：$dp[i] = max(dp[i], dp[i-len[j]]+price[j])$

​		初始值：i=1...len, j=0...num2.len

**总结：如果是完全背包问题，即数组中的元素可重复使用，nums放在外循环，target在内循环。且内循环正序：**

```python
for num in nums:
  for i in range(num, target+1):
    
```

**如果组合问题需要考虑元素之间的顺序，需要将target放在外循环，将nums放在内循环**

```python
for i in range(1, target+1):
  for num in nums:
    
```

[LeetCode322 零钱兑换](https://leetcode-cn.com/problems/coin-change/) ：最少硬币个数

```python
def coinChange(self, coins: List[int], amount: int) -> int:
    """
    动态规划：
    状态定义：dp[i]表示组成金额i所需最少的硬币
    状态转移方程：dp[i] = min(dp[i], dp[i-coins[j]]+1)
    时间复杂度：O(Sn) S是金额，n是面额数
    空间负责度：O(S)
    """
    Max = amount + 1
    dp = [Max for i in range(amount+1)]
    dp[0] = 0

    for i in range(1, amount+1):
        for j in range(len(coins)):
            if i >= coins[j]:
                dp[i] = min(dp[i], dp[i-coins[j]]+1)
    return dp[amount] if dp[amount] <= amount else -1
```

[LeetCode518 零钱兑换二](https://leetcode-cn.com/problems/coin-change/) ：凑成总金额的硬币组合数

```python
def change(self, amount: int, coins: List[int]) -> int:
    """
    动态规划：跳台阶问题的思考方式一样，将每次跳台阶数变成了输入数组coins
    1.定义动态：dp[j] 代表装满容量为j的背包有几种硬币组合
    2.动态转移方程：dp[j] = dp[j] + dp[j - coin]
    3.初始化：dp[0] = 1
    4.输出结果: dp[amount]
    """
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for j in range(coin, amount+1):
            dp[j] += dp[j - coin]
    return dp[amount]
```

[LeetCode剑指 Offer 14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/) ：长度n的绳子剪成m段，得到最大乘积是多少

```python
def cuttingRope(self, n: int) -> int:
    """
    动态规划：
    1.状态定义：dp[i]表示剪断长度为i的绳子得到最大的乘积
    2.状态转移方程：dp[i] = max(dp[i], dp[i-j]*j), j=1...i (i>=j, i的初始值为3)
    3.初始值：初始值全为1，dp[3] = 2
    4.输出：dp[n]
    """
    dp = [1] * (n+1)
    if n == 3: return 2

    for i in range(3, n+1):
        for j in range(1, i+1):
            dp[i] = max(dp[i], dp[i-j]*j)
    return dp[n]
```

[LeetCode剑指 Offer 49.丑数](https://leetcode-cn.com/problems/chou-shu-lcof/) ：只包含因子2、3、5的数称作丑数，求按从小到大的顺序的第n个丑数

```python
def nthUglyNumber(self, n: int) -> int:
    """
    动态规划：
    1.定义状态：dp[i]表示第i个丑数的值，除此之外要定义三个变量，分别保存了当前{2, 3, 5}三个数的个数num2, num3, num5
    2.定义状态转移方程：dp[i] = min(dp[num2]*2, min(dp[num3]*3, dp[num5]*5))
    3.初始化：num2 = 0, num3 = 0, num5 = 0
    4.输出：dp[n-1]
    """
    dp = [0] * n
    num2 = num3 = num5 = 0
    dp[0] = 1

    for i in range(1, n):
        dp[i] = min(dp[num2] * 2, min(dp[num3] * 3, dp[num5] * 5))
        if dp[i] == dp[num2] * 2:
            num2 += 1
        if dp[i] == dp[num3] * 3:
            num3 += 1
        if dp[i] == dp[num5] * 5:
            num5 += 1
    return dp[n-1]
```



#### 7.2.4 回文子序列与最长字符串系列

##### 7.2.4.1 子序列问题

​		**子序列问题不要求数组元素连续**

**1.最长回文子序列**

```python
# 由于方程式中含有i+1，所以初始时要倒序遍历i，i=len-1...0
# 包含j-1, 所以要整虚遍历j，j=i+1...len
if s[i] == s[j]:
  dp[i][j] = dp[i+1][j-1] + 2 
else:
  dp[i][j] = max(dp[i+1][j], dp[i][j-1])
```

[LeetCode 516.最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)：返回最长回文子序列的长度

```python
def longestPalindromeSubseq(self, s: str) -> int:
    """
    动态规划:
    1.状态定义：dp[i][j]从第i个字符到第j个字符之间最长回文子序列的长度
    2.状态转移方程：dp[i][j] = dp[i+1][j-1] + 2             s[i] == s[j]
                            max(dp[i+1][j], dp[i][j-1])   s[i] != s[j]
        如果s[i] == s[j]，则等于第i+1到j-1字符之间的最长子串加上2
        如果s[i] != s[j]，则需要计算i和j之间子串的最大值，也即是i向左移j向右移
        由于在计算dp[i][j]时要提前知道dp[i+1]，所以i从字符串最后往前遍历
        同理要知道dp[j-1]所以j需要从前往后遍历
    3.初始化：dp[i][i] = 1 单个字符的最长回文序列是1
    4.输出： dp[0][s.len-1]
    """
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n-1, -1, -1):
        dp[i][i] = 1
        for j in range(i+1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    return dp[0][n-1]
```

**2.最长公共子序列**

**3.最长递增子序列（定差）**

```python
# 只需要判断后一个值大于前一个值：
if nums[i] > nums[j]:
  dp[i] = max(dp[i], dp[j]+1)
```

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
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] +1)
    return max(dp)
```



##### 7.2.4.2 子字符串问题

**1.最长回文子串**

```python
# 初始时i=len-1...0,  j=i...len
dp[i][j] = (s[i] == s[j]) and (j-i < 3 or dp[i+i][j-1])

或者
if s[i] == s[j]:
  dp[i][j] = True and (j-i < 3 or dp[i+i][j-1])
else:
  dp[i][j] = False
```

[LeetCode 5.最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)：返回s中最长的回文子串

```python
def longestPalindrome(self, s: str) -> str:
    """
    动态规划
    1.状态定义：dp[i][j]表示为起始位置分别为i到j组成的子串s[i:j]是否是回文字符串，bool类型
    2.状态转移方程：dp[i][j] = (s[i] == s[j]) & (j-i<3 || dp[i+1][j-1])
         边界条件：子串长度j-1 - (i+1) + 1如果小于2，要么长度为0的空串，要么长度为1的单个字符串，都是回文子串
    """
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    start = end = 0

    for i in range(n-1, -1, -1):
        dp[i][i] = True
        for j in range(i, n):
            if s[i] == s[j]:
                dp[i][j] = True and (j-i<3 or dp[i+1][j-1])
            else:
                dp[i][j] = False
            if dp[i][j] and j-i+1 >= end - start:
                start = i
                end = j + 1
    return s[start:end]
```

**2.最长公共子串**

**3.最长递增子串**



**4.最大子数组和**

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



**5.乘积最大子数组**

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



**6.三角形最小路径和**

[LeetCode120](https://leetcode-cn.com/problems/triangle/)

```python
def minimumTotal(self, triangle: List[List[int]]) -> int:
    """
    动态规划：
    1.定义状态：dp[i,j]: bottom ——>[i,j] path sum min
    2.动态转移方程：dp[i,j] = min(dp(i+1, j), dp(i+1, j+1)) + triangle[i,j]
              初始化：dp[m-1,j] = triangle[m-1,j]
    """
    if not triangle: return 0

    res = triangle[-1]
    for i in range(len(triangle)-2, -1, -1):
        for j in range(len(triangle[i])):
            res[j] = min(res[j], res[j+1]) + triangle[i][j]
    return res[0]
```



**7.买卖股票最佳时机**

[LeetCode121](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)    [LeetCode122](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)    [LeetCode123](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)   [LeetCode188](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

```python
def maxProfit(self, k: int, prices: List[int]) -> int:
    """
    动态规划：
    1.动态定义：mp[i][k][j]：表示第i天进行k次交易获取的最大利润
              i：表示股票在第i天的价格0～n-1
              k：表示第k笔交易0～k
              j：0:当前手上没有股票；1:当前手上持有一支股票
    2.动态转移方程：
       mp[i][k][0] = max( 
                         mp[i-1][k][0]   # 不动
                         mp[i-1][k-1][1] + a[i] # 卖掉           
       )
       mp[i][k][1] = max(
                         mp[i-1][k][1]   #不动
                         mp[i-1][k][0] - a[i]  # 买入
       )
    """
    if not prices: return 0

    n = len(prices)
    k = min(k, n // 2)
    maxprof = 0

    profit = [[[0 for _ in range(2)] for _ in range(k+1)]for _ in range(n)]

    profit[0][0][0], profit[0][0][1] = 0, -prices[0]

    for m in range(1, k+1):
        profit[0][m][0] = profit[0][m][1] = float("-inf")

    for i in range(1, n):
        profit[i][0][1] = max(profit[i-1][0][1], profit[i-1][0][0] - prices[i])
        for m in range(1, k+1):
            profit[i][m][0] = max(profit[i-1][m][0], profit[i-1][m-1][1] + prices[i])
            profit[i][m][1] = max(profit[i-1][m][1], profit[i-1][m][0] - prices[i])
            maxprof = max(profit[i][m][0], profit[i][m][1], maxprof)
    return maxprof
```



8.无限背包问题





