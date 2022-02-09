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













