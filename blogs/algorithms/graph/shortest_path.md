---
title: Shortest Path
description: Floyd, Dijkstra, Bellman-Ford, SPFA
level: three
cat: graph
---


# Shortest Path

**最短路径定义**，在一个图G中，点u到点v有若干种走法，那么定义u到v的最短路径为路径权值和最小的走法。

有以下几种情况：

图：

1. 有向图
2. 无向图

边：

1. 图中无负边
2. 图中有负边，无负环
3. 图中有负边，有负环

目标：

1）求最短路径长度，

2）求具体的最短路径


## Floyd Warshall Algorithm

> 时间复杂度 **O(V^3)**
> 
> 空间复杂度 **O(V^2)**
> 
> 有向图无向图均可使用
> 
> 允许有负边，无负环
> 
> 多源最短路径
> 
> 基于动态规划



The Floyd Warshall Algorithm is for solving the **All Pairs Shortest Path** problem. The problem is to find shortest distances between every pair of vertices in a given edge **weighted directed** Graph. 

1) We initialize the solution matrix same as the input graph matrix as a first step. 

2) Then we update the solution matrix by considering all vertices as an intermediate vertex. The idea is to one by one pick all vertices and updates all shortest paths which include the picked vertex as an intermediate vertex in the shortest path. When we pick vertex number k as an intermediate vertex, we already have considered vertices {0, 1, 2, .. k-1} as intermediate vertices. For every pair (i, j) of the source and destination vertices respectively, there are two possible cases. 

a) k is not an intermediate vertex in shortest path from i to j. We keep the value of dist[i][j] as it is. 

b) k is an intermediate vertex in shortest path from i to j. We update the value of dist[i][j] as dist[i][k] + dist[k][j] if dist[i][j] > dist[i][k] + dist[k][j]



定义状态，划分阶段，我们定义dist[i][j][k]为经过前k的节点，从i到j所能得到的最短路径，dist[i][j][k]可以从dist[i][j][k-1]转移过来，即不经过第k个节点，也可以从dist[i][k][k-1]+dist[k][j][k-1]转移过来，即经过第k个节点。观察一下这个状态的定义，满足不满足最优子结构和无后效性原则。

最优子结构：图结构中一个显而易见的定理：最短路径的子路径仍然是最短路径 ,这个定理显而易见，比如一条从a到e的最短路a->b->c->d->e 那么 a->b->c 一定是a到c的最短路c->d->e一定是c到e的最短路，反过来，如果说一条最短路必须要经过点k，那么i->k的最短路加上k->j的最短路一定是i->j 经过k的最短路，因此，最优子结构可以保证。

无后效性：状态dist[i][j][k]由dist[i][j][k-1]，dist[i][k]][k-1]以及dist[k][j][k-1]转移过来，很容易可以看出，dist[k] 的状态完全由dist[k-1]转移过来，只要我们把k放倒最外层循环中，就能保证无后效性。

满足以上两个要求，我们即证明了Floyd算法是正确的。我们最后得出一个状态转移方程：${\displaystyle \mathrm {shortestPath} (i,j,k)=}{\displaystyle \mathrm {min} {\Big (}\mathrm {shortestPath} (i,j,k-1)}, {\displaystyle \mathrm {shortestPath} (i,k,k-1)+\mathrm {shortestPath} (k,j,k-1){\Big )}}$ ，可以观察到，这个三维的状态转移方程可以使用滚动数组进行优化。



```python

def floydWarshall(V, graph):

    dist = [[[sys.maxsize]*(1+V) for _ in range(1+V)] for _ in range(1+V)]

    for i in range(V):
        for j in range(V):
            dist[i][j][0] = graph[i][j]

    for k in range(1, V+1):
        # pick all vertices as source one by one
        for i in range(1, 1+V):
            # Pick all vertices as destination for the above picked source
            for j in range(1， 1+V):
                # If vertex k is on the shortest path from i to j, then update the value of dist[i][j]
                dist[i][j][k] = min(dist[i][j][k-1], dist[i][k][k-1] + dist[k][j][k-1])

    return dist
```

**滚动数组优化** scrolling array optimization

滚动数组是一种动态规划中常见的降维优化的方式，应用广泛（背包dp等），可以极大的减少空间复杂度。在我们求出的状态转移方程中，我们在更新dist[k]层状态的时候，用到dist[k-1]层的值，dist[k-2], dist[k-3]层的值就直接废弃了。所以我们大可让最内层的大小从n变成2。

再进一步，我们在dist[k]层更新dist[i][j][k]的时候，只用到了dist[i][k][k-1]和dist[k][j][k-1]。我们如果能保证，在更新k层另外一组路径m->n(类似于i->j)的时候，不受前面更新过的dist[i][j][k]的影响，就可以把第一维度去掉了。

假设去掉第一个维度，就是dist[i][j]依赖dist[i][k] + dist[k][j] 我们在更新dist[m][n]的时候，用到dist[m][k] + dist[k][n] 假设dist[i][j]的更新影响到了dist[m][k] 或者 dist[k][m] 即 {m=i,k=j} 或者 {k=i,n=j} 这时候有两种情况，j和k是同一个点，或者i和k是同一个点，那么 dist[i][j] = dist[i][j] + dist[j][j]，或者dist[i][j] = dist[i][i] + dist[i][j] 这时候，我们所谓的“前面更新的(i->j)点对”还是这两个点本来的路径，也就是说，唯一两种在某一层先更新的点，影响到后更新的点的情况，是完全合理的，所以说，我们即时把第一维去掉，也满足无后效性原则。

优化之后的状态转移方程即为：dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

```python
def floydWarshall(V, graph):

    # dist = copy.deepcopy(graph)
    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))
  
    for k in range(V):
        # pick all vertices as source one by one
        for i in range(V):
            # Pick all vertices as destination for the above picked source
            for j in range(V):
                # If vertex k is on the shortest path from i to j, then update the value of dist[i][j]
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist
```

**求具体路径** 

只需要记录下来在更新dist[i][j]的时候，用到的中间节点是哪个就可以了。假设我们用 $path[i][j]$ 记录从i到j松弛的节点k，那么从i到j,肯定是先从i到k，然后再从k到j， 那么我们在找出path[i][k] , path[k][j]即可，即 i到k的最短路是 i -> path[i][k] -> k -> path[k][j] -> k 然后求path[i][k]和path[k][j] ，一直到某两个节点没有中间节点为止。

```python
# 更新路径的时候
tmp = dist[i][k] + dist[k][j]
if tmp < dist[i][j]：
    dist[i][j] = tmp
    path[i][j] = k
```

```python
# 求路径的时候
res = []
def get_path(path, i, j):
    if (path[i][j] == -1)
        return []
    else:
        k = path[i][j]
        return [i] + get_path(path, i, k) + get_path(path, k, j) + [j]
```




## Dijkstra算法

> 时间复杂度 **O(V^2)** OR **O(E*log V)**
> 
> 空间复杂度 **O(V^2)** OR **O(E+V)**
> 
> 无负权边，无负环
> 
> 有向图无向图均可使用
> 
> 单源最短路算法
> 
> 基于贪心思想

Algorithm 

1) Create a set spt_set (shortest path tree set) that keeps track of vertices included in the shortest-path tree, i.e., whose minimum distance from the source is calculated and finalized. Initially, this set is empty. 

2) Assign a distance value to all vertices in the input graph. Initialize all distance values as INFINITE. Assign distance value as 0 for the source vertex so that it is picked first. 

3) While spt_set doesn’t include all vertices 

a) Pick a vertex u which is not there in spt_set and has a minimum distance value. 

b) Include u to spt_set. 

c) Update distance value of all adjacent vertices of u. To update the distance values, iterate through all adjacent vertices. For every adjacent vertex v, if the sum of distance value of u (from source) and weight of edge u-v, is less than the distance value of v, then update the distance value of v. 

**普通版本，邻接矩阵，时间复杂度O(V^2)**

```python
class Graph():
 
    def __init__(self, vertices):
        # adjacency matrix
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]
 
    # A utility function to find the vertex with minimum distance value, from the set of vertices not yet included in shortest path tree
    def min_dist_node(self, dist, spt_set):
        min_d = sys.maxsize
        # Search for the nearest vertex not in the shortest path tree
        for u in range(self.V):
            if dist[u] < min_d and spt_set[u] == False:
                min_d = dist[u]
                min_index = u
 
        return min_index
 
    # Function that implements Dijkstra's single source shortest path algorithm for a graph represented using adjacency matrix representation
    def dijkstra(self, src):
 
        dist = [sys.maxsize] * self.V # update it everytime when we add new node into the spt_set
        dist[src] = 0
        spt_set = [False] * self.V 
 
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from the set of vertices not yet processed. 
            # x is always equal to src in first iteration
            x = self.min_dist_node(dist, spt_set)
 
            # Put the minimum distance vertex in the shortest path tree
            spt_set[x] = True
 
            # Update dist value of the adjacent vertices of the picked vertex only if the current distance is greater than new distance and the vertex in not in the shortest path tree
            for y in range(self.V):
                if self.graph[x][y] > 0 and spt_set[y] == False and dist[y] > dist[x] + self.graph[x][y]:
                    dist[y] = dist[x] + self.graph[x][y]
 
        return dist
```


**Path information**

Use a parent array, update the parent array when distance is updated and and use it to show the shortest path from source to different vertices.

```python
# initialization
parent = [i for i in range(V)]
# ...............
            # Update dist value of the adjacent vertices of the picked vertex only if the current distance is greater than new distance and the vertex in not in the shortest path tree
            for y in range(self.V):
                if self.graph[x][y] > 0 and spt_set[y] == False and dist[y] > dist[x] + self.graph[x][y]:
                    dist[y] = dist[x] + self.graph[x][y]
                    parent[y] = x
#................

def get_path(src, des):
    parent = des
    path = [des]
    while parent != src:
        path.append(parent):
    path.append(src)
    return path
```

**邻接表存储，Binary Heap for Priority Queue 实现**，time complexity: $O(E*log V)$

Time complexity can be reduced to O(E + VLogV) using Fibonacci Heap. The reason is, Fibonacci Heap takes O(1) time for decrease-key operation while Binary Heap takes O(Logn) time.

Algorithm

1) Create a Min Heap of size V where V is the number of vertices in the given graph. Every node of min heap contains vertex number and distance value of the vertex. 
   
2) Initialize Min Heap with source vertex as root (the distance value assigned to source vertex is 0). The distance value assigned to all other vertices is INF (infinite). 
   
3) While Min Heap is not empty, do following: 
   
a) Extract the vertex with minimum distance value node from Min Heap. Let the extracted vertex be u. 

b) For every adjacent vertex v of u, check if v is in Min Heap. If v is in Min Heap and distance value is more than weight of u->v plus distance value of u, then update the distance value of v.

```python
class Heap():
  
    def __init__(self):
        self.array = [] # (v, dist), v is node index, dist is the dist value from v to source
        self.pos = []
        self.size = 0
    
    def isEmpty(self):
        return True if self.size == 0 else False

    # A utility function to swap two nodes of min heap. Needed for min heapify
    def swapMinHeapNode(self, a, b): # a, b is the idx in heap
        t = self.array[a]
        self.array[a] = self.array[b]
        self.array[b] = t
  
    # A standard function to heapify (down) at given idx. This function also updates position of nodes when they are swapped.
    def downHeapify(self, idx):
        smallest = idx
        left = 2*idx + 1
        right = 2*idx + 2
  
        if left < self.size and self.array[left][1] < self.array[smallest][1]:
            smallest = left
  
        if right < self.size and self.array[right][1] < self.array[smallest][1]:
            smallest = right
  
        # The nodes to be swapped in min heap if idx is not smallest
        if smallest != idx:
            # Swap positions
            self.pos[self.array[smallest][0]] = idx
            self.pos[self.array[idx][0]] = smallest
            # Swap nodes
            self.swapMinHeapNode(smallest, idx)
  
            self.downHeapify(smallest)
  
    # Standard function to extract minimum node from heap
    def pop(self): 
        # Return NULL wif heap is empty
        if self.isEmpty() == True:
            return None

        # Put last node in root then down heapify
        root = self.array[0]
        lastNode = self.array[self.size - 1]
        self.array[0] = lastNode
  
        # Update position of root and last node
        self.pos[lastNode[0]] = 0
        self.pos[root[0]] = self.size - 1
  
        # Reduce heap size and heapify root
        self.size -= 1
        self.downHeapify(0)
  
        return root
  
    def upHeapify(self, v, dist): # up heapify
  
        # Get the index of v in  heap array
        i = self.pos[v]
  
        # Get the node and update its dist value
        self.array[i][1] = dist
  
        # Travel up while the complete tree is not hepified. This is a O(Logn) loop
        while i > 0 and self.array[i][1] < self.array[(i - 1) / 2][1]:
            # Swap this node with its parent
            self.pos[ self.array[i][0] ] = (i-1)/2
            self.pos[ self.array[(i-1)/2][0] ] = i
            self.swapMinHeapNode(i, (i - 1)/2)
            # move to parent index
            i = (i - 1) / 2
  
    # A utility function to check if a given vertex 'v' is in min heap or not
    def isInMinHeap(self, v):
        if self.pos[v] < self.size:
            return True
        return False


class Graph():
    def dijkstra(self, src):
  
        V = self.V  # Get the number of vertices in graph
        dist = []   # dist values used to pick minimum weight edge in cut
  
        # minHeap represents set E
        minHeap = Heap()
  
        #  Initialize min heap with all vertices dist value
        for v in range(V):
            dist.append(sys.maxsize)
            minHeap.array.append((v, dist[v]))
            minHeap.pos.append(v)
  
        # Make dist value of src vertex as 0 so that it is extracted first
        minHeap.pos[src] = src
        dist[src] = 0
        minHeap.upHeapify(src, dist[src])
  
        # Initially size of min heap is equal to V
        minHeap.size = V
  
        # In the following loop, min heap contains all nodes whose shortest distance is not yet finalized.
        while not minHeap.isEmpty():
  
            # Extract the vertex with minimum distance value
            newHeapNode = minHeap.pop()
            u = newHeapNode[0]
  
            # Traverse through all adjacent vertices of u (the extracted vertex) and update their distance values
            for neighbor in self.graph[u]:
                v = neighbor[0]
                # If shortest distance to v is not finalized yet, and distance to v through u is less than its previously calculated distance
                if minHeap.isInMinHeap(v) and dist[u] != sys.maxsize and neighbor[1] + dist[u] < dist[v]:
                        dist[v] = neighbor[1] + dist[u]
                        # update distance value in min heap also
                        minHeap.upHeapify(v, dist[v])
  
        return dist
```

**邻接表存储，heapq 实现**，time complexity: $O(E*log V)$ 

上面的priority queue采用自己写的heap进行实现，优点是采用了pos数组存储了vertices在heap array中的位置，用函数isInMinHeap()来判断neighbor是否在堆中。

如果想使用heapq包中的数据结构，将没有isInMinHeap()也没有那么upHeapify()来更新堆中元素的信息。那么，另一种方法是：

Do not update a key, but insert one more copy of it. So we allow multiple instances of same vertex in priority queue. This approach doesn’t require decrease key operation and has below important properties.

1）Whenever distance of a vertex is reduced, we add one more instance of vertex in priority_queue. Even if there are multiple instances, we only consider the instance with minimum distance and ignore other instances.

2）The time complexity remains O(ELogV)) as there will be at most O(E) vertices in priority queue and O(Log E) is same as O(Log V)

Below is algorithm based on above idea.

     1) Initialize distances of all vertices as infinite.
     2) Create an empty priority_queue pq.  Every item of pq is a pair (weight, vertex). Weight (or distance) is used as first item of pair, and by default used to compare two pairs.
     3) Insert source vertex into pq and make its distance as 0.
     4) While either pq doesn't become empty
     a) Extract minimum distance vertex from pq. Let the extracted vertex be u.
     b) Loop through all adjacent of u and do following for every neighbor v.
            If dist[u] + weight(u, v) < dist[v] // If there is a shorter path to v through u. 
                 (i) Update distance of v, i.e., do dist[v] = dist[u] + weight(u, v)
                 (ii) Insert v into the pq (Even if v is already there)


```python
def dijkstra(self, src):

    visited = set()
    dist = [sys.maxsize]*V   # dist values used to pick minimum weight edgE
    minHeap = []  # minHeap represents set E that is unvisited

    # Make dist value of src vertex as 0 so that it is extracted first
    visited.add(src)
    dist[src] = 0
    heappush(minHeap, (0, src))

    # In the following loop, min heap contains all nodes whose shortest distance is not yet finalized.
    while minHeap:
        # Extract the vertex with minimum distance value
        newHeapNode = heappop(minHeap)
        src_to_u, u = newHeapNode # src_to_u == dist[u]

        if u not in visited: 
        # if src_to_u <= dist[u]: # only consider the neighbors of u when src_to_u is smaller than current distance in dist[u]
            # Traverse through all adjacent vertices of u (the extracted vertex) and update their distance values
            for neighbor in self.graph[u]:
                v, weight = neighbor[0]
                # If shortest distance to v is not finalized yet, and distance to v through u is less than its previously calculated distance
                if src_to_u + weight < dist[v]:
                    dist[v] = src_to_u + weight
                    # Push vertex v into heap
                    heappush(minHeap, (dist[v], v))

    return dist
```


## Bellman–Ford Algorithm

> 时间复杂度 **O(V*E)**
> 
> 空间复杂度 **O(V^2)**
> 
> 有负权边，有负环
> 
> 有向图无向图均可使用
> 
> 单源最短路算法
> 
> 基于动态规划

Algorithm 

    Input: Graph and a source vertex src
    Output: Shortest distance to all vertices from src. If there is a negative weight cycle, then shortest distances are not calculated, negative weight cycle is reported.
    1) Create an array dist[] of size V with all values as infinite except dist[src] where src is source vertex.
    2) Do following V-1 times where V is the number of vertices in given graph. 
        a) Do following for each edge u-v 
        If dist[v] > dist[u] + weight of edge uv, then update dist[v] 
        dist[v] = dist[u] + weight of edge uv
    3) This step reports if there is a negative weight cycle in graph. Do following for each edge u-v:
        If dist[v] > dist[u] + weight of edge uv, then “Graph contains negative weight cycle” 

```python
class Graph:
    def __init__(self, vertices):
        self.V = vertices # No. of vertices
        self.graph = [] # Use list to store all edges in graph
 
    # function to add an edge to graph 
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    # The main function that finds shortest distances from src to all other vertices using Bellman-Ford algorithm. The function also detects negative weight cycle
    def BellmanFord(self, src):
        # Step 1: Initialize distances from src to all other vertices as INFINITE
        dist = [sys.maxsize] * self.V
        dist[src] = 0
 
        # Step 2: Relax all edges |V| - 1 times. A simple shortest path from src to any other vertex can have at-most |V| - 1 edges
        for _ in range(self.V - 1):
            # Update dist value and parent index of the adjacent vertices of the picked vertex. Consider only those vertices which are still in queue
            for u, v, w in self.graph:
                if dist[u] != sys.maxsize and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
 
        # Step 3: check for negative-weight cycles. The above step guarantees shortest distances if graph doesn't contain negative weight cycle. If we get a shorter path, then there is a cycle.
        for u, v, w in self.graph:
            if dist[u] != sys.maxsize and dist[u] + w < dist[v]:
                print("Graph contains negative weight cycle")
                return None
                         
        return dist
```

为什么这样就可以了？

**做 n - 1 次已经足够了**。证明如下：

从原点开始走，到第 x 个节点，这中间只有 x - 1 条边（不考虑环路）。如果一个图有 n 个节点，那么即使用最啰嗦的走法，到达一个点顶多需要走 n - 1 条边（不考虑环路）。也就是顶多把所有的节点都经过一遍。

在第一轮对所有的边进行松弛的时候，被松弛的点其实只有从原点可以一步到达的点。其他的点所在的边 Edge( u -> v ) 中，u.distance 都是 ∞，v 无法被松弛。只有 start_point.distance 为 0，Edge( start_point -> v ) 中的 v 才可能被松弛。

以此类推，在第 i 轮中，被松弛的点只可能是距离原点 i 步的点。他们利用到的边是 Edge( vi - 1 -> vi)，其中 vi - 1 在上一轮松弛的过程中已经被松弛过，如果他能到达原点的话，vi - 1.distance 就不会是 ∞。

有些点可能有多种不同的到达方式，并且在第 i 步之前也松弛过。这其实没关系。如果第 i 步是最后一次到达他，所有能用来到达这个点的边都已经被计算机探索过（不然这就不是最后一次到达），所以这次松弛也将是它最后一次被松弛，之后到达他的 distance 就已经是最终结果值了。

根据上面提到的 2，不可能有节点出现 n - 1 步还到达不了的地方，即使一个点有多条路径可以到达（除非这个点真的无法到达），他的最多步数路径上的边也都被计算机探索过了。也就是说，他的最后一次被访问已经发生过，他的 distance 肯定已经是最终结果值了。没有任何一个点可以例外。所以 n - 1 次循环已经足够。

算法中的那个操作为什么要叫「松弛」呢？

根据我们之前对 n - 1 次松弛操作的作用的证明，可以想像，在一遍一遍的循环中，最短路径由假设的 ∞，逐渐减小到最终的结果值。这就好像是一个用 ∞ 距离撑起来的图，最开始张力非常大，马上就要撑爆了的感觉；然后一点一点的释放这些张力，让整个图「松弛」下来。


## Shortest Path Faster Algorithm 

> 时间复杂度 **O(k*E)**
> 
> 空间复杂度 **O(V+E)**
> 
> 有负权边，有负环
> 
> 有向图无向图均可使用
> 
> 单源最短路算法
> 
> 基于Bellman-Ford

Bellman-Ford算法属于一种暴力的算法，即，每次将所有的边都松弛一遍，这样肯定能保证顺序，但是仔细分析不难发现，源点s到达其他的点的最短路径中的第一条边，必定是源点s与s的邻接点相连的边，因此，第一次松弛，我们只需要将这些边松弛一下即可。第二条边必定是第一次松弛的时候的邻接点与这些邻接点的邻接点相连的边。因此我们可以这样进行优化：设置一个队列，初始的时候将源点s放入，然后s出队，松弛s与其邻接点相连的边，**将松弛成功的点放入队列中**，然后再次取出队列中的点，松弛该点与该点的邻接点相连的边，如果松弛成功，看这个邻接点是否在队列中，没有则进入，有则不管，这里要说明一下，如果发现某点u的邻接点v已经在队列中，那么将点v再次放到队列中是没有意义的。因为即时你不放入队列中，点v的邻接点相连的边也会被松弛，只有松弛成功的边相连的邻接点，且这个点没有在队列中，这时候稍后对其进行松弛才有意义。因为该点已经更新，需要重新松弛。

The shortest path faster algorithm is based on Bellman-Ford algorithm where every vertex is used to relax its adjacent vertices but in SPF algorithm, a queue of vertices is maintained and a vertex is added to the queue only if that vertex is relaxed. This process repeats until no more vertex can be relaxed. 

    1) Create an array d[] to store the shortest distance of all vertex from the source vertex. Initialize this array by infinity except for d[S] = 0 where S is the source vertex.
    2) Create a queue Q and push starting source vertex in it. 
        while Queue is not empty, do the following for each edge(u, v) in the graph 
        A) If d[v] > d[u] + weight of edge(u, v)
        b) d[v] = d[u] + weight of edge(u, v)
        c) If vertex v is not present in Queue, then push the vertex v into the Queue.

```python
# Function to compute the SPF algorithm
def shortestPathFaster(src, V):
 
    # Create array d to store shortest distance
    d = [sys.maxsize]*V
    in_queue = [False]*V # Boolean array to check if vertex is present in queue or not
    q = deque()
    
    d[src] = 0
    in_queue[src] = True
    q.append(src)
    n = 0 # used to check if there are negtive circles

    while q and n < V-1:
        # Take the front vertex from Queue
        u = q.popleft()
        in_queue[u] = False
 
        # Relaxing all the adjacent edges of vertex taken from the Queue
        for v, weight in graph[u]:
            if (d[v] > d[u] + weight):
                d[v] = d[u] + weight
                # Check if vertex v is in Queue or not, if not then append it into the Queue
                if (in_queue[v] == False):
                    q.append(v)
                    in_queue[v] = True

        n += 1
    
    if n == V-1 and q:
        print("Graph contains negative weight cycle")
        return None

    return d
```
