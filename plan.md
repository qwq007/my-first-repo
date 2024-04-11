# plan

[TOC]

## 知识点

数学：**线性筛法**，**逆元**，组合数，扩展欧几里得算法，欧拉函数

基础：**倍增**，博弈枚举（dfs搜索）,双向搜索（折半搜索）

DP：背包，区间，树形，状压*

字符串：**字符串哈希**，KMP算法

数据结构：**并查集**，**堆（优先队列）**，**链表**，**单调栈**，**单调队列**，**ST表**，树状数组...，**线段树**，主席树

图论：拓扑排序，**最小生成树**，**最短路**，二分图（建图），**网络流（不好学）**，强连通分量（还行）

杂项：离散化，双指针

## 快读

```c++
int read()
{
    char c=getchar();int x=0,f=1;
    while(c<'0'||c>'9'){if(c=='-')f=-1;c=getchar();}
    while(c>='0'&&c<='9'){x=x*10+c-'0';c=getchar();}
    return x*f;
}
```

## 字符串哈希

```c++
const int B = 23317, N = 1e9 + 7, M = 1e9 + 9;
pair<int, int> get_hash(string temp)
{
    int res1 = 0, res2 = 0;
    for (int i = 0 ; i < temp.size() ; i ++)
    {
        res1 = (res1 * B + temp[i]) % N;
        res2 = (res2 * B + temp[i]) % M;
    }
    return pair<int, int>(res1, res2);
}
```

## 线性筛法

又名欧拉筛法，O(n)，还可得到了每个数的最小质因子。

```c++
int pri[X], indPri;
void initPri(int nPri) //n >= 2
{
    indPri = 1;
    int visPri[nPri + 5];
    for (int i = 2 ; i <= nPri ; i ++)
        visPri[i] = 0;

    for (int i = 2 ; i <= nPri ; i ++)
    {
        if (!visPri[i])
            pri[indPri ++] = i;
        for (int j = 1 ; j < indPri ; j ++)
        {
            if (i * pri[j] > nPri)
                break;
            visPri[i * pri[j]] = 1;
            if (i % pri[j] == 0)
                break;
        }
    }
}
```

## 逆元

1. b为素数，则1/a的逆元为a^b-2^ mod b
2. 阶乘的逆元：n！逆元 = (n + 1) ! * (n + 1) % mod

## 并查集

1. 初始化：fa[i] = i

2. 查询

   ```python
   def find(i):
       if i == fa[i]:
           return i
       fa[i] = find(fa[i])
       return fa[i]
   ```

3. 合并

   ```python
   def union(i, j):
       fa[find(i)] = find(j)
   ```

4. 星球大战：https://www.luogu.com.cn/problem/P1197

   1. 题解：

      ```python
      [n, m] = list(map(int, input().split()))
      fa = [i for i in range(n)]#标记每个点的父级
      flag = [True for i in range(n)]#标记每个点是否完好
      v = [[] for i in range(n)]#标记每条边
      def find(c):#找祖级函数
          if c == fa[c]:
              return c
          fa[c] = find(fa[c])
          return fa[c]
      def union(c1, c2):#合并函数
          fa[find(c1)] = find(c2)
      for i in range(m):#输入边
          [x, y] = list(map(int, input().split()))
          v[x].append(y)
          v[y].append(x)
      k = int(input())
      bloN = []#记录被摧毁的点
      for i in range(k):
          temp = int(input())
          bloN.append(temp)
          flag[temp] = False#相应的点标记为被摧毁
      tot = n - k#最开始的连通体数目等于完好点的个数
      for fro in range(len(v)):#遍历每条边，连接完好点之间的边
          for to in v[fro]:
              if flag[fro] and flag[to] and find(fro) != find(to):#若两个点都完好且不是一个连通体（同一个连通体的两个点相连没有意义）
                  tot -= 1#因两点不在一个连通体，则相连后连通体个数减一
                  union(fro, to)#合并
      ans = [tot]#完好点之间的边相连好后的连通体数为最后一个答案
      for i in range(k - 1, -1 ,-1):#遍历每个被摧毁的点，从后往前依次修复
          #修复一个点，即连通体加一，修改该点状态为完好
          tot += 1
          flag[bloN[i]] = True
          #遍历与该点相连的边
          for to in v[bloN[i]]:
              if flag[to] and find(to) != find(bloN[i]):#边另一端的点完好且不在一个两连通体
                  tot -= 1#联通体减一
                  union(bloN[i], to)#合并
          ans.append(tot)#记录该点修复后的连通体数
      for i in range(len(ans) - 1, -1, -1):#反向输出答案
          print(ans[i])
      ```

5. 过家家：https://www.luogu.com.cn/problem/P1682

## 单调队列

1. 滑动窗口【模板】单调队列 https://www.luogu.com.cn/problem/P1886

   C++：

   ```c++
   #include <bits/stdc++.h>
   #define int long long
   using namespace std;
   const int X = 1e6 + 5;
   int n, k;
   int num[X];
   vector<int> ans1;
   deque<int> q1;
   vector<int> ans2;
   deque<int> q2;
   signed main()
   {
       cin >> n >> k;
       for (int i = 0 ; i < n ; i ++)
           cin >> num[i];
       for (int i = 0 ; i < n ; i ++)
       {
           while (q1.size() && q1.front() <= i - k)//从队头操作，剔除所有非窗口内的元素
               q1.pop_front();
           while (q1.size() && num[i] >= num[q1.back()])//从队尾操作，剔除所有比当前所遍历的元素小或等于的元素，维护单调递增队列
               q1.pop_back();
           q1.push_back(i);//队尾插入当前所遍历元素的索引
           if (i >= k - 1)//当遍历到索引大于k-1的元素时，记录答案
               ans1.push_back(num[q1.front()]);//队列单调递增，则最大值永远是队头的元素
           
           //最小值同理
           while (q2.size() && q2.front() <= i - k)
               q2.pop_front();
           while (q2.size() && num[i] <= num[q2.back()])
               q2.pop_back();
           q2.push_back(i);
           if (i >= k - 1)
               ans2.push_back(num[q2.front()]);
       }
       for (int i = 0 ; i < ans2.size() ; i ++)
           cout << ans2[i] << " ";
       cout << endl;
       for (int i = 0 ; i < ans1.size() ; i ++)
           cout << ans1[i] << " ";
       return 0;
   }
   ```

   python (TLE)：

   ```python
   from collections import deque
   [n, k] = list(map(int, input().split()))
   lit = list(map(int, input().split()))
   q1 = deque()
   ansMax = []
   q2 = deque()
   ansMin = []
   for i in range(n):
       while q1 and q1[0] <= i - k:
           q1.popleft()
       while q1 and lit[i] >= lit[q1[-1]]:
           q1.pop()
       q1.append(i)
       if i >= k - 1:
           ansMax.append(str(lit[q1[0]]))
   
       while q2 and q2[0] <= i - k:
           q2.popleft()
       while q2 and lit[i] <= lit[q2[-1]]:
           q2.pop()
       q2.append(i)
       if i >= k - 1:
           ansMin.append(str(lit[q2[0]]))
   
   print(' '.join(ansMin))
   print(' '.join(ansMax))
   ```

2. [Luogu P2698 Flowerpot S](https://www.luogu.com.cn/problem/P2698)

   ```c++
   #include <bits/stdc++.h>
   #define int long long
   using namespace std;
   const int X = 1e5 + 5;
   deque<int> qn;
   deque<int> qx;
   int N, D, ans = 2e9, L;
   struct node
   {
       int x, y;
   } num[X];
   bool cmp(node a, node b)
   {
       return a.x < b.x;
   }
   signed main()
   {
       cin >> N >> D;
       for (int i = 0 ; i < N ; i ++)
           cin >> num[i].x >> num[i].y;
       sort(num, num + N, cmp);
   
       for (int R = 0 ; R < N ; R ++)
       {
           while (qn.size() && num[R].y <= num[qn.back()].y)
               qn.pop_back();
           qn.push_back(R);
           while (qx.size() && num[R].y >= num[qx.back()].y)
               qx.pop_back();
           qx.push_back(R);
           while (L <= R && num[qx.front()].y - num[qn.front()].y >= D)
           {
               ans = min(ans, num[R].x - num[L].x);
               L ++;
               while (qn.size() && qn.front() < L)
                   qn.pop_front();
               while (qx.size() && qx.front() < L)
                   qx.pop_front();
           }
       }
       if (ans < 2e9)
           cout << ans;
       else
           cout << -1;
       return 0;
   }
   ```

## 单调栈

- [洛谷 P5788【模板】单调栈](https://www.luogu.com.cn/problem/P5788)

```c++
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int X = 3e6 + 5;
int z[X], ind, n, a[X];
int ans[X];
signed main()
{
    cin >> n;
    for (int i = 1 ; i <= n ; i ++)
        cin >> a[i];
    for (int i = 1 ; i <= n ; i ++)
    {
        while (ind > 0 && a[z[ind - 1]] < a[i])
        {
            ans[z[ind - 1]] = i;
            ind --;
        }
        z[ind ++] = i;
    }
    for (int i = 1 ; i <= n ; i ++)
        cout << ans[i] << " ";
    return 0;
}

```

## ST表

解决**可重复贡献问题**（对于运算opt，满足x opt x = x，则对应的区间询问就是一个可重复贡献问题，如max(x, x) = x, gcd(x, x) = x)

预处理O(nlogn)，O(1)回答每个查询，但不支持修改操作。

- 预处理二维数组st\[n]\[20]（如果n的数据范围到1e5，一般往后预处理到2^20^）

  遍历每个起点，先处理长度为2^1^，然后2^2^…………

- 预处理每个数的k（小于该数的最大2^k^）的k[n]（或者需要时直接logn）

- 每次查询[l, r]的最大值=max(st\[ l ][ k[len] ], st\[ r - (1 << k\[len]) + 1 ][ k[len] ])

[【模板】 ST表](https://www.luogu.com.cn/problem/P3865)

```c++
int st[X][50], kSt[X];
void initSt(int *eleA, int aSize)
{
    for (int j = 0 ; j <= 20 ; j ++)
        for (int i = 1 ; i + (1 << j) - 1 <= aSize ; i ++)
            if (j == 0)
                st[i][j] = eleA[i];
            else
                st[i][j] = max(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);

    int kindSt = 2;
    int ansSt = 0;
    for (int i = 1 ; i <= aSize ; i ++)
    {
        if (kindSt == i)
        {
            kindSt *= 2;
            ansSt ++;
        }
        kSt[i] = ansSt;
    }
}
int searchSt(int lSt, int rSt)
{
    int lenSt = rSt - lSt + 1;
    return max(st[lSt][kSt[lenSt]], st[rSt - (1 << kSt[lenSt]) + 1][kSt[lenSt]]);
}

```

## 堆

### 二叉堆

1. 向上调整（小根堆）

   ```c++
   void up(int now)
   {
       if (now <= 1)
           return ;
       int fa = now >> 1;
       if (heap[now] < heap[fa])
       {
           swap(heap[now], heap[fa]);
           up(fa);
       }
   }
   ```

2. 向下调整（小根堆）

   ```c++
   void down(int now)
   {
       if ((now << 1) >= ind)
           return ;
       int c = now << 1;
       if ((c | 1) < ind && heap[c | 1] < heap[c])
           c |= 1;
       if (heap[now] > heap[c])
       {
           swap(heap[now], heap[c]);
           down(c);
       }
   }
   ```

3. 构建堆（向下调整更快， 叶子节点无需调整）

   ```c++
   void build_heap_2() {
     for (i = n / 2 ; i >= 1 ; i--) down(i);
   }
   ```

### 对顶堆

动态维护一个序列上第k大的数，k值可能发生变化。

例题：[SP16254 RMID2 - Running Median Again](https://www.luogu.com.cn/problem/SP16254)

## 链表

### 双向链表

构建

```c++
int s[X], ind = n + 1, l[X], r[X];
for (int i = 1 ; i <= n ; i ++)
{
    l[i] = i - 1;
    r[i] = i + 1;
}
```

删除

```c++
r[l[i]] = r[i];
l[r[i]] = l[i];
```

插入

```c++
r[ind] = r[i];
l[ind] = i;
r[i] = ind;
l[i] = ind;
```

### 邻接表（同理vector\<type\> e[n]）

```c++
int head[n], cut = 1;
struct edge
{
    int to, dis, next;
} ed[m];
void add_edge(int u, int v, int w)
{
    ed[cut].to = v;
    ed[cut].dis = w;
    ed[cut].next = head[u];
    head[u] = cut ++;
}
for (int i = head[u] ; i ; i = ed[i].next)
    ...
```

## 线段树

维护区间信息的数据结构

线段树可以在O(log N)的时间复杂度内实现单点修改、区间修改、区间查询（区间求和，求区间最大值，求区间最小值)等操作。

[luogu P3372【模板】线段树 1](https://www.luogu.com.cn/problem/P3372)

```c++
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int X = 1e5 + 5;
int s[X], t[X<<2], n, m, lz[X<<2];
void build(int l, int r, int ind)
{
    if (l == r)
    {
        t[ind] = s[l];
        return ;
    }
    int mid = (l + r) / 2;
    build(l, mid, ind * 2);
    build(mid + 1, r, ind * 2 + 1);
    t[ind] = t[ind * 2] + t[ind * 2 + 1];
}

void down(int l, int r, int ind)
{
    if (l == r)
        return ;
    if (lz[ind])
    {
        int mid = (l + r) / 2;
        t[ind * 2] += (mid - l + 1) * lz[ind];
        t[ind * 2 + 1] += (r - mid) * lz[ind];
        lz[ind * 2] += lz[ind];
        lz[ind * 2 + 1] += lz[ind];
        lz[ind] = 0;
    }
}

void add(int al, int ar, int l, int r, int ind, int num)
{
    if (al <= l && r <= ar)
    {
        lz[ind] += num;
        t[ind] += num * (r - l + 1);
        return ;
    }
    int mid = (l + r) / 2;
    down(l, r, ind);
    if (al <= mid)
        add(al, ar, l, mid, ind * 2, num);
    if (ar >= mid + 1)
        add(al, ar, mid + 1, r, ind * 2 + 1, num);
    t[ind] = t[ind * 2] + t[ind * 2 + 1];
}
int getnum(int sl, int sr, int l, int r, int ind)
{
    if (sl <= l && r <= sr)
        return t[ind];
    int mid = (l + r) / 2, sum = 0;
    down(l ,r, ind);
    if (sl <= mid)
        sum += getnum(sl, sr, l, mid, ind * 2);
    if (sr >= mid + 1)
        sum += getnum(sl, sr, mid + 1, r, ind * 2 + 1);
    return sum;
}
signed main()
{
    cin >> n >> m;
    for (int i = 1 ; i <= n ; i ++)
        cin >> s[i];
    build(1, n, 1);
    for (int i = 1 ; i <= m ; i ++)
    {
        int op, x, y, k;
        cin >> op;
        if (op == 1)
        {
            cin >> x >> y >> k;
            add(x, y, 1, n, 1, k);
        }
        else if (op == 2)
        {
            cin >> x >> y;
            cout << getnum(x, y, 1, n, 1) << endl;
        }
    }
    return 0;
}

```

## 最小生成树

Kruskal 算法：一直加入最小边（贪心），直至已经加入了n-1条边。

找最小边用**堆**维护，加边时判断是否会成环用**并查集**维护。

[模板 最小生成树](https://www.luogu.com.cn/problem/P3366)

题解：

```c++
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int X = 2e5 + 5;
int n, m, x, y, z, ednum, ans;
int fa[5005];
int fi(int c)
{
    if (c == fa[c])
        return c;
    return fa[c] = fi(fa[c]);
}

void un(int c1, int c2)
{
    fa[fi(c1)] = fi(c2);
}

struct edge
{
    int u, v, dis;
} heap[X<<2];
int ind = 1;

void up(int now)
{
    if (now == 1)
        return ;
    int fa = now >> 1;
    if (heap[now].dis < heap[fa].dis)
    {
        swap(heap[now], heap[fa]);
        up(fa);
    }
}

void down(int now)
{
    if ((now << 1) >= ind)
        return ;
    int c = now << 1;
    if ((c | 1) < ind && heap[c | 1].dis < heap[c].dis)
        c |= 1;
    if (heap[now].dis > heap[c].dis)
    {
        swap(heap[now], heap[c]);
        down(c);
    }
}

void in(edge e)
{
    heap[ind ++] = e;
    up(ind - 1);
}

void pop()
{
    heap[1] = heap[-- ind];
    down(1);
}

signed main()
{
    cin >> n >> m;
    for (int i = 1 ; i <= n ; i ++)
        fa[i] = i;
    for (int i = 1 ; i <= m ; i ++)
    {
        cin >> x >> y >> z;
        in(edge{x, y, z});
    }
    while (ednum < n - 1 && ind != 1)
    {
        if (fi(heap[1].u) == fi(heap[1].v))
            pop();
        else
        {
            un(heap[1].u, heap[1].v);
            ednum ++;
            ans += heap[1].dis;
            pop();
        }
    }
    if (ednum == n - 1)
        cout << ans << endl;
    else
        cout << "orz" << endl;
    return 0;
}

```



## 树状数组

（单点修改，区间查询）、代码量小、维护的信息要满足**结合律**和**可差分**

## 博弈论

Nim游戏

![image-20230730113011125](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230730113011125.png)

![image-20230730113024076](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230730113024076.png)

## 网络流

### 最大流

#### Ford-Fulkerson 算法

- 初始化残量网络，所有边的空闲量等于容量

  while 存在起点到达终点的简单路径

  - 找到一条从起点到终点的路径
  - 找到该路径上的最小权重x
  - 该路径上的所有边都减去x
  - 添加从终点到起点的反向路径，所有边的权重都为x

- 每天边的流量 = 容量 - 剩余量

- 所有汇入终点的流量总和（或所有流出起点的流量总和）为最大流

  时间复杂度：O(f * m)（最大流的大小f，边数m）

#### Edmonds-Karp 算法

（Ford-Fulkerson 算法的特例，不依赖最大流的大小，有更好的时间复杂度）

算法步骤与Ford-Fulkerson 算法相似，区别在于路径找的是最短路径（图视为无权有向图）

时间复杂度：O(m^2^ * n)（边数m，点数n）

#### Dinic's 算法

时间复杂度：O(m * n^2^)（边数m，点数n）

- 初始化残量网络，所有边的空闲量等于容量

- while

  根据当前的残量网络构建level graph（去除无用路段）；

  寻找level graph中的阻塞流blocking flow（可用简单算法寻找），没有则终止循环；

  残量网路上减去找出的阻塞流（删掉剩余量为0的边），并生成相同权重的反向路径；

- 用当前的残量网络计算最大流

```c++
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int X = 1e5 + 5, inf = 2e9;
int n, m, s, t, ans, dis[X];
int cut = 2, now[X], head[X];

struct edge
{
    int to, next, dis;
} ed[X];

void add(int u, int v, int w)
{
    ed[cut].to = v;
    ed[cut].dis = w;
    ed[cut].next = head[u];
    head[u] = cut ++;

    ed[cut].to = u;
    ed[cut].dis = 0;
    ed[cut].next = head[v];
    head[v] = cut ++;
}

int bfs()
{
    for(int i = 1 ; i <= n ; i ++)
        dis[i] = inf;
    queue<int> q;
    q.push(s);
    dis[s] = 0;
    now[s] = head[s];
    while(!q.empty())
    {
        int x = q.front();
        q.pop();
        for(int i = head[x] ; i ; i = ed[i].next)
        {
            int v = ed[i].to;
            if(ed[i].dis > 0 && dis[v] == inf)
            {
                q.push(v);
                now[v] = head[v];
                dis[v] = dis[x] + 1;
                if(v == t)
                    return 1;
            }
        }
    }
    return 0;
}

int dfs(int x, int sum)
{
    if(x == t)
        return sum;
    int k, res = 0;
    for(int i = now[x] ; i && sum ; i = ed[i].next)
    {
        now[x] = i;
        int v = ed[i].to;
        if(ed[i].dis > 0 && dis[v] == dis[x] + 1)
        {
            k = dfs(v, min(sum, ed[i].dis));
            if(k == 0)
                dis[v] = inf;
            ed[i].dis -= k;
            ed[i ^ 1].dis += k;
            res += k;
            sum -= k;
        }
    }
    return res;
}

signed main()
{
    cin >> n >> m >> s >> t;
    while (bfs())
        ans += dfs(s, inf);
    cout << ans << endl;
    return 0;
}

```

[【模板】网络最大流](https://www.luogu.com.cn/problem/P3376)

### 最小割

最大流量 = 最小割的容量

## 最短路

#### Floyd 算法

任何图，最短路必须存在，不能有负环

```c++
for (int k = 1 ; k <= n ; k ++)
    for (int i = 1 ; i <= n ; i ++)
        for (int j = 1 ; j <= n ; j ++)
            if (m[i][k] + m[k][j] < m[i][j])
                m[i][j] = m[i][k] + m[k][j];
```

#### Bellman-Ford 算法

基于松弛操作，可求有负权图的最短路，可对最短路不存在的情况进行判断

#### SPFA

Bellman-Ford 算法的优化，时间复杂度O(nm)

```c++
struct edge {
  int v, w;
};

vector<edge> e[maxn];
int dis[maxn], cnt[maxn], vis[maxn];
queue<int> q;

bool spfa(int n, int s) {
  memset(dis, 63, sizeof(dis));
  dis[s] = 0, vis[s] = 1;
  q.push(s);
  while (!q.empty()) {
    int u = q.front();
    q.pop(), vis[u] = 0;
    for (auto ed : e[u]) {
      int v = ed.v, w = ed.w;
      if (dis[v] > dis[u] + w) {
        dis[v] = dis[u] + w;
        cnt[v] = cnt[u] + 1;  // 记录最短路经过的边数
        if (cnt[v] >= n) return false;
        // 在不经过负环的情况下，最短路至多经过 n - 1 条边
        // 因此如果经过了多于 n 条边，一定说明经过了负环
        if (!vis[v]) q.push(v), vis[v] = 1;
      }
    }
  }
  return true;
}

```

#### Dijkstra 算法

求解非负权图上单源最短路径的算法

![image-20230801121546675](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230801121546675.png)

```c++
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int X = 1e5 + 5, inf = 2e9;
int n, m, s, u, v, w, mini, minn;
int dis[X], vis[X], head[X], cut = 1;
struct edge
{
    int to, dis, next;
} ed[X*5];

void addEdge(int tfrom, int tto, int tdis)
{
    ed[cut].to = tto;
    ed[cut].dis = tdis;
    ed[cut].next = head[tfrom];
    head[tfrom] = cut ++;
}

signed main()
{
    cin >> n >> m >> s;
    for (int i = 1 ; i <= n ; i ++)
        dis[i] = inf;
    dis[s] = 0;
    vis[s] = 1;
    for (int i = 1 ; i <= m ; i ++)
    {
        cin >> u >> v >> w;
        addEdge(u, v, w);
    }
    while (1)
    {
        mini = -1;
        minn = inf;
        for (int i = 1 ; i <= n ; i ++)
        {
            if (vis[i] == 0 && dis[i] < minn)
            {
                mini = i;
                minn = dis[i];
            }
        }
        if (mini == -1)
            break;
        vis[mini] = 1;
        for (int i = head[mini] ; i ; i = ed[i].next)
            if (dis[ed[i].to] > minn + ed[i].dis)
                dis[ed[i].to] = minn + ed[i].dis;
    }
    for (int i = 1 ; i <= n ; i ++)
        cout << dis[i] << " ";
    return 0;
}

```

![image-20230801121557103](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230801121557103.png)

[【模板】单源最短路径（标准版）](https://www.luogu.com.cn/problem/P4779)

```c++
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int X = 1e5 + 5;
int n, m, s, u, v, w, inf = 2e9;
int ind = 1, dis[X], vis[X], cut = 1;
int head[X];
struct edge
{
    int to, dis, next;
} ed[X<<1];

void addEdge(int u1, int v1, int w1)
{
    ed[cut].to = v1;
    ed[cut].dis = w1;
    ed[cut].next = head[u1];
    head[u1] = cut ++;
}

struct node
{
    int pos, dis;
} heap[X<<2], mine;

void up(int now)
{
    if (now == 1)
        return ;
    int fa = now >> 1;
    if (heap[now].dis < heap[fa].dis)
    {
        swap(heap[now], heap[fa]);
        up(fa);
    }
}

void down(int now)
{
    if (now << 1 >= ind)
        return ;
    int c = now << 1;
    if ((c | 1) < ind && heap[(c | 1)].dis < heap[c].dis)
        c |= 1;
    if (heap[now].dis > heap[c].dis)
    {
        swap(heap[now], heap[c]);
        down(c);
    }
}

void build()
{
    for (int i = n / 2 ; i >= 1 ; i --)
        down(i);
}

void pop()
{
    heap[1] = heap[-- ind];
    down(1);
}

void push(node no)
{
    heap[ind ++] = no;
    up(ind - 1);
}

signed main()
{
    cin >> n >> m >> s;
    for (int i = 1 ; i <= n ; i ++)
    {
        dis[i] = inf;
        heap[i] = (node){i, dis[i]};
    }
    dis[s] = 0;
    vis[s] = 1;
    heap[s] = (node){s, 0};
    ind = n + 1;
    build();
    for (int i = 1 ; i <= m ; i ++)
    {
        cin >> u >> v >> w;
        addEdge(u, v, w);
    }
    while (ind != 1)
    {
        mine = heap[1];
        pop();
        if (vis[mine.pos])
            continue;
        vis[mine.pos] = 1;
        for (int i = head[mine.pos] ; i ; i = ed[i].next)
        {
            if (dis[ed[i].to] > dis[mine.pos] + ed[i].dis)
            {
                dis[ed[i].to] = dis[mine.pos] + ed[i].dis;
                if (!vis[ed[i].to])
                    push((node){ed[i].to, dis[ed[i].to]});
            }
        }
    }
    for (int i = 1 ; i <= n ; i ++)
        cout << dis[i] << " ";
    return 0;
}

```

![image-20230801121608116](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230801121608116.png)

