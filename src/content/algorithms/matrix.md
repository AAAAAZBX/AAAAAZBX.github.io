---
title: "矩阵学习笔记"
id: "2025-12-01"
date: "2025-12-01"
description: "主要讲解矩阵乘法，线性基，高斯消元几种常见的矩阵相关算法"
tags: ["矩阵", "矩阵乘法", "线性基", "高斯消元"]
---

## 矩阵乘法

### 牛客例题-E斐波

**题意**

> 定义$fib_n$为斐波那契数列第$n$项，$fib_1=fib_2=1$，$fib_n=fib_{n-1}+fib_{n-2}$
>
> 给定可重集$S$，定义$f(S)=\sum_{T \subseteq S }fib(\sum_{s\isin T}s)^2$,
>
> 支持两种操作：
>
> (1)单点修改$a_p\rightarrow v$
>
> (2)区间查询$\sum_{l\le i\le j \le r}f(\{a_{i},a_{i+1},...,a_j\})$.

**题解**

首先有定理：两个可以线性递推的公式相乘所得到公式一定是可以线性递推的。令 $g(n)=fib^2(n)$ 根据下面的推导：
$$
g(n) =fib^2(n)

        \\ =(fib(n-1)+fib(n-2))^2

    \\  =g(n-1)+g(n-2)+2fib(n-1)fib(n-2)

  \\     =g(n-1)+g(n-2)+2g(n-2)+2fib(n-2)fib(n-3)
$$
使用$g(n-1)$消去原来的式子中的$2fib(n-2)fib(n-3)$.

最终得到 $g(n)$ 的线性递推式为 $g(n)=2g(n-1)+2g(n-2)-g(n-3)$ .可以通过矩阵快速幂快求出第$n$项的值：

$$
\begin{bmatrix}
g(n)   & g(n-1) & g(n-2)
\end{bmatrix}
=
\begin{bmatrix}
g(n-1) & g(n-2) & g(n-3)
\end{bmatrix}
\begin{bmatrix}
2 & 1 & 0 \\\\
2 & 0 & 1 \\\\
-1 & 0 & 0
\end{bmatrix}
$$
对于公式$f(S)=\sum_{T \subseteq S }fib(\sum_{s\isin T}s)^2=\sum_{T \subseteq S }g(\sum_{s\isin T}s)$.

根据昨天求区间斐波那契函数和的经验，对于要使用线段树维护的值可以线性递推的情况，可以令其为矩阵形式，所以令 $G(S)$ 为$[g(n),g(n-1),g(n-2)]$. 原式变成了$F(S)=\sum_{T\subseteq S}G(\sum_{s \isin T}s )$.

为了方便求 $G(s)$ ,可以令 $G(0)=[g(0),g(-1),g(-2)]$ ,则 $G(s)=G(0)*base^s$.

然后观察 $f(S)$ 的性质，在$S$中新加入一个元素$a$后可以得到$f(S \cup \{a\})=f(S)+\sum_{T \subseteq S }g(a+\sum_{s\isin T}s)$。

又知 $F(\phi)=G(0)$， 所以拓展到矩阵 $F(S)$ 的形式上可以得到 :
$$
F(S \cup \{a\})
\\=F(\phi)*[F(S)+\sum_{T \subseteq S }G(a+\sum_{s\isin T}s)]
\\=F(\phi)*F(S)+F(\phi)*F(S)*base^a
\\=F(\phi)*F(S)*(E+base^a)
\\=G(0)*F(S)*(E+base^a)
$$
所以对于区间维护$\sum_{l\le i\le j \le r}f(\{a_{i},a_{i+1},...,a_j\})$的值，就是在每个区间上维护$\prod(E+base^{a_i})$，这样就转变为了一颗数据结构问题：
$$
\sum_{l\le i\le j \le r}F(\{a_{i},a_{i+1},...,a_j\})
\\=G(0)\sum_{l\le i \le j \le r}\prod_{k=i}^j(E+base^{a_k})
$$
在线段树中需要维护四个矩阵（先令$v_i=(E+base_{a_i})$以便后续表示），所求即为$G(0)\sum_{l\le i \le j \le r}\prod_{k=i}^jv_k$：

> 整个区间的总乘积矩阵$sum=\prod_{i=l}^{r}v_i$.
>
> 区间从左端点开始得到的所有前缀字段成绩之和$suml=\sum_{i=l}^{r}\prod_{k=l}^iv_k$.
>
> 区间从右端点开始得到的所有后缀字段成绩之和$sumr=\sum_{i=l}^{r}\prod_{k=i}^rv_k$.
>
> 区间答案 $ans=\sum_{l\le i\le j\le r}\prod_{k=l}^iv_k$.

有了上面四个矩阵便可以进行区间合并了

> $u.sum=l.sum*r.sum$.
>
> $u.suml=l.suml+l.sum*r.suml$.
>
> $u.sumr=r.sumr+r.sum*l.sumr$.
>
> $u.ans=l.ans+r.ans+l.sumr*r.suml$.

综上便可以使用线段树维护区间矩阵乘法解决这个问题，本题实现方面其实就是多个技巧凑在一起的，赛场上遇到需要攻克不少难点，比如需要知道两个可以线性递推的公式相乘所得到公式一定是可以线性递推的，线段树可以维护的值能用矩阵乘法做出来的话考虑使用矩阵代替要维护的信息，在数据结构的四个矩阵设计上也比较巧妙，做完能学到不少技巧，具体代码如下：

```c++
#include <bits/stdc++.h>
using namespace std;
const int mod = 998244353;
typedef long long ll;
typedef double ld;
const int N = 1e5 + 10;
struct Matrix {
    int n = 3;
    ll a[3][3];
    Matrix() {memset(a,0,sizeof a);}
    Matrix operator*(const Matrix& t) const {
        Matrix ans;
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                for (int k = 0;k < n;k++) {
                    ans.a[i][j] = (ans.a[i][j] + this->a[i][k] * t.a[k][j] % mod) % mod;
                }
            }
        }
        return ans;
    }
    Matrix operator*(ll x) const {
        Matrix ans;
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                ans.a[i][j] = this->a[i][j] * x % mod;
            }
        }
        return ans;
    }
    Matrix operator+(const Matrix& t) const {
        Matrix ans;
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                ans.a[i][j] = (this->a[i][j] + t.a[i][j]) % mod;
            }
        }
        return ans;
    }
    Matrix power(ll k) {
        Matrix ans;
        Matrix base = *this;
        for (int i = 0;i < n;i++) ans.a[i][i] = 1ll;
        for (;k;k >>= 1, base = base * base) {
            if (k & 1) {
                ans = ans * base;
            }
        }
        return ans;
    }
    bool operator!=(const Matrix& t) {
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                if (a[i][j] != t.a[i][j])return 1;
            }
        }
        return 0;
    }
    void output() {
        cout << "matrix\n";
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                cout << a[i][j] << ' ';
            }
            cout << '\n';
        }
    }
}I, G, base;
struct Node {
    int l, r;
    Matrix sum, suml, sumr, ans;
};
vector<Node>tr(N << 2);
int a[N];
int n, m;
inline void pushup(Node& u, const Node& l, const Node& r) {
    u.sum = l.sum * r.sum;
    u.suml = l.suml + l.sum * r.suml;
    u.sumr = r.sumr + r.sum * l.sumr;
    u.ans = l.ans + r.ans + l.sumr * r.suml;
}
inline void pushup(int u) {
    pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
}
void build(int u, int l, int r) {
    tr[u] = { l,r };
    if (l == r) {
        int x;cin >> x;
        tr[u].sum = tr[u].suml = tr[u].sumr = tr[u].ans = I + base.power(x);
        return;
    }
    int mid = l + r >> 1;
    build(u << 1, l, mid);
    build(u << 1 | 1, mid + 1, r);
    pushup(u);
}
void modify(int u, int x, ll v) {
    if (tr[u].l == tr[u].r) {
        tr[u].sum = tr[u].suml = tr[u].sumr = tr[u].ans = I + base.power(v);
        return;
    }
    int mid = tr[u].l + tr[u].r >> 1;
    if (x <= mid)modify(u << 1, x, v);
    else modify(u << 1 | 1, x, v);
    pushup(u);
}
Node query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) return tr[u];
    int mid = tr[u].l + tr[u].r >> 1;
    if (r <= mid)return query(u << 1, l, r);
    else if (l > mid)return query(u << 1 | 1, l, r);
    else {
        Node res;
        pushup(res, query(u << 1, l, r), query(u << 1 | 1, l, r));
        return res;
    }
}
void init() {
    for (int i = 0;i < 3;i++)I.a[i][i] = 1;
    G.a[0][0] = 0;
    G.a[0][1] = 1;
    G.a[0][2] = 1;
    base.a[0][0] = 2, base.a[0][1] = 1, base.a[0][2] = 0;
    base.a[1][0] = 2, base.a[1][1] = 0, base.a[1][2] = 1;
    base.a[2][0] = 998244352, base.a[2][1] = 0, base.a[2][2] = 0;
}
int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> n >> m;
    init();
    build(1, 1, n);
    while (m--) {
        int op, l, r;
        cin >> op >> l >> r;
        if (op == 1)modify(1, l, r);
        else cout << (G * query(1, l, r).ans).a[0][0] << '\n';
    }
    return 0;
}
```

### 2021 新疆省赛I Fibonacci sequence

**题意**：$f(x)=f(x-1)+f(x-2)+2\sqrt {3+f(n-1)f(n-2)}$.，给定$f(0),f(1),M,n,x_i$，求$\prod_{i=1}^nf(x_i)mod M$.（保证$ \sqrt {3+f(n-1)f(n-2)}$是整数）.

**题解**：首先令$s_n=\sqrt{3+f(n)f(n-1)}$，根据推导可知：
$$
s_n^2=3+f(n)f(n-1)
\\=3+(f(n-1)+f(n-2)+2\sqrt{3+f(n-1)f(n-2)})f(n-1)
\\=s_{n-1}^2+f^2(n-1)+2f(n-1)\sqrt{3+f(n-1)f(n-2)}
\\=s_{n-1}^2+2f(n-1)s_{n-1}+f^2(n-1)
\\=(s_{n-1}+f(n-1))^2
$$
所以$s_n=s_{n-1}+f(n-1)$.

将$f(n)=s_{n+1}-s_n$代入原式子$f(n)=f(n-1)+f(n-2)+2s_{n-1}$可以得到：

$s_{n+1}=2s_n+2s_{n-1}-s_{n-2}$，进而可以得到$f(n)=2f(n-1)+2f(n-2)-f(n-3)$.

> ps: 很奇怪，这个题目的线性递推式和前面第一道题的斐波那契额平方函数的递推式子居然一模一样，上午留的代码的转移矩阵都不用变，gpt的解释有好多不认识的术语（特征多项式，特征根，Binet公式），搞不懂。

但是还是不够，如果每一项都做一次矩阵乘法的话，时间复杂度为$O(n\cdot3^3\cdot log(v))$，大约为8e8，TLE的风险很大，所以需要做一些优化。

这里又学了一个技巧：光速幂，假设底数base固定，但是幂次是$10^9$级别的，可以令$B=\sqrt{10^9}+1$,预处理两个数组，$fl[i]$ 表示$base^{i*B}$,$fr[i]$表示$base^i$，数组大小均为B，这样对于任意的$0 \le x \le 10^9$,都可以通过$fl[x/B]*fr[x\%B]$来得到$base^x$的值。

经过优化后，预处理部分的复杂度为$O(B)$，总时间复杂度为$O(B+n\cdot 3^3)$.

```c++
#include <bits/stdc++.h>
using namespace std;
int mod = 998244353;
typedef long long ll;
typedef double ld;
const int N = 1e5 + 10;
struct Matrix {
    int n = 3;
    ll a[3][3];
    Matrix() {memset(a,0,sizeof a);}
    Matrix operator*(const Matrix& t) const {
        Matrix ans;
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                for (int k = 0;k < n;k++) {
                    ans.a[i][j] = (ans.a[i][j] + this->a[i][k] * t.a[k][j] % mod) % mod;
                }
            }
        }
        return ans;
    }
    Matrix operator*(ll x) const {
        Matrix ans;
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                ans.a[i][j] = this->a[i][j] * x % mod;
            }
        }
        return ans;
    }
    Matrix operator+(const Matrix& t) const {
        Matrix ans;
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                ans.a[i][j] = (this->a[i][j] + t.a[i][j]) % mod;
            }
        }
        return ans;
    }
    Matrix power(ll k) {
        Matrix ans;
        Matrix base = *this;
        for (int i = 0;i < n;i++) ans.a[i][i] = 1ll;
        for (;k;k >>= 1, base = base * base) {
            if (k & 1) {
                ans = ans * base;
            }
        }
        return ans;
    }
    bool operator!=(const Matrix& t) {
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                if (a[i][j] != t.a[i][j])return 1;
            }
        }
        return 0;
    }
    void output() {
        cout << "matrix\n";
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                cout << a[i][j] << ' ';
            }
            cout << '\n';
        }
    }
}I, G, base;
int a, b, n, B, c;
Matrix fl[100010], fr[100010];
void init() {
    B = sqrt(1e9 + 7) + 1;
    c = 2 * sqrtl(3 + 1ll * a * b) + a + b;
    for (int i = 0;i < 3;i++)I.a[i][i] = 1;
    G.a[0][0] = c;
    G.a[0][1] = b;
    G.a[0][2] = a;
    base.a[0][0] = 2, base.a[0][1] = 1, base.a[0][2] = 0;
    base.a[1][0] = 2, base.a[1][1] = 0, base.a[1][2] = 1;
    base.a[2][0] = mod - 1, base.a[2][1] = 0, base.a[2][2] = 0;
    fl[0] = fr[0] = I;
    for (int i = 1;i <= B;i++)fr[i] = fr[i - 1] * base;
    for (int i = 1;i <= B;i++)fl[i] = fl[i - 1] * fr[B];
}
int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> a >> b >> mod >> n;
    init();
    ll ans = 1;
    for (int i = 1, x;i <= n;i++) {
        cin >> x;
        if (x <= 2)ans = ans * (x == 1 ? b : c) % mod;
        else{
            x -= 2;
            ans = ans * (G * fl[x / B] * fr[x - x / B * B]).a[0][0] % mod;
        }
    }
    cout << ans << '\n';
    return 0;
}
```

## 线性基

### 牛客例题-F POJ1222 EXTENDED LIGHTS OUT

给定一个5*6的01矩阵，每次操作一个点会使得自己和周围4个灯变化，求应该按下哪些开关，才能使得整个矩阵变为全0。

很早以前在acwing上做过，当时是枚举第一行的$2^6$种操作可能，然后顺序操作每一行。

学完高斯消元以后再来做有又了一种思路，30个点对应30个方程组，直接使用高斯消元求解异或线性方程组即可。

```c++
#include <bits/stdc++.h>
using namespace std;
int dx[5] = { 0,0,0,-1,1 };
int dy[5] = { 0,1,-1,0,0 };
int n;
int main() {
    int _;
    cin >> _;
    for (int __ = 1; __ <= _;__++) {
        vector a(5, vector<int>(6));
        for (auto& u : a)for (auto& v : u)cin >> v;
        vector<bitset<30>>g(30);vector<int>w(30), ans(30);
        bitset<30>b;
        for (int i = 0;i < 5;i++) {
            for (int j = 0;j < 6;j++) {
                b.reset();
                for (int k = 0;k < 5;k++) {
                    int x = dx[k] + i, y = dy[k] + j;
                    if (x >= 0 && x < 5 && y >= 0 && y < 6) {
                        b.set(x * 6 + y);
                    }
                }
                g[i * 6 + j] = b;
                w[i * 6 + j] = a[i][j];
            }
        }
        for (int c = 0;c < 30;c++) {
            int p = -1;
            for (int i = c;i < 30;i++) {
                if (g[i][c]) { p = i; break; }
            }
            if (p != c)swap(g[p], g[c]), swap(w[p], w[c]);
            for (int i = c + 1;i < 30;i++) {
                if (g[i][c]) {
                    g[i] ^= g[c], w[i] ^= w[c];
                }
            }
        }
        for (int i = 29;i >= 0;i--) {
            for (int j = i - 1;j >= 0;j--) {
                if (g[j][i])w[j] ^= w[i];
            }
        }
        cout << "PUZZLE #" << __;
        for (int i = 0;i < 30;i++) {
            if (i % 6 == 0)cout << '\n';
            cout << w[i] << " ";
        }
        cout << '\n';
    }
    return 0;
}
```

### 牛客例题-I CF1101G (Zero XOR Subset)-less

**题意：**给出一个长度为$n$的序列$a_i$，将其划分为尽可能多的非空子段，满足每一个元素出现且仅出现在其中一个子段中，且在这些子段中任取若干子段，它们包含的所有数的异或和不能为0．

**题解：**考虑使用前缀和，对于每个子段$[l,r]$,其异或和为$S(r) \oplus S(l-1)$,假设每段的右端点为$v_1,v_2,...v_k=n$,那么这些序列的异或和分别为$S(v_1),S(v_2)\oplus S(v_1),...S(n)\oplus S(v_k)$，这些数的信息（这个信息可以用线性代数中的秩来理解）可以用$S(v_1),S(v_2),...,S(v_k)$来表示，题目要求异或和不为0，即这些向量组线性无关，所以得到这些向量的秩后直接输出即可。

```c++
#include <bits/stdc++.h>
using namespace std;
struct LB { // Linear Basis
    using i64 = long long;
    const int BASE = 63;
    vector<i64> d, p;
    int cnt, flag;
    LB() {
        d.resize(BASE + 1);
        p.resize(BASE + 1);
        cnt = flag = 0;
    }
    bool insert(i64 val) {
        for (int i = BASE - 1; i >= 0; i--) {
            if (val & (1ll << i)) {
                if (!d[i]) {
                    d[i] = val;
                    return true;
                }
                val ^= d[i];
            }
        }
        flag = 1; //可以异或出0
        return false;
    }
    bool check(i64 val) { // 判断 val 是否能被异或得到
        for (int i = BASE - 1; i >= 0; i--) {
            if (val & (1ll << i)) {
                if (!d[i]) {
                    return false;
                }
                val ^= d[i];
            }
        }
        return true;
    }
    i64 ask_max() {
        i64 res = 0;
        for (int i = BASE - 1; i >= 0; i--) {
            if ((res ^ d[i]) > res) res ^= d[i];
        }
        return res;
    }
    void rebuild() { // 第k小值独立预处理
        for (int i = BASE - 1; i >= 0; i--) {
            for (int j = i - 1; j >= 0; j--) {
                if (d[i] & (1ll << j)) d[i] ^= d[j];
            }
        }
        for (int i = 0; i <= BASE - 1; i++) {
            if (d[i]) p[cnt++] = d[i];
        }
    }
    i64 kthquery(i64 k) { // 查询能被异或得到的第 k 小值, 如不存在则返回 -1
        if (flag) k--; // 特判 0, 如果不需要 0, 直接删去
        if (!k) return 0;
        i64 res = 0;
        if (k >= (1ll << cnt)) return -1;
        for (int i = BASE - 1; i >= 0; i--) {
            if (k & (1LL << i)) res ^= p[i];
        }
        return res;
    }
    bool solve(i64 x, i64 y) {
        for (int i = BASE - 1;i >= 0;i--) {
            int xi = x >> i & 1, yi = y >> i & 1;
            if (xi != yi)x ^= d[i];
        }
        return x == y;
    }
    void Merge(const LB& b) { // 合并两个线性基
        for (int i = BASE - 1; i >= 0; i--) {
            if (b.d[i]) {
                insert(b.d[i]);
            }
        }
    }
}lb;
int n, m;
int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    int sum = 0;
    lb.insert(sum);
    for (int i = 1, x;i <= n;i++) {
        cin >> x;
        sum ^= x;
        lb.insert(sum);
    }
    if (sum == 0)cout << "-1" << '\n';
    else {
        lb.rebuild();
        cout << lb.cnt << '\n';
    }
    return 0;
}
```

## 高斯消元解异或方程组

### 2024ICPC昆明区域赛E Extracting Weights

把长度为k的路径全部找出来，使用bitset记录每个路径上有哪些点，记录起点和终点后面用来询问，然后正着做一遍（因为这个时候$w_i$还未知），得到矩阵的秩，如果无解，即有些变量无法求解，直接输出No即可。

然后询问刚刚消元得到的前n个线性无关的线程方程组的起终点，重新做一遍高斯消元即可（正着反着都做一遍），注意原始的起终点信息要随着消元的过程进行交换以便用于后面询问，但是线性方程组要重新构建，不能使用已经正着做过一遍的旧的线性方程组，时间复杂度$O(n^3/w)$.

```c++
#include <bits/stdc++.h>
#define pb push_back
using namespace std;
const int N = 300;
int n, k;
int dep[N], fa[N];
vector<int>e[N];
void dfs(int u, int f) {
    dep[u] = dep[f] + 1;
    for (auto v : e[u]) {
        if (v ^ f) {
            fa[v] = u;
            dfs(v, u);
        }
    }
}
void find(int st, int ed, vector<int>& path) {
    if (dep[st] < dep[ed])swap(st, ed);
    while (dep[st] != dep[ed]) {
        path.pb(st);
        st = fa[st];
    }
    while (st != ed) {
        path.pb(st), path.pb(ed);
        st = fa[st], ed = fa[ed];
    }
    path.pb(st);
}
int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> n >> k;
    for (int i = 1, u, v;i < n;i++) {
        cin >> u >> v;
        e[u].pb(v), e[v].pb(u);
    }
    dfs(1, 0);
    vector<bitset<250>>g;
    vector<int>w(n), st, ed;
    bitset<250>b;
    b.set(0);
    st.pb(1), ed.pb(1);
    g.pb(b);
    for (int i = 1;i <= n;i++) {
        for (int j = i + 1;j <= n;j++) {
            vector<int>path;
            find(i, j, path);
            b.reset();
            if (path.size() == k + 1) {
                for (auto t : path)b.set(t - 1);
                g.pb(b), st.pb(i), ed.pb(j);
            }
        }
    }
    for (int c = 0;c < n;c++) {
        int p = -1;
        for (int j = c;j < g.size();j++) {
            if (g[j][c]) {
                p = j;
                break;
            }
        }
        if (p == -1) { cout << "NO\n";cout.flush();return 0; }
        if (p != c)swap(g[p], g[c]), swap(st[p], st[c]), swap(ed[p], ed[c]);
        for (int i = c + 1;i < g.size();i++) {
            if (g[i][c]) {
                g[i] ^= g[c];
            }
        }
    }
    cout << "Yes" << endl;
    cout << "? " << n - 1;
    for (int i = 1;i < n;i++)cout << " " << st[i] << " " << ed[i];
    cout << endl;
    cout.flush();
    g.clear();
    b.reset();b.set(0);g.pb(b);
    for (int i = 1;i < n;i++) {
        cin >> w[i];
        vector<int>path;
        b.reset();
        find(st[i], ed[i], path);
        for (auto t : path)b.set(t - 1);
        g.pb(b);
    }
    for (int c = 0;c < n;c++) {
        int p = -1;
        for (int j = c;j < g.size();j++) {
            if (g[j][c]) {p = j;break;}
        }
        if (p != c)swap(g[p], g[c]), swap(w[p], w[c]);
        for (int i = c + 1;i < g.size();i++) {
            if (g[i][c]) {
                g[i] ^= g[c], w[i] ^= w[c];
            }
        }
    }
    for (int c = n - 1;c >= 0;c--) {
            for (int j = c - 1;j >= 0;j--) {
            if (g[j][c])w[j] ^= w[c];
        }
    }
    cout << "! ";
    for (int i = 1;i < n;i++)cout << w[i] << " ";
    cout << endl;
    cout.flush();
    return 0;
}
```



