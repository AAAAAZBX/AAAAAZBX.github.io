---
title: "Codeforces Edu学习 (ITMO Academy: pilot course) SegmentTree part 1-2 全"
id: "2025-06-30"
date: "2025-06-30"
description: "线段树"
tags: ["线段树", "懒标记", "线段树上二分"]
---

博主比较菜，从23年9月份开始学习Codeforces edu上的线段树专题，中间做的断断续续，到25年6月份才翻出来最后一题并AK掉了，主要为个人复习回顾用，所以一些地方不会讲的特别详细，建议再参考一下网上其他人的题解，后面博主有时间的话会把一些题目的题解补全。

<div style="display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap; justify-content:center;">
  <img src="/algorithms/codeforces-edu.png" alt="image-20260314211128851" style="height:300px; width:auto; max-width:100%;" />
  <img src="/algorithms/segmenttree.png" alt="image-202603142128851" style="height:300px; width:auto; max-width:100%;" />
</div>

## [Segment Tree, part 1](https://codeforces.com/edu/course/2/lesson/4)

### Step 1: [theory](https://codeforces.com/edu/course/2/lesson/4/1), [practice](https://codeforces.com/edu/course/2/lesson/4/1/practice): 3 of 3

**A. Segment Tree for the Sum**

**题意**：线段树处理区间和

没什么好说的，线段树处理区间加。

```cpp
typedef long long ll;
struct Node{
    int l,r;
    ll sum;
}tr[400010];
ll a[100010];
int n, m, k, l, r;
void pushup(int u){
    tr[u].sum = (ll)tr[u << 1].sum + (ll)tr[u << 1 | 1].sum;
}
void build(int u, int l, int r){
    if (l == r)tr[u] = { l,r,a[l] };
    else {
        tr[u] = { l,r };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int x, int v){
    if (tr[u].l == x && tr[u].r == x)tr[u] = { x,x,v };
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
ll query(int u, int l, int r){
    if (tr[u].l >= l && tr[u].r <= r)return tr[u].sum;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        ll sum = 0;
        if (r <= mid)return query(u << 1, l, r);
        else if (l > mid)return query(u << 1 | 1, l, r);
        else {
            auto left = query(u << 1, l, r);
            auto right = query(u << 1 | 1, l, r);
            return left + right;
        }
    }
}
int main(){
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++)scanf("%lld", &a[i]);
    build(1, 1, n);
    while (m--){
        scanf("%d%d%d", &k, &l, &r);
        if (k == 1)modify(1, l + 1, r);
        else printf("%lld\n", query(1, l + 1, r));
    }
    return 0;
}
```

B. Segment Tree for the Minimum

**题意**：线段树处理区间最小值

线段树处理区间最小值。

```cpp
typedef long long ll;
struct Node{
    int l,r;
    ll mn;
}tr[400010];
ll a[100010];
int n, m, k, l, r;
void pushup(int u){
    tr[u].mn = min(tr[u << 1].mn , (ll)tr[u << 1 | 1].mn);
}
void build(int u, int l, int r){
    if (l == r)tr[u] = { l,r,a[l] };
    else {
        tr[u] = { l,r };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int x, int v){
    if (tr[u].l == x && tr[u].r == x)tr[u] = { x,x,v };
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
ll query(int u, int l, int r){
    if (tr[u].l >= l && tr[u].r <= r)return tr[u].mn;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (r <= mid)return query(u << 1, l, r);
        else if (l > mid)return query(u << 1 | 1, l, r);
        else {
            auto left = query(u << 1, l, r);
            auto right = query(u << 1 | 1, l, r);
            return min(left, right);
        }
    }
}
int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++)scanf("%lld", &a[i]);
    build(1, 1, n);
    while (m--)
    {
        scanf("%d%d%d", &k, &l, &r);
        if (k == 1)modify(1, l + 1, r);
        else printf("%lld\n", query(1, l + 1, r));
    }
    return 0;
}
```

**C. Number of Minimums on a Segment**

**题意**：线段树处理区间最小值数量

记录**mn**为最小值，**num**为最小值数量

```cpp
const int N = 100010;
struct Node {
    int l, r;
    int mn, num;
}tr[N << 2];
int n, m;
int a[N];
void pushup(Node& u, Node& l, Node& r) {
    if (l.mn == r.mn)u.mn = l.mn, u.num = l.num + r.num;
    else if (l.mn < r.mn)u.mn = l.mn, u.num = l.num;
    else u.mn = r.mn, u.num = r.num;
}
void pushup(int u) {
    pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,l,a[l],1 };
    else {
        tr[u] = { l,r };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int x, int v) {
    if (tr[u].l == tr[u].r)tr[u].mn = v;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
Node query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)return tr[u];
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (r <= mid)return query(u << 1, l, r);
        else if (l > mid)return query(u << 1 | 1, l, r);
        else {
            Node res = { 0,0,0,0 };
            Node left = query(u << 1, l, r);
            Node right = query(u << 1 | 1, l, r);
            pushup(res, left, right);
            return res;
        }
    }
}
int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++)cin >> a[i];
    build(1, 1, n);
    while (m--) {
        int op, l, r;
        cin >> op >> l >> r; 
        l++;
        if (op == 1)modify(1, l, r);
        else {
            Node res = query(1, l, r);
            cout << res.mn << " " << res.num << endl;
        }
    }
    return 0;
}
```

### Step 2: [theory](https://codeforces.com/edu/course/2/lesson/4/2), [practice](https://codeforces.com/edu/course/2/lesson/4/2/practice): 4 of 4

**A. Segment with the Maximum Sum**

**题意**：线段树处理区间最大子段和

线段树维护区间最大字段和，单点修改，维护mx作为最大字段和，lmx为从左边向右延伸的最大字段和，rmx为从区间最右边向左延伸的最大字段和。

```cpp
const int N = 100010;
struct Node {
    int l, r;
    ll mx, lmx, rmx;
    ll sum;
}tr[N << 2];
int n, m;
int a[N];
void pushup(Node& u, Node& l, Node& r) {
    u.sum = l.sum + r.sum;
    u.lmx = max(l.lmx, l.sum + r.lmx);
    u.rmx = max(r.rmx, r.sum + l.rmx);
    u.mx = max(max(r.mx, l.mx), l.rmx + r.lmx);
}
void pushup(int u) {
    pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,l,a[l],a[l],a[l],a[l] };    
    else {
        tr[u] = { l,r };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int x, int v) {
    if (tr[u].l == tr[u].r)tr[u] = { x,x,v,v,v,v };
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
Node query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)return tr[u];
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (r <= mid)return query(u << 1, l, r);
        else if (l > mid)return query(u << 1 | 1, l, r);
        else {
            Node res = { 0,0,0,0 };
            Node left = query(u << 1, l, r);
            Node right = query(u << 1 | 1, l, r);
            pushup(res, left, right);
            return res;
        }
    }
}
int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++)cin >> a[i];
    build(1, 1, n);
    cout << max(tr[1].mx,0ll) << endl;
    while (m--) {
        int x, v;
        cin >> x >> v; x++;
        modify(1, x, v);
        cout << max(tr[1].mx,0ll) << endl;
    }
    return 0;
}
```

**B. K-th one**

**题意**：线段树查找01序列第k个1的位置。

线段树上二分的板子题，和树状数组一样可以将双log优化为单log。

```cpp
typedef pair<double, double> PDD;
typedef pair<int, int>PII;
typedef long long ll;
const int N = 100010;
struct Node {
    int l, r;
    int sum;
}tr[N << 2];
int n, m;
int a[N];
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,l,a[l] };
    else {
        tr[u] = { l,r };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int x) {
    if (tr[u].l == tr[u].r)tr[u].sum ^= 1;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x);
        else modify(u << 1 | 1, x);
        pushup(u);
    }
}
int query(int u, int x) {
    if (tr[u].l == tr[u].r)return tr[u].l;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= tr[u<<1].sum)return query(u << 1, x);
        else return query(u << 1 | 1, x-tr[u<<1].sum);
    }
}
int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++)cin >> a[i];
    build(1, 1, n);
    while (m--) {
        int op, x;
        cin >> op >> x; x++;
        if (op == 1)modify(1, x);
        else cout << query(1, x)-1 << endl;
    }
    return 0;
}
```

**C. First element at least X**

**题意**：线段树查找第一个大于等于$X$的位置。

同上题，也是线段树上二分。

```cpp
const int N = 100010;
struct Node {
    int l, r;
    int mx;
}tr[N << 2];
int n, m;
int a[N];
void pushup(int u) {
    tr[u].mx = max(tr[u << 1].mx , tr[u << 1 | 1].mx);
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,l,a[l] };
    else {
        tr[u] = { l,r };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int x, int v) {
    if (tr[u].l == tr[u].r)tr[u].mx = v;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
int query(int u, int x) {
    if (tr[u].l == tr[u].r) {
        if (tr[u].mx >= x)return tr[u].l;
        else return 0;
    }
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= tr[u<<1].mx)return query(u << 1, x);
        else return query(u << 1 | 1, x);
    }
}
int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++)cin >> a[i];
    build(1, 1, n);
    while (m--) {
        int op, x, v;
        cin >> op;
        if (op == 1) {
            cin >> x >> v; x++;
            modify(1, x, v);
        }
        else {
            cin >> x;
            cout << query(1, x) - 1 << endl;
        }
    }
    return 0;
}
```

**D. First element at least X - 2**

**题意**：线段树查找从$l$开始第一个大于等于$X$的位置。

和上一题一样，只不过添加了左边界必须是$l$这一条件，注意一下query函数实现即可。

```cpp
const int N = 100010;
struct Node {
    int l, r;
    int mx;
}tr[N << 2];
int n, m;
int a[N];
void pushup(int u) {
    tr[u].mx = max(tr[u << 1].mx , tr[u << 1 | 1].mx);
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,l,a[l] };
    else {
        tr[u] = { l,r };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int x, int v) {
    if (tr[u].l == tr[u].r)tr[u].mx = v;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
int query(int u, int l, int x) {
    if (tr[u].mx < x)return 0;
    if (tr[u].r < l)return 0;
    if (tr[u].l == tr[u].r)return tr[u].l;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        int res = query(u << 1, l, x);
        if (res == 0)
            return query(u << 1 | 1, l, x);
        return res;
    }
}
int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++)cin >> a[i];
    build(1, 1, n);
    while (m--) {
        int op, x, v;
        cin >> op;
        if (op == 1) {
            cin >> x >> v; x++;
            modify(1, x, v);
        }
        else {
            cin >> x >> v;
            cout << query(1,v+1,x) - 1 << endl;
        }
    }
    return 0;
}
```

### Step 3: [theory](https://codeforces.com/edu/course/2/lesson/4/3), [practice](https://codeforces.com/edu/course/2/lesson/4/3/practice): 5 of 5

**A. Inversions**

**题意**：已知由 $n$ 个元素组成的排列 $p_i$ ，求每个 $i$ 中 $j$ 的个数，使得 $j < i$ 和 $p_j > p_i$ .

正常模拟即可。

```cpp
const int N = 100010;
struct Node {
    int l, r;
    int sum;
}tr[N << 2];
int n, m;
int a[N];
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,l,0 };
    else {
        tr[u] = { l,r,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int x, int v) {
    if (tr[u].l == tr[u].r)tr[u].sum = v;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
int query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)return tr[u].sum;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        int res = 0;
        if (l <= mid)res = query(u << 1, l, r);
        if (r > mid)res += query(u << 1 | 1, l, r);
        return res;
    }
}
int main() {
    cin >> n;
    for (int i = 1; i <= n; i++)cin >> a[i];
    build(1, 1, n);
    ll ans = 0;
    for (int i = 1; i <= n; i++) {
        //cout << i << endl;
        modify(1, a[i], 1);
        cout << query(1, a[i], n) - 1 << " ";
    }
    return 0;
}
```

**B. Inversions 2**

**题意**：这个问题是上一个问题的反转版本。有一个由 $n$ 个元素组成的排列 $p_i$ ，我们为每个 $i$ 写下了 $a_i$ 个数， $j < i$ 和 $p_j > p_i$ 的 $j$ 个数。恢复给定的 $a_i$ 的原始排列。

```cpp
const int N = 100010;
struct Node {
    int l, r;
    int sum;
}tr[N << 2];
int n, m;
int a[N];
int ans[N];
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,l,1 };
    else {
        tr[u] = { l,r };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int x, int v) {
    if (tr[u].l == tr[u].r)tr[u].sum = v;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
int query(int u, int x) {
    if (tr[u].l == tr[u].r)return tr[u].l;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= tr[u << 1].sum)return query(u << 1, x);
        else return query(u << 1 | 1, x - tr[u << 1].sum);
    }
}
int main() {
    cin >> n;
    for (int i = 1; i <= n; i++)cin >> a[i];
    build(1, 1, n);
    for (int i = n; i; i--) {
        ans[i] = query(1, i - a[i]);
        modify(1, ans[i], 0);
    }
    for (int i = 1; i <= n; i++)cout << ans[i] << " ";
    return 0;
}
```

**C. Nested Segments**

**题意**：给定一个由 $2n$ 个数字组成的数组，其中从 1 到 $n$ 的每个数字都正好出现两次。如果数字 $y$ 的两次出现都位于数字 $x$ 的两次出现之间，我们就可以说 $y$ 嵌套在 $x$ 中。求每个线段 $i$ 内嵌套了多少个线段。

```cpp
const int N = 200010;
struct Node {
    int l, r;
    int sum;
}tr[N << 2];
int n, m;
int a[N], l[N];
int ans[N];
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,l,0 };
    else {
        tr[u] = { l,r,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int x) {
    if (tr[u].l == tr[u].r)tr[u].sum = 1;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x);
        else modify(u << 1 | 1, x);
        pushup(u);
    }
}
int query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)return tr[u].sum;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        int res = 0;
        if (l <= mid)res = query(u << 1, l, r);
        if (r > mid)res += query(u << 1 | 1, l, r);
        return res;
    }
}
int main() {
    cin >> n;
    n <<= 1;
    build(1, 1, n);
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        if (l[a[i]]) {
            ans[a[i]] = query(1, l[a[i]], n);
            modify(1, l[a[i]]);
        }
        else l[a[i]] = i;
    }
    n >>= 1;
    for (int i = 1; i <= n; i++)cout << ans[i] << " ";
    return 0;
}
```

**D. Intersecting Segments**

**题意**：给定一个由 $2n$ 个数字组成的数组，其中从 1 到 $n$ 的每个数字都恰好出现两次。如果在数字 $x$ 出现的次数之间正好有一个数字 $y$ 出现，我们就说线段 $y$ 与线段 $x$ 相交。求每条线段 $i$ 与多少条线段相交。

```cpp
const int N = 200010;
struct Node {
    int l, r;
    int sum;
}tr[N << 2];
int n, m;
int a[N], l[N];
int ans[N];
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,l,0 };
    else {
        tr[u] = { l,r,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int x, int v) {
    if (tr[u].l == tr[u].r)tr[u].sum = v;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
int query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)return tr[u].sum;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        int res = 0;
        if (l <= mid)res = query(u << 1, l, r);
        if (r > mid)res += query(u << 1 | 1, l, r);
        return res;
    }
}
int main() {
    cin >> n;
    n <<= 1;
    build(1, 1, n);
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        if (l[a[i]]) {
            ans[a[i]] = query(1, l[a[i]]+1, n);
            modify(1, l[a[i]], 0);
        }
        else {
            l[a[i]] = i;
            modify(1, i, 1);
        }
    }
    build(1, 1, n);
    for (int i = 1; i <= n; i++)l[i] = 0;
    for (int i = n; i; i--) {
        if (l[a[i]]) {
            ans[a[i]] += query(1, 1, l[a[i]] - 1);
            modify(1, l[a[i]], 0);
        }
        else {
            l[a[i]] = i;
            modify(1, i, 1);
        }
    }
    n >>= 1;
    for (int i = 1; i <= n; i++)cout << ans[i] << " ";
    return 0;
}
```

**E. Addition to Segment**

**题意**

有一个由 $n$ 元素组成的数组，初始填充为零。你需要编写一个数据结构来处理两种类型的查询：

- 在从 $l$ 到 $r-1$ 的段落中添加数字 $v$ 、
- 查找元素 $i$ 的当前值。

也可以用懒标记线段树做，但是没有访问区间和，所以可以转化为差分形式，这样单点查询就变为了查询差分数组的前缀和。

```cpp
const int N = 200010;
struct Node {
    int l, r;
    ll sum;
}tr[N << 2];
int n, m;
int a[N], l[N];
int ans[N];
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,l,0 };
    else {
        tr[u] = { l,r,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int x, int v) {
    if (tr[u].l == tr[u].r)tr[u].sum += v;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
ll query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)return tr[u].sum;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        ll res = 0;
        if (l <= mid)res = query(u << 1, l, r);
        if (r > mid)res += query(u << 1 | 1, l, r);
        return res;
    }
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    int op, l, r, x;
    while (m--) {
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x; l++;
            modify(1, l, x);
            if (r + 1 <= n)modify(1, r + 1, -x);
        }
        else {
            cin >> x; x++;
            cout << query(1, 1, x) << endl;
        }
    }
    return 0;
}
```

### Step 4: [theory](https://codeforces.com/edu/course/2/lesson/4/4), [practice](https://codeforces.com/edu/course/2/lesson/4/4/practice): 5 of 5

**A. Sign alternation**

**题意**：实现由 $n$ 元素 $a_1, a_2 \ldots a_n$ 组成的数据结构，并进行以下操作：

- 为元素 $a_i$ 赋值 $j$ ；
- 在 $l$ 到 $r$ 之间（包括 $a_l - a_{l+1} + a_{l + 2} - \ldots \pm a_{r}$ ）查找交替符号和。

```cpp
typedef long long ll;
const int N = 200010;
struct Node {
    int l, r;
    ll sum;
}tr[N << 2];
int n, m;
int a[N], l[N];
int ans[N];
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,l,a[l]};
    else {
        tr[u] = { l,r };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int x, int v) {
    if (tr[u].l == tr[u].r)tr[u].sum = v;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
ll query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)return tr[u].sum;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        ll res = 0;
        if (l <= mid)res = query(u << 1, l, r);
        if (r > mid)res += query(u << 1 | 1, l, r);
        return res;
    }
}
int main() {
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        if (!(i % 2))a[i] *= -1;
    }
    cin >> m;
    build(1, 1, n);
    int op, l, r;
    while (m--) {
        cin >> op >> l >> r;
        if (op == 0)modify(1, l, l % 2 ? r : -r);
        else {
            if (l % 2)cout << query(1, l, r) << endl;
            else cout << -query(1, l, r) << endl;
        }
    }
    return 0;
}
```

**B. Cryptography**

**题意**：给定大小为 $2\times 2$ 的 $n$ 矩阵 $A_1, A_2, \ldots, A_n$ 。你需要计算几个查询的矩阵 $A_i, A_{i+1}, \ldots, A_j$ 的乘积。所有计算都以 $r$ 为模数来进行。

```cpp
typedef long long ll;
const int N = 200010;
struct Node {
    int l, r;
    int a[2][2];
}tr[N << 2];
int n, m, mod;
int a[N], l[N];
int ans[N];
Node operator*(const Node p, const Node q) {
    Node res = { p.l,q.r,0,0,0,0};
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                res.a[i][j] = (res.a[i][j] + p.a[i][k] * q.a[k][j] % mod) % mod;
    return res;
}
void build(int u, int l, int r) {
    if (l == r) {
        tr[u] = { l,r };
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                cin >> tr[u].a[i][j];
    }
    else {
        tr[u] = { l,r };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        tr[u] = tr[u << 1] * tr[u << 1 | 1];
    }
}
Node query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)return tr[u];
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (r <= mid)return query(u << 1, l, r);
        else if (l > mid)return query(u << 1 | 1, l, r);
        return query(u << 1, l, r) * query(u << 1 | 1, l, r);
    }
}
int main() {
    cin >> mod >> n >> m;
    build(1, 1, n);
    while (m--) {
        int l, r; cin >> l >> r;
        Node res = query(1, l, r);
        for (int i = 0; i < 2; i++, puts(""))
            for (int j = 0; j < 2; j++)
                cout << res.a[i][j] << " ";
        puts("");
    }
    return 0;
}
```

**C. Number of Inversions on Segment**

**题意**：给定一个由小整数（ $1 \leq a_i \leq 40$ ）组成的数组 $a$ 。你需要建立一个数据结构来处理两种类型的查询：

1. 查找段上的逆序对数量。
2. 更改数组的元素。

```cpp
typedef long long ll;
const int N = 100010;
struct Node {
    int l, r;
    ll inv, pos;
    int num[40];
}tr[N << 2];
int n, m, mod;
int a[N], l[N];
int ans[N];
void pushup(Node& u, Node l, Node r) {
    u.inv = l.inv + r.inv;
    ll pre[40] = { 0 };
    pre[0] = r.num[0];
    for (int i = 0; i < 40; i++) {
        u.num[i] = l.num[i] + r.num[i];
        if (i) {
            u.inv += l.num[i] * pre[i - 1];
            pre[i] = pre[i - 1] + r.num[i];
        }
    }
}
void pushup(int u) {
    pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
}
void build(int u, int l, int r) {
    tr[u] = { l,r,0 };
    if (l == r) {
        tr[u].num[a[l]]++;
        tr[u].pos = a[l];
    }
    else {
        tr[u] = { l,r,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int x, int v) {
    if (tr[u].l == tr[u].r) {
        tr[u].num[tr[u].pos]--;
        tr[u].num[v]++;
        tr[u].pos = v;
    }
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
Node query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)return tr[u];
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (r <= mid)return query(u << 1, l, r);
        else if (l > mid)return query(u << 1 | 1, l, r);
        Node res = { 0,0,0,0 };
        pushup(res, query(u << 1, l, r), query(u << 1 | 1, l, r));
        return res;
    }
}
int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++)cin >> a[i], a[i]--;
    build(1, 1, n);
    while (m--) {
        int op, l, r;
        cin >> op >> l >> r;
        if (op == 1)cout << query(1, l, r).inv << endl;
        else modify(1, l, r - 1);
    }
    return 0;
}
```

**D. Number of Different on Segment**

**题意**：给定一个由小整数（ $1 \leq a_i \leq 40$ ）组成的数组 $a$ 。你需要建立一个数据结构来处理两种类型的查询：

1. 查找段上不同元素的数量。
2. 更改数组中的元素。

```cpp
typedef long long ll;
const int N = 100010;
struct Node {
    int l, r;
    ll sum;
}tr[N << 2];
int n, m, mod;
int a[N], l[N];
int ans[N];
int f(ll x) {
    int res = 0;
    while (x)x = x & (x-1), res++;
    return res;
}
void pushup(Node& u, Node l, Node r) {
    u.sum = l.sum | r.sum;
}
void pushup(int u) {
    pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,1ll << a[l] };
    else {
        tr[u] = { l,r,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int x, int v) {
    if (tr[u].l == tr[u].r) {
        tr[u].sum = 1ll << v;
    }
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
ll query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)return tr[u].sum;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (r <= mid)return query(u << 1, l, r);
        else if (l > mid)return query(u << 1 | 1, l, r);
        return query(u << 1, l, r) | query(u << 1 | 1, l, r);
    }
}
int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++)cin >> a[i];
    build(1, 1, n);
    while (m--) {
        int op, l, r;
        cin >> op >> l >> r;
        if (op == 1)cout << f(query(1, l, r)) << endl;
        else modify(1, l, r);
    }
    return 0;
}
```

**E. Earthquakes**

**题意**：

城市是由从 $0$ 到 $n-1$ 编号的 $n$ 个单元组成的序列。最初，所有单元格都是空的。然后，两种类型中的 $m$ 个事件依次发生：

- 在 $i$ 单元中正在建造一栋强度为 $h$ 的建筑（如果该单元中已有建筑，则该建筑将被拆除并替换为新建筑）、
-  在 $l$ 到 $r-1$ 的区间内发生了威力为 $p$ 的地震，摧毁了所有强度不超过 $p$ 的建筑物。 你的任务是针对每次地震说出它将摧毁多少座建筑物。 

```cpp
using i64 = long long;
const i64 inf = 0x3f3f3f3f3f3f3f3f;
i64 n, m;
i64 segtree[100010 << 2];
inline void update(int i, int l, int r, int x, int y) {
    if (l == r) {
        segtree[i] = y;
        return;
    }
    int mid = (l + r) >> 1;
    if (x <= mid) {
        update(i << 1, l, mid, x, y);
    }
    else {
        update(i << 1 | 1, mid + 1, r, x, y);
    }
    segtree[i] = min(segtree[i << 1], segtree[i << 1 | 1]);
}
i64 calc(int i, int l, int r, int x, int y, int p) {
    if (segtree[i] > p) {
        return 0;
    }
    if (l == r) {
        segtree[i] = inf;
        return 1;
    }
    int mid = (l + r) >> 1;
    i64 res = 0;
    if (x <= mid && segtree[i << 1] <= p) {
        res += calc(i << 1, l, mid, x, y, p);
    }
    if (y > mid && segtree[i << 1 | 1] <= p) {
        res += calc(i << 1 | 1, mid + 1, r, x, y, p);
    }
    segtree[i] = min(segtree[i << 1], segtree[i << 1 | 1]);
    return res;
}
int main() {
    cin >> n >> m;
    for (int i = 0; i < (100010 << 2); i++) {
        segtree[i] = inf;
    }
    for (int i = 0; i < m; i++) {
        int op;
        cin >> op;
        if (op == 1) {
            i64 x, y;
            cin >> x >> y;
            x++;
            update(1, 1, n, x, y);
        }
        else {
            int left, right;
            i64 p;
            cin >> left >> right >> p;
            left++;
            cout << calc(1, 1, n, left, right, p) << '\n';
        }
    }
    return 0;
}
```

## [Segment Tree, part 2](https://codeforces.com/edu/course/2/lesson/5)

### Step 1: [theory](https://codeforces.com/edu/course/2/lesson/5/1), [practice](https://codeforces.com/edu/course/2/lesson/5/1/practice): 3 of 3

**A. Addition to Segment**

**题意**：有一个由 $n$ 元素组成的数组，最初填充的是零。你需要编写一个数据结构来处理两种类型的查询：

- 将数字 $v$ 添加到从 $l$ 到 $r-1$ 的段中。
- 查找元素 $i$ 的当前值。

```cpp
typedef long long ll;
const int N = 1e5 + 10;
struct Node {
    int l, r;
    ll sum, tag;
}tr[N << 2];
int n, m;
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void pushdown(int u) {
    if (tr[u].tag) {
        tr[u << 1].sum += (tr[u << 1].r - tr[u << 1].l + 1) * tr[u].tag;
        tr[u << 1].tag += tr[u].tag;
        tr[u << 1 | 1].sum += (tr[u << 1 | 1].r - tr[u << 1 | 1].l + 1) * tr[u].tag;
        tr[u << 1 | 1].tag += tr[u].tag;
        tr[u].tag = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,0,0 };
    else {
        tr[u] = { l,r,0,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum += (tr[u].r - tr[u].l + 1) * x;
        tr[u].tag += x;
    }
    else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)modify(u << 1, l, r, x);
        if (r > mid)modify(u << 1 | 1, l, r, x);
        pushup(u);
    }
}
ll query(int u, int x) {
    if (tr[u].l == tr[u].r)return tr[u].sum;
    int mid = tr[u].l + tr[u].r >> 1;
    pushdown(u);
    if (x <= mid)return query(u << 1, x);
    else return query(u << 1 | 1, x);
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r;ll x;
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x;
            modify(1, l + 1, r, x);
        }
        else {
            cin >> x;
            x++;
            cout << query(1,x) << endl;
        }
    }
    return 0;
}
```

**B. Applying MAX to Segment**

**题意**：有一个由 $n$ 个元素组成的数组，初始填充为零。你需要编写一个数据结构来处理两种类型的查询：

- 对于从 $l$ 到 $r-1$ 的所有 $i$ 执行 $a_i = \max(a_i, v)$ 操作、
- 查找元素 $i$ 的当前值。

```cpp
typedef long long ll;
const int N = 1e5 + 10;
struct Node {
    int l, r;
    ll mx, tag;
}tr[N << 2];
int n, m;
void pushup(int u) {
    tr[u].mx = max(tr[u << 1].mx, tr[u << 1 | 1].mx);
}
void pushdown(int u) {
    if (tr[u].tag) {
        tr[u << 1].mx = max(tr[u << 1].mx, tr[u].tag);
        tr[u << 1].tag = max(tr[u].tag, tr[u << 1].tag);
        tr[u << 1 | 1].mx = max(tr[u << 1 | 1].mx, tr[u].tag);
        tr[u << 1 | 1].tag = max(tr[u << 1 | 1].tag, tr[u].tag);
        tr[u].tag = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,0,0 };
    else {
        tr[u] = { l,r,0,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].mx = max(tr[u].mx, x);
        tr[u].tag = max(x, tr[u].tag);
    }
    else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)modify(u << 1, l, r, x);
        if (r > mid)modify(u << 1 | 1, l, r, x);
        pushup(u);
    }
}
ll query(int u, int x) {
    if (tr[u].l == tr[u].r)return tr[u].mx;
    int mid = tr[u].l + tr[u].r >> 1;
    pushdown(u);
    if (x <= mid)return query(u << 1, x);
    else return query(u << 1 | 1, x);
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r;ll x;
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x;
            modify(1, l + 1, r, x);
        }
        else {
            cin >> x;
            x++;
            cout << query(1,x) << endl;
        }
    }
    return 0;
}
```

**C. Assignment to Segment**

**题意**：有一个由 $n$ 个元素组成的数组，初始填充为零。你需要编写一个数据结构来处理两种类型的查询：

- 为从 $l$ 到 $r-1$ 段上的所有元素赋值 $v$ 、
- 查找元素 $i$ 的当前值。

```cpp
typedef long long ll;
const int N = 1e5 + 10;
struct Node {
    int l, r;
    ll v, tag;
}tr[N << 2];
int n, m;
void pushdown(int u) {
    if (tr[u].tag) {
        tr[u << 1].v = tr[u].tag;
        tr[u << 1].tag = tr[u].tag;
        tr[u << 1 | 1].v = tr[u].tag;
        tr[u << 1 | 1].tag = tr[u].tag;
        tr[u].tag = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,1,0 };
    else {
        tr[u] = { l,r,1,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].v = x;
        tr[u].tag = x;
    }
    else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)modify(u << 1, l, r, x);
        if (r > mid)modify(u << 1 | 1, l, r, x);
    }
}
ll query(int u, int x) {
    if (tr[u].l == tr[u].r)return tr[u].v;
    int mid = tr[u].l + tr[u].r >> 1;
    pushdown(u);
    if (x <= mid)return query(u << 1, x);
    return query(u << 1 | 1, x);
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r;ll x;
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x; x++;
            modify(1, l + 1, r, x);
        }
        else {
            cin >> x;
            x++;
            cout << query(1, x)-1 << endl;
        }
    }
    return 0;
}
```

### Step 2: [theory](https://codeforces.com/edu/course/2/lesson/5/2), [practice](https://codeforces.com/edu/course/2/lesson/5/2/practice): 6 of 6

**A. Addition and Minimum**

**题意**：有一个由 $n$ 个元素组成的数组，初始填充为零。你需要编写一个数据结构来处理两种类型的查询：

- 在从 $l$ 到 $r-1$ 的线段上添加 $v$ 。
- 查找从 $l$ 到 $r-1$ 的线段上的最小值。

```cpp
typedef long long ll;
const int N = 1e5 + 10;
struct Node {
    int l, r;
    ll mn, sum, tag;
}tr[N << 2];
int n, m;
void pushup(int u) {
    tr[u].mn = min(tr[u << 1].mn, tr[u << 1 | 1].mn);
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void pushdown(int u) {
    if (tr[u].tag) {
        tr[u << 1].sum += (tr[u << 1].r - tr[u << 1].l + 1) * tr[u].tag;
        tr[u << 1].mn += tr[u].tag;
        tr[u << 1].tag += tr[u].tag;
        tr[u << 1 | 1].sum += (tr[u << 1 | 1].r - tr[u << 1 | 1].l + 1) * tr[u].tag;
        tr[u << 1 | 1].mn += tr[u].tag;
        tr[u << 1 | 1].tag += tr[u].tag;
        tr[u].tag = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,0,0,0 };
    else {
        tr[u] = { l,r,0,0,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum += (tr[u].r - tr[u].l + 1) * x;
        tr[u].mn += x;
        tr[u].tag += x;
    }
    else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)modify(u << 1, l, r, x);
        if (r > mid)modify(u << 1 | 1, l, r, x);
        pushup(u);
    }
}
ll query(int u, int l,int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u].mn;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    ll mn = 4e18;
    if (l <= mid)mn = min(mn, query(u << 1, l, r));
    if (r > mid)mn = min(mn, query(u << 1 | 1, l, r));
    return mn;
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r;ll x;
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x;
            modify(1, l + 1, r, x);
        }
        else {
            cin >> l >> r;
            cout << query(1, l + 1, r) << endl;
        }
    }
    return 0;
}
```

**B. Multiplication and Sum**

**题意**：有一个由 $n$ 个元素组成的数组，最初填的是 1。你需要编写一个数据结构来处理两种类型的查询：

- 用数字 $v$ 乘从 $l$ 到 $r-1$ 段上的所有元素。
- 求从 $l$ 到 $r-1$ 的线段上的总和。

这两个运算都以 $10^9+7$ 为模来进行。

```cpp
typedef long long ll;
const int N = 1e5 + 10;
const int mod = 1e9 + 7;
struct Node {
    int l, r;
    ll sum, tag;
}tr[N << 2];
int n, m;
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
    tr[u].sum %= mod;
}
void pushdown(int u) {
    if (tr[u].tag!=1) {
        tr[u << 1].sum *= tr[u].tag;
        tr[u << 1].sum %= mod;
        tr[u << 1].tag *= tr[u].tag;
        tr[u << 1].tag %= mod;
        tr[u << 1 | 1].sum *= tr[u].tag;
        tr[u << 1 | 1].sum %= mod;
        tr[u << 1 | 1].tag *= tr[u].tag;
        tr[u << 1 | 1].tag %= mod;
        tr[u].tag = 1;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,1,1 };
    else {
        tr[u] = { l,r,0,1 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum *= x;
        tr[u].sum %= mod;
        tr[u].tag *= x;
        tr[u].tag %= mod;
    }
    else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)modify(u << 1, l, r, x);
        if (r > mid)modify(u << 1 | 1, l, r, x);
        pushup(u);
    }
}
ll query(int u, int l,int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u].sum;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    ll sum = 0;
    if (l <= mid)sum = query(u << 1, l, r);
    if (r > mid)sum += query(u << 1 | 1, l, r);
    sum %= mod;
    return sum;
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r;ll x;
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x;
            modify(1, l + 1, r, x);
        }
        else {
            cin >> l >> r;
            cout << query(1, l + 1, r) << endl;
        }
    }
    return 0;
}
```

**C. Bitwise OR and AND**

**题意**：有一个由 $n$ 个元素组成的数组，初始填充为零。你需要编写一个数据结构来处理两种类型的查询：

- 对从 $l$ 到 $r-1$ 的所有元素执行 $a_i = a_i | v$ （按位 OR）操作
- 查找从 $l$ 到 $r-1$ 范围内元素的位操作 AND。

```cpp
typedef long long ll;
const int N = 1e5 + 10;
const int mod = 1e9 + 7;
struct Node {
    int l, r;
    ll sum, tag;
}tr[N << 2];
int n, m;
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum & tr[u << 1 | 1].sum;
}
void pushdown(int u) {
    if (tr[u].tag) {
        tr[u << 1].sum |= tr[u].tag;
        tr[u << 1].tag |= tr[u].tag;
        tr[u << 1 | 1].sum |= tr[u].tag;
        tr[u << 1 | 1].tag |= tr[u].tag;
        tr[u].tag = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,0,0 };
    else {
        tr[u] = { l,r,0,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum |= x;
        tr[u].tag |= x;
    }
    else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)modify(u << 1, l, r, x);
        if (r > mid)modify(u << 1 | 1, l, r, x);
        pushup(u);
    }
}
ll query(int u, int l,int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u].sum;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (r <= mid)return query(u << 1, l, r);
    else if (l > mid)return query(u << 1 | 1, l, r);
    return query(u << 1, l, r) & query(u << 1 | 1, l, r);
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r;ll x;
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x;
            modify(1, l + 1, r, x);
        }
        else {
            cin >> l >> r;
            cout << query(1, l + 1, r) << endl;
        }
    }
    return 0;
}
```

**D. Addition and Sum**

**题意**：有一个由 $n$ 元素组成的数组，初始填充为零。你需要编写一个数据结构来处理两种类型的查询：

- 将 $v$ 添加到从 $l$ 到 $r-1$ 的数据段中。
- 求从 $l$ 到 $r-1$ 的线段上的总和。

```cpp
typedef long long ll;
const int N = 1e5 + 10;
const int mod = 1e9 + 7;
struct Node {
    int l, r;
    ll sum, tag;
}tr[N << 2];
int n, m;
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void pushdown(int u) {
    if (tr[u].tag) {
        tr[u << 1].sum += tr[u].tag * (tr[u << 1].r - tr[u << 1].l + 1);
        tr[u << 1].tag += tr[u].tag;
        tr[u << 1 | 1].sum += tr[u].tag * (tr[u << 1 | 1].r - tr[u << 1 | 1].l + 1);
        tr[u << 1 | 1].tag += tr[u].tag;
        tr[u].tag = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,0,0 };
    else {
        tr[u] = { l,r,0,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum += x * (tr[u].r - tr[u].l + 1);
        tr[u].tag += x;
    }
    else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)modify(u << 1, l, r, x);
        if (r > mid)modify(u << 1 | 1, l, r, x);
        pushup(u);
    }
}
ll query(int u, int l,int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u].sum;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (r <= mid)return query(u << 1, l, r);
    else if (l > mid)return query(u << 1 | 1, l, r);
    return query(u << 1, l, r) + query(u << 1 | 1, l, r);
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r;ll x;
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x;
            modify(1, l + 1, r, x);
        }
        else {
            cin >> l >> r;
            cout << query(1, l + 1, r) << endl;
        }
    }
    return 0;
}
```

**E. Assignment and Minimum**

有一个由 $n$ 个元素组成的数组，初始填充为零。你需要编写一个数据结构来处理两种类型的查询：

- 为从 $l$ 到 $r-1$ 的线段上的所有元素赋值 $v$ 、
- 查找从 $l$ 到 $r-1$ 的线段上的最小值。

```cpp
typedef long long ll;
const int N = 1e5 + 10;
const int mod = 1e9 + 7;
struct Node {
    int l, r;
    ll mn, tag;
}tr[N << 2];
int n, m;
void pushup(int u) {
    tr[u].mn = min(tr[u << 1].mn, tr[u << 1 | 1].mn);
}
void pushdown(int u) {
    if (tr[u].tag) {
        tr[u << 1].mn = tr[u].tag;
        tr[u << 1].tag = tr[u].tag;
        tr[u << 1 | 1].mn = tr[u].tag;
        tr[u << 1 | 1].tag = tr[u].tag;
        tr[u].tag = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,1,0 };
    else {
        tr[u] = { l,r,1,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].mn = x;
        tr[u].tag = x;
    }
    else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)modify(u << 1, l, r, x);
        if (r > mid)modify(u << 1 | 1, l, r, x);
        pushup(u);
    }
}
ll query(int u, int l,int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u].mn;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (r <= mid)return query(u << 1, l, r);
    else if (l > mid)return query(u << 1 | 1, l, r);
    return min(query(u << 1, l, r), query(u << 1 | 1, l, r));
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r;ll x;
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x; x++;
            modify(1, l + 1, r, x);
        }
        else {
            cin >> l >> r;
            cout << query(1, l + 1, r)-1 << endl;
        }
    }
    return 0;
}
```

**F. Assignment and Sum**

有一个由 $n$ 个元素组成的数组，初始填充为零。你需要编写一个数据结构来处理两种类型的查询：

- 为 $l$ 至 $r-1$ 段中的所有元素赋值 $v$ 。
- 求从 $l$ 到 $r-1$ 的线段上的总和。

```cpp
typedef long long ll;
const int N = 1e5 + 10;
const int mod = 1e9 + 7;
struct Node {
    int l, r;
    ll sum, tag;
}tr[N << 2];
int n, m;
ll qmi(ll a, ll k) {
    ll res = 1;
    a %= mod;
    while (k) {
        if (k & 1)res = res * a % mod;
        k >>= 1;
        a = a * a % mod;
    }
    return res;
}
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void pushdown(int u) {
    if (tr[u].tag) {
        tr[u << 1].sum = tr[u].tag * (tr[u << 1].r - tr[u << 1].l + 1);
        tr[u << 1].tag = tr[u].tag;
        tr[u << 1 | 1].sum = tr[u].tag * (tr[u << 1 | 1].r - tr[u << 1 | 1].l + 1);
        tr[u << 1 | 1].tag = tr[u].tag;
        tr[u].tag = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,0,0 };
    else {
        tr[u] = { l,r,0,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum = x * (tr[u].r - tr[u].l + 1);
        tr[u].tag = x;
    }
    else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)modify(u << 1, l, r, x);
        if (r > mid)modify(u << 1 | 1, l, r, x);
        pushup(u);
    }
}
ll query(int u, int l,int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u].sum;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (r <= mid)return query(u << 1, l, r);
    else if (l > mid)return query(u << 1 | 1, l, r);
    return query(u << 1, l, r)+query(u << 1 | 1, l, r);
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r;ll x;
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x;
            modify(1, l + 1, r, x);
        }
        else {
            cin >> l >> r;
            cout << query(1, l + 1, r) << endl;
        }
    }
    return 0;
}
```

### Step 3: [theory](https://codeforces.com/edu/course/2/lesson/5/3), [practice](https://codeforces.com/edu/course/2/lesson/5/3/practice): 3 of 3

**A. Assignment and Maximal Segment**

**题意**：有一个由 $n$ 个元素组成的数组，初始填充为零。你需要编写一个数据结构来处理两种类型的查询：

- 为从 $l$ 到 $r-1$ 的线段上的所有元素赋值 $v$ 。
- 找出总和最大的线段。

```cpp
typedef long long ll;
const int N = 1e5 + 10;
const ll INF = 4e18;
struct Node {
    int l, r;
    ll sum, lmx, rmx, mx, tag;
}tr[N << 2];
int n, m;
void pushup(Node &u,Node l,Node r) {
    u.sum = l.sum + r.sum;
    u.lmx = max(l.lmx, l.sum + r.lmx);
    u.rmx = max(r.rmx, l.rmx + r.sum);
    u.mx = max(max(l.mx, r.mx), l.rmx + r.lmx);
}
void pushup(int u) {
    pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
}
void pushdown(int u) {
    if (tr[u].tag!=INF) {
        if (tr[u].tag > 0) {
            tr[u << 1].mx = tr[u << 1].lmx = tr[u << 1].rmx = tr[u << 1].sum = tr[u].tag * (tr[u << 1].r - tr[u << 1].l + 1);
            tr[u << 1 | 1].mx = tr[u << 1 | 1].lmx = tr[u << 1 | 1].rmx = tr[u << 1 | 1].sum = tr[u].tag * (tr[u << 1 | 1].r - tr[u << 1 | 1].l + 1);
            tr[u << 1].tag = tr[u << 1 | 1].tag = tr[u].tag;
        }
        else
        {
            tr[u << 1].mx = tr[u << 1].lmx = tr[u << 1].rmx = 0;
            tr[u << 1].sum = tr[u].tag * (tr[u << 1].r - tr[u << 1].l + 1);
            tr[u << 1 | 1].mx = tr[u << 1 | 1].lmx = tr[u << 1 | 1].rmx = 0;
            tr[u << 1 | 1].sum = tr[u].tag * (tr[u << 1 | 1].r - tr[u << 1 | 1].l + 1);
            tr[u << 1].tag = tr[u << 1 | 1].tag = tr[u].tag;
        }
        tr[u].tag = INF;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,0,0,0,0,INF };
    else {
        tr[u] = { l,r,0,0,0,0,INF };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        if (x > 0) {
            tr[u].sum = x * (tr[u].r - tr[u].l + 1);
            tr[u].mx = tr[u].lmx = tr[u].rmx = tr[u].sum;
            tr[u].tag = x;
        }
        else {
            tr[u].sum = x * (tr[u].r - tr[u].l + 1);
            tr[u].mx = tr[u].lmx = tr[u].rmx = 0;
            tr[u].tag = x;
        }
    }
    else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)modify(u << 1, l, r, x);
        if (r > mid)modify(u << 1 | 1, l, r, x);
        pushup(u);
    }
}
Node query(int u, int l,int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u];
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (r <= mid)return query(u << 1, l, r);
    else if (l > mid)return query(u << 1 | 1, l, r);
    Node res = { 0,0,0,0,0,0,0 };
    Node left = query(u << 1, l, r);
    Node right = query(u << 1 | 1, l, r);
    pushup(res, left, right);
    return res;
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int l, r, x;
        cin >> l >> r >> x;
        l++;
        modify(1, l, r, x);
        cout << max(tr[1].mx, 0ll) << endl;
    }
    return 0;
}
```

**B. Inverse and K-th one**

**题意**：有一个由 $n$ 个布尔数组组成的数组，初始值为零。您需要编写一个数据结构来处理两种类型的查询：

- 将 $l$ 到 $r-1$ 段中所有元素的值改为相反值。
- 找到 $k-th$ 索引。

```cpp
typedef long long ll;
const int N = 1e5 + 10;
const ll INF = 4e18;
struct Node {
    int l, r;
    int sum, tag;
}tr[N << 2];
int n, m;
void pushup(Node &u,Node l,Node r) {
    u.sum = l.sum + r.sum;
}
void pushup(int u) {
    pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
}
void pushdown(int u) {
    if (tr[u].tag) {
        tr[u<<1].sum = tr[u<<1].r - tr[u<<1].l + 1 - tr[u<<1].sum;
        tr[u << 1].tag ^= 1;
        tr[u << 1 | 1].sum = tr[u << 1 | 1].r - tr[u << 1 | 1].l + 1 - tr[u << 1 | 1].sum;
        tr[u << 1 | 1].tag ^= 1;
        tr[u].tag = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,0,0 };
    else {
        tr[u] = { l,r,0,0, };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum = tr[u].r - tr[u].l + 1 - tr[u].sum;
        tr[u].tag ^= 1;
    }
    else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)modify(u << 1, l, r);
        if (r > mid)modify(u << 1 | 1, l, r);
        pushup(u);
    }
}
int query(int u, int x) {
    if (tr[u].l == tr[u].r)return tr[u].l;
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (x <= tr[u << 1].sum)return query(u << 1, x);
    return query(u << 1 | 1, x - tr[u << 1].sum);
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r, x;
        cin >> op;
        if (op == 1)cin>>l>>r,modify(1, l+1, r);
        else cin >> x, cout << query(1, x+1)-1 << endl;
    }
    return 0;
}
```

**C. Addition and First element at least X**

**题意**：有一个由 $n$ 个元素组成的数组，初始填充为零。你需要编写一个数据结构来处理两种类型的查询：

- 将 $v$ 添加到从 $l$ 到 $r-1$ 的线段上的所有元素、
- 找出最小索引 $j$ ，使得 $j \ge l$ 和 $a[j] \ge x$ .

```cpp
typedef long long ll;
const int N = 2e5 + 10;
const ll INF = 4e18;
struct Node {
    int l, r;
    ll mx, tag;
}tr[N << 2];
int n, m;
void pushup(Node &u,Node l,Node r) {
    u.mx = max(l.mx, r.mx);
}
void pushup(int u) {
    pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
}
void pushdown(int u) {
    if (tr[u].tag) {
        tr[u << 1].mx += tr[u].tag;
        tr[u << 1].tag += tr[u].tag;
        tr[u << 1 | 1].mx += tr[u].tag;
        tr[u << 1 | 1].tag += tr[u].tag;
        tr[u].tag = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,0,0 };
    else {
        tr[u] = { l,r,0,0, };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].mx += x;
        tr[u].tag += x;
    }
    else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)modify(u << 1, l, r, x);
        if (r > mid)modify(u << 1 | 1, l, r, x);
        pushup(u);
    }
}
int query(int u,int l,int r, ll x) {
    pushdown(u);
    if (tr[u].mx < x)return 0;
    if (tr[u].r < l)return 0;
    if (tr[u].l == tr[u].r)return tr[u].l;
    int mid = tr[u].l + tr[u].r >> 1;
    int res = query(u << 1, l, r, x);
    if (res == 0)return query(u << 1 | 1, l, r, x);
    return res;
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r, x;
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> x;
            modify(1, l + 1, r, x);
        }
        else {
            cin >> x >> l;
            cout << query(1, l + 1, n, x)-1 << endl;
        }
    }
    return 0;
}
```

### Step 4: [theory](https://codeforces.com/edu/course/2/lesson/5/4), [practice](https://codeforces.com/edu/course/2/lesson/5/4/practice): 6 of 6

**A. Assignment, Addition, and Sum**

**题意**：有一个由 $n$ 个元素组成的数组，初始填充为零。你需要编写一个数据结构来处理三种类型的查询：

- 为从 $l$ 到 $r-1$ 段上的所有元素赋值 $v$ 、
- 将 $v$ 添加到从 $l$ 到 $r-1$ 区间的所有元素、
- 求从 $l$ 到 $r-1$ 的线段上的和。

```cpp
typedef long long ll;
const int N = 2e5 + 10;
const ll INF = 4e18;
struct Node {
    int l, r;
    ll sum, asstag, addtag;
}tr[N << 2];
int n, m;
void pushup(Node &u,Node l,Node r) {
    u.sum = l.sum + r.sum;
}
void pushup(int u) {
    pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
}
void pushdown(int u) {
    if (tr[u].asstag != INF) {
        tr[u<<1].sum = tr[u].asstag * (tr[u << 1].r - tr[u << 1].l + 1);
        tr[u<<1|1].sum = tr[u].asstag * (tr[u << 1|1].r - tr[u << 1|1].l + 1);
        tr[u << 1].asstag = tr[u].asstag;
        tr[u << 1 | 1].asstag = tr[u].asstag;
        tr[u << 1].addtag = INF;
        tr[u << 1 | 1].addtag = INF;
        tr[u].asstag = INF;
    }
    else if (tr[u].addtag != INF) {
        tr[u << 1].sum += tr[u].addtag * (tr[u << 1].r - tr[u << 1].l + 1);
        tr[u << 1 | 1].sum += tr[u].addtag * (tr[u << 1 | 1].r - tr[u << 1 | 1].l + 1);
        if (tr[u << 1].asstag != INF) {
            tr[u << 1].asstag += tr[u].addtag;
            tr[u << 1].addtag = INF;
        }
        else if (tr[u << 1].addtag == INF)
            tr[u << 1].addtag = tr[u].addtag;
        else tr[u << 1].addtag += tr[u].addtag;
        if (tr[u << 1 | 1].asstag != INF) {
            tr[u << 1 | 1].asstag += tr[u].addtag;
            tr[u << 1 | 1].addtag = INF;
        }
        else if (tr[u << 1 | 1].addtag == INF)
            tr[u << 1 | 1].addtag = tr[u].addtag;
        else tr[u << 1 | 1].addtag += tr[u].addtag;
        tr[u].addtag = INF;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,0,INF,INF };
    else {
        tr[u] = { l,r,0,INF,INF };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void assign(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum = x * (tr[u].r - tr[u].l + 1);
        tr[u].asstag = x;
        tr[u].addtag = INF;
        return;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (l <= mid)assign(u << 1, l, r, x);
    if (r > mid)assign(u << 1 | 1, l, r, x);
    pushup(u);
}
void add(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum += x * (tr[u].r - tr[u].l + 1);
        if (tr[u].asstag != INF)
            tr[u].asstag += x, tr[u].addtag = INF;
        else if (tr[u].addtag != INF)tr[u].addtag += x;
        else tr[u].addtag = x;
        return;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (l <= mid)add(u << 1, l, r, x);
    if (r > mid)add(u << 1 | 1, l, r, x);
    pushup(u);
}
ll query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u].sum;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    ll res = 0;
    if (l <= mid)res = query(u << 1, l, r);
    if (r > mid)res += query(u << 1 | 1, l, r);
    return res;
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r;ll x;
        cin >> op >> l >> r; l++;
        if (op == 1) {
            cin >> x;
            assign(1, l, r, x);
        }
        else if (op == 2) {
            cin >> x;
            add(1, l, r, x);
        }
        else cout << query(1, l, r) << endl;
    }
    return 0;
}
```

**B. Add Arithmetic Progression On Segment**

**题意**：给你一个数组 $x$ ，由等于 $0$ 的 $n$ 个元素和两种类型的 $m$ 个查询组成：

- 为一个数段添加算术级数：查询用四个整数 $l, r, a, d$ 来描述--每个 $l \le i \le r$ 都要执行 $x_i += a + d \cdot (i - l)$ ；
- 打印给定元素的当前值。

```cpp
typedef long long ll;
const int N = 2e5 + 10;
const ll INF = 4e18;
struct Node {
    int l, r;
    ll sum, tag;
}tr[N << 2];
int n, m;
void pushup(Node &u,Node l,Node r) {
    u.sum = l.sum + r.sum;
}
void pushup(int u) {
    pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
}
void pushdown(int u) {
    if (tr[u].tag) {
        tr[u << 1].sum += tr[u].tag * (tr[u << 1].r - tr[u << 1].l + 1);
        tr[u << 1].tag += tr[u].tag;
        tr[u << 1 | 1].sum += tr[u].tag * (tr[u << 1 | 1].r - tr[u << 1 | 1].l + 1);
        tr[u << 1 | 1].tag += tr[u].tag;
        tr[u].tag = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,0,0 };
    else {
        tr[u] = { l,r,0,0 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum += x * (tr[u].r - tr[u].l + 1);
        tr[u].tag += x;
        return;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (l <= mid)modify(u << 1, l, r, x);
    if (r > mid)modify(u << 1 | 1, l, r, x);
    pushup(u);
}
ll query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u].sum;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    ll res = 0;
    if (l <= mid)res = query(u << 1, l, r);
    if (r > mid)res += query(u << 1 | 1, l, r);
    return res;
}
int main() {
    cin >> n >> m;
    build(1, 1, n);
    while (m--) {
        int op, l, r; ll a, d;
        cin >> op;
        if (op == 1) {
            cin >> l >> r >> a >> d;
            modify(1, l, l, a);
            if (r > l)modify(1, l + 1, r, d);
            if (r < n)modify(1, r + 1, r + 1, -a + (l - r) * d);
        }
        else {
            cin >> d;
            cout << query(1, 1, d) << endl;
        }
    }
    return 0;
}
```

**C. Painter**

**题意**：

意大利抽象艺术家 F. Mandarino 对绘制一维黑白画感兴趣。他试图找到画面中黑色部分的位置和数量。为此，他在线条上画出了白段和黑段，在每次操作之后，他都想知道最终画面中黑色段的数量及其总长度。

最初，线条是白色的。您的任务是编写一个程序，在每次操作后输出艺术家感兴趣的数据。

```cpp
typedef long long ll;
const int N = 1e6 + 10;
const ll INF = 4e18;
struct Node {
    int l, r;
    int num, sum;
    int lv, rv, tag;
}tr[N << 2];
int n, m;
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
    tr[u].num = tr[u << 1].num + tr[u << 1 | 1].num;
    if (tr[u << 1].rv && tr[u << 1 | 1].lv)tr[u].num--;
    tr[u].lv = tr[u << 1].lv;
    tr[u].rv = tr[u << 1 | 1].rv;
}
void pushdown(int u) {
    if (~tr[u].tag) {
        if (tr[u].tag == 0) {
            tr[u << 1].sum = 0;
            tr[u << 1].lv = tr[u << 1].rv = 0;
            tr[u << 1].num = 0;
            tr[u << 1].tag = 0;
            tr[u << 1 | 1].sum = 0;
            tr[u << 1 | 1].lv = tr[u << 1 | 1].rv = 0;
            tr[u << 1 | 1].num = 0;
            tr[u << 1 | 1].tag = 0;
        }
        else {
            tr[u << 1].sum = tr[u << 1].r - tr[u << 1].l + 1;
            tr[u << 1].lv = tr[u << 1].rv = 1;
            tr[u << 1].num = 1;
            tr[u << 1].tag = 1;
            tr[u << 1 | 1].sum = tr[u << 1 | 1].r - tr[u << 1 | 1].l + 1;
            tr[u << 1 | 1].lv = tr[u << 1 | 1].rv = 1;
            tr[u << 1 | 1].num = 1;
            tr[u << 1 | 1].tag = 1;
        }
        tr[u].tag = -1;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,0,0,0,0,-1 };
    else {
        tr[u] = { l,r,0,0,0,0,-1 };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void modify(int u, int l, int r, int x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        if (x) {
            tr[u].num = 1;
            tr[u].lv = tr[u].rv = 1;
            tr[u].sum = tr[u].r - tr[u].l + 1;
            tr[u].tag = 1;
        }
        else {
            tr[u].num = 0;
            tr[u].lv = tr[u].rv = 0;
            tr[u].sum = 0;
            tr[u].tag = 0;
        }
        return;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (l <= mid)modify(u << 1, l, r, x);
    if (r > mid)modify(u << 1 | 1, l, r, x);
    pushup(u);
}
int main() {
    cin >> m;
    build(1, -500001, 500001);
    while (m--) {
        char op; int l, r;
        cin >> op >> l >> r;
        modify(1, l + 1, r + l, op=='B');
        pushdown(1);
        cout << tr[1].num << " " << tr[1].sum << endl;
    }
    return 0;
}
```

**D. Problem About Weighted Sum**

**题意**：在这个问题中，你要回答关于给定数组的加权和的查询。从形式上看，你将得到一个长度为 $n$ 的数组 $a[1 \dots n]$ 。你要回答两种类型的查询：

- 分段变化：给定三个整数 $l, r, d$ ，在数组的每个元素 $i$ 中添加 $d$ ，使得 $l \le i \le r$ 
- 计算加权和：给定两个整数 $l, r$ 计算并打印 $a[l] \cdot 1 + a[l + 1] \cdot 2 + \dots \ a[r] \cdot (r - l + 1)$ 

```cpp
typedef long long ll;
const int N = 1e6 + 10;
const ll INF = 4e18;
struct Node {
    int l, r;
    ll sum, mul, addtag;
}tr[N << 2];
ll a[N];
int n, m;
void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
    tr[u].mul = tr[u << 1].mul + tr[u << 1 | 1].mul;
}
void pushdown(int u) {
    if (tr[u].addtag) {
        tr[u << 1].sum += tr[u].addtag * (tr[u << 1].r - tr[u << 1].l + 1);
        tr[u << 1 | 1].sum += tr[u].addtag * (tr[u << 1 | 1].r - tr[u << 1 | 1].l + 1);
        tr[u << 1].mul += tr[u].addtag * (tr[u << 1].r - tr[u << 1].l + 1) * (tr[u << 1].l + tr[u << 1].r) / 2;
        tr[u << 1|1].mul += tr[u].addtag * (tr[u << 1|1].r - tr[u << 1|1].l + 1) * (tr[u << 1|1].l + tr[u << 1|1].r) / 2;
        tr[u << 1].addtag += tr[u].addtag;
        tr[u << 1 | 1].addtag += tr[u].addtag;
        tr[u].addtag = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r)tr[u] = { l,r,a[l],a[l]*l,0};
    else {
        tr[u] = { l,r };
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int l, int r, ll x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum += x * (tr[u].r - tr[u].l + 1);
        tr[u].mul += x * (tr[u].r - tr[u].l + 1) * (tr[u].l + tr[u].r) / 2;
        tr[u].addtag += x;
        return;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (l <= mid)modify(u << 1, l, r, x);
    if (r > mid)modify(u << 1 | 1, l, r, x);
    pushup(u);
}
ll querysum(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)return tr[u].sum;
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    ll res = 0;
    if (l <= mid)res = querysum(u << 1, l, r);
    if (r > mid)res += querysum(u << 1 | 1, l, r);
    return res;
}
ll querymul(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r)return tr[u].mul;
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    ll res = 0;
    if (l <= mid)res = querymul(u << 1, l, r);
    if (r > mid)res += querymul(u << 1 | 1, l, r);
    return res;
}
int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++)cin >> a[i];
    build(1, 1, n);
    while (m--) {
        int op, l, r; ll x;
        cin >> op >> l >> r;
        if (op == 2) {
            cout << querymul(1, l, r)-querysum(1,l,r)*(l-1) << endl;
        }
        else {
            cin >> x;
            modify(1, l, r, x);
        }
    }
    return 0;
}
```

**E. Wall**

**题意**：

建嘉正在用大小相同的砖块堆砌一堵墙。这堵墙由 $n$ 列砖组成，从左到右编号为 $0$ 至 $(n - 1)$ 。这些砖柱可能有不同的高度。柱子的高度就是其中砖块的数量。

建甲砌墙的过程如下。一开始，任何一列都没有砖块。然后，"健嘉 "会经历 $k$ 个添加或移除砖块的阶段。当所有 $k$ 个阶段都完成后，建造过程结束。在每个阶段中，建甲都会得到一系列连续的砖柱和高度 $h$ ，他的操作步骤如下：

- 在加砖阶段，"佳佳 "会在给定范围内砖量少于 $h$ 的柱子上加砖，使其砖量正好为 $h$ 。他对有 $h$ 个或更多砖块的列不做任何操作。
- 在删除阶段，贾健从给定范围内有多于 $h$ 块砖的列中删除砖块，使它们正好有 $h$ 块砖。他对砖块数为 $h$ 或更少的列不做任何操作。

你的任务是确定墙的最终形状。

```c++
#include<iostream>
#include<fstream>
#include<bitset>
#include<vector>
#include<map>
#include<set>
#include<unordered_set>
#define endl '\n'
using namespace std;
const int INF = 0x3f3f3f3f, N = 2e6 + 10;
struct Node {
    int l, r;
    int mn, mx;
}tr[N << 2];
int n, k;
void pushdown(int u) {
    tr[u << 1].mx = max(tr[u << 1].mx, tr[u].mx);
    tr[u << 1].mx = min(tr[u << 1].mx, tr[u].mn);

    tr[u << 1].mn = min(tr[u << 1].mn, tr[u].mn);
    tr[u << 1].mn = max(tr[u << 1].mn, tr[u].mx);

    tr[u << 1 | 1].mx = max(tr[u << 1 | 1].mx, tr[u].mx);
    tr[u << 1 | 1].mx = min(tr[u << 1 | 1].mx, tr[u].mn);

    tr[u << 1 | 1].mn = min(tr[u << 1 | 1].mn, tr[u].mn);
    tr[u << 1 | 1].mn = max(tr[u << 1 | 1].mn, tr[u].mx);

    tr[u].mx = 0, tr[u].mn = INF;
}
void build(int u, int l, int r) {
    if (l == r) { tr[u] = { l,r,INF,0 }; }
    else {
        tr[u] = { l,r,INF,0 };
        int mid = l + r >> 1;
        build(u<<1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }
}
void dfs(int u) {
    if (tr[u].l == tr[u].r) 
    {cout << tr[u].mx << "\n";return;}
    pushdown(u);
    dfs(u << 1);
    dfs(u << 1 | 1);
}
void MAX(int u, int l, int r, int x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].mx = max(tr[u].mx, x);
        tr[u].mn = max(tr[u].mn, x);
        return;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (l <= mid)MAX(u << 1, l, r, x);
    if (r > mid)MAX(u << 1 | 1, l, r, x);
}
void MIN(int u, int l, int r, int x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].mx = min(tr[u].mx, x);
        tr[u].mn = min(tr[u].mn, x);
        return;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (l <= mid)MIN(u << 1, l, r, x);
    if (r > mid)MIN(u << 1 | 1, l, r, x);
}
int main() {
    cin >> n >> k;
    build(1, 0, n - 1);
    while (k--) {
        int op, l, r, h;
        cin >> op >> l >> r >> h;
        if (op == 1)MAX(1, l, r, h);
        else MIN(1, l, r, h);
    }
    dfs(1);
    return 0;
}
```

**F. Mountain**

**题意**：这是 IOI 2005 中的一个问题，也是使用段树求解的 山上游乐园开设了一个全新的模拟过山车。模拟轨道由 $n$ 条端对端连接的轨道组成，第一条轨道的起点固定在高程 0 处。操作员拜特曼可以通过调整若干连续轨道的高程变化来随意重新配置轨道。其他轨道的高程变化不受影响。每次调整轨道时，后面的轨道都会根据需要升高或降低，以连接轨道，同时将起点保持在海拔 0。也就是说，只要轨道的高度不超过 $h$ ，只要没有到达轨道的终点，小车就会继续行驶。给定当天所有的乘车记录和轨道配置变化，计算每次乘车时小车在停止前所经过的轨道数。模拟器内部将轨道表示为 $n$ 升高变化 $d_i$ 的序列。最初轨道是水平的，即所有 $i$ 都是 $d_i = 0$ 。在一天中，乘车和重新配置是交错进行的。每次重新配置由三个数字指定： $a$ 、 $b$ 和 $D$ 。需要调整的区段由 $a$ 至 $b$ 轨道组成。(包括在内）。区段中每条轨道的高程变化设置为 $D$ 。每趟列车都由一个数字 $h$ 指定--即列车可以达到的最大高度。

```c++
#define ls tr[u].lson
#define rs tr[u].rson
#define mid (L+R>>1)

using namespace std;
typedef long long ll;
typedef pair<int, int>PII;

const int N = 1000010;
const int INF = 0x3f3f3f3f;

int n, tot = 1;
struct Node {
    int lson, rson;
    ll lsum, sum;
}tr[(int)(N*4.5)];
int tag[(int)(N*4.5)];
ll ans;

inline void check(int u, int L, int R) {
    if (!ls)ls = ++tot, tag[ls] = -INF;
    if (!rs)rs = ++tot, tag[rs] = -INF;
}

void pushup(int u) {
    tr[u].sum = tr[ls].sum + tr[rs].sum;
    tr[u].lsum = max(tr[ls].sum + tr[rs].lsum, tr[ls].lsum);
}

void pushdown(int u,int L,int R) {
    if (tag[u] != -INF) {
        tag[ls] = tag[rs] = tag[u];
        tr[ls].sum = 1ll * tag[u] * (mid - L + 1);
        tr[rs].sum = 1ll * tag[u] * (R - mid);
        tr[ls].lsum = max(0ll, tr[ls].sum);
        tr[rs].lsum = max(0ll, tr[rs].sum);
        tag[u] = -INF;
    }
}

void modify(int u, int L, int R, int l, int r, int v) {
    if (L >= l && R <= r) {
        tag[u] = v;
        tr[u].sum = 1ll * v * (R - L + 1);
        tr[u].lsum = max(0ll, tr[u].sum);
        return;
    }
    check(u, L, R);
    pushdown(u, L, R);
    if (l <= mid)modify(ls, L, mid, l, r, v);
    if (r > mid)modify(rs, mid + 1, R, l, r, v);
    pushup(u);
}

int find(int u, int L, int R, ll h) {
    if (L == R) {
        return h >= tr[u].lsum ? L : L - 1;
    }
    check(u, L, R);
    pushdown(u, L, R);
    if (h >= tr[ls].lsum) {
        return find(rs, mid + 1, R, h - tr[ls].sum);
    }
    else {
        return find(ls, L, mid, h);
    }
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    string op;
    int l, r, h;
    while ((cin >> op) && op != "E") {
        if (op[0] == 'Q') {
            cin >> h;
            cout << find(1, 1, n, h) << '\n';
        }
        else if (op[0] == 'I') {
            cin >> l >> r >> h;
            modify(1, 1, n, l, r, h);
        }
    }
    return 0;
}
```
