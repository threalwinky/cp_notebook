





































/**
Table of Contents
Dynamic Programming
    Matrix Mul
    Knapsack (bitset)
    DP Broken Profile
    SOS (Subset sum convolution, SOS, Generalized SOS)
    FWHT (XOR, AND, OR)
    DP Digit
    DP Connected Component
    DP Open Close Interval Trick
    DP Convex Hull Trick (CHT, LiChao, Line Container)
    DP Divide and Conquer
    DP Knuth
    Slope Trick

Graph
    Find Bridge
    Find Arti (DFS Tree)
    SCC (Kosaraju)
    2SAT
    Bridge Tree
    Block cut Tree
    Flow (Edmond, Dinic, Matching, MinCost, Konig, Generalized Matching)
    Dijkstra
    BellmanFord
    Tracing Negative Cycle
    Euler Cycle

Tree Algorithms
    LCA
    Heavy Light Decomp
    Centroid Decomp (Count and Query)
    Tree Isomorphism (rooted and unrooted)
    DP Reroot
    DP Knapsack on Tree

Combinatorics and Math
    Derangements
    Stars and Bars (Chia keo euler)
    Euler phi
    Rho
    Inclusion Exclusion
    FFT
    Gauss
    Xor basis
    Lagrange

Geometry Algorithms
    Orientation
    Line Segment Intersecion
    Polygon Area
    Pick Theorem
    Point in Polygon
    Convex Hull
    Mincowski sum
    Time is Money Trick
String Algorithms
    Hash
    Z + KMP
    DP KMP
    Trie (string and bit)
    Aho Corasick
Data Structure
    Mo Algorithm
    Segment Tree lazy (polynomial / range)
    Fenwick 2d lazy sum
    Fenwick 2d compress
    Persistent Segment Tree
    Merge Sort Tree
    Sparse Table
Misc
    DSU rollback (offline dynamic connectivity)
    Parallel Binary Search
    Xor hashing
    Katcl Geometry
**/

// Problem A - Matrix Multiplication
// Note nhan doc chia ngang
const int MOD = 1e9+7;
const int maxn = 500;
long long n;
const int siz = 2;
struct matrix
{
    long long a[siz][siz];
    matrix()
    {
        memset(a,0,sizeof(a));
    }
    void ident()
    {
        for(int i = 0; i < siz; i++)
            a[i][i] = 1;
    }

};
matrix iden;
matrix mul(matrix a, matrix b)
{
    matrix ans;
    for(int i = 0; i < siz; i++)
        for(int j = 0; j < siz; j++)
            for(int k = 0; k < siz; k++)
    {
        ans.a[i][j] += a.a[i][k] * b.a[k][j];
        ans.a[i][j] %= MOD;
    }
    return ans;
}
matrix binPow(matrix a, long long k)
{
    iden.ident();
    if(k == 0) return iden;
    matrix ans = iden;
    while(k)
    {
        if(k%2)
        {
            ans = mul(ans,a);
        }
        a = mul(a,a);
        k /= 2;
    }
    return ans;
}
void solve()
{
    matrix trans;
    trans.a[0][0] = 1;
    trans.a[0][1] = 1;
    trans.a[1][0] = 1;
    trans = binPow(trans,n - 1);
    matrix result;
    result.a[1][0] = 1;
    result = mul(result,trans);
    cout << result.a[1][0];
}

// Note cho B-D Knapsack: log trick, harmonic, bitset
bitset<100005> dp;
void solve()
{
    dp[0] = 1;
    for(int i = 1; i <= cnt; i++)
    {
        dp |= dp<<w[i];
    }

    for(int i = 1; i <= n; i++)
        cout << dp[i];
}

// E - Dp Broken Profile: Count 1x2 and 2x1 tiles in grid
long long dp[1 << 10][1005];
void solve() {
    // n small m big
    int n, m;
    cin >> n >> m;
    dp[0][0] = 1;
    for (int j = 0; j < m; j++) for (int i = 0; i < n; i++) {
        for (int mask = 0; mask < (1 << n); mask++) {
            dp[mask ^ (1 << i)][j+1] = dp[mask][j]; // Vertical/no tile
        }
        for (int mask = 0; mask < (1 << n); mask++) {
            if (i+1 < n && !(mask & (1 << i)) && !(mask & (1 << (i + 1)))) // Horizontal tile
                dp[mask ^ (1 << (i + 1))][j+1] += dp[mask][j];


        }
        for (int mask = 0; mask < (1 << n); mask++)
        {
            while (dp[mask][j+1] >= MOD) dp[mask][j+1] -= MOD;
           // cout << mask << ' ' << i <<  ' ' << j << ' ' << dp[mask][j+1] << '\n';
            dp[mask][j] = dp[mask][j+1];
        }
    }
    cout << dp[0][m];
}

// F - SOS DP
// Normal SOS, submask and supermask
void normalSOS()
{
      for(int i = 0; i < 20; i++)
        for(int mask = 0; mask < maxn; mask++)
            if((mask&(1<<i)) == 0) dp[mask] += dp[mask ^ (1<<i)];
            else dp2[mask] += dp2[mask ^ (1<<i)];
    // Inverse SOS
    for(int i = 0; i < 20; i++)
        for(int mask = 0; mask < maxn; mask++)
            if((mask&(1<<i)) == 0) dp[mask] -= dp[mask ^ (1<<i)];
            else dp2[mask] -= dp2[mask ^ (1<<i)];
    // dp is supermask, dp2 is submask
}
// Subset Convolution, Combining to function such that H(mask) = F(i) * G(mask \ i)
void subset_convo()
{
  // Make fhat[][] = {0} and ghat[][] = {0}
  for (int mask = 0; mask < (1 << n); mask++) {
    fhat[__builtin_popcount(mask)][mask] = d[mask];
    ghat[__builtin_popcount(mask)][mask] = e[mask];
  }
  // Apply zeta transform on fhat[][] and ghat[][]
  for (int i = 0; i <= n; i++) {
    for (int j = 0; j <= n; j++) {
      for (int mask = 0; mask < (1 << n); mask++) {
        if ((mask & (1 << j)) != 0) {
          fhat[i][mask] += fhat[i][mask ^ (1 << j)];
          if (fhat[i][mask] >= MOD) fhat[i][mask] -= MOD;
          ghat[i][mask] += ghat[i][mask ^ (1 << j)];
          if (ghat[i][mask] >= MOD) ghat[i][mask] -= MOD;
        }
      }
    }
  }
  // Do the convolution and store into h[][] = {0}
  for (int mask = 0; mask < (1 << n); mask++) {
    for (int i = 0; i <= n; i++) {
      for (int j = 0; j <= i; j++) {
        h[i][mask] += 1LL * fhat[j][mask] * ghat[i - j][mask] % MOD;
        if (h[i][mask] >= MOD) h[i][mask] -= MOD;
      }
    }
  }
  // Apply inverse SOS dp on h[][]
  for (int i = 0; i <= n; i++) {
    for (int j = 0; j <= n; j++) {
      for (int mask = 0; mask < (1 << n); mask++) {
        if ((mask & (1 << j)) != 0) {
          h[i][mask] -= h[i][mask ^ (1 << j)];
          if (h[i][mask] < 0) h[i][mask] += MOD;
        }
      }
    }
  }
  for (int mask = 0; mask < (1 << n); mask++)  a[mask] = h[__builtin_popcount(mask)][mask];
}
// FWHT Convo
int fpow(int n, long long k, int p = (int) 1e9 + 7) {
    int r = 1;
    for (; k; k >>= 1) {
        if (k & 1) r = (long long) r * n % p;
        n = (long long) n * n % p;
    }
    return r;
}

/*
 * matrix:
 * +1 +1
 * +1 -1
 */
void XORFFT(int a[], int n, int p, int invert) {
    for (int i = 1; i < n; i <<= 1) {
        for (int j = 0; j < n; j += i << 1) {
            for (int k = 0; k < i; k++) {
                int u = a[j + k], v = a[i + j + k];
                a[j + k] = u + v;
                if (a[j + k] >= p) a[j + k] -= p;
                a[i + j + k] = u - v;
                if (a[i + j + k] < 0) a[i + j + k] += p;
            }
        }
    }
    if (invert) {
        long long inv = fpow(n, p - 2, p);
        for (int i = 0; i < n; i++) a[i] = a[i] * inv % p;
    }
}
/*
 * Matrix:
 * +1 +1
 * +1 +0
 */
void ORFFT(int a[], int n, int p, int invert) {
    for (int i = 1; i < n; i <<= 1) {
        for (int j = 0; j < n; j += i << 1) {
            for (int k = 0; k < i; k++) {
                int u = a[j + k], v = a[i + j + k];
                if (!invert) {
                    a[j + k] = u + v;
                    a[i + j + k] = u;
                    if (a[j + k] >= p) a[j + k] -= p;
                }
                else {
                    a[j + k] = v;
                    a[i + j + k] = u - v;
                    if (a[i + j + k] < 0) a[i + j + k] += p;
                }
            }
        }
    }
}
/*
 * matrix:
 * +0 +1
 * +1 +1
 */
void ANDFFT(int a[], int n, int p, int invert) {
    for (int i = 1; i < n; i <<= 1) {
        for (int j = 0; j < n; j += i << 1) {
            for (int k = 0; k < i; k++) {
                int u = a[j + k], v = a[i + j + k];
                if (!invert) {
                    a[j + k] = v;
                    a[i + j + k] = u + v;
                    if (a[i + j + k] >= p) a[i + j + k] -= p;
                }
                else {
                    a[j + k] = v - u;
                    if (a[j + k] < 0) a[j + k] += p;
                    a[i + j + k] = u;
                }
            }
        }
    }
}
// Generalized SOS
void generalizeSOS()
{
    for(int i = 1; i <= 6; i++)
    {
        for(int mask = maxn - 1; mask >= 0; mask--)
        {
            int p = getdigit(mask,i);

            for(int x = 0; x < p; x++)
            {
                int newmask = mask - p * pow1[i-1] + x * pow1[i-1];
          //      cout << mask << ' ' << newmask << '\n';
                dp[mask] += dp[newmask];
            }
        }
    }
    // Inverse SOS
    for(int i = 1; i <= 6; i++)
    {
        for(int mask = 0; mask < maxn; mask++)
        {
            int p = getdigit(mask,i);

            for(int x = 0; x < p; x++)
            {
                int newmask = mask - p * pow1[i-1] + x * pow1[i-1];
          //      cout << mask << ' ' << newmask << '\n';
                dp[mask] -= dp[newmask];
            }
        }
    }
}

// I - DP DIGIT
// Note can change representation of base in the number in question is too large
long long dp[25][11][2][2];
long long a,b;
int digit[25];
long long comp(int pos, int last,int tight, int isPos)
{
    if(pos < 0) return 1;
    if(dp[pos][last][tight][isPos] != -1) return dp[pos][last][tight][isPos];
    long long ans = 0;
    int bound = digit[pos];
    if(!tight) bound = 9;
    for(int x = 0; x <= bound; x++)
    {
        if(isPos && x == last) continue;
        int newtight = tight && (x == digit[pos]);
        int newPos = isPos || (x > 0);
        ans += comp(pos-1, x, newtight, newPos);
    }
    return dp[pos][last][tight][isPos] = ans;
}
long long calc(long long x)
{
    if(x < 0) return 0;
    int cnt = -1;
    while(x)
    {
        digit[++cnt] = x % 10;
        x /= 10;
    }
    memset(dp,-1,sizeof(dp));
    return comp(cnt,0,1,0);
}
void solve()
{
    cin >> a >> b;
    cout << calc(b) - calc(a-1);
}
//  J - DP Connected Component
// Build by inserting element in permutation in increasing order into connected components
void DPCC()
{
        # Create a new component
        DP[i+1][j+1] += DP[i][j] * ( j + 1 )

        # Add at the beggining or the end of a component
        DP[i+1][j] += DP[i][j] * ( 2 * j )

        # Merge two existing components
        DP[i+1][j-1] += DP[i][j] * ( j - 1 )
}
// K - DP Open Close Interval Trick
// Manages Intervals using DP
void DPOC()
{
    dp[0][offset] = 1;
    for(int i = 1; i <= n; i++)
    {
        for(int k = 0; k <= 50; k++)
        for(int j = 0; j <= 2*offset; j++)
        {
            if(dp[k][j])
            {
                tmp[k][j] = (tmp[k][j] + dp[k][j] * (k+1)) % MOD;
                if(j - a[i] >= 0) tmp[k+1][j-a[i]] = (tmp[k+1][j-a[i]] + dp[k][j]) % MOD;
                if(j + a[i] <= 2 * offset && k > 0) tmp[k-1][j+a[i]] = (tmp[k-1][j+a[i]] + k*dp[k][j]) % MOD;
          //      cout << k << ' ' << j << ' ';
               // cout << tmp[k+1][j] << '\n';
            }

        }
        for(int k = 0; k <= 50; k++)
            for(int j = 0; j <= 2 * offset; j++)
                dp[k][j] = tmp[k][j], tmp[k][j] = 0;

    }
}
// L M N - DP Convex Hull Trick Related Techniques
// Add lines in form of ax + b and find the maximal value for x
// Always try to transform into ax+b from by expanding the function, such as (a-x)^2
// Convex Hull Trick
// Decreasing Insertion, Query Min
struct CHT {
      vector<long long> a, b;

      bool cross(int i, int j, int k) {
            return 1.d*(a[j] - a[i])*(b[k] - b[i]) >= 1.d*(a[k] - a[i])*(b[j] - b[i]);
      }

      void add(long long A, long long B) {
            a.push_back(A);
            b.push_back(B);

            while (a.size() > 2 && cross(a.size() - 3, a.size() - 2, a.size() - 1)) {
            a.erase(a.end() - 2);
            b.erase(b.end() - 2);
        }
      }

      long long query(long long x) {
            int l = 0, r = a.size() - 1;

            while (l < r) {
                  int mid = l + (r - l)/2;
            long long f1 = a[mid] * x + b[mid];
            long long f2 = a[mid + 1] * x + b[mid + 1];

            if (f1 > f2) l = mid + 1;
            else r = mid;
            }

            return a[l]*x + b[l];
      }
};
// Line Container, max query
struct Line {
    mutable ll k, m, p;
    bool operator<(const Line& o) const { return k < o.k; }
    bool operator<(ll x) const { return p < x; }
};

struct LineContainer : multiset<Line, less<>> {
    // (for doubles, use inf = 1/.0, div(a,b) = a/b)
    static const ll inf = LLONG_MAX;
    ll div(ll a, ll b) { // floored division
        return a / b - ((a ^ b) < 0 && a % b); }
    bool isect(iterator x, iterator y) {
        if (y == end()) return x->p = inf, 0;
        if (x->k == y->k) x->p = x->m > y->m ? inf : -inf;
        else x->p = div(y->m - x->m, x->k - y->k);
        return x->p >= y->p;
    }
    void add(ll k, ll m) {
        auto z = insert({k, m, 0}), y = z++, x = y;
        while (isect(y, z)) z = erase(z);
        if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
        while ((y = x) != begin() && (--x)->p >= y->p)
            isect(x, erase(y));
    }
    ll query(ll x) {
        assert(!empty());
        auto l = *lower_bound(x);
        return l.k * x + l.m;
    }
};
// Li Chao Tree
const int N = 2e5 + 5;

struct line {
    ll a, b;
    ll val(ll x) { return a * x + b; }
};

line seg[4 * N];

void upd(int id, int l, int r, line lin) {
    if (l == r) {
        if (lin.val(l) > seg[id].val(l))
            seg[id] = lin;
        return;
    }
    int mid = (l + r) >> 1;
    if (seg[id].a > lin.a)
        swap(seg[id], lin);
    // sap xep lai do doc duong thang de tong quat hoa bai toan
    if (lin.val(mid) > seg[id].val(mid)) {
        swap(seg[id], lin);
        upd(id << 1, l, mid, lin);
    } else
        upd(id << 1 | 1, mid + 1, r, lin);
}

void upd_range(int id, int l, int r, int u, int v, line lin) {
    if (l > v || r < u)
        return;
    if (l >= u && r <= v) {
        upd(id, l, r, lin);
        return;
    }
    int mid = (l + r) >> 1;
    upd_range(id << 1, l, mid, u, v, lin);
    upd_range(id << 1 | 1, mid + 1, r, u, v, lin);
}

ll get(int id, int l, int r, int pos) {
    if (l == pos && l == r) {
        return seg[id].val(pos);
    }
    int mid = (l + r) >> 1;
    if (mid >= pos)
        return max(seg[id].val(pos), get(id << 1, l, mid, pos));
    else
        return max(seg[id].val(pos), get(id << 1 | 1, mid + 1, r, pos));
}

// O - DP Divide and Conquer
long long cost(int l, int r)
{
    long long x = a[r] - a[l-1];
    return x * x;
}
void split(int level, int l, int r, int optL, int optR)
{
   // cout << l << ' ' << r << endl;
    if(l > r) return;

    int optCut = optL;
    int mid = (l+r)/2;
    dp[mid][level] = dp[optCut][level-1] + cost(optCut+1,mid);
    for(int cut = optL; cut <= min(mid,optR); cut++)
    {
        if(dp[mid][level] > dp[cut][level-1] + cost(cut+1,mid))
        {
            dp[mid][level] = dp[cut][level-1] + cost(cut+1,mid);
            optCut = cut;
        }
    }
 //   cout << dp[mid][level] << '\n';
    split(level,l,mid-1,optL,optCut);
    split(level,mid+1,r,optCut,optR);
}

// P - DP Knuth (2D1D)
/*
* Complexity: O(N^2)
* f[i][j] = min(f[i][k] + f[k][j] + c[i][j], i < k < j)
* a[i][j] = min(k | i < k < j && f[i][j] = f[i][k] + f[k][j] + c[i][j])
* Sufficient condition: a[i][j - 1] <= a[i][j] <= a[i + 1][j] or
* c[x][z] + c[y][t] <= c[x][t] + c[y][z] (quadrangle inequality) and c[y][z] <= c[x][t] (monotonicity), x <= y <= z <= t
*/

void knuth() {
    for (int i = 1; i <= n; i++) {
        f[i][i] = 0;
        a[i][i] = i;
    }
    for(int len = 1; len <= n-1;len++)
            for(int i = 1; i <= n-len;i++)
        {
            int j = i + len;
            f[i][j] = INF;
            for (int k = a[i][j - 1]; k <= a[i + 1][j]; k++)
            {
                if (f[i][j] > f[i][k-1] + f[k][j] + c[i][j])
                {
                    f[i][j] = f[i][k-1] + f[k][j] + c[i][j];
                    a[i][j] = k;
                }
            }
        }
    cout << f[1][n] << '\n';
}

// T - DP Slope Trick
long long a[300005];
long long b[300005];
void solve()
{
    long long ans = 0;
    cin >> n;
    for(int i = 1; i <= n; i++)
    {
        long long x;
        cin >> x;
        b[i] = x;
        x -= i;
        slope.push(x);
        ans += slope.top() - x;
        a[i] =  slope.top() - x + b[i];
        slope.pop();
        slope.push(x);
    }
    for(int i = n-1; i >= 1; i--)
    {
        long long v = a[i+1] - 1;
    //    cout << a[i] << ' ' << v << '\n';
        a[i] = min(a[i],v);
    }
    cout << ans << '\n';
    for(int i = 1; i <= n; i++)
    {
        cout << a[i] << ' ';
    }

}

// V - Find Bridge
// DFS Tree moment, use tree algo to solve problems
void dfs(int u, int par)
{
    tin[u] = low[u] = ++timer;
    for(auto v: adj[u])
    {
        if(v != par)
        {
            if(tin[v] == 0)
            {
                dfs(v,u);
                low[u] = min(low[u], low[v]);
                if(low[v] > tin[u])
                {
                    bridge.insert({min(u,v),max(u,v)});
                }
            }
            else low[u] = min(low[u],tin[v]);
        }
    }
}

// W - Find Arti
void dfs(int u, int par)
{
    int child = 0;
    tin[u] = low[u] = ++timer;
    for(auto v: adj[u])
    {
        if(v != par)
        {
            if(tin[v] == 0)
            {
                child++;
                dfs(v,u);
                low[u] = min(low[u], low[v]);
               // cout << v << ' ' << u << ' ' << low[v] << ' ' << tin[u] << '\n';
                if(low[v] >= tin[u] && par != -1)
                {
                    arti.insert(u);
                }
            }
            else low[u] = min(low[u],tin[v]);
        }
    }
    if(par == -1 && child > 1) arti.insert(u);
}

// NKPOLICE
#include <bits/stdc++.h>

using namespace std;

const int maxN  = 100010;

int n, m, q;
int timeDfs = 0;
int low[maxN], num[maxN], tail[maxN];
int depth[maxN], p[maxN][20];
bool joint[maxN];
vector <int> g[maxN];

/* Tính mảng p */
void calP() {
    p[1][0] = 1;
    for (int j = 1; j <= 19; j++)
        for (int i = 1; i <= n; i++)
            p[i][j] = p[p[i][j - 1]][j - 1];
}

/* Tìm tổ tiên của đỉnh u là con trực tiếp của đỉnh par */
int findParent(int u, int par) {
    for (int i = 19; i >= 0; i--)
        if (depth[p[u][i]] > depth[par]) u = p[u][i];
    return u;
}

/* Tìm khớp cầu */
void dfs(int u, int pre) {
    int child = 0;
    num[u] = low[u] = ++timeDfs;
    for (int v : g[u]){
        if (v == pre) continue;
        if (!num[v]) {
            child++;
            p[v][0] = u;
            depth[v] = depth[u] + 1;
            dfs(v, u);
            low[u] = min(low[u], low[v]);
            if (u == pre) {
                if (child > 1) joint[u] = true;
            }
            else if (low[v] >= num[u]) joint[u] = true;
        }
        else low[u] = min(low[u], num[v]);
    }
    tail[u] = timeDfs;
}

/* Kiểm tra xem đỉnh u có nằm trong cây con DFS gốc root hay không? */
bool checkInSubtree(int u, int root) {
    return num[root] <= num[u] && num[u] <= tail[root];
}

/* Xử lí truy vấn 1 */
bool solve1(int a, int b, int g1, int g2) {
    /* Vì ta coi g2 là con trực tiếp của g1 nên khi g1 là con của g2,
    ta phải đổi chỗ 2 giá trị g1 và g2 cho nhau */
    if (num[g1] > num[g2]) swap(g1, g2);

    /* Kiểm tra nếu cạnh (g1, g2) không phải là cầu */
    if (low[g2] != num[g2]) return true;

    if (checkInSubtree(a, g2) && !checkInSubtree(b, g2)) return false;
    if (checkInSubtree(b, g2) && !checkInSubtree(a, g2)) return false;
    return true;
}

/* Xử lí truy vấn 2 */
bool solve2(int a, int b, int c) {
    if (!joint[c]) return true;
    int pa = 0, pb = 0;
    if (checkInSubtree(a, c)) pa = findParent(a, c);
    if (checkInSubtree(b, c)) pb = findParent(b, c);

    if (!pa && !pb) return true;
    if (pa == pb) return true;
    if (!pa && low[pb] < num[c]) return true;
    if (!pb && low[pa] < num[c]) return true;
    if (pa && pb && low[pa] < num[c] && low[pb] < num[c]) return true;

    return false;
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= m; i++) {
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    depth[1] = 1;
    dfs(1, 1);
    calP();
    cin >> q;
    while (q--) {
        int type, a, b, c, g1, g2;
        cin >> type;
        if (type == 1) {
            cin >> a >> b >> g1 >> g2;
            cout << (solve1(a, b, g1, g2) ? "yes\n" : "no\n");
        }
        else {
            cin >> a >> b >> c;
            cout << (solve2(a, b, c) ? "yes\n" : "no\n");
        }
    }
}

// X - SCC
// Note nen scc thanh dp dag
vector<int> adj[maxn+5];
vector<int> adjRev[maxn+5];
vector<int> topo;
bool visit[maxn+5];
int col[maxn+5];
int n,m;
void dfs(int u)
{
    visit[u] = 1;
    for(auto v: adj[u])
    {
        if(visit[v] == 0)
            dfs(v);
    }
    topo.push_back(u);
}
int timer = 0;
void dfs2(int u)
{
    col[u] = timer;
    for(auto v: adjRev[u])
    {
        if(col[v] == 0)
        dfs2(v);
    }
}
void solve()
{
    cin >> n >> m;
    for(int i = 1; i <= m; i++)
    {
        int u,v;
        cin >> u >> v;
        adj[u].push_back(v);
        adjRev[v].push_back(u);
    }
    for(int i = 1; i <= n; i++)
    {
        if(visit[i] == 0)
        {
            dfs(i);
        }
    }
    reverse(topo.begin(),topo.end());
    for(auto u: topo)
    {
        if(col[u] == 0)
        {
            timer++;
            dfs2(u);
        }
    }
    cout << timer << '\n';
    for(int i = 1; i <= n; i++)
        cout << col[i] << ' ';
}

// Z - 2SAT
bool visit[maxn+5];
int comp[maxn+5];
int col = 0;
vector<int> adj[maxn+5];
vector<int> adjrev[maxn+5];
vector<int> topo;
int neg(int x)
{
    return (x + n) % (2 * n);
}
void dfs(int u)
{
    visit[u] = true;
    for(auto v: adj[u])
    {
        if(!visit[v]) dfs(v);
    }
    topo.push_back(u);
}
void dfs_scc(int u)
{
    comp[u] = col;
    for(auto v: adjrev[u])
        if(comp[v] == 0) dfs_scc(v);
}

void solve()
{
    cin >> m >> n;
    for(int i = 1; i <= m; i++)
    {
        char c;
        cin >> c;
        int u;
        cin >> u;
        u--;
        if(c == '-') u = neg(u);
        cin >> c;
        int v;
        cin >> v;
        v--;
        if(c == '-') v = neg(v);
        adj[u].push_back(neg(v));
        adj[v].push_back(neg(u));
        adjrev[neg(v)].push_back(u);
        adjrev[neg(u)].push_back(v);
    }
    for(int i = 0; i < 2 * n; i++)
        if(!visit[i]) dfs(i);
    reverse(topo.begin(),topo.end());
    for(auto u: topo)
    {
        if(comp[u] == 0)
        {
            //cout << u << ' ' << col << '\n';
            col++;
            dfs_scc(u);
        }
    }
    vector<char> ans;
    for(int i = 0; i < n; i++)
    {
        if(comp[i] == comp[i+n])
        {
            cout << "IMPOSSIBLE";
            return;
        }
        if(comp[i] < comp[i+n]) ans.push_back('+');
        else ans.push_back('-');
    }
    for(auto x: ans)
        cout << x << ' ';


}

// AA - Bridge Tree
const int maxn = 1e5;
int n,m;
int tin[maxn+5], low[maxn+5];
int timer = 0;
int comp[maxn+5];
int col = 0;
vector<int> adj[maxn+5];
vector<int> adjTree[maxn+5];
stack<int> st;
void dfs(int u, int par)
{
    tin[u] = low[u] = ++timer;
    st.push(u);
    for(auto v: adj[u])
    {
        if(v != par)
        {
            if(!tin[v])
            {
                dfs(v,u);
                low[u] = min(low[u],low[v]);
                if(low[v] > tin[u])
                {
                    col++;
                    while(st.top() != v) comp[st.top()] = col, st.pop();
                    comp[st.top()] = col;
                    st.pop();
                }
            }
            else low[u] = min(low[u],tin[v]);
        }
    }
}
int dist[maxn+5];
void dfs_tree(int u, int par)
{
    for(auto v: adjTree[u])
    {
        if(v != par)
        {
            dist[v] = dist[u] + 1;
            dfs_tree(v,u);
        }
    }
}
void solve()
{
    cin >> n >> m;
    for(int i = 1; i <= m; i++)
    {
        int u,v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    dfs(1,1);
    if(!st.empty())
    {
        col++;
        while(!st.empty())
        {
            comp[st.top()] = col;
            st.pop();
        }

    }
    int cnt = 0;
    for(int i = 1; i <= n; i++)
        for(auto v: adj[i])
        {
            if(comp[i] != comp[v])
            {
                 adjTree[comp[i]].push_back(comp[v]);
            //    cout << "edge " << comp[i] << ' ' << comp[v] << '\n';
            }

        }
    dist[1] = 0;
    dfs_tree(1,1);
    int root = 1;
    for(int i = 2; i <= col; i++)
        if(dist[root] < dist[i]) root = i;
    dist[root] = 0;
    dfs_tree(root,root);
    int diameter = 0;
    for(int i = 1; i <= col; i++)
        diameter = max(diameter,dist[i]);
    cout << col - diameter - 1 << '\n';
    for(int i = 1; i <= n; i++)
        adj[i].clear();
    for(int i = 1; i <= col; i++)
        adjTree[i].clear();
    for(int i = 1; i <= n; i++)
        tin[i] = low[i] = 0;
    timer = 0;
    col = 0;


}

// AB - Block Cut Tree
const int N = 4e5 + 9;

int T, low[N], dis[N], art[N], sz;
vector<int> g[N], bcc[N], st;
void dfs(int u, int pre = 0) {
  low[u] = dis[u] = ++T;
  st.push_back(u);
  for(auto v: g[u]) {
    if(!dis[v]) {
      dfs(v, u);
      low[u] = min(low[u], low[v]);
      if(low[v] >= dis[u]) {
        sz ++;
        int x;
        do {
          x = st.back();
          st.pop_back();
          bcc[x].push_back(sz);
        } while(x ^ v);
        bcc[u].push_back(sz);
      }
    } else if(v != pre) low[u] = min(low[u], dis[v]);
  }
}

int dep[N], par[N][20], cnt[N], id[N];
vector<int> bt[N];
void dfs1(int u, int pre = 0) {
  dep[u] = dep[pre] + 1;
  cnt[u] = cnt[pre] + art[u];
  par[u][0] = pre;
  for(int k = 1; k <= 18; k++) par[u][k] = par[par[u][k - 1]][k - 1];
  for(auto v: bt[u]) if(v != pre) dfs1(v, u);
}

int lca(int u, int v) {
  if(dep[u] < dep[v]) swap(u, v);
  for(int k = 18; k >= 0; k--) if(dep[par[u][k]] >= dep[v]) u = par[u][k];
  if(u == v) return u;
  for(int k = 18; k >= 0; k--) if(par[u][k] != par[v][k]) u = par[u][k], v = par[v][k];
  return par[u][0];
}

int dist(int u, int v) {
  int lc = lca(u, v);
  return cnt[u] + cnt[v] - 2 * cnt[lc] + art[lc];
}

int32_t main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  int n, m;
  cin >> n >> m;
  while(m--) {
    int u, v;
    cin >> u >> v;
    g[u].push_back(v);
    g[v].push_back(u);
  }
  dfs(1);
  for(int u = 1; u <= n; u++) {
    if(bcc[u].size() > 1) { //AP
      id[u] = ++sz;
      art[id[u]] = 1; //if id in BCT is an AP on real graph or not
      for(auto v: bcc[u]) {
        bt[id[u]].push_back(v);
        bt[v].push_back(id[u]);
      }
    } else if(bcc[u].size() == 1) id[u] = bcc[u][0];
  }

  dfs1(1);
  int q;
  cin >> q;
  while(q--) {
    int u, v;
    cin >> u >> v;
    int ans;
    if(id[u] == id[v]) ans = 0;
    else ans = dist(id[u], id[v]) - art[id[u]] - art[id[v]];
    cout << ans << '\n';//number of articulation points in the path from u to v except u and v
    //u and v are in the same bcc if ans == 0
  }
  return 0;
}
// AC to  AH - Max Flow type of problems

// Edmond Karp
long long cap[maxn+5][maxn+5];
long long flow[maxn+5][maxn+5];
int pre[maxn+5];
int n,m;
long long remaining(int u, int v)
{
    return cap[u][v] - flow[u][v];
}
queue<int> q;
bool canFind()
{
    pre[1] = 1;
    q.push(1);
    for(int i = 2; i <= n; i++)
        pre[i] = 0;
    while(!q.empty())
    {
        int u = q.front();
        q.pop();
        for(auto v: adj[u])
        {
            if(pre[v] == 0 && remaining(u,v) > 0)
            {
                pre[v] = u;
                q.push(v);
            }
        }
    }
    return pre[n] != 0;
}
void max_flow()
{
    long long ans = 0;
    while(canFind())
    {
        long long cur = 1e18;
        for(int u = n; u != 1; u = pre[u])
        {
            int last = pre[u];
            cur = min(cur,remaining(last,u));
        }
        for(int u = n; u != 1; u = pre[u])
        {
            int last = pre[u];
            flow[last][u] += cur;
            flow[u][last] -= cur;
        }
        ans += cur;
    }

    cout << ans;

}
void solve()
{
    cin >> n >> m;
    for(int i = 1; i <= m; i++)
    {
        long long u,v,c;
        cin >> u >> v >> c;
        cap[u][v] += c;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    max_flow();

}
// Dinic
struct FlowEdge {
    int v, u;
    long long cap, flow = 0;
    FlowEdge(int v, int u, long long cap) : v(v), u(u), cap(cap) {}
};

struct Dinic {
    const long long flow_inf = 1e18;
    vector<FlowEdge> edges;
    vector<vector<int>> adj;
    int n, m = 0;
    int s, t;
    vector<int> level, ptr;
    queue<int> q;

    Dinic(int n, int s, int t) : n(n), s(s), t(t) {
        adj.resize(n);
        level.resize(n);
        ptr.resize(n);
    }

    void add_edge(int v, int u, long long cap) {
        edges.emplace_back(v, u, cap);
        edges.emplace_back(u, v, 0);
        adj[v].push_back(m);
        adj[u].push_back(m + 1);
        m += 2;
    }

    bool bfs() {
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            for (int id : adj[v]) {
                if (edges[id].cap - edges[id].flow < 1)
                    continue;
                if (level[edges[id].u] != -1)
                    continue;
                level[edges[id].u] = level[v] + 1;
                q.push(edges[id].u);
            }
        }
        return level[t] != -1;
    }

    long long dfs(int v, long long pushed) {
        if (pushed == 0)
            return 0;
        if (v == t)
            return pushed;
        for (int& cid = ptr[v]; cid < (int)adj[v].size(); cid++) {
            int id = adj[v][cid];
            int u = edges[id].u;
            if (level[v] + 1 != level[u] || edges[id].cap - edges[id].flow < 1)
                continue;
            long long tr = dfs(u, min(pushed, edges[id].cap - edges[id].flow));
            if (tr == 0)
                continue;
            edges[id].flow += tr;
            edges[id ^ 1].flow -= tr;
            return tr;
        }
        return 0;
    }

    long long flow() {
        long long f = 0;
        while (true) {
            fill(level.begin(), level.end(), -1);
            level[s] = 0;
            q.push(s);
            if (!bfs())
                break;
            fill(ptr.begin(), ptr.end(), 0);
            while (long long pushed = dfs(s, flow_inf)) {
                f += pushed;
            }
        }
        return f;
    }
};

// Distinct Routes
using namespace std;
const int N = 505, INF = 1e9 + 5;
int n, m, f[505][505],c[505][505], par[N];
int id = 0;
vector<int> adj[N], res[N];
int maxflow(){
    queue<int> q; q.push(1);
    memset(par, 0, sizeof par);
    par[1] = -1;
    while(!q.empty()){
        int i = q.front();
        q.pop();
        if(i == n) return 1;
        for(int x : adj[i]){
            if(par[x] || f[i][x] >= c[i][x]) continue;
            par[x] = i;
            q.push(x);
        }
    }
    return 0;
}
void dfs(int v)
{
    for(auto x: adj[v])
    {
        if(f[v][x] == -1)
        {
            f[v][x]++;
            res[id].push_back(x);
            dfs(x);
            return;
        }
    }

}
signed main()
{
    cin.tie(0) -> sync_with_stdio(0);
    cin >> n >> m;
    for(int i = 1, x, y; i <= m; i++){
        cin >> x >> y;
        adj[x].pb(y);
        adj[y].pb(x);
        c[x][y] = 1;
    }
    int ans = 0;
    while((maxflow())){
        int add = 1e9;
        for(int i = n; i != 1; i = par[i]){
            int x = par[i];
            add = min(add,c[x][i]-f[x][i]);
        }
        for(int i = n; i != 1; i = par[i]){
            int x = par[i];
            f[x][i] += add;
            f[i][x] -= add;
        }
        ans += add;
    }
    cout << ans << '\n';
    for(auto x: adj[n])
    {
        if(f[n][x] == -1)
        {
            ++id;
            res[id].push_back(n);
            res[id].push_back(x);
            f[n][x]++;
            dfs(x);
            reverse(res[id].begin(),res[id].end());
        }
    }
    for(int i = 1; i <= id; i++){
        cout << res[i].size() << '\n';
        for(int x : res[i]) cout << x << ' ';
        cout << '\n';
    }
    return 0;
}
// Mincost MaxFlow
#define MCMF MincostMaxflow
#define flow_t int
#define cost_t int
const flow_t foo = (flow_t)1e9;
const cost_t coo = (cost_t)1e9;
namespace MincostMaxflow {
const int maxv = 1e5 + 5;
const int maxe = 1e6 + 5;
int n, s, t, E;
int adj[maxe], nxt[maxe], lst[maxv], frm[maxv];
flow_t cap[maxe], flw[maxe], totalFlow;
cost_t cst[maxe], pot[maxe], dst[maxv], totalCost;

void init(int nn, int ss, int tt) {
  n = nn, s = ss, t = tt;
  fill_n(lst, n, -1), E = 0;
}
void add(int u, int v, flow_t ca, cost_t co) {
  adj[E] = v, cap[E] = ca, flw[E] = 0, cst[E] = +co, nxt[E] = lst[u],
  lst[u] = E++;
  adj[E] = u, cap[E] = 0, flw[E] = 0, cst[E] = -co, nxt[E] = lst[v],
  lst[v] = E++;
}
void bellman() {
  fill_n(pot, n, 0);
  while (1) {
    int found = 0;
    for (int u = 0; u < n; u++)
      for (int e = lst[u]; e != -1; e = nxt[e])
        if (flw[e] < cap[e]) {
          int v = adj[e];
          if (pot[v] > pot[u] + cst[e]) {
            pot[v] = pot[u] + cst[e];
            found = 1;
          }
        }
    if (!found)
      break;
  }
}
int dijkstra() {
  priority_queue<pair<cost_t, int>> que;
  fill_n(dst, n, coo), dst[s] = 0;
  que.push(make_pair(-dst[s], s));
  while (que.size()) {
    cost_t dnow = -que.top().first;
    int u = que.top().second;
    que.pop();
    if (dst[u] < dnow)
      continue;
    for (int e = lst[u]; e != -1; e = nxt[e])
      if (flw[e] < cap[e]) {
        int v = adj[e];
        cost_t dnxt = dnow + cst[e] + pot[u] - pot[v];
        if (dst[v] > dnxt) {
          dst[v] = dnxt;
          frm[v] = e;
          que.push(make_pair(-dnxt, v));
        }
      }
  }
  return dst[t] < coo;
}


cost_t mincost() {
  totalCost = 0, totalFlow = 0;
  bellman();
  while (1) {
    if (!dijkstra())
      break;
    flow_t mn = foo;
    for (int v = t, e = frm[v]; v != s; v = adj[e ^ 1], e = frm[v])
      mn = min(mn, cap[e] - flw[e]);
    for (int v = t, e = frm[v]; v != s; v = adj[e ^ 1], e = frm[v]) {
      flw[e] += mn;
      flw[e ^ 1] -= mn;
    }
    totalFlow += mn;
    totalCost += mn * (dst[t] - pot[s] + pot[t]);
    for (int u = 0; u < n; u++)
      pot[u] += dst[u];
  }
  return totalCost;
}
}
// Matching
int iter = 0;
bool dfs(int u)
{
    was[u] = iter;
    for(auto v: adj[u])
    {
        if(pb[v] == 0)
        {
            pa[u] = v;
            pb[v] = u;
            return true;
        }
    }
    for(auto v: adj[u])
    {
        if(was[pb[v]] != iter && dfs(pb[v]))
        {
            pa[u] = v;
            pb[v] = u;
            return true;
        }
    }
    return false;
}
void MaxMatching()
{
   cin >> n >> n1 >> m;
   for(int i = 1; i <= m; i++)
   {
       int u,v;
       cin >> u >> v;
       adj[u].push_back(v);
   }
   int cnt = 0;
   while(true)
   {
       iter++;
       int add = 0;
       for(int i = 1; i <= n; i++)
       {
           if(pa[i] == 0 && dfs(i))
            add++;
       }
       if(add == 0) break;
       cnt += add;
   }
   cout << cnt << '\n';
   for(int i = 1; i <= n; i++)
    if(pa[i] != 0) cout << i << ' ' << pa[i] << '\n';

}

// Konig - MIS and MVC
void cal(int node, bool col){
    if(col)
    {
        if(visitb[node]) return;
    }
    else
    {
        if(visita[node]) return;
    }
    if(col)
        visitb[node] = 1;
    else
        visita[node] = 1;
    if(col){     // node from the right side, can only traverse matched edge
        cal(pb[node], col ^ 1);
        return;
    }
    for(auto nx:adj[node]){
        //cout << nx << ' ' << pa[node] << '\n';
        if(nx==pa[node])continue;
        cal(nx, col ^ 1);
    }
}
int main(){
   Matching();
   for(int i=1;i<=n;i++){
        if(pa[i])continue;       // matched node from the left side
        cal(i,0);
    }
    vector<int> MaxISa, MaxISb, MVCa, MVCb; // find max cover and minimum cover
    for(int i=1;i<=n1;i++){
        if(visita[i]) MaxISa.pb(i); // Minimum indepedent set is visted on the left
        else MVCa.pb(i); // Max vertex cover is not visited on left
    }
    for(int i=1;i<=n2;i++){
        if(!visitb[i])MaxISb.pb(i); // Minimum indepedent set is not visted on the right
        else MVCb.pb(i); // Max vertex cover is visited on right
    }
}

// AJ - Dijkstra
// Note that dijkstra can solve hard transition DP problems and that Dijikstra = DP
priority_queue<pair<long long, long long>> q;
long long dist[maxn+5];
long long cnt[maxn+5];
long long minn[maxn+5], maxx[maxn+5];
void solve()
{
    cin >> n >> m;
    for(int i = 1; i <= m; i++)
    {
        long long u,v,w;
        cin >> u >> v >> w;
        adj[u].push_back({v,w});
    }
    cnt[1] = 1;
    for(int i = 2; i <= n; i++)
        dist[i] = minn[i] = 1e18;
    q.push({0,1});
    while(!q.empty())
    {
        int u = q.top().second;
        long long cost = -q.top().first;
        q.pop();
        if(cost != dist[u]) continue;
        for(auto x: adj[u])
        {
            int v = x.first;
            long long w = x.second;
            if(dist[v] > dist[u] + w)
            {
                dist[v] = dist[u] + w;
                cnt[v] = cnt[u];
                minn[v] = minn[u] + 1;
                maxx[v] = maxx[u] + 1;
                q.push({-dist[v],v});
            }
            else if(dist[v] == dist[u] + w)
            {
                cnt[v] = (cnt[v] + cnt[u]) % MOD;
                minn[v] = min(minn[v], minn[u] + 1);
                maxx[v] = max(maxx[v], maxx[u] + 1);
            }
        }
    }
}

// AK - BellmanFord
vector<int> adj[maxn+5];
vector<int> adjRev[maxn+5];
vector<pair<pair<int,int>,long long>> edge;
bool visit[maxn+5][3];
void dfs(int u, int typ)
{
    visit[u][typ] = true;
    if(typ == 1)
    {
        for(auto v: adj[u])
            if(!visit[v][typ]) dfs(v,typ);
    }
    else
    {
        for(auto v: adjRev[u])
            if(!visit[v][typ]) dfs(v,typ);
    }

}
void solve()
{
    cin >> n >> m;
    for(int i = 2; i <= n; i++)
        dist[i] = -1e18;
    for(int i = 1; i <= m; i++)
    {
        long long u,v,w;
        cin >> u >> v >> w;
        edge.push_back({{u,v},w});
        adj[u].push_back(v);
        adjRev[v].push_back(u);
    }
    dfs(1,1);
    dfs(n,2);
    for(int i = 1; i < n; i++)
    {
        for(auto e: edge)
        {
            int u = e.first.first;
            int v = e.first.second;
            long long w = e.second;
            if(dist[u] != -1e18)
            {
                dist[v] = max(dist[v],dist[u] + w);
            }
        }
    }
    for(auto e: edge)
    {
        int u = e.first.first;
        int v = e.first.second;
        long long w = e.second;
        if(dist[u] != -1e18)
        {
            if(dist[v] < dist[u] + w && visit[v][1] && visit[v][2])
            {
                cout << -1;
                return;
            }
        }
    }
    cout << dist[n];

}

// Tracing Negative Cycle
void solve()
{
    cin >> n >> m;
    for(int i = 1; i <= m; i++)
    {
        long long u,v,w;
        cin >> u >> v >> w;
        if(u == v && w < 0)
        {
            cout << "YES\n";
            cout << u << ' ' << v;
            return;
        }
        edge.push_back({{u,v},w});
    }
    for(int i = 1; i < n; i++)
    {
        for(auto e: edge)
        {
            int u = e.first.first;
            int v = e.first.second;
            long long w = e.second;
            if(dist[v] > dist[u] + w)
            {
                dist[v] = min(dist[v],dist[u] + w);
                par[v] = u;
            }
        }
    }
    for(auto e: edge)
    {
        int u = e.first.first;
        int v = e.first.second;
        long long w = e.second;
            if(dist[v] > dist[u] + w)
            {
                cout << "YES\n";
                par[v] = u;
                int go = v;
                for(int i = 1; i <= n; i++)
                    go = par[go];
                vector<int> ans;
                ans.push_back(go);
                for(int u = par[go]; u != go; u = par[u])
                    ans.push_back(u);
                ans.push_back(go);
                reverse(ans.begin(),ans.end());
                for(auto x: ans)
                    cout << x << ' ';
                return;
            }
    }
    cout << "NO";

}


//  AQ - Euler Cycle
set<int> adj[maxn+5];
int deg[maxn+5];
vector<int> ans;
    int n,m;
void dfs(int u)
{
    while(adj[u].size())
    {
        m--;
        int v = *adj[u].begin();
        adj[u].erase(v);
        adj[v].erase(u);
        dfs(v);
    }
    ans.push_back(u);
}
void solve()
{


    cin >> n >> m;
    for(int i = 1; i <= m; i++)
    {
        int u,v;
        cin >> u >> v;
        adj[u].insert(v);
        adj[v].insert(u);
        deg[u]++;
        deg[v]++;

    }
     for(int i = 1; i <= n; i++)
    {
       if(deg[i] % 2)
       {
           cout << "IMPOSSIBLE";
           return;
       }
    }
    dfs(1);
    if(m != 0) cout << "IMPOSSIBLE";
    else{
    reverse(ans.begin(),ans.end());
    for(auto u: ans)
        cout << u << ' ';}
}
// AV - LCA stuff
// LCA does also include combining stuff, pretty useful for path query, especially Lubenica
void dfs(int u, int par)
{
    binLift[u][0] = par;
    for(int x = 1; x <= K; x++)
    {
        binLift[u][x] = binLift[binLift[u][x-1]][x-1];
    }

    for(auto v: adj[u])
    {
        if(v != par)
        {
            h[v] = h[u] + 1;
            dfs(v,u);
        }
    }
}
int lca(int u, int v)
{
    if(h[u] < h[v]) swap(u,v);
    for(int p = K; p >= 0; p--)
        if(h[u]-(1<<p) >= h[v]) u = binLift[u][p];
    if(u == v) return u;
    for(int p = K; p >= 0; p--)
        if(binLift[u][p] != binLift[v][p])
    {
        u = binLift[u][p];
        v = binLift[v][p];
    }
    return binLift[u][0];
}
// BA - Heavy Light Deccomposition
long long segtree[4*maxn+5];
void update(int idx, long long val) {
    segtree[idx += maxn] = val;
    for (idx /= 2; idx; idx /= 2)
        segtree[idx] = max(segtree[2 * idx], segtree[2 * idx + 1]);
}

long long query(int lo, int hi) {
    long long ra = 0, rb = 0;
    for (lo += maxn, hi += maxn + 1; lo < hi; lo /= 2, hi /= 2) {
        if (lo & 1)
            ra = max(ra, segtree[lo++]);
        if (hi & 1)
            rb = max(rb, segtree[--hi]);
    }
    return max(ra, rb);
}
void dfs(int u, int par)
{
    sub[u] = 1;
    p[u] = par;
    for(auto v: adj[u])
    {
        if(v != par)
        {
            h[v] = h[u] + 1;
            dfs(v,u);
            sub[u] += sub[v];
        }
    }
}
void dfs_hld(int u, int par, int x)
{
    id[u] = ++timer;
    update(id[u],a[u]);
    head[u] = x;
    int child = 0;
    for(auto v: adj[u])
    {
        if(v != par)
        {
            if(sub[child] < sub[v]) child = v;
        }
    }
    if(child == 0) return;
    dfs_hld(child,u,x);
    for(auto v: adj[u])
    {
        if(v != par && v != child)
        {
            dfs_hld(v,u,v);
        }
    }

}
long long path(int x, int y)
{
    long long ans = 0;
    while(head[x] != head[y])
    {
        if(h[head[x]] < h[head[y]]) swap(x,y);
        ans = max(ans,query(id[head[x]],id[x]));
       // cout << x << ' ' << ans << '\n';
        x = p[head[x]];
    }
    if(h[x] > h[y]) swap(x,y);
    ans = max(ans,query(id[x],id[y]));
    return ans;
}
// BB - Centroid Decomposition for counting
void update(int idx,long long v)
{
    for(int i = idx; i <= maxn; i += i & (-i))
        fen[i] += v;
}
long long query(int idx)
{
    if(idx<0) return 0;
    long long ans = 0;
    for(int i = idx; i > 0; i -= i & (-i))
        ans += fen[i];
    return ans+1;
}
void dfs(int u,int par)
{
    subtree[u] = 1;
    for(auto v: adj[u])
    {
        if(v != par  && !process[v])
        {
            dfs(v,u);
            subtree[u] += subtree[v];
        }
    }
}
void get_ans(int u, int par, int depth)
{
    if(depth>k2) return;
    res += query(k2-depth) - query(k1-1-depth);
   // cout << u << ' ' << depth << ' ' << res << '\n';
    for(auto v: adj[u])
    {
        if(v!=par && !process[v])
            get_ans(v,u,depth+1);
    }

}
void amount(int u, int par, int depth,int s)
{
    update(depth,s);
    for(auto v: adj[u])
    {
        if(v!=par && !process[v])
            amount(v,u,depth+1,s);
    }
}
long long find_Centroid(int u,int par, int req)
{
    for(auto v: adj[u])
        if(v!=par && !process[v])
            if(subtree[v] >= req) return find_Centroid(v,u,req);
    return u;
}
void centroid_decomp(int node = 1)
{
    dfs(node,node);
    int centroid = find_Centroid(node,node,subtree[node]/2);
    process[centroid] = true;
   // cout << centroid << '\n';
    for(auto v: adj[centroid])
    {
        if(!process[v])
        {
           // cout << v << "hi\n";
            get_ans(v,centroid,1);
            amount(v,centroid,1,1);
        }
    }
    for(auto v: adj[centroid])
    {
        if(!process[v])
        {
            amount(v,centroid,1,-1);
        }
    }
    for (int v : adj[centroid]) if (!process[v]) centroid_decomp(v);
}
// BC - Centroid Decomposition for Query
void dfs(int u,int par)
{
    subtree[u] = 1;
    for(auto v: adj[u])
    {
        if(v != par  && !process[v])
        {
            dfs(v,u);
            subtree[u] += subtree[v];
        }
    }
}
long long find_Centroid(int u,int par, int req)
{
    for(auto v: adj[u])
        if(v!=par && !process[v])
            if(subtree[v] >= req) return find_Centroid(v,u,req);
    return u;
}
int p[maxn+5];
void centroid_decomp(int node, int par)
{
    dfs(node,node);
    int centroid = find_Centroid(node,node,subtree[node]/2);
    process[centroid] = true;
    p[centroid] = par;
   // cout << centroid << '\n';
    for (int v : adj[centroid]) if (!process[v]) centroid_decomp(v,centroid);
}
int h[maxn+5];
const int K = 19;
int binLift[maxn+5][K+2];
void dfs_pre(int u, int par)
{
    binLift[u][0] = par;
    for(int x = 1; x <= K; x++)
    {
        binLift[u][x] = binLift[binLift[u][x-1]][x-1];
    }

    for(auto v: adj[u])
    {
        if(v != par)
        {
            h[v] = h[u] + 1;
            dfs_pre(v,u);
        }
    }
}
int lca(int u, int v)
{
    if(h[u] < h[v]) swap(u,v);
    for(int p = K; p >= 0; p--)
        if(h[u]-(1<<p) >= h[v]) u = binLift[u][p];
    if(u == v) return u;
    for(int p = K; p >= 0; p--)
        if(binLift[u][p] != binLift[v][p])
    {
        u = binLift[u][p];
        v = binLift[v][p];
    }
    return binLift[u][0];
}
long long dist(int u, int v)
{
    return h[u] + h[v] - 2 * h[lca(u,v)];
}
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    cin >> n >> q;
    for(int i = 1; i < n;i++)
    {
        int u,v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    centroid_decomp(1,0);
    dfs_pre(1,1);
    for(int i = 1; i <= n; i++)
        ans[i] = 1e18;
    for(int x = 1; x != 0; x = p[x])
    {
        ans[x] = min(ans[x], dist(1,x));
    }
    for(int i = 1; i <= q; i++)
    {
        int op;
        cin >> op;
        if(op == 1)
        {
            int a;
            cin >> a;
            for(int x = a; x != 0; x = p[x])
            {
                ans[x] = min(ans[x], dist(a,x));
            }
        }
        else
        {
            int a;
            cin >> a;
            long long total = 1e18;
            for(int x = a; x != 0; x = p[x])
            {
                total = min(total, ans[x] + dist(a,x));

            }
            cout << total << '\n';
        }
    }
    return 0;
}

// BD - DP Reroot
// Calculate dp such that dp[u] is answer for all subtree of u and then dp2 is thinking about how to make u the root of that tree
void dfs(int u, int par)
{
    for(auto v: adj[u])
    {
        if(v != par)
        {
            dfs(v,u);
            if(dp[v] + 1 >= dp[u])
            {
                dp2[u] = dp[u];
                dp[u] = dp[v] + 1;
                choose[u] = v;
            }
            else dp2[u] = max(dp2[u],dp[v] + 1);
        }
    }
}
void dfs2(int u, int par)
{
    for(auto v: adj[u])
    {
        if(v != par)
        {
            ans[v] = ans[u] + 1;
            if(v != choose[u]) ans[v] = max(ans[v], dp[u] + 1);
            else ans[v] = max(ans[v], dp2[u] + 1);
            dfs2(v,u);
        }
    }
}
void solve()
{
    cin >> n;
    for(int i = 1; i < n; i++)
    {
        int u,v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    dfs(1,1);
    ans[1] = -1;
    dfs2(1,1);
    for(int i = 1; i <= n; i++)
        cout << max(dp[i],ans[i]) << ' ';



}

// BE - DP Knapsack on Tree
// Note if it is siz[u] + siz[v] then it is n^3 -> n ^ 2, if it is max(siz[u],siz[v]) it is n ^ 2 -> n
long long tmp[5005][2];
void combine(int u,int v)
{
    for(int i = 0; i <= sub[u]+sub[v]; i++) tmp[i][0] = tmp[i][1] = INF;
    for(int i = 0; i  <= sub[u]; i++)
        for(int j = 0; j <= sub[v]; j++)
    {
        tmp[i+j][0] = min(tmp[i+j][0], dp[u][i][0] + dp[v][j][0]);
        tmp[i+j][1] = min(tmp[i+j][1], dp[u][i][1] + dp[v][j][1]);
    }
    for(int i = 0; i <= sub[u]+sub[v]; i++) dp[u][i][0] = tmp[i][0], dp[u][i][1] = tmp[i][1];
}
void dfs(int u, int par)
{
    sub[u] = 1;
    dp[u][0][0] = 0;
    dp[u][1][0] = normal[u];
    dp[u][1][1] = discount[u];
    for(auto v: adj[u])
    {
        if(v!=par)
        {
            dfs(v,u);
            combine(u,v);
            sub[u] += sub[v];
        }
    }
    for(int i = 0; i <= sub[u]; i++)
    {
        //cout << u << ' ' <<  i << ' '  << dp[u][i][0] << ' ' << dp[u][i][1] << '\n';
        dp[u][i][1] = min(dp[u][i][1],dp[u][i][0]);
    }

}
// BH - Tree isomorphism
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
long long RNG(long long l, long long r)
{
    return uniform_int_distribution<long long>(l,r)(rng);
}
const int MOD = 998244353;
long long check[maxn+5];
struct tree
{
    vector<int> adj[maxn+5];
    int sub[maxn+5], h[maxn+5];
    long long hashs[maxn+5];
    void dfs(int u, int par)
    {
        sub[u] = 1;
        hashs[u] = 1;
        for(auto v: adj[u])
        {
            if(v != par)
            {
                h[u] = max(h[v],h[u]);
                dfs(v,u);

            }
        }
        h[u]++;
        for(auto v: adj[u])
        {
            if(v != par)
            {
                hashs[u] = (check[h[u]] + hashs[v]) * hashs[u] % MOD;

            }
        }
    }


};
tree a, b;
void solve()
{

    int n;
    cin >> n;
    for(int i = 0; i <= n; i++)
        check[i] = RNG(2, MOD-1);
    a.h[1] = 0;
    b.h[1] = 0;
    for(int i = 1; i < n; i++)
    {
        int u,v;
        cin >> u >> v;
        a.adj[u].push_back(v);
        a.adj[v].push_back(u);
    }
     for(int i = 1; i < n; i++)
    {
        int u,v;
        cin >> u >> v;
        b.adj[u].push_back(v);
        b.adj[v].push_back(u);
    }
    a.dfs(1,1);
    b.dfs(1,1);
    if(a.hashs[1] == b.hashs[1]) cout << "YES\n";
    else cout << "NO\n";
    for(int i = 1; i <= n; i++)
        a.adj[i].clear(), b.adj[i].clear();
}

// Other Isomorphic Algo
long long RNG(long long l, long long r)
{
    return uniform_int_distribution<long long>(l,r)(rng);
}
int n;
map<pair<long long, long long>,long long> track;
struct Tree
{
    vector<int> adj[maxn+5];
    long long ha[maxn+5];
    long long suba[maxn+5];
    long long hasha[maxn+5];
    void init(int n)
    {
        for(int i = 1; i <= n; i++)
        {
            ha[i] = 0;
            suba[i] = 0;
            hasha[i] = 0;
            adj[i].clear();
        }
    }
    vector<int> dfs(int u, int par, int target)
    {
        hasha[u] = 0;
        suba[u] = 1;
        vector<int> now;
        bool isCent = true;
        for(auto v: adj[u])
        {
            if(v != par)
            {
                ha[v] = ha[u] + 1;
                vector<int> tmp = dfs(v,u,target);
                for(auto x: tmp)
                    now.push_back(x);
                if(suba[v] > target)
                {
                    isCent = false;
                }

                suba[u] += suba[v];
                hasha[u] ^= hasha[v];
            }
        }
        if(n - suba[u] > target) isCent = false;
        if(isCent) now.push_back(u);
        if(!track.count({ha[u],suba[u]})) track[{ha[u],suba[u]}] = RNG(1,2e18);
        hasha[u] ^= track[{ha[u],suba[u]}];
        return now;
        //cout << "check " << u << ' ' << track[{ha[u],suba[u]}] << '\n';
    }
};
Tree a;
Tree b;
bool isGood(int t1,int t2)
{
    a.ha[t1] = 0;
    b.ha[t2] = 0;
    a.dfs(t1,t1,n/2);
    b.dfs(t2,t2,n/2);
    sort(a.hasha + 1, a.hasha + 1 + n);
    sort(b.hasha + 1, b.hasha + 1 + n);
    bool flag = true;
    for(int i = 1; i <= n; i++)
    {
        flag &= (a.hasha[i] == b.hasha[i]);
      //  cout << i << ' ' << a.ha[i] << ' ' << b.ha[i] << '\n';
    }
    return flag;
}
void solve()
{

    cin >> n;
    a.init(n);
    b.init(n);
    for(int i = 1; i < n; i++)
    {
        int u,v;
        cin >> u >> v;
        a.adj[u].push_back(v);
        a.adj[v].push_back(u);
    }
    for(int i = 1; i < n; i++)
    {
        int u,v;
        cin >> u >> v;
        b.adj[u].push_back(v);
        b.adj[v].push_back(u);
    }
    vector<int> c1 = a.dfs(1,1,n/2);
    vector<int> c2 = b.dfs(1,1,n/2);
   // cout << c1.size() << ' ' << c2.size() << '\n';
    assert(c1.size() <= 2 && c2.size() <= 2);
    bool flag = false;
    int x = c1[0];
    for(auto c: c2)
    {
        flag |= isGood(x,c);
    }

    if(flag) cout << "YES\n";
    else cout << "NO\n";
    track.clear();
}
// Combinatorics
// Stars and Bars Euler Formula = C(n+m-1,m)
// Catalan = (1 / (n+1)) * C(2n,n)
// Derangement = D(n) = (n−1)(D(n−1)+D(n−2))
// Burnside Lemma =  sum of 1/n * m ^ (gcd(i,n)) for all i from 1 to n

// Little Fermat x ^ (p-1) = 1 (mod p) with p is prime
// Phi euler is n * (1 - 1 / prime1) * (1 - 1 / prime2) * .. * (1 - primen) for every prime of n
int phi(int n) {
    int result = n;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0)
                n /= i;
            result -= result / i;
        }
    }
    if (n > 1)
        result -= result / n;
    return result;
}
void phi_1_to_n(int n) {
    vector<int> phi(n + 1);
    for (int i = 0; i <= n; i++)
        phi[i] = i;

    for (int i = 2; i <= n; i++) {
        if (phi[i] == i) {
            for (int j = i; j <= n; j += i)
                phi[j] -= phi[j] / i;
        }
    }
}
//
/**
On unrooted Tree
# on n vertices: nn−2
# on k existing trees of size ni: n1n2 · · · nknk−2
# with degrees di: (n − 2)!/((d1 − 1)! · · · (dn − 1)!)
**/

// BM - BN Inclusion Exclusion (PIE)

// Basically the gist is what if we have a weak condition (at least) so that  so that we can do bitmask or something to remove them
// Kinversion
long long binPow(long long n, long long k)
{
    if(k == 0) return 1;
    long long v = binPow(n,k/2);
    if(k % 2) return v * v % MOD * n % MOD;
    else return v * v % MOD;
}
const int maxn = 2e5;
long long fact[maxn+5];
long long inv[maxn+5];
long long C(long long n, long long k)
{
    return fact[n] * inv[k] % MOD * inv[n-k] % MOD;
}
int dp[maxn+5][455];
long long n,k;
long long euler(long long n, long long k)
{
    return C(n+k-1,n-1);
}
void solve()
{
    //cin >> a >> b;

    cin >> n >> k;
    fact[0] = 1;
    inv[0] = 1;
    for(int i = 1; i <= n+k; i++)
        fact[i] = fact[i-1] * i % MOD;
    for(int i = 1; i <= n+k; i++)
        inv[i] = binPow(fact[i],MOD-2);
    dp[0][0] = 1;
    // Counting the number of increasing Partition of n
    for(int x = 1; x <= k; x++)
    {
        for(int i = 1; i <= 450; i++)
        {
             if(x >= i)
             dp[x][i] += (dp[x-i][i] + dp[x-i][i-1]) % MOD;
             if(x >= n+1)
             {
                 dp[x][i] = (dp[x][i] - dp[x-(n+1)][i-1]) % MOD;
             }
             dp[x][i] %= MOD;
             if(dp[x][i] < 0) dp[x][i] += MOD;
        }
    }

    long long ans = euler(n,k);
    for(int i = 1; i <= 450; i++)
    {
        for(long long p = 1; p <= k; p++)
        {
            if(i % 2) ans -= dp[p][i] * euler(n,k-p) % MOD;
            else ans += dp[p][i] * euler(n,k-p) % MOD;
          //  cout << dp[p][i] * euler(n,k-p) << '\n';
            ans %= MOD;
            if(ans < 0) ans += MOD;
        }
    }
    cout << ans;


}

// Gauss Elimination
// Gauss with Mod, Solving Linear Combinations such that ax1 + bx2 modulo a (mod p)
void gauss()
{
    int m = n;
    vector<int> where (m, -1);
    for (int col=0, row=0; col<m && row<n; ++col)
    {
        int current = row;
        for(int k = row; k < n; k++)
            if(a[k][col] > a[current][col]) current = k;
        if(a[current][col] == 0) continue;
        for(int k = col; k <= m; k++)
            swap(a[current][k],a[row][k]);
        where[col] = row;
        for (int i=0; i<n; ++i)
        {
            if (i != row)
            {
                int c = a[row][col];
                int c2 = a[i][col];

                for (int j=col; j<=m; ++j)
                    a[i][j] = (((long long) a[i][j] * c - (long long) a[row][j] * c2) % t + t) % t;
            }
        }
        row++;
    }
    long long sum = 1;
    for(auto k: where)
        if(k==-1) sum = (sum * t) % MOD;
    cout << "Case " << ++cnt << ": " << sum << '\n';

}

void solve()
{
    cin >> n >> m >> t;
    for(int i = 0; i <= n; i++)
        for(int j = 0; j <= n; j++)
            a[i][j] = 0;
    for(int i = 0; i < n; i++)
        a[i][i] = t-1;
    for(int i = 1; i <= m; i++)
    {
        int u,v;
        cin >> u >> v;
        u--;
        v--;
        a[u][v] = 1;
        a[v][u] = 1;
    }
    gauss();

}
// BO - Normal Gauss
const double EPS = 1e-9;
const int INF = 2; // it doesn't actually have to be infinity or a big number

int gauss (vector < vector<double> > a, vector<double> & ans) {
    int n = (int) a.size();
    int m = (int) a[0].size() - 1;

    vector<int> where (m, -1);
    for (int col=0, row=0; col<m && row<n; ++col) {
        int sel = row;
        for (int i=row; i<n; ++i)
            if (abs (a[i][col]) > abs (a[sel][col]))
                sel = i;
        if (abs (a[sel][col]) < EPS)
            continue;
        for (int i=col; i<=m; ++i)
            swap (a[sel][i], a[row][i]);
        where[col] = row;

        for (int i=0; i<n; ++i)
            if (i != row) {
                double c = a[i][col] / a[row][col];
                for (int j=col; j<=m; ++j)
                    a[i][j] -= a[row][j] * c;
            }
        ++row;
    }

    ans.assign (m, 0);
    for (int i=0; i<m; ++i)
        if (where[i] != -1)
            ans[i] = a[where[i]][m] / a[where[i]][i];
    for (int i=0; i<n; ++i) {
        double sum = 0;
        for (int j=0; j<m; ++j)
            sum += ans[j] * a[i][j];
        if (abs (sum - a[i][m]) > EPS)
            return 0;
    }

    for (int i=0; i<m; ++i)
        if (where[i] == -1)
            return INF;
    return 1;
}

// Lagrange Interpolation
// Given (k+1) answers of an unknown polynomial equation of kth degree (p^k max) find its answer for arbitary x
long long n,k,a[maxn],fac[maxn],ifac[maxn],prf[maxn],suf[maxn];
void build()
{
    fac[0] = ifac[0] = 1;
    for (int i = 1; i < maxn; i++)
    {
        fac[i] = fac[i - 1] * i % MOD;
        ifac[i] = binPow(fac[i], MOD - 2);
    }
}
    //Calculate P(x) of degree k - 1, k values form 1 to k
    //P(i) = a[i]
long long calc(long long x, long long k)
{
        if(x <= k)
        {
            return a[x];
        }
        prf[0] = suf[k + 1] = 1;
        for (long long i = 1; i <= k; i++) {
            prf[i] = prf[i - 1] * (x - i + MOD) % MOD;
        }
        for (long long i = k; i >= 1; i--) {
            suf[i] = suf[i + 1] * (x - i + MOD) % MOD;
        }
        long long res = 0;
        for (long long i = 1; i <= k; i++) {
            if (!((k - i) & 1)) {
                res = (res +  prf[i - 1] * suf[i + 1] % MOD
                        * ifac[i - 1] % MOD * ifac[k - i] % MOD * a[i]) % MOD;
            }
            else {
                res = (res -  prf[i - 1] * suf[i + 1] % MOD
                        * ifac[i - 1] % MOD * ifac[k - i] % MOD * a[i] % MOD + MOD) % MOD;
            }
        }
        return res;
}
void solve()
{
    cin >> n >> k;
    build();
    for(int i = 1; i <= k+2; i++)
        a[i] = (a[i-1]+binPow(i,k))%MOD;
    cout << calc(n,k+2);
}


// BP - Xor Basis
long long basis[maxn+5];
const int k = 62;
void add(long long x)
{
    for(int p = k; p >= 0; p--)
    {
        if(basis[p] != 0 && x & (1ll<<p))
        {
            x ^= basis[p];
            continue;
        }
        if(x & (1ll<<p))
        {
            basis[p] = x;
            return;
        }
    }
}
long long get()
{
    long long ans = 0;
     for(int p = k; p >= 0; p--)
    {
        if((ans & (1ll<<p))) continue;
        ans ^= basis[p];
    }
    return ans;
}

// Minimum Xor Path (Xor basis + Cycle Basis)
void dfs(int u)
{
    visit[u] = 1;
    for(auto x: adj[u])
    {
        int v = x.first;
        long long c = x.second;
        if(!visit[v])
        {
            cycle[v] = cycle[u] ^ c;
            dfs(v);
        }
        else
        {
            add(cycle[u] ^ cycle[v] ^ c);
        }
    }
}
long long query(long long x)
{
    for(int p = k; p >= 0; p--)
    {
        if(x & (1ll<<p))
        {
            if(basis[p]) x ^= basis[p];
        }
    }
    return x;
}
void solve()
{
    cin >> n >> m;
    for(int i = 1; i <= m; i++)
    {
        long long u,v,c;
        cin >> u >> v >> c;
        adj[u].push_back({v,c});
        adj[v].push_back({u,c});
    }
    dfs(1);
    cout << query(cycle[n]);

}

// Grundy Game
// Basically Mex of all position, winning position is not 0
// xor all games
// Also usually grundy games if pass a threshold is always winning
//
const int maxn = 2520;
set<int> dp[maxn+5];
int ans[maxn+5];
void precalc()
{
    for(int i = 3; i <= maxn; i++)
    {
        for(int x = 1; 2*x < i; x++)
            dp[i].insert(ans[i-x] ^ ans[x]);
        for(int x = 0; x <= maxn+1; x++)
            if(dp[i].count(x) == 0)
        {
            ans[i] = x;
           // if(i <= 10) cout << i << ' ' << ans[i] << '\n';
            break;
        }
    }
}
void solve()
{
    int n;
    cin >> n;
    if(n <= 2520)
    {if(ans[n] == 0) cout << "second\n";
    else cout << "first\n";}
    else cout << "first\n";

}

// Chinese Remainder Theorem
//Given m1, m2, r1, r2, find x :
//- x mod m1 = r1
//- x mod m2 = r2
pair<int,int> ExEuclid(int a, int b){
    int x0 = 1, y0 = 0;
    int x1 = 0, y1 = 1;
    int x2, y2;
    while (b){
        int q = a / b;
        int r = a % b;
        a = b; b = r;
        x2 = x0 - q * x1;
        y2 = y0 - q * y1;
        x0 = x1; y0 = y1;
        x1 = x2; y1 = y2;
    }
    return {x0, y0};
}
int cal_crt(int r1, int r2, int m1, int m2)){
    ii ans = ExEuclid(m1, m2);
    bool f = false;
    int g = __gcd(m1, m2);
    if ((r2 - r1) % g) return 1e9; //No solution

    int k = ans.x * ((r2 - r1) / g);
    k %= (m2 / g);
    return (r1 + k * m1) % lc;
    //all_ans = {ans + k*LCM(m1, m2)}
}
// Rho factorization
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
vector <int> b;
ll binpower(ll base, ll e, ll mod) {
    ll result = 1;
    base %= mod;
    while (e){
        if (e & 1)result = (ll)result * base % mod;
        base = (ll)base * base % mod;e >>= 1;
    }
    return result;
}
bool check_composite(ll n, ll a, ll d, int s){
    ll x = binpower(a, d, n);
    if (x == 1 or x == n - 1)return false;
    for (int r = 1; r < s; r++) {
        x = (ll)x * x % n;
        if (x == n - 1)return false;
    }return true;
}

bool MillerRabin_checkprime(ll n) {
    if (n < 2)return false;
    int r = 0;
    ll d = n - 1;
    while ((d & 1) == 0) {
        d >>= 1;
        r++;
    }
    for (int a : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
        if (n == a)return true;
        if (check_composite(n, a, d, r))return false;
    }
    return true;
}

int mult(int a, int b, int mod) {
    return (ll)a * b % mod;
}

int F(int x, int c, int mod) {
    return (mult(x, x, mod) + c) % mod;
}
int rho(int n, int x0=2, int c=1) {
    int x = x0;
    int y = x0;
    int g = 1;
    if (n % 2 == 0) return 2;
    while (g == 1) {
        x = F(x, c, n);
        y = F(y, c, n);
        y = F(y, c, n);
        g = gcd(abs(x - y), n);
    }
    return g;
}

void Rho_factorization(int n){
    set <int> s;
    if (n == 1) {return;}
    while (!MillerRabin_checkprime(n)){
        ll k;
        while (1){
            int p = (rng()%(n-2))+2, q = (rng()%(n-1))+1;
            k = rho(n,p,q);
            if (MillerRabin_checkprime(k)) break;
        }
        s.insert(k);
        n/=k;
    }
    s.insert(n);
    fora(i,s) b.pb(i);
}

// FFT with Arbitary Mod
using namespace std;

using ll = long long;
using db = long double;  // or double, if TL is tight
using str = string;	  // yay python!

using vl = vector<ll>;
using vi = vector<int>;

#define tcT template <class T
#define tcTU tcT, class U
tcT > using V = vector<T>;
tcT, size_t SZ > using AR = array<T, SZ>;
tcT > using PR = pair<T, T>;

// pairs
#define mp make_pair
#define f first
#define s second

#define sz(x) int((x).size())

tcT > void re(T &x) { cin >> x; }
tcTUU > void re(T &t, U &...u) {
    re(t);
    re(u...);
}
tcT > void re(V<T> &x) { each(a, x) re(a); }


#define rep(i, a, b) for (int i = a; i < (b); ++i)
typedef pair<int, int> pii;

/**
 * Author: Ludo Pulles, chilli, Simon Lindholm
 * Date: 2019-01-09
 * License: CC0
 * Source: http://neerc.ifmo.ru/trains/toulouse/2017/fft2.pdf (do read, it's
 excellent) Accuracy bound from http://www.daemonology.net/papers/fft.pdf
 * Description: fft(a) computes $\hat f(k) = \sum_x a[x] \exp(2\pi i \cdot k x /
 N)$ for all k$. N must be a power of 2. Useful for convolution:
   \texttt{conv(a, b) = c}, where c[x] = \sum a[i]b[x-i]$.
   For convolution of complex numbers or more than two vectors: FFT, multiply
   pointwise, divide by n, reverse(start+1, end), FFT back.
   Rounding is safe if $(\sum a_i^2 + \sum b_i^2)\log_2{N} < 9\cdot10^{14}$
   (in practice ^{16}$; higher for random inputs).
   Otherwise, use NTT/FFTMod.
 * Time: O(N \log N) with N = |A|+|B|$ ($\tilde 1s$ for N=2^{22}$)
 * Status: somewhat tested
 * Details: An in-depth examination of precision for both FFT and FFTMod can be
 found
 * here
 (https://github.com/simonlindholm/fft-precision/blob/master/fft-precision.md)
 */

typedef complex<double> C;
void fft(vector<C> &a) {
    int n = sz(a), L = 31 - __builtin_clz(n);
    static vector<complex<long double>> R(2, 1);
    static vector<C> rt(2, 1);  // (^ 10% faster if double)
    for (static int k = 2; k < n; k *= 2) {
        R.resize(n);
        rt.resize(n);
        auto x = polar(1.0L, acos(-1.0L) / k);
        rep(i, k, 2 * k) rt[i] = R[i] = i & 1 ? R[i / 2] * x : R[i / 2];
    }
    vi rev(n);
    rep(i, 0, n) rev[i] = (rev[i / 2] | (i & 1) << L) / 2;
    rep(i, 0, n) if (i < rev[i]) swap(a[i], a[rev[i]]);
    for (int k = 1; k < n; k *= 2)
        for (int i = 0; i < n; i += 2 * k) rep(j, 0, k) {
                // C z = rt[j+k] * a[i+j+k]; // (25% faster if hand-rolled)  ///
                // include-line
                auto x = (double *)&rt[j + k],
                     y = (double *)&a[i + j + k];  /// exclude-line
                C z(x[0] * y[0] - x[1] * y[1],
                    x[0] * y[1] + x[1] * y[0]);  /// exclude-line
                a[i + j + k] = a[i + j] - z;
                a[i + j] += z;
            }
}

typedef vector<ll> vl;
template <int M> vl convMod(const vl &a, const vl &b) {
    if (a.empty() || b.empty()) return {};
    vl res(sz(a) + sz(b) - 1);
    int B = 32 - __builtin_clz(sz(res)), n = 1 << B, cut = int(sqrt(M));
    vector<C> L(n), R(n), outs(n), outl(n);
    rep(i, 0, sz(a)) L[i] = C((int)a[i] / cut, (int)a[i] % cut);
    rep(i, 0, sz(b)) R[i] = C((int)b[i] / cut, (int)b[i] % cut);
    fft(L), fft(R);
    rep(i, 0, n) {
        int j = -i & (n - 1);
        outl[j] = (L[i] + conj(L[j])) * R[i] / (2.0 * n);
        outs[j] = (L[i] - conj(L[j])) * R[i] / (2.0 * n) / 1i;
    }
    fft(outl), fft(outs);
    rep(i, 0, sz(res)) {
        ll av = ll(real(outl[i]) + .5), cv = ll(imag(outs[i]) + .5);
        ll bv = ll(imag(outl[i]) + .5) + ll(real(outs[i]) + .5);
        res[i] = ((av % M * cut + bv) % M * cut + cv) % M;
    }
    return res;
}

int main() {
    vector<long long> a;
    for(int i = 0; i <= k; i++)
        a.push_back(0);
    for(int p = 1; p <= k; p++)
        a[p] = 1;
    a[0] = 1;
    auto ret = convMod<MOD>(a,a);
}

// Geometry Stuff that I coded
// Orientation
long long cross(long long x1,  long long y1, long long x2, long long y2)
{
    return x1 * y2 - x2 * y1;
}
void solve()
{
    long long x1,y1,x2,y2,x3,y3;
    cin >> x1 >> y1 >> x2 >> y2 >> x3 >> y3;
    long long prod = cross(x2 - x1,y2 - y1,x3 - x2,y3 - y2);
    if(prod == 0) cout << "TOUCH\n";
    else if(prod < 0) cout << "RIGHT\n";
    else cout << "LEFT\n";
}

// Line Segment Intersection
struct pt
{
    long long x,y;
};
long long S(pt x, pt y, pt z)
{
    long long x1 = y.x - x.x;
    long long x2 = z.x - y.x;
    long long x3 = x.x - z.x;
    long long y1 = y.y + x.y;
    long long y2 = z.y + y.y;
    long long y3 = x.y + z.y;
    long long area = x1 * y1 + x2 * y2 + x3 * y3;
    if(area > 0) return 1;
    else if(area == 0) return 0;
    else  return -1;
}
long long intersect(long long a, long long b, long long c, long long d)
{
    if(a < b) swap(a,b);
    if(c < d) swap(c,d);
    return min(a,c) < max(b,d);
}

// Polygon Area
pair<long long,long long> p[maxn+5];
int n;
// 2 * area
long long area()
{
    long long total = 0;
    for(int i = 0; i < n; i++)
    {
        total += (p[i].first - p[(i+1)%n].first) * (p[i].second + p[(i+1)%n].second);
    }
    return llabs(total);
}
void solve()
{

    cin >> n;
    for(int i = 0; i < n; i++)
        cin >> p[i].first >> p[i].second;
    cout << area();
}
// Point in General Polygon in O(n)
const int maxN = 1001;

struct Point {
    int x, y;
    void read(){ scanf("%d %d", &x, &y); }
    Point operator +(const Point& b) const { return Point{x+b.x, y+b.y}; }
    Point operator -(const Point& b) const { return Point{x-b.x, y-b.y}; }
    ll operator *(const Point& b) const { return (ll) x * b.y - (ll) y * b.x; }
    void operator +=(const Point& b) { x += b.x; y += b.y; }
    void operator -=(const Point& b) { x -= b.x; y -= b.y; }
    void operator *=(const int k) { x *= k; y *= k; }

    ll cross(const Point& b, const Point& c) const {
        return (*this-b) * (c-*this);
    }
};

int N, M;
Point P[maxN];

bool pointlineintersect(Point P1, Point P2, Point P3){
    if(P1.cross(P2, P3) != 0)   return false;
    return (min(P2.x, P3.x) <= P1.x && P1.x <= max(P2.x, P3.x))
        && (min(P2.y, P3.y) <= P1.y && P1.y <= max(P2.y, P3.y));
}

void pointinpolygon(){
    int cnt = 0;
    bool bound = false;
    for(int i = 1; i <= N; i++)
    {
        int j = i+1;
        if(i==N) j = 1;
        if(pointlineintersect(P[0],P[i],P[j]))
        {
            bound = true;
            break;
        }
        Point p1 = P[i];
        Point p2 = P[j];
        if(p1.x < p2.x) swap(p1,p2);
        if(p2.x <= P[0].x && P[0].x < p1.x && P[0].cross(p1,p2) < 0) cnt++;
    }
    if(bound) cout << "BOUNDARY\n";
    else if(cnt%2) cout << "INSIDE\n";
    else cout << "OUTSIDE\n";
}
// Pick Theorem
struct pt
{
    long long x,y;
};
pt poly[maxn+5];
long long area()
{
    long long total = 0;
    for(int i = 0; i < n; i++)
    {
        long long dx = poly[(i+1) % n].x - poly[i].x;
        long long dy = poly[(i+1) % n].y + poly[i].y;
        total += dx * dy;
    }
    return llabs(total);
}
long long boundary()
{
 long long ans = 0;
 for(int i = 0; i < n; i++)
    {
        long long dx = poly[(i+1) % n].x - poly[i].x;
        long long dy = poly[(i+1) % n].y - poly[i].y;
        ans += __gcd(abs(dx),abs(dy));
    }
    return ans;
}
void solve()
{
    cin >> n;
    for(int i = 0; i < n; i++)
    {
        cin >> poly[i].x >> poly[i].y;
    }
    // S = I + B/2 - 1
    // 2S + 2 = 2I + B

    // Area = number inside + (number on boundary) / 2 - 1
    long long bound = boundary();
    long long inside = (area() + 2 - bound) / 2;
    cout << inside << ' ' << bound;

}

// CA - Closest Pairs Euclid
struct pt
{
    long long x,y;
};
pt a[maxn+5];
bool cmp(pt a, pt b)
{
    if(a.x == b.x) return a.y < b.y;
    return a.x < b.x;
}
set<pair<long long, long long>> track;
long long calc(pair<long long, long long> a, pair<long long, long long> b)
{
    long long dx = a.first - b.first;
    long long dy = a.second - b.second;
    return dx * dx + dy * dy;
}
void solve()
{
    cin >> n;
    for(int i = 1; i <= n; i++)
    {
        cin >> a[i].x >> a[i].y;
    }
    sort(a + 1, a + 1 + n, cmp);
    int j = 1;
    long long ans = 8e18;
    for(int i = 1; i <= n; i++)
    {
        long long dist = ceil(sqrt(ans));
        while(j < i && a[i].x - a[j].x > dist) track.erase({a[j].y,a[j].x}), j++;

        auto itL = track.lower_bound({a[i].y - dist, 0ll});
        auto itR = track.upper_bound({a[i].y + dist, 0ll});
        pair<long long, long long> cur = {a[i].y,a[i].x};
        for(auto x = itL; x != itR; x++)
        {
            auto p = *x;

            ans = min(ans,calc(p,cur));
        }
        track.insert(cur);

    }
    cout << ans;


}

// Convex Hull
pair<long long, long long> pt[maxn+5];
vector<pair<long long, long long>> lower,upper,hull;
long long cross(pair<long long, long long> x, pair<long long, long long> y)
{
    return x.first * y.second - x.second * y.first;
}
pair<long long, long long> sub(pair<long long, long long> x, pair<long long, long long> y)
{
    return {x.first - y.first, x.second - y.second};
}
int n;
void solve()
{
    cin >> n;
    for(int i = 1; i <= n; i++)
        cin >> pt[i].first >> pt[i].second;
    sort(pt + 1, pt + 1 + n);
    lower.push_back(pt[1]);
    lower.push_back(pt[2]);
    for(int i = 3; i <= n; i++)
    {
        while(lower.size() >= 2 && cross(sub(pt[i],lower[lower.size()-1]),sub(lower[lower.size()-1],lower[lower.size()-2])) < 0) lower.pop_back();
        lower.push_back(pt[i]);
    }
    upper.push_back(pt[n]);
    upper.push_back(pt[n-1]);
     for(int i = n-2; i >= 1; i--)
    {
        while(upper.size() >= 2 && cross(sub(pt[i],upper[upper.size()-1]),sub(upper[upper.size()-1],upper[upper.size()-2])) < 0) upper.pop_back();
        upper.push_back(pt[i]);
    }
    for(auto x: lower)
        hull.push_back(x);
    hull.pop_back();
    for(auto x: upper)
        hull.push_back(x);
    hull.pop_back();
    cout << hull.size() << '\n';
    for(auto x: hull)
        cout << x.first << ' ' << x.second << '\n';

}
// CC - CD Sweepline
// Area of Rectangle
const int maxn = 2e6 + 5;
const int offset = 1e6 + 1;
vector<pair<int,pair<int,int>>> event[maxn+5];
int segCnt[4*maxn+5];
int seg[4 * maxn+5];
void update(int node, int l, int r, int u, int v, int x)
{
    if(l > v || r < u) return;
    if(u <= l && r <= v)
    {
        segCnt[node] += x;
        if(segCnt[node] == 0)
        {
            if(l < r) seg[node] = seg[node * 2] + seg[node * 2 + 1];
            else seg[node] = 0;
        }
        else if(segCnt[node] == 1) seg[node] = (r-l+1);
        return;
    }
    int mid = (l+r)/2;
    update(node*2,l,mid,u,v,x);
    update(node*2+1,mid+1,r,u,v,x);
    if(segCnt[node] == 0) seg[node] = seg[node*2] + seg[node*2+1];
}
void solve()
{
    cin >> n;
    for(int i = 1; i <= n; i++)
    {
        int x1,y1,x2,y2;
        cin >> x1 >> y1 >> x2 >> y2;
        x1 += offset;
        y1 += offset;
        x2 += offset;
        y2 += offset;
        event[x1].push_back({1,{y1+1,y2}});
        event[x2].push_back({-1,{y1+1,y2}});
    }
    long long ans = 0;
    for(int i = 1; i <= maxn; i++)
    {
        for(auto x: event[i])
            update(1,1,maxn,x.second.first,x.second.second,x.first);
        ans += seg[1];
       // if(seg[1]) cout << seg[1] << '\n';
    }
    cout << ans;

}

// CE - Rotate
// Manhattan Rotation (x,y) = (x+y,x-y), can generalized all dimension by adding 2 ^ k, turning sum into max
void solve()
{
    int n,k;
    cin >> n >> k;
    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= n; j++)
    {
        long long x;
        cin >> x;
        pre[i+j][i-j+n] += x;
    }
    for(int i = 1; i <= 2*n; i++)
        for(int j = 1; j <= 2*n; j++)
            pre[i][j] += pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1];
    long long ans = 0;
    for(int i = 1; i <= 2 * n; i++)
        for(int j = 1; j <= 2 * n; j++)
        {
            ans = max(ans,get(max(1,i-2*k),max(1,j-2*k),i,j));
            //cout << get(max(1,i-2*k),max(1,j-2*k),i,j) << '\n';
        }

    cout << ans;


}

// CI Mincowski Sum
// Making a polygon using the sum point of every pair of points of 2 polygon
const int maxn = 1e5;
struct pt {
    long long x, y;
    pt() {}
    pt(long long _x, long long _y) : x(_x), y(_y) {}
    pt operator+(const pt &p) const { return pt(x + p.x, y + p.y); }
    pt operator-(const pt &p) const { return pt(x - p.x, y - p.y); }
    long long cross(const pt &p) const { return x * p.y - y * p.x; }
    long long dot(const pt &p) const { return x * p.x + y * p.y; }
    long long cross(const pt &a, const pt &b) const { return (a - *this).cross(b - *this); }
    long long dot(const pt &a, const pt &b) const { return (a - *this).dot(b - *this); }
    long long sqrLen() const { return this->dot(*this); }
};
vector<pt> check;
vector<pt> query;
bool lexComp(const pt &l, const pt &r) {
    return l.x < r.x || (l.x == r.x && l.y < r.y);
}

int sgn(long long val) { return val > 0 ? 1 : (val == 0 ? 0 : -1); }

vector<pt> seq;
pt translation;
int n;

bool pointInTriangle(pt a, pt b, pt c, pt point) {
    long long s1 = abs(a.cross(b, c));
    long long s2 = abs(point.cross(a, b)) + abs(point.cross(b, c)) + abs(point.cross(c, a));
    return s1 == s2;
}

void prepare(vector<pt> &points) {
    n = points.size();
    int pos = 0;
    for (int i = 1; i < n; i++) {
        if (lexComp(points[i], points[pos]))
            pos = i;
    }
    rotate(points.begin(), points.begin() + pos, points.end());

    n--;
    seq.resize(n);
    for (int i = 0; i < n; i++)
        seq[i] = points[i + 1] - points[0];
    translation = points[0];
}

bool pointInConvexPolygon(pt point) {
    point = point - translation;
    if (seq[0].cross(point) != 0 &&
            sgn(seq[0].cross(point)) != sgn(seq[0].cross(seq[n - 1])))
        return false;
    if (seq[n - 1].cross(point) != 0 &&
            sgn(seq[n - 1].cross(point)) != sgn(seq[n - 1].cross(seq[0])))
        return false;
    if (seq[0].cross(point) == 0)
        return seq[0].sqrLen() >= point.sqrLen();

    int l = 0, r = n - 1;
    while (r - l > 1) {
        int mid = (l + r) / 2;
        int pos = mid;
        if (seq[pos].cross(point) >= 0)
            l = mid;
        else
            r = mid;
    }
    int pos = l;
    return pointInTriangle(seq[pos], seq[pos + 1], pt(0, 0), point);
}
void reorder_polygon(vector<pt> & P){
    size_t pos = 0;
    for(size_t i = 1; i < P.size(); i++){
        if(P[i].y < P[pos].y || (P[i].y == P[pos].y && P[i].x < P[pos].x))
            pos = i;
    }
    rotate(P.begin(), P.begin() + pos, P.end());
}

vector<pt> minkowski(vector<pt> P, vector<pt> Q){
    // the first vertex must be the lowest
    reorder_polygon(P);
    reorder_polygon(Q);
    // we must ensure cyclic indexing
    P.push_back(P[0]);
    P.push_back(P[1]);
    Q.push_back(Q[0]);
    Q.push_back(Q[1]);
    // main part
    vector<pt> result;
    size_t i = 0, j = 0;
    while(i < P.size() - 2 || j < Q.size() - 2){
        result.push_back(P[i] + Q[j]);
        auto cross = (P[i + 1] - P[i]).cross(Q[j + 1] - Q[j]);
        if(cross >= 0)
            ++i;
        if(cross <= 0)
            ++j;
    }
    return result;
}
vector<pt> res;
void solve()
{
    for(int i = 0; i < 3; i++)
    {
        int m;
        cin >> m;
        vector<pt> current;
        for(int j = 1;  j <= m; j++)
        {
            pt a;
            cin >> a.x >> a.y;
            current.push_back(a);
        }
        if(i != 0)
        res = minkowski(res,current);
        else res = current;
    }
    int q;
    cin >> q;
    prepare(res);
    for(int i = 1; i <= q; i++)
    {
        pt p;
        cin >> p.x >> p.y;
        p.x *= 3;
        p.y *= 3;
        if(pointInConvexPolygon(p)) cout << "YES\n";
        else cout << "NO\n";
    }

}
// Time is Money Trick
// To optimize sum a * sum b
// Basically give all weight to A and give all weight to B, then find the points on the convex hull of these two points
int n,m;
struct edge
{
    int u,v,t,c;
    long long coef;
};
bool cmp(edge x, edge y)
{
    return x.coef < y.coef;
}
vector<edge> edges;
long long dot(pair<long long, long long> x, pair<long long, long long> y)
{
    return x.first * y.first + x.second * y.second;
}
long long cross(pair<long long, long long> x, pair<long long, long long> y)
{
    return x.first * y.second - x.second * y.first;
}
long long area(pair<long long, long long> a, pair<long long, long long> b, pair<long long, long long> c)
{
    return cross({b.first - a.first, b.second - a.second}, {c.first - b.first, c.second - b.second});
}
int dsu[maxn+5];
void reset()
{
    for(int i = 0; i < n; i++)
        dsu[i] = i;
}
int par(int u)
{
    if(dsu[u] == u) return u;
    return dsu[u] = par(dsu[u]);
}
bool unite(int u, int v)
{
    u = par(u);
    v = par(v);
    if(u == v) return false;
    dsu[v] = u;
    return true;
}
vector<edge> print;
pair<long long, long long> f(pair<long long, long long> axis, bool yes)
{
    for(auto &e: edges)
        e.coef = dot(axis,{e.t,e.c});
    sort(edges.begin(),edges.end(),cmp);
    pair<long long, long long> ans = {0,0};
    reset();
    for(auto e: edges)
    {
        if(unite(e.u,e.v))
        {
            ans.first += e.t;
            ans.second += e.c;
            if(yes) print.push_back(e);
        }
    }
    return ans;

}
vector<pair<long long, long long>> ans;
vector<pair<long long, long long>> trace;
void divide(pair<long long, long long> lp, pair<long long, long long> rp, pair<long long, long long>  l, pair<long long, long long> r)
{

    pair<long long, long long> axis = {lp.second - rp.second, rp.first - lp.first};
    auto mp = f(axis,0);
  //  cout << mp.first << ' ' << mp.second << '\n';
    if(area(lp,mp,rp) <= 0)
    {
        ans.push_back(lp);
        trace.push_back(l);
        ans.push_back(rp);
        trace.push_back(r);
        return;
    }
    divide(lp,mp,l,axis);
    divide(mp,rp,axis,r);
}
void solve()
{
    cin >> n >> m;
    for(int i = 1; i <= m; i++)
    {
        edge e;
        cin >> e.u >> e.v >> e.t >> e.c;
        edges.push_back(e);
    }
    pair<int,int> x = {1,0};
    pair<int,int> y = {0,1};
    divide(f(x,0),f(y,0),x,y);
    long long total = 1e18;
    pair<long long, long long> opt;
    pair<long long, long long> optax;
    for(int i = 0; i < ans.size(); i++)
    {
        if(total > ans[i].first * ans[i].second)
        {
            total =  ans[i].first * ans[i].second;
            opt = ans[i];
            optax = trace[i];
        }
    }
    cout << opt.first << ' ' << opt.second << '\n';
    f(optax,1);
    for(auto e: print)
    {
        cout << e.u << ' ' << e.v << '\n';
    }


}

// CJ - Hash
// Remember to use 2 MOD
const int maxn = 1e6;
string s,t;
const int base = 331;
const long long MOD = 1e9+9;
long long Pow[maxn+5], hashs[maxn+5];
int n;
long long get(int l, int r)
{
    return (hashs[r] - hashs[l-1] * Pow[r-l+1] + MOD  * MOD) % MOD;
}
void solve()
{
    cin >> s >> t;
    if(t.size() > s.size())
    {
        cout << 0;
        return;
    }
    Pow[0] = 1;
    int x = t.size();
    n = s.size();
    for(int i = 1; i <= n; i++)
        Pow[i] = Pow[i-1] * base % MOD;
    s = ' ' + s;
    for(int i = 1; i <= n; i++)
    {
        hashs[i] = (hashs[i-1] * base + s[i] - 'a' + 1) % MOD;
     }
    long long target = 0;
    for(auto c: t)
    {
        target = (target * base + c - 'a' + 1) % MOD;
    }
    int cnt = 0;
    for(int i = 1; i + x - 1 <= n; i++)
        if(get(i,i+x-1) == target) cnt++;

    cout << cnt;

}

// CK - Z + KMP
long long z[maxn+5], kmp[maxn+5];
int n;
void solve()
{
    string s;
    cin >> s;
    n = s.size();
    long long l = 0, r = 0;
    for(int i = 1; i < n; i++)
    {
        if(i <= r) z[i] = min(r-i+1,z[i-l]);
        while(z[i] + i < n && s[z[i]+i] == s[z[i]]) z[i]++;
        if(i + z[i] - 1 > r)
        {
            l = i;
            r = i + z[i] - 1;
        }
    }
    for(int i = 0; i < n; i++)
        cout << z[i] << ' ';
    cout << '\n';
    for(int i = 1; i < n; i++)
    {
        int last = kmp[i-1];
        while(last != 0 && s[i] != s[last]) last = kmp[last-1];
        kmp[i] = last;
        if(s[i] == s[last]) kmp[i]++;
    }
    for(int i = 0; i < n; i++)
        cout << kmp[i] << ' ';



}

// CL - DP KMP
long long z[maxn+5], kmp[maxn+5];
int n;
void solve()
{
    string s;
    cin >> s;
    n = s.size();
    long long l = 0, r = 0;
    for(int i = 1; i < n; i++)
    {
        if(i <= r) z[i] = min(r-i+1,z[i-l]);
        while(z[i] + i < n && s[z[i]+i] == s[z[i]]) z[i]++;
        if(i + z[i] - 1 > r)
        {
            l = i;
            r = i + z[i] - 1;
        }
    }
    for(int i = 0; i < n; i++)
        cout << z[i] << ' ';
    cout << '\n';
    for(int i = 1; i < n; i++)
    {
        int last = kmp[i-1];
        while(last != 0 && s[i] != s[last]) last = kmp[last-1];
        kmp[i] = last;
        if(s[i] == s[last]) kmp[i]++;
    }
    for(int i = 0; i < n; i++)
        cout << kmp[i] << ' ';



}

// CM - CN - Trie

// Trie for words
void add(string s)
{
    int root = 0;
    for(auto c: s)
    {
        if(trie[root][c - 'a'] == 0) trie[root][c - 'a'] = ++timer;
        root = trie[root][c - 'a'];
    }
    isEnd[root]++;
}
string s;
long long calc(int pos)
{
    if(pos == n+1) return 1;
    if(dp[pos] != -1) return dp[pos];
    int root = 0;
    long long ans = 0;
    for(int x = pos; x <= n; x++)
    {
        if(trie[root][s[x]-'a'] == 0) break;
        root = trie[root][s[x] - 'a'];
        //cout << pos << ' ' << x << ' ' << root << '\n';
        ans = (ans + isEnd[root] * calc(x+1)) % MOD;
    }
   // cout << pos << ' ' << ans << '\n';
    return dp[pos] = ans;

}

// Trie bits
void add(int mask)
{
    int root = 0;
    for(int x = k - 1; x >= 0; x--)
    {
        int bit = (mask & (1<<x)) > 0;
        if(trie[root][bit] == 0) trie[root][bit] = ++timer;
        root = trie[root][bit];
        sub[root]++;
    }
}
void del(int mask)
{
    int root = 0;
    for(int x = k - 1; x >= 0; x--)
    {
        int bit = (mask & (1<<x)) > 0;
        root = trie[root][bit];
        sub[root]--;
    }
}
int calc(int mask)
{
    int ans = 0;
    int root = 0;
    for(int x = k - 1; x >= 0; x--)
    {
        int bit = (mask & (1<<x)) > 0;
        if(sub[trie[root][bit]] == 0)
        {
            root = trie[root][bit ^ 1];
            ans |= (1<<x);
            continue;
        }
        root = trie[root][bit];
    }
    return ans;
}

// Aho Corasick

struct Node{
    int next[26];
    vector<int> leaf;
    int p = -1;
    int pch;
    int link = -1, speclink = -1;
    int go[26];
    long long good;
    Node(int p=-1, int ch='$') : p(p), pch(ch) {
        fill(begin(next), end(next), -1);
        fill(begin(go), end(go), -1);
        good = 0;
    }
};

vector<Node> trie(1);
long long s[205][205];
int siz[205];
long long cost[205];

void add_string(int i) {
    int v = 0;
    for (int j = siz[i]-1; j >= 0; j--) {
        int c = s[i][j];
        if (trie[v].next[c] == -1) {
            trie[v].next[c] = trie.size();
            trie.emplace_back(v, c);
        }
        v = trie[v].next[c];
    }
    trie[v].good += cost[i];
    trie[v].leaf.push_back(i);
}

int go(int v, int ch);

int get_link(int v) {
    if (trie[v].link == -1) {
        if (v == 0 || trie[v].p == 0) trie[v].link = 0;
        else trie[v].link = go(get_link(trie[v].p), trie[v].pch);
    }
    return trie[v].link;
}

int go(int v, int ch) {
    int c = ch;
    if (trie[v].go[c] == -1) {
        if (trie[v].next[c] != -1) trie[v].go[c] = trie[v].next[c];
        else trie[v].go[c] = v == 0 ? 0 : go(get_link(v), ch);
    }
 //   trie[trie[v].go[c]].good |= trie[v].good;
    return trie[v].go[c];
}
int get_speclink(int v) {
    if (trie[v].speclink == -1) {
        int br = get_link(v);
        if (v == 0 && br == 0) trie[v].speclink = 0;
        else if (sz(trie[br].leaf)) trie[v].speclink = br;
        else trie[v].speclink = get_speclink(br);
    }
    return trie[v].speclink;
}
const int maxn = 200;
int dp[205][205][505][2];
const int mod = 1e9+7;
long long n,m,k;
long long a[maxn+5];
long long l[maxn+5], r[maxn+5];
long long calc(int pos, int state, int cnt, bool tight, bool isPos)
{
    if(cnt > k) return 0;
    if(pos == -1) return (cnt <= k);
    if(isPos && dp[pos][state][cnt][tight] != -1) return dp[pos][state][cnt][tight];
    long long ans = 0;
    int newlim = tight ? a[pos] : (m-1);
    for(int c = 0; c <= newlim; c++)
    {
        int newstate = go(state,c);
        int check = trie[newstate].good;
        int v = newstate;
        bool newpos = isPos || (c > 0);
        bool newtight = tight && (c == a[pos]);
        while(v)
        {
            v = get_speclink(v);
            check += trie[v].good;
        }
        if(!newpos) check = 0, newstate = 0;
       // cout << cnt << ' ' << check << '\n';
        ans += calc(pos-1,newstate,cnt + check, newtight, newpos);
        assert(ans >= 0);
        ans %= mod;
    }
    if(isPos) dp[pos][state][cnt][tight] = ans;
    return ans;
}
signed main() {

    ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);
    cin >> n >> m >> k;
    int x,y;
    cin >> x;
    for(int i = x-1; i >= 0; i--)
        cin >> l[i];
    cin >> y;
    for(int i = y-1; i >= 0; i--)
        cin >> r[i];
    for1(i,1,n) {
        int p;
        cin >> p;
        siz[i] = p;
        for(int j = 0; j < p; j++)
            cin >> s[i][p-j-1];
        cin >> cost[i];
        add_string(i);
    }
    long long ans = 0;
    int cnt = 0;
    int state = 0;
    for(int i = x-1; i >= 0; i--)
    {
        state = go(state,l[i]);
        int v = state;
        cnt += trie[state].good;
        while(v)
        {
            v = get_speclink(v);
            cnt += trie[v].good;
        }
        if(cnt > k) break;
    }
    if(cnt <= k) ans++;
    memset(dp,-1,sizeof(dp));
    for(int i = 0; i < x; i++)
        a[i] = l[i];
    ans -= calc(x-1,0,0,1,0);
    for(int i = 0; i < y; i++)
        a[i] = r[i];
    memset(dp,-1,sizeof(dp));
   ans += calc(y-1,0,0,1,0);

    cout << (ans % MOD + MOD) % MOD;
}


// CQ - Mo offline
struct query
{
    int l,r,idx;
} ask[maxn+5];
bool cmp(query x, query y)
{
    if(x.l / BLOCK == y.l / BLOCK) return x.r < y.r;
    return x.l / BLOCK < y.l / BLOCK;
}
long long ans[maxn+5];
long long total = 0;
void solve()
{
    cin >> n >> q;
    for(int i = 1; i <= n; i++)
        cin >> a[i], p[a[i]]++;
    for(auto &x: p)
        x.second = ++cnt;
    for(int i = 1; i <= n; i++)
        a[i] = p[a[i]];
    for(int i = 1; i <= q; i++)
    {
        cin >> ask[i].l >> ask[i].r;
        ask[i].idx = i;
    }
    sort(ask + 1, ask + 1 + q, cmp);
    int curl = 1;
    int curr = 0;
    for(int i = 1; i <= q; i++)
    {
        while(curl > ask[i].l)
        {
            curl--;
            track[a[curl]]++;
            if(track[a[curl]] == 1) total++;
        }
        while(curl < ask[i].l)
        {
            track[a[curl]]--;
            if(track[a[curl]] == 0) total--;
            curl++;
        }
        while(curr > ask[i].r)
        {
            track[a[curr]]--;
            if(track[a[curr]] == 0) total--;
            curr--;
        }
        while(curr < ask[i].r)
        {
            curr++;
            track[a[curr]]++;
            if(track[a[curr]] == 1) total++;
        }
        ans[ask[i].idx] = total;
    }
    for(int i = 1; i <= q; i++)
        cout << ans[i] << '\n';
}

// CT - Segment Tree lazy (add, set, sum)
void down(int node, int l, int r)
{
    if(lazy2[node] != 0)
    {
        long long mid = (l+r)/2;
        long long p = lazy2[node];
        seg[node*2] = p * (mid-l+1);
        seg[node*2+1] = p * (r-mid);
        lazy2[node * 2] = lazy2[node];
        lazy2[node * 2 + 1] = lazy2[node];
        lazy2[node] = 0;
        lazy[node * 2] = 0;
        lazy[node * 2 + 1] = 0;
    }
    long long mid = (l+r)/2;
    long long p = lazy[node];
        seg[node*2] += p * (mid-l+1);
        seg[node*2+1] += p * (r-mid);
        lazy[node * 2] += p;
        lazy[node * 2 + 1] += p;
        lazy[node] = 0;
}
void update(int node, int l, int r, int u, int v, long long val, int typ)
{
    if(l > v || r < u) return;
    if(u <= l && r <= v)
    {
        if(typ == 1)
        {
            long long cnt = r-l+1;

            seg[node] += cnt * val;
            lazy[node] += val;
        }
        else
        {
            long long cnt = r - l + 1;
            seg[node] = cnt * val;
            //cout << l << ' ' << r << ' ' << cnt * val << '\n';
            lazy2[node] = val;
            lazy[node] = 0;
        }
        return;
    }
    int mid = (l+r)/2;
    down(node,l,r);
    update(node*2,l,mid,u,v,val,typ);
    update(node*2+1,mid+1,r,u,v,val,typ);
    seg[node] = seg[node*2] + seg[node*2+1];
}
long long query(int node, int l, int r, int u, int v)
{
    if(l > v || r < u) return 0;
    if(u <= l && r <= v) return seg[node];
    down(node,l,r);
    int mid = (l+r)/2;
    long long le = query(node*2,l,mid,u,v);
    long long ri = query(node*2+1,mid+1,r,u,v);
    return le + ri;
}

// CU - Segment tree lazy, polynomial query
void down(int node, int l, int r)
{
    long long p = lazy[node];
    long long p2 = lazy2[node];
    int mid = (l+r)/2;

    // left
    long long cntL = (mid - l + 1);
    seg[node * 2] += p * cntL * (cntL + 1) / 2 + p2 * cntL;
    lazy[node * 2] += p;
    lazy2[node * 2] += p2;
    // right
    long long cntR = (r - mid);
    seg[node * 2 + 1] += p * cntR * (cntR + 1) / 2 + p2 * cntR + p * cntL * cntR;
    lazy[node * 2 + 1] += p;
    lazy2[node * 2 + 1] += p * cntL + p2;
    lazy[node] = lazy2[node] = 0;
}
void update(int node, int l, int r, int u, int v, long long val, int typ)
{
    if(l > v || r < u) return;
    if(u <= l && r <= v)
    {
        if(typ == 1)
        {
            long long cnt = r-l+1;
            seg[node] += cnt * (cnt + 1) / 2 * val + cnt * (l-u) * val;
            lazy[node] += val;
            lazy2[node] += (l-u) * val;
        }
        return;
    }
    int mid = (l+r)/2;
    down(node,l,r);
    update(node*2,l,mid,u,v,val,typ);
    update(node*2+1,mid+1,r,u,v,val,typ);
    seg[node] = seg[node*2] + seg[node*2+1];
}
long long query(int node, int l, int r, int u, int v)
{
    if(l > v || r < u) return 0;
    if(u <= l && r <= v) return seg[node];
    down(node,l,r);
    int mid = (l+r)/2;
    long long le = query(node*2,l,mid,u,v);
    long long ri = query(node*2+1,mid+1,r,u,v);
    return le + ri;
}

// CV - Segment Tree Walk, nearest to 1 number that is larger than target
void update(int node, int l, int r, int idx, long long val)
{
    if(l > idx || r < idx) return;
    if(l == r && l == idx)
    {
        seg[node] += val;
        return;
    }
    int mid = (l+r)/2;
    update(node*2,l,mid,idx,val);
    update(node*2+1,mid+1,r,idx,val);
    seg[node] = max(seg[node*2], seg[node*2+1]);
}
long long query(int node, int l, int r, long long target)
{
    if(seg[node] < target) return 0;
    if(l == r) return l;
    int mid = (l+r)/2;
    if(seg[node * 2] >= target)
        return query(node*2,l,mid,target);
    return query(node*2+1,mid+1,r,target);
}

// CX - Segment Tree 2d
void build_y(int vx, int lx, int rx, int vy, int ly, int ry) {
    if (ly == ry) {
        if (lx == rx)
            segtree[vx][vy] = 0;
        else
            segtree[vx][vy] = max(segtree[vx*2][vy],segtree[vx*2+1][vy]);
    } else {
        int my = (ly + ry) / 2;
        build_y(vx, lx, rx, vy*2, ly, my);
        build_y(vx, lx, rx, vy*2+1, my+1, ry);
        segtree[vx][vy] = max(segtree[vx][vy*2],segtree[vx][vy*2+1]);
    }
}

void build_x(int vx, int lx, int rx) {
    if (lx != rx) {
        int mx = (lx + rx) / 2;
        build_x(vx*2, lx, mx);
        build_x(vx*2+1, mx+1, rx);
    }
    build_y(vx, lx, rx, 1, 0, m - 1);
}
int query_y(int vx, int vy, int tly, int try_, int ly, int ry) {
    if (ly > ry)
        return 0;
    if (ly == tly && try_ == ry)
        return segtree[vx][vy];
    int tmy = (tly + try_) / 2;
    return max(query_y(vx, vy*2, tly, tmy, ly, min(ry, tmy))
        ,query_y(vx, vy*2+1, tmy+1, try_, max(ly, tmy+1), ry));
}

int query_x(int vx, int tlx, int trx, int lx, int rx, int ly, int ry) {
    if (lx > rx)
        return 0;
    if (lx == tlx && trx == rx)
        return query_y(vx, 1, 0, m-1, ly, ry);
    int tmx = (tlx + trx) / 2;
    return max(query_x(vx*2, tlx, tmx, lx, min(rx, tmx), ly, ry)
         ,query_x(vx*2+1, tmx+1, trx, max(lx, tmx+1), rx, ly, ry));
}
void update_y(int vx, int lx, int rx, int vy, int ly, int ry, int x, int y, int new_val) {
    if (ly == ry) {
        if (lx == rx)
            segtree[vx][vy] = new_val;
        else
            segtree[vx][vy] = max(segtree[vx*2][vy], segtree[vx*2+1][vy]);
    } else {
        int my = (ly + ry) / 2;
        if (y <= my)
            update_y(vx, lx, rx, vy*2, ly, my, x, y, new_val);
        else
            update_y(vx, lx, rx, vy*2+1, my+1, ry, x, y, new_val);
        segtree[vx][vy] = max(segtree[vx][vy*2],segtree[vx][vy*2+1]);
    }
}

void update_x(int vx, int lx, int rx, int x, int y, int new_val) {
    if (lx != rx) {
        int mx = (lx + rx) / 2;
        if (x <= mx)
            update_x(vx*2, lx, mx, x, y, new_val);
        else
            update_x(vx*2+1, mx+1, rx, x, y, new_val);
    }
    update_y(vx, lx, rx, 1, 0, m-1, x, y, new_val);
}

// Lazy 2d Sum update Fenwick
int n, m, A[N][N], B[N][N][4];
void upd(int x, int y, int v) {
  for(int i = x ; i <= n ; i += lowbit(i)) {
    for(int j = y ; j <= m ; j += lowbit(j)) {
      B[i][j][0] += v;
      B[i][j][1] += x * v;
      B[i][j][2] += y * v;
      B[i][j][3] += x * y * v;
    }
  }
}
int qry(int x, int y) {
  int ans = 0;
  for(int i = x ; i > 0 ; i -= lowbit(i)) {
    for(int j = y ; j > 0 ; j -= lowbit(j)) {
      ans += (x + 1) * (y + 1) * B[i][j][0] - (y + 1) * B[i][j][1] - (x + 1) * B[i][j][2] + B[i][j][3];
    }
  }
  return ans;
}
void update(int x1, int y1, int x2, int y2, int v) {
  upd(x1, y1, v);
  upd(x1, y2 + 1, -v);
  upd(x2 + 1, y1, -v);
  upd(x2 + 1, y2 + 1, v);
}
int query(int x1, int y1, int x2, int y2) {
  return qry(x2, y2) - qry(x1 - 1, y2) - qry(x2, y1 - 1) + qry(x1 - 1, y1 - 1);
}
void init() {
  for(int i = 1 ; i <= n ; ++i) {
    for(int j = 1 ; j <= m ; ++j) {
      upd(i, j, A[i][j]);
    }
  }
}

// Compress 2d Fenwick
void fakeUpdate(int u, int v) {
    for(int x = u; x <= cnt; x += x & -x)
        nodes[x].push_back(v);
}

void fakeGet(int u, int v) {
    for(int x = u; x > 0; x -= x & -x)
        nodes[x].push_back(v);
}

// Add point (u, v)
void update(int u, int v,long long k) {
    for(int x = u; x <= cnt; x += x & -x)
        for(int y = lower_bound(nodes[x].begin(),nodes[x].end(),v) - nodes[x].begin(); y < nodes[x].size(); y += y & -y)
            f[x][y] = max(f[x][y],k);
}

// Get number of point in rectangle with corners at (1, 1) and (u, v)
long long get(int u, int v) {
    long long res = 0;
    for(int x = u; x > 0; x -= x & -x)
        for(int y = lower_bound(nodes[x].begin(), nodes[x].end(), v) - nodes[x].begin(); y > 0; y -= y & -y)
            res = max(res,f[x][y]);
    return res;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cin >> n;
    for(int i = 1; i <= n; i++)
    {
        cin >> p[i].first >> p[i].second;
        dict.push_back(p[i].first);
    }
    sort(dict.begin(),dict.end());
    dict.resize(unique(dict.begin(),dict.end()) - dict.begin());
    cnt = dict.size();
    for(int i = 1; i <= n; i++)
    {
        p[i].first = (lower_bound(dict.begin(),dict.end(),p[i].first) - dict.begin()) + 1;
        fakeUpdate(p[i].first,p[i].second);
        fakeGet(p[i].first-1,p[i].second-1);
    }
    for(int i = 1; i <= cnt; i++)
    {
        nodes[i].push_back(-1e18);
        sort(nodes[i].begin(),nodes[i].end());
        nodes[i].resize(unique(nodes[i].begin(),nodes[i].end()) - nodes[i].begin());
        f[i].resize(nodes[i].size()+1,0);
    }
    long long maxx = 1;
    for(int i = 1; i <= n; i++)
    {
        long long x = p[i].first, y = p[i].second;
     //   cout << x << ' ' << y << '\n';
        long long v = 1 + get(x-1,y-1);
        maxx = max(v,maxx);
        update(x,y,v);
    }
    cout << maxx;
}

// DA - Persistent Segment Tree
// Half Interval
struct segment_tree {
#define m (l + r) / 2
    long long hsh;
    segment_tree *lc, *rc;

    segment_tree(long long _hsh = 0, segment_tree *_lc = nullptr, segment_tree *_rc = nullptr) {
        hsh = _hsh;
        lc = _lc;
        rc = _rc;
    }

    segment_tree* update(int l, int r, int u, long long h) {
        segment_tree* nw = new segment_tree(hsh + h, lc, rc);
        if (l == r - 1) {
            return nw;
        }
        if (u < m) {
            if (lc == nullptr) {
                lc = new segment_tree();
            }
            nw->lc = lc->update(l, m, u, h);
        } else {
            if (rc == nullptr) {
                rc = new segment_tree();
            }
            nw->rc = rc->update(m, r, u, h);
        }
        return nw;
    }

    long long query(int l, int r, int L, int R) {
        if (l >= R || r <= L || L >= R) {
            return 0;
        } else if (L <= l && r <= R) {
            return hsh;
        } else {
            return (lc == nullptr ? 0 : lc->query(l, m, L, R))
                + (rc == nullptr ? 0 : rc->query(m, r, L, R));
        }
    }
};

int main() {
    int n; cin >> n;
    int q; cin >> q;
    vector<int> a(n);
    vector<segment_tree*> wtf;
    segment_tree *rt = new segment_tree();
    map<int, vector<int>> ma;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        rt = rt->update(0, n, i, a[i]);
    }
    wtf.push_back(rt);
    while (q--) {
        int op;
        cin >> op;
        if(op == 1)
        {
            int k,a,x; cin >> k >> a >> x;
            a--;
            k--;
            int now = wtf[k]->query(0,n,a,a+1);
            wtf[k] = wtf[k]->update(0, n, a, x - now);
        }
        else if(op == 2)
        {
            int k,l,r;
            cin >> k >> l >> r;
            k--;
            l--;
            cout << wtf[k]->query(0,n,l,r) << '\n';
        }
        else
        {
            int k;
            cin >> k;
            k--;
            segment_tree* root = wtf[k];
            wtf.push_back(root);
        }


    }
}
// DB - Merge Sort Tree
vector<int> combine (vector<int> l,vector<int>  r)
{
    vector<int>  cur;
    merge(l.begin(), l.end(), r.begin(), r.end(),
              back_inserter(cur));
    return cur;
}
void build(int node,int start,int finish)
{
    if(start>finish) return;
    if(start==finish)
    {
         segtree[node] = vector<int>(1, a[start]);
         //cout << start << ' ' << segtree[node][0] << '\n';
         return;
    }
    int mid = (start+finish)/2;
    build(node*2,start,mid);
    build(node*2+1,mid+1,finish);
    segtree[node] = combine(segtree[node*2],segtree[node*2+1]);
   /* cout << node << ' ';
    for(auto i: segtree[node])
        cout << i << ' ';
    cout << '\n';*/
}
long long query(int node,int l,int r,int u,int v,int k)
{
    //cout << node << ' ' << l << ' ' << r << ' ';
    if(u > r || v < l) return 0;
    if(u <= l && r <= v) return segtree[node].end()-upper_bound(segtree[node].begin(),segtree[node].end(),k);
    int mid = (l+r)/2;
    long long left = query(2*node,l,mid,u,v,k);
    long long right = query(2*node+1,mid+1,r,u,v,k);
    return left+right;


}

// DD - Sparse Table
int query(int l, int r)
{
    int x = jump[r-l+1];
  //  cout << "check " << l << ' ' << r << ' ' << x << '\n';
    return min(st[l][x],st[r-(1<<x) + 1][x]);
}
void solve()
{
    int n,q;
    cin >> n >> q;
    for(int i = 1; i <= n; i++)
        cin >> a[i];
    for(int i = 2; i <= n; i++)
        jump[i] = jump[i/2] + 1;
    for(int i = 1; i <= n; i++)
        st[i][0] = a[i];
    for(int x = 1; x <= k; x++)
    {
        for(int p = 1; p + (1<<x) - 1 <= n; p++)
        {
           // cout << "check " << x << ' ' << p << '\n';
            st[p][x] = min(st[p][x-1], st[p+(1<<(x-1))][x-1]);
        }
    }
    for(int i = 1; i <= q; i++)
    {
        int l,r;
        cin >> l >> r;
        l++;
        //cout << l << ' ' << st[l][0] << '\n';
        cout << query(l,r) << '\n';
    }
}

// DE - DSU Rollback (add and delete query)
int dsu[maxn+5];
int siz[maxn+5];
stack<pair<pair<int,int>,pair<int,int>>> roll;
int par(int u)
{
    if(dsu[u] == u) return u;
    return par(dsu[u]);
}
bool unite(int u, int v)
{
    u = par(u);
    v = par(v);
    if(u == v) return false;
    if(siz[u] < siz[v]) swap(u,v);
    roll.push({{u,siz[u]},{v,siz[v]}});
    dsu[v] = u;
    siz[u] += siz[v];
    siz[v] = siz[u];
    return true;
}
int n,m,q;
int compo;
long long ans[maxn+5];
map<pair<int,int>,int> exist;
vector<pair<int,int>> seg[4 * maxn];
void update(int node, int l, int r, int u, int v, pair<int,int> edge)
{
    if(l > v || r < u) return;
    if(u <= l && r <= v)
    {
        seg[node].push_back(edge);
        return;
    }
    int mid = (l+r)/2;
    update(node*2,l,mid,u,v,edge);
    update(node*2+1,mid+1,r,u,v,edge);
}
void dfs(int node, int l, int r)
{
    int times = 0;
    for(auto e: seg[node])
    {
        //cout << e.first << ' ' << e.second << '\n';
        int add = unite(e.first,e.second);
        compo -= add;
        times += add;
    }
    if(l == r)
    {
        ans[l] = compo;
    }
    else
    {
        int mid = (l+r)/2;
        dfs(node*2,l,mid);
        dfs(node*2+1,mid+1,r);
    }
    while(times--)
    {
        int u = roll.top().first.first;
        int sizu = roll.top().first.second;
        int v = roll.top().second.first;
        int sizv = roll.top().second.second;
        dsu[u] = u;
        dsu[v] = v;
        siz[v] = sizv;
        siz[u] = sizu;
        compo++;
        roll.pop();
    }
}
void solve()
{
    cin >> n >> m >> q;
    compo = n;
    for(int i = 1; i <= n; i++)
        dsu[i] = i, siz[i] = 1;
    for(int i = 1; i <= m; i++)
    {
        int a,b;
        cin >> a >> b;
        if(a > b) swap(a,b);
        exist[{a,b}] = 1;
    }
    for(int i = 2; i <= q + 1; i++)
    {
        int op, u, v;
        cin >> op >> u >> v;
        if(u > v) swap(u,v);
        if(op == 1)
        {
            exist[{u,v}] = i;
        }
        else
        {
            update(1,1,q+1,exist[{u,v}],i-1, {u,v});
            exist[{u,v}] = 0;
        }
    }
    for(auto e: exist)
        if(e.second != 0)
    {
        update(1,1,q+1,e.second,q+1, e.first);
    }
    dfs(1,1,q+1);
    for(int i = 1; i <= q+1; i++)
        cout << ans[i] << ' ';

}

// Bipartite DSU with rollback
void make_set(int v) {
    parent[v] = {v,0};
    ranks[v] = 0;
    bipartite[v] = true;
}

pair<int, int> find_set(int v, int x) {
    if (v == parent[v].first) {
        return {v,x};
    }
    return find_set(parent[v].first, x ^ parent[v].second);
}
void rollback()
{
    query x = st.top();
    st.pop();
    parent[x.u] = x.paru;
    ranks[x.u] = x.ranku;
    bipartite[x.u] = x.bipa_u;
    parent[x.v] = x.parv;
    ranks[x.v] = x.rankv;
    bipartite[x.v] = x.bipa_v;

}
int add_edge(int a, int b) {
    pair<int, int> pa = find_set(a,0);
    a = pa.first;
    int x = pa.second;

    pair<int, int> pb = find_set(b,0);
    b = pb.first;

    int y = pb.second;
    query roll;
    roll.u = a;
    roll.v = b;
    roll.paru = pa;
    roll.parv = pb;
    roll.bipa_u = bipartite[a];
    roll.bipa_v = bipartite[b];
    roll.ranku = ranks[a];
    roll.rankv = ranks[b];
    st.push(roll);
    if (a == b) {
        if (x == y)
            bipartite[a] = false;
    } else {
        if (ranks[a] < ranks[b])
            swap (a, b);
        parent[b] = make_pair(a, x^y^1);
        bipartite[a] &= bipartite[b];
        if (ranks[a] == ranks[b])
            ++ranks[a];
    }
    return bipartite[a];
}

bool is_bipartite(int v) {
    return bipartite[find_set(v,0).first];
}

// DH - Parallel Binary Search
long long target[maxn+5];
vector<int> station[maxn+5];
pair<pair<int,int>,long long> up[maxn+5];
vector<int> event[maxn+5];
int le[maxn+5],ri[maxn+5], ans[maxn+5];
void bs()
{
    memset(fen,0,sizeof(fen));
    for(int i = 1; i <= n; i++)
    {
        if(le[i] > ri[i]) continue;
        event[(le[i]+ri[i])/2].push_back(i);
    }
    for(int i = 1; i <= q; i++)
    {
        int l = up[i].first.first;
        int r = up[i].first.second;
        long long x = up[i].second;
        update(l,x);
        update(r+1,-x);
        if(l > r) update(1,x);
        for(auto e: event[i])
        {
            long long cnt = 0;
            for(auto x: station[e])
            {
                cnt += get(x);
                if(cnt >= target[e])
                {
                    break;
                }
            }
            if(cnt >= target[e])
            {
                ans[e] = i;
                ri[e] = i-1;
            }
            else le[e] = i+1;
        }
        event[i].clear();
    }
}
void solve()
{
    cin >> n >> m;
    for(int i = 1; i <= m; i++)
    {
        int x;
        cin >> x;
        station[x].push_back(i);
    }
    for(int i = 1; i <= n; i++)
    {
        cin >> target[i];
    }
    cin >> q;
    for(int i = 1; i <= n; i++)
    {
        le[i] = 1;
        ri[i] = q;
        ans[i] = -1;
    }
    for(int i = 1; i <= q; i++)
    {
        cin >> up[i].first.first >> up[i].first.second >> up[i].second;
    }
    for(int x = 1; x <= 20; x++)
        bs();
    for(int i = 1; i <= n; i++)
    {
        if(ans[i] == -1) cout << "NIE\n";
        else cout << ans[i] << '\n';
    }
}

// DI - Xor hashing
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
long long RNG(long long l, long long r)
{
    return uniform_int_distribution<long long>(l,r)(rng);
}
long long a[maxn+5];
long long hashs[maxn+5];
long long pre[maxn+5];
long long pref[maxn+5], suf[maxn+5];
set<pair<int,int>> ans;
void solve()
{
    cin >> n;
    for(int i = 1; i <= n; i++)
        hashs[i] = RNG(1,1e18);
    for(int i = 1; i <= n; i++)
    {
        pre[i] = pre[i-1] ^ hashs[i];
        cin >> a[i];
        pref[i] = pref[i-1] ^ hashs[a[i]];
   //     if(i == n) cout << (hashs[a[n]] ^ hashs[a[n-1]]) << ' ' << pre[2] << '\n';
    }
    for(int i = 1; i <= n; i++)
    {
        if(a[i] == 1)
        {
            long long upper = 1;
            ans.insert({i,i});
            for(int x = i+1; x <= n; x++)
            {
                if(a[x] == 1) break;
                upper = max(upper,a[x]);
                int l = x - upper + 1;
                int r = x;
              //  cout << i << ' ' << l << ' ' << r << '\n';
                if(l <= i && l >= 1)
                {
                    //cout << l << ' ' << r << ' ' << (pre[r] ^ pre[l-1]) << ' ' << hashs[r-l+1] << '\n';
                    if((pref[r] ^ pref[l-1]) == pre[r-l+1]) ans.insert({l,r});
                }
            }
        }
    }
    for(int i = n; i  >= 1; i--)
    {
        if(a[i] == 1)
        {
            long long upper = 1;
            for(int x = i-1; x >= 1; x--)
            {
                if(a[x] == 1) break;
                upper = max(upper,a[x]);
                int l = x;
                int r = x + upper - 1;
                if(r >= i && r <= n)
                {
                    if((pref[r] ^ pref[l-1]) == pre[r-l+1]) ans.insert({l,r});
                }
            }
        }
    }
}

// Random KATCL Geometry Template
5 Geometry
5.1 Angle
/**
* Description: A class for ordering angles (as represented
by int points and
* a number of rotations around the origin). Useful for
rotational sweeping.
* Sometimes also represents points or vectors.
* Usage:
* vector<Angle> v = {w[0], w[0].t360() ...}; // sorted
* int j = 0; rep(i,0,n) { while (v[j] < v[i].t180()) ++j; }
* // sweeps j such that (j-i) represents the number of
positively oriented triangles with vertices at 0 and i
* Status: Used, works well
*/
#pragma once
struct Angle {
int x, y;
int t;
Angle(int x, int y, int t=0) : x(x), y(y), t(t) {}
Angle operator-(Angle b) const { return {x-b.x, y-b.y, t};
}
int half() const {
assert(x || y);
return y < 0 || (y == 0 && x < 0);
}
Angle t90() const { return {-y, x, t + (half() && x >= 0)};
}
Angle t180() const { return {-x, -y, t + half()}; }
Angle t360() const { return {x, y, t + 1}; }
};
bool operator<(Angle a, Angle b) {
// add a.dist2() and b.dist2() to also compare distances
return make_tuple(a.t, a.half(), a.y * (ll)b.x) <
make_tuple(b.t, b.half(), a.x * (ll)b.y);
}
// Given two points, this calculates the smallest angle between
// them, i.e., the angle that covers the defined line segment.
pair<Angle, Angle> segmentAngles(Angle a, Angle b) {
if (b < a) swap(a, b);
return (b < a.t180() ?
make_pair(a, b) : make_pair(b, a.t360()));
}
Angle operator+(Angle a, Angle b) { // point a + vector b
Angle r(a.x + b.x, a.y + b.y, a.t);
if (a.t180() < r) r.t--;
return r.t180() < a ? r.t360() : r;
}
Angle angleDiff(Angle a, Angle b) { // angle b - angle a
int tu = b.t - a.t; a.t = b.t;
return {a.x*b.x + a.y*b.y, a.x*b.y - a.y*b.x, tu - (b < a)
};
}
5.2 CircleIntersection
/**
* Description: Computes the pair of points at which two
circles intersect. Returns false in case of no
intersection.
* Status: stress-tested
*/
#pragma once
#include "Point.h"
typedef Point<double> P;
bool circleInter(P a,P b,double r1,double r2,pair<P, P>* out) {
if (a == b) { assert(r1 != r2); return false; }
P vec = b - a;
double d2 = vec.dist2(), sum = r1+r2, dif = r1-r2,
p = (d2 + r1*r1 - r2*r2)/(d2*2), h2 = r1*r1 - p*p*d2;
if (sum*sum < d2 || dif*dif > d2) return false;
P mid = a + vec*p, per = vec.perp() * sqrt(fmax(0, h2) / d2
);
*out = {mid + per, mid - per};
return true;
}
5.3 CircleLine
/**
* Description: Finds the intersection between a circle and
a line.
* Returns a vector of either 0, 1, or 2 intersection points
.
* P is intended to be Point<double>.
*/
#pragma once
#include "Point.h"
template<class P>
vector<P> circleLine(P c, double r, P a, P b) {
P ab = b - a, p = a + ab * (c-a).dot(ab) / ab.dist2();
double s = a.cross(b, c), h2 = r*r - s*s / ab.dist2();
if (h2 < 0) return {};
if (h2 == 0) return {p};
P h = ab.unit() * sqrt(h2);
return {p - h, p + h};
}
5.4 CirclePolygonIntersection
/**
* Description: Returns the area of the intersection of a
circle with a
* ccw polygon.
* Time: O(n)
* Status: Tested on GNYR 2019 Gerrymandering, stress-tested
*/
#pragma once
#include "../../content/geometry/Point.h"
typedef Point<double> P;
#define arg(p, q) atan2(p.cross(q), p.dot(q))
double circlePoly(P c, double r, vector<P> ps) {
auto tri = [&](P p, P q) {
auto r2 = r * r / 2;
P d = q - p;
auto a = d.dot(p)/d.dist2(), b = (p.dist2()-r*r)/d.dist2()
;
auto det = a * a - b;
if (det <= 0) return arg(p, q) * r2;
auto s = max(0., -a-sqrt(det)), t = min(1., -a+sqrt(det));
if (t < 0 || 1 <= s) return arg(p, q) * r2;HSG 10
P u = p + d * s, v = p + d * t;
return arg(p,u) * r2 + u.cross(v)/2 + arg(v,q) * r2;
};
auto sum = 0.0;
rep(i,0,sz(ps))
sum += tri(ps[i] - c, ps[(i + 1) % sz(ps)] - c);
return sum;
}
5.5 CircleTangents
/**
* Description: Finds the external tangents of two circles,
or internal if r2 is negated.
* Can return 0, 1, or 2 tangents -- 0 if one circle
contains the other (or overlaps it, in the internal
case, or if the circles are the same);
* 1 if the circles are tangent to each other (in which case
.first = .second and the tangent line is perpendicular
to the line between the centers).
* .first and .second give the tangency points at circle 1
and 2 respectively.
* To find the tangents of a circle with a point set r2 to
0.
* Status: tested
*/
#pragma once
#include "Point.h"
template<class P>
vector<pair<P, P>> tangents(P c1, double r1, P c2, double r2
) {
P d = c2 - c1;
double dr = r1 - r2, d2 = d.dist2(), h2 = d2 - dr * dr;
if (d2 == 0 || h2 < 0) return {};
vector<pair<P, P>> out;
for (double sign : {-1, 1}) {
P v = (d * dr + d.perp() * sqrt(h2) * sign) / d2;
out.push_back({c1 + v * r1, c2 + v * r2});
}
if (h2 == 0) out.pop_back();
return out;
}
5.6 circumcircle
/**
* Description:\
* The circumcirle of a triangle is the circle intersecting
all three vertices. ccRadius returns the radius of the
circle going through points A, B and C and ccCenter
returns the center of the same circle.
* Status: tested
*/
#pragma once
#include "Point.h"
typedef Point<double> P;
double ccRadius(const P& A, const P& B, const P& C) {
return (B-A).dist()*(C-B).dist()*(A-C).dist()/
abs((B-A).cross(C-A))/2;
}
P ccCenter(const P& A, const P& B, const P& C) {
P b = C-A, c = B-A;
return A + (b*c.dist2()-c*b.dist2()).perp()/b.cross(c)/2;
}
5.7 ClosestPair
/**
* Source: https://codeforces.com/blog/entry/58747
* Description: Finds the closest pair of points.
*/
#pragma once
#include "Point.h"
typedef Point<ll> P;
pair<P, P> closest(vector<P> v) {
assert(sz(v) > 1);
set<P> S;
sort(all(v), [](P a, P b) { return a.y < b.y; });
pair<ll, pair<P, P>> ret{LLONG_MAX, {P(), P()}};
int j = 0;
for (P p : v) {
P d{1 + (ll)sqrt(ret.first), 0};
while (v[j].y <= p.y - d.x) S.erase(v[j++]);
auto lo = S.lower_bound(p - d), hi = S.upper_bound(p + d);
for (; lo != hi; ++lo)
ret = min(ret, {(*lo - p).dist2(), {*lo, p}});
S.insert(p);
}
return ret.second;
}
5.8 ConvexHull
/**
Returns a vector of the points of the convex hull in counter
-clockwise order.
Points on the edge of the hull between two other points are
not considered part of the hull.
*/
#pragma once
#include "Point.h"
typedef Point<ll> P;
vector<P> convexHull(vector<P> pts) {
if (sz(pts) <= 1) return pts;
sort(all(pts));
vector<P> h(sz(pts)+1);
int s = 0, t = 0;
for (int it = 2; it--; s = --t, reverse(all(pts)))
for (P p : pts) {
while (t >= s + 2 && h[t-2].cross(h[t-1], p) <= 0) t--;
h[t++] = p;
}
return {h.begin(), h.begin() + t - (t == 2 && h[0] == h[1])
};
}
5.9 DelaunayTriangulation
/**
* Description: Computes the Delaunay triangulation of a set
of points.
* Each circumcircle contains none of the input points.
* If any three points are collinear or any four are on the
same circle, behavior is undefined.
* Time: O(n^2)
* Status: stress-tested
*/
#pragma once
#include "Point.h"
#include "3dHull.h"
template<class P, class F>HSG 11
void delaunay(vector<P>& ps, F trifun) {
if (sz(ps) == 3) { int d = (ps[0].cross(ps[1], ps[2]) < 0);
trifun(0,1+d,2-d); }
vector<P3> p3;
for (P p : ps) p3.emplace_back(p.x, p.y, p.dist2());
if (sz(ps) > 3) for(auto t:hull3d(p3)) if ((p3[t.b]-p3[t.a
]).
cross(p3[t.c]-p3[t.a]).dot(P3(0,0,1)) < 0)
trifun(t.a, t.c, t.b);
}
5.10 FastDelaunay
/**
* Description: Fast Delaunay triangulation.
* Each circumcircle contains none of the input points.
* There must be no duplicate points.
* If all points are on a line, no triangles will be
returned.
* Should work for doubles as well, though there may be
precision issues in ’circ’.
* Returns triangles in order \{t[0][0], t[0][1], t[0][2], t
[1][0], \dots}, all counter-clockwise.
* Time: O(n \log n)
*/
#pragma once
#include "Point.h"
typedef Point<ll> P;
typedef struct Quad* Q;
typedef __int128_t lll; // (can be ll if coords are < 2e4)
P arb(LLONG_MAX,LLONG_MAX); // not equal to any other point
struct Quad {
Q rot, o; P p = arb; bool mark;
P& F() { return r()->p; }
Q& r() { return rot->rot; }
Q prev() { return rot->o->rot; }
Q next() { return r()->prev(); }
} *H;
bool circ(P p, P a, P b, P c) { // is p in the circumcircle?
lll p2 = p.dist2(), A = a.dist2()-p2,
B = b.dist2()-p2, C = c.dist2()-p2;
return p.cross(a,b)*C + p.cross(b,c)*A + p.cross(c,a)*B >
0;
}
Q makeEdge(P orig, P dest) {
Q r = H ? H : new Quad{new Quad{new Quad{new Quad{0}}}};
H = r->o; r->r()->r() = r;
rep(i,0,4) r = r->rot, r->p = arb, r->o = i & 1 ? r : r->r
();
r->p = orig; r->F() = dest;
return r;
}
void splice(Q a, Q b) {
swap(a->o->rot->o, b->o->rot->o); swap(a->o, b->o);
}
Q connect(Q a, Q b) {
Q q = makeEdge(a->F(), b->p);
splice(q, a->next());
splice(q->r(), b);
return q;
}
pair<Q,Q> rec(const vector<P>& s) {
if (sz(s) <= 3) {
Q a = makeEdge(s[0], s[1]), b = makeEdge(s[1], s.back());
if (sz(s) == 2) return { a, a->r() };
splice(a->r(), b);
auto side = s[0].cross(s[1], s[2]);
Q c = side ? connect(b, a) : 0;
return {side < 0 ? c->r() : a, side < 0 ? c : b->r() };
}
#define H(e) e->F(), e->p
#define valid(e) (e->F().cross(H(base)) > 0)
Q A, B, ra, rb;
int half = sz(s) / 2;
tie(ra, A) = rec({all(s) - half});
tie(B, rb) = rec({sz(s) - half + all(s)});
while ((B->p.cross(H(A)) < 0 && (A = A->next())) ||
(A->p.cross(H(B)) > 0 && (B = B->r()->o)));
Q base = connect(B->r(), A);
if (A->p == ra->p) ra = base->r();
if (B->p == rb->p) rb = base;
#define DEL(e, init, dir) Q e = init->dir; if (valid(e)) \
while (circ(e->dir->F(), H(base), e->F())) { \
Q t = e->dir; \
splice(e, e->prev()); \
splice(e->r(), e->r()->prev()); \
e->o = H; H = e; e = t; \
}
for (;;) {
DEL(LC, base->r(), o); DEL(RC, base, prev());
if (!valid(LC) && !valid(RC)) break;
if (!valid(LC) || (valid(RC) && circ(H(RC), H(LC))))
base = connect(RC, base->r());
else
base = connect(base->r(), LC->r());
}
return { ra, rb };
}
vector<P> triangulate(vector<P> pts) {
sort(all(pts)); assert(unique(all(pts)) == pts.end());
if (sz(pts) < 2) return {};
Q e = rec(pts).first;
vector<Q> q = {e};
int qi = 0;
while (e->o->F().cross(e->F(), e->p) < 0) e = e->o;
#define ADD { Q c = e; do { c->mark = 1; pts.push_back(c->p)
; \
q.push_back(c->r()); c = c->next(); } while (c != e); }
ADD; pts.clear();
while (qi < sz(q)) if (!(e = q[qi++])->mark) ADD;
return pts;
}
5.11 HullDiameter
/**
* Description: Returns the two points with max distance on
a convex hull (ccw,
* no duplicate/collinear points).
*/
#pragma once
#include "Point.h"
typedef Point<ll> P;
array<P, 2> hullDiameter(vector<P> S) {
int n = sz(S), j = n < 2 ? 0 : 1;
pair<ll, array<P, 2>> res({0, {S[0], S[0]}});
rep(i,0,j)
for (;; j = (j + 1) % n) {
res = max(res, {(S[i] - S[j]).dist2(), {S[i], S[j]}});
if ((S[(j + 1) % n] - S[j]).cross(S[i + 1] - S[i]) >= 0)
break;
}
return res.second;
}
5.12 InsidePolygon
/**
* Description: Returns true if p lies within the polygon.
If strict is true,
* it returns false for points on the boundary. The
algorithm uses
* products in intermediate steps so watch out for overflow.
* Time: O(n)
*/
#pragma once
#include "Point.h"
#include "OnSegment.h"
#include "SegmentDistance.h"
template<class P>
bool inPolygon(vector<P> &p, P a, bool strict = true) {
int cnt = 0, n = sz(p);
rep(i,0,n) {
P q = p[(i + 1) % n];
if (onSegment(p[i], q, a)) return !strict;
//or: if (segDist(p[i], q, a) <= eps) return !strict;
cnt ^= ((a.y<p[i].y) - (a.y<q.y)) * a.cross(p[i], q) > 0;
}
return cnt;
}
5.13 linearTransformation
/**
Apply the linear transformation (translation, rotation and
scaling) which takes line p0-p1 to line q0-q1 to point
r.
*/
#pragma once
#include "Point.h"
typedef Point<double> P;
P linearTransformation(const P& p0, const P& p1,
const P& q0, const P& q1, const P& r) {
P dp = p1-p0, dq = q1-q0, num(dp.cross(dq), dp.dot(dq));
return q0 + P((r-p0).cross(num), (r-p0).dot(num))/dp.dist2
();
}
5.14 lineDistance
/**
* Returns the signed distance between point p and the line
containing points a and b. Positive value on left side
and negative on right as seen from a towards b. a==b
gives nan. P is supposed to be Point<T> or Point3D<T>
where T is e.g. double or long long. It uses products
in intermediate steps so watch out for overflow if
using int or long long. Using Point3D will always give
a non-negative distance. For Point3D, call .dist on the
result of the cross product.
* Status: tested
*/
#pragma once
#include "Point.h"
template<class P>
double lineDist(const P& a, const P& b, const P& p) {
return (double)(b-a).cross(p-a)/(b-a).dist();
}
5.15 lineIntersection
/**
If a unique intersection point of the lines going through s1
,e1 and s2,e2 exists \{1, point} is returned.
If no intersection point exists \{0, (0,0)} is returned and
if infinitely many exists \{-1, (0,0)} is returned.
The wrong position will be returned if P is Point<ll> and
the intersection point does not have integer
coordinates.
Products of three coordinates are used in intermediate steps
so watch out for overflow if using int or ll.
* Usage:
* auto res = lineInter(s1,e1,s2,e2);
* if (res.first == 1)
* cout << "intersection point at " << res.second << endl;
* Status: stress-tested, and tested through half-plane
tests
*/
#pragma once
#include "Point.h"
template<class P>
pair<int, P> lineInter(P s1, P e1, P s2, P e2) {
auto d = (e1 - s1).cross(e2 - s2);
if (d == 0) // if parallel
return {-(s1.cross(e1, s2) == 0), P(0, 0)};
auto p = s2.cross(e1, e2), q = s2.cross(e2, s1);
return {1, (s1 * p + e1 * q) / d};
}
5.16 LineProjectionReflection
/**
* Description: Projects point p onto line ab. Set refl=true
to get reflection
* of point p across line ab insted. The wrong point will be
returned if P is
* an integer point and the desired point doesn’t have
integer coordinates.
* Products of three coordinates are used in intermediate
steps so watch out
* for overflow.
*/
#pragma once
#include "Point.h"
template<class P>
P lineProj(P a, P b, P p, bool refl=false) {
P v = b - a;
return p - v.perp()*(1+refl)*v.cross(p-a)/v.dist2();
}
5.17 ManhattanMST
/**
* Description: Given N points, returns up to 4*N edges,
which are guaranteed
* to contain a minimum spanning tree for the graph with
edge weights w(p, q) =
* |p.x - q.x| + |p.y - q.y|. Edges are in the form (
distance, src, dst). Use a
* standard MST algorithm on the result to find the final
MST.
* Time: O(N \log N)
*/
#pragma once
#include "Point.h"
typedef Point<int> P;
vector<array<int, 3>> manhattanMST(vector<P> ps) {
vi id(sz(ps));
iota(all(id), 0);
vector<array<int, 3>> edges;
rep(k,0,4) {
sort(all(id), [&](int i, int j) {
return (ps[i]-ps[j]).x < (ps[j]-ps[i]).y;});
map<int, int> sweep;
for (int i : id) {
for (auto it = sweep.lower_bound(-ps[i].y);
it != sweep.end(); sweep.erase(it++)) {
int j = it->second;
P d = ps[i] - ps[j];
if (d.y > d.x) break;
edges.push_back({d.y + d.x, i, j});
}
sweep[-ps[i].y] = i;
}
for (P& p : ps) if (k & 1) p.x = -p.x; else swap(p.x, p.y)
;
}
return edges;
}
5.18 MinimumEnclosingCircle
/**
* Description: Computes the minimum circle that encloses a
set of points.
* Time: expected O(n)
* Status: stress-tested
*/
#pragma once
#include "circumcircle.h"
pair<P, double> mec(vector<P> ps) {
shuffle(all(ps), mt19937(time(0)));
P o = ps[0];
double r = 0, EPS = 1 + 1e-8;
rep(i,0,sz(ps)) if ((o - ps[i]).dist() > r * EPS) {
o = ps[i], r = 0;
rep(j,0,i) if ((o - ps[j]).dist() > r * EPS) {
o = (ps[i] + ps[j]) / 2;
r = (o - ps[i]).dist();
rep(k,0,j) if ((o - ps[k]).dist() > r * EPS) {
o = ccCenter(ps[i], ps[j], ps[k]);
r = (o - ps[i]).dist();
}
}
}
return {o, r};
}
5.19 OnSegment
/**
* Description: Returns true iff p lies on the line segment
from s to e.
* Use \texttt{(segDist(s,e,p)<=epsilon)} instead when using
Point<double>.
* Status:
*/
#pragma once
#include "Point.h"
template<class P> bool onSegment(P s, P e, P p) {
return p.cross(s, e) == 0 && (s - p).dot(e - p) <= 0;
}
5.20 Point
/**
* Description: Class to handle points in the plane.
* T can be e.g. double or long long. (Avoid int.)
* Status: Works fine, used a lot
*/
#pragma once
template <class T> int sgn(T x) { return (x > 0) - (x < 0);
}
template<class T>
struct Point {
typedef Point P;
T x, y;
explicit Point(T x=0, T y=0) : x(x), y(y) {}
bool operator<(P p) const { return tie(x,y) < tie(p.x,p.y);
}
bool operator==(P p) const { return tie(x,y)==tie(p.x,p.y);
}
P operator+(P p) const { return P(x+p.x, y+p.y); }
P operator-(P p) const { return P(x-p.x, y-p.y); }
P operator*(T d) const { return P(x*d, y*d); }
P operator/(T d) const { return P(x/d, y/d); }
T dot(P p) const { return x*p.x + y*p.y; }
T cross(P p) const { return x*p.y - y*p.x; }
T cross(P a, P b) const { return (a-*this).cross(b-*this);
}
T dist2() const { return x*x + y*y; }
double dist() const { return sqrt((double)dist2()); }
// angle to x-axis in interval [-pi, pi]
double angle() const { return atan2(y, x); }
P unit() const { return *this/dist(); } // makes dist()=1
P perp() const { return P(-y, x); } // rotates +90 degrees
P normal() const { return perp().unit(); }
// returns point rotated ’a’ radians ccw around the origin
P rotate(double a) const {
return P(x*cos(a)-y*sin(a),x*sin(a)+y*cos(a)); }
friend ostream& operator<<(ostream& os, P p) {
return os << "(" << p.x << "," << p.y << ")"; }
};
5.21 PointInsideHull
/**
* Description: Determine whether a point t lies inside a
convex hull (CCW
* order, with no collinear points). Returns true if point
lies within
* the hull. If strict is true, points on the boundary aren’
t included.
* Time: O(\log N)
*/
#pragma once
#include "Point.h"
#include "sideOf.h"
#include "OnSegment.h"
typedef Point<ll> P;
bool inHull(const vector<P>& l, P p, bool strict = true) {
int a = 1, b = sz(l) - 1, r = !strict;
if (sz(l) < 3) return r && onSegment(l[0], l.back(), p);
if (sideOf(l[0], l[a], l[b]) > 0) swap(a, b);
if (sideOf(l[0], l[a], p) >= r || sideOf(l[0], l[b], p)<= -
r)
return false;
while (abs(a - b) > 1) {
int c = (a + b) / 2;
(sideOf(l[0], l[c], p) > 0 ? b : a) = c;
}
return sgn(l[a].cross(l[b], p)) < r;
}
5.22 PolygonArea
/**
* Description: Returns twice the signed area of a polygon.
* Clockwise enumeration gives negative area. Watch out for
overflow if using int as T!
* Status: Stress-tested and tested on kattis:polygonarea
*/
#pragma once
#include "Point.h"
template<class T>
T polygonArea2(vector<Point<T>>& v) {
T a = v.back().cross(v[0]);
rep(i,0,sz(v)-1) a += v[i].cross(v[i+1]);
return a;
}
5.23 PolygonCenter
/**
* Description: Returns the center of mass for a polygon.
* Time: O(n)
* Status: Tested
*/
#pragma once
#include "Point.h"
typedef Point<double> P;
P polygonCenter(const vector<P>& v) {
P res(0, 0); double A = 0;
for (int i = 0, j = sz(v) - 1; i < sz(v); j = i++) {
res = res + (v[i] + v[j]) * v[j].cross(v[i]);
A += v[j].cross(v[i]);
}
return res / A / 3;
}
5.24 PolygonCut
/**
Returns a vector with the vertices of a polygon with
everything to the left of the line going from s to e
cut away.
* Usage:
* vector<P> p = ...;
* p = polygonCut(p, P(0,0), P(1,0));
*/
#pragma once
#include "Point.h"
#include "lineIntersection.h"
typedef Point<double> P;
vector<P> polygonCut(const vector<P>& poly, P s, P e) {
vector<P> res;
rep(i,0,sz(poly)) {
P cur = poly[i], prev = i ? poly[i-1] : poly.back();
bool side = s.cross(e, cur) < 0;
if (side != (s.cross(e, prev) < 0))
res.push_back(lineInter(s, e, cur, prev).second);
if (side)
res.push_back(cur);
}
return res;
}
5.25 PolygonUnion
/**
* Description: Calculates the area of the union of n$
polygons (not necessarily
* convex). The points within each polygon must be given in
CCW order.
* (Epsilon checks may optionally be added to sideOf/sgn,
but shouldn’t be needed.)
* Time: O(N^2)$, where N$ is the total number of points
* Status: stress-tested, Submitted on ECNA 2017 Problem A
*/
#pragma once
#include "Point.h"
#include "sideOf.h"
typedef Point<double> P;
double rat(P a, P b) { return sgn(b.x) ? a.x/b.x : a.y/b.y;
}
double polyUnion(vector<vector<P>>& poly) {
double ret = 0;
rep(i,0,sz(poly)) rep(v,0,sz(poly[i])) {
P A = poly[i][v], B = poly[i][(v + 1) % sz(poly[i])];
vector<pair<double, int>> segs = {{0, 0}, {1, 0}};
rep(j,0,sz(poly)) if (i != j) {
rep(u,0,sz(poly[j])) {
P C = poly[j][u], D = poly[j][(u + 1) % sz(poly[j])];
int sc = sideOf(A, B, C), sd = sideOf(A, B, D);
if (sc != sd) {
double sa = C.cross(D, A), sb = C.cross(D, B);
if (min(sc, sd) < 0)
segs.emplace_back(sa / (sa - sb), sgn(sc - sd));
} else if (!sc && !sd && j<i && sgn((B-A).dot(D-C))>0){
segs.emplace_back(rat(C - A, B - A), 1);
segs.emplace_back(rat(D - A, B - A), -1);
}
}
}
sort(all(segs));
for (auto& s : segs) s.first = min(max(s.first, 0.0), 1.0)
;
double sum = 0;
int cnt = segs[0].second;
rep(j,1,sz(segs)) {
if (!cnt) sum += segs[j].first - segs[j - 1].first;
cnt += segs[j].second;
}
ret += A.cross(B) * sum;
}
return ret / 2;
}
5.26 SegmentDistance
/**
Returns the shortest distance between point p and the line
segment from point s to e.
* Usage:
* Point<double> a, b(2,2), p(1,1);
* bool onSegment = segDist(a,b,p) < 1e-10;
* Status: tested
*/
#pragma once
#include "Point.h"
typedef Point<double> P;
double segDist(P& s, P& e, P& p) {
if (s==e) return (p-s).dist();
auto d = (e-s).dist2(), t = min(d,max(.0,(p-s).dot(e-s)));
return ((p-s)*d-(e-s)*t).dist()/d;
}HSG 15
5.27 SegmentIntersection
/**
If a unique intersection point between the line segments
going from s1 to e1 and from s2 to e2 exists then it is
returned.
If no intersection point exists an empty vector is returned.
If infinitely many exist a vector with 2 elements is
returned, containing the endpoints of the common line
segment.
The wrong position will be returned if P is Point<ll> and
the intersection point does not have integer
coordinates.
Products of three coordinates are used in intermediate steps
so watch out for overflow if using int or long long.
* Usage:
* vector<P> inter = segInter(s1,e1,s2,e2);
* if (sz(inter)==1)
* cout << "segments intersect at " << inter[0] << endl;
* Status: stress-tested, tested on kattis:intersection
*/
#pragma once
#include "Point.h"
#include "OnSegment.h"
template<class P> vector<P> segInter(P a, P b, P c, P d) {
auto oa = c.cross(d, a), ob = c.cross(d, b),
oc = a.cross(b, c), od = a.cross(b, d);
// Checks if intersection is single non-endpoint point.
if (sgn(oa) * sgn(ob) < 0 && sgn(oc) * sgn(od) < 0)
return {(a * ob - b * oa) / (ob - oa)};
set<P> s;
if (onSegment(c, d, a)) s.insert(a);
if (onSegment(c, d, b)) s.insert(b);
if (onSegment(a, b, c)) s.insert(c);
if (onSegment(a, b, d)) s.insert(d);
return {all(s)};
}
5.28 sideOf
/**
* Description: Returns where p$ is as seen from s$
towards e$. 1/0/-1 $\Leftrightarrow$ left/on line/
right. If the optional argument eps$ is given 0 is
returned if p$ is within distance eps$ from the line.
P is supposed to be Point<T> where T is e.g. double or
long long. It uses products in intermediate steps so
watch out for overflow if using int or long long.
* Usage:
* bool left = sideOf(p1,p2,q)==1;
* Status: tested
*/
#pragma once
#include "Point.h"
template<class P>
int sideOf(P s, P e, P p) { return sgn(s.cross(e, p)); }
template<class P>
int sideOf(const P& s, const P& e, const P& p, double eps) {
auto a = (e-s).cross(p-s);
double l = (e-s).dist()*eps;
return (a > l) - (a < -l);
}
