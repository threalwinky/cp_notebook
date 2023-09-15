# Graph
## Depth First Search
```cpp
bool c[N];
void dfs(int u){
    tmp.push_back(u);
    c[u] = 1;
    for (auto x : adj[u]){
        if (!c[x]){
            dfs(x);
        }
    }
}
```
## Breadth First Search
```cpp
bool c[N];
void bfs(int u){
    queue<int> q;
    q.push(u);
    c[u] = 1;
    while (!q.empty()){
        int t = q.front();
        tmp.push_back(t);
        q.pop();
        for (auto x : adj[t]){
            if (!c[x]){
                c[x] = 1;
                q.push(x);
            }
        }
    }
}
```

## Minimum Spanning Tree
```cpp
struct Edge{
    int u, v, w;
};
vector<Edge> E;
struct DSU{
    vi par, sz;
    void init(int n){
        par.assign(n + 9, 0);
        sz.assign(n + 9, 0);
        for (int i=1; i<=n; i++){
            par[i] = i;
            sz[i] = 1;
        }
    }
    int find(int u){
        return (u == par[u])?u:(par[u]=find(par[u]));
    }
    bool join(int u, int v){
        u = find(u), v = find(v);
        if (u == v) return 0;
        if (sz[u] < sz[v]) swap(u, v);
        par[v] = u;
        sz[u] += sz[v];
        return 1;
    }
};
bool cmp(Edge a, Edge b){
    return a.w < b.w;
}
signed main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    opf(0);
    int n, m;
    cin >> n >> m;
    for (int i=0; i<m; i++){
        int u, v, w;
        cin >> u >> v >> w;
        E.push_back({u, v, w});
    }
    sort(E.begin(), E.end(), cmp);
    DSU dsu;
    dsu.init(n);
    int ans = 0;
    for (auto it : E){
        if (!dsu.join(it.u, it.v)){ continue; }
        ans += it.w;
    }
    cout << ans;
}

```
## Dijkstra (Shortest path)
```cpp
struct Edge{
    int v;
    ll w;
};

struct Node{
    int u;
    ll D_u;
};

struct cmp{
    bool operator () (const Node a, const Node b){
        return a.D_u > b.D_u;
    }
};

V<Edge> E[100009];
vll d(100009, maxll);
vi trace(100009, -1);
vll shortest_path(int s){
    V<bool> P(100009, 0);
    priority_queue<Node, V<Node>, cmp> pq;
    d[s] = 0;
    pq.push({s, d[s]});
    while (!pq.empty()){
        Node x = pq.top();
        pq.pop();
        int u = x.u;
        if (P[u]) continue;
        P[u] = 1;
        for (auto e : E[u]){
            int v = e.v;
            ll w = e.w;
            if (d[v] > d[u] + w){
                d[v] = d[u] + w;
                pq.push({v, d[v]});
                trace[v] = u;
            }
        }
    }
    return d;
}
```

## Topological Sort
```cpp
vi adj[100006];
vi tmp;
vb check(100006, 0);
void dfs(int u){
    check[u] = 1;
    tmp.push_back(u);
    for (auto x : adj[u]){
        if (!check[x]) dfs(x);
    }
}

vi topo(int n){
    for (int i=1; i<=n; i++){
        if (!check[i]) dfs(i);
    }
    return tmp;
}
```

## Kosaraju (Strong Connected Components)

```cpp
vi adj[100006], radj[100006];
vb check(100006, 0);
stack<int> st;
int cnt = 0, c = 0;

void dfs(int u, int p){
    check[u] = 1;
    for (auto it : p ? adj[u] : radj[u]){
        if (!check[it]){
            dfs(it, p);
        }
    }
    if (p) st.push(u);
    return;
}

int scc(int n){
    for (int i=1; i<=n; i++){
        if (!check[i]){
            dfs(i, 1);
        }
    }
    check.assign(100006, 0);
    int ans = 0;
    while (!st.empty()){
        int t = st.top();
        if (!check[t]){
            dfs(t, 0);
            ans++;
        }
        st.pop();
    }
    return ans;
}
```