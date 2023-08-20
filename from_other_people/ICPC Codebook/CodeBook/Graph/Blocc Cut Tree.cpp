struct block_cut_tree {
    int h[maxn], p[20][maxn], in[maxn], low[maxn], cnt = 0, node;
    vector<int> adj2[maxn], st;
    void DFS(int u = 1, int p = 0)
    {
        in[u] = low[u] = ++cnt;
        st.emplace_back(u);
        for (int v: adj[u])
        {
            if (v == p) continue;
            if (in[v]) low[u] = min(low[u], in[v]);
            else
            {
                DFS(v, u);
                low[u] = min(low[u], low[v]);
                if (low[v] >= in[u])
                {
                    node++;
                    int x;
                    do
                        x = st.back(), st.pop_back(),
                        adj2[node].emplace_back(x),
                        adj2[x].emplace_back(node);
                    while (x != v);
                    adj2[node].emplace_back(u);
                    adj2[u].emplace_back(node);
                }
            }
        }
    }
    void DFS2(int u = 1, int pa = 0)
    {
        for (int v: adj2[u])
        {
            if (v == pa) continue;
            h[v] = h[u] + 1;
            p[0][v] = u;
            DFS2(v, u);
        }
    }
    int lca(int u, int v)
    {
        if (h[u] > h[v]) swap(u, v);
        int dis = h[v] - h[u];
        for (int i=19; i>=0; i--) if ((dis>>i)&1) v = p[i][v];
        if (v == u) return u;
        for (int i=19; i>=0; i--) if (p[i][u] != p[i][v]) u = p[i][u], v = p[i][v];
        return p[0][u];
    }
    int dis(int u, int v) {return h[u] + h[v] - 2 * h[lca(u, v)];}
    int query(int u, int v)
    {
        return dis(u, v)/2 + 1;
    }
    void init()
    {
        memset(p, -1, sizeof p);
        memset(in, 0, sizeof in);
        memset(low, 0, sizeof low);
        memset(h, 0, sizeof h);
        st.clear();
        for (int i=1; i<=n; i++) adj2[i].clear();
        node = n;
        DFS();
        DFS2();
        for (int i=1; (1<<i) <= node; i++)
            for (int j=1; j<=node; j++)
                if (p[i-1][j] != -1) p[i][j] = p[i-1][p[i-1][j]];
    }

} bctree;