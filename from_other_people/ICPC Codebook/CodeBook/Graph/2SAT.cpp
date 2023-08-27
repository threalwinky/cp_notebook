//task : n people, each people have 2 request : + x or - x
//Ask : Is there a way build array m elements that for each people,
//at least one of two request is satisfied. If yes, print it.

void tarjan(int u) {
    st[++top] = u;
    in[u] = low[u] = ++tme;
    was_tarjan[u] = true;
    for (int v : g[u]) {
        if (!was_tarjan[v]) {
            tarjan(v);
            low[u] = min(low[u], low[v]);
        }
        else {
            low[u] = min(low[u], in[v]);
        }
    }

    if (low[u] == in[u]) {
        ++scc_cnt;
        while (st[top] != u) {
            scc_id[st[top]] = scc_cnt;
            in[st[top]] = low[st[top]] = N;
            --top;
        }
        scc_id[st[top--]] = scc_cnt;
        in[u] = low[u] = N;
    }
}

void compress_scc_to_dag() {
    for (int u = 1; u <= m; ++u)
        for (int v : g[u]) if (scc_id[v] != scc_id[u])
            g2[scc_id[u]].emplace_back(scc_id[v]);
}

void dfs(int u) { // toposort
    was[u] = true;
    for (int v : g2[u]) if (!was[v]) dfs(v);

    topo[u] = top--;
}

void toposort_g2() {
    top = scc_cnt;
    fill(was + 1, was + scc_cnt + 1, false);
    for (int i = 1; i <= scc_cnt; ++i) if (!was[i]) dfs(i);
}

int main() {
	//io()
	m = n << 1;
    for (int i = 1; i <= m; ++i) if (!was_tarjan[i]) tarjan(i);

    for (int i = 1; i <= n; ++i) if (scc_id[i] == scc_id[i + n]) {
        cout << "IMPOSSIBLE";
        return 0;
    }

    compress_scc_to_dag();
    toposort_g2();

    for (int i = 1; i <= n; ++i) {
        cout << (topo[scc_id[i]] > topo[scc_id[i + n]] ? '+' : '-') << ' ';
    }
}