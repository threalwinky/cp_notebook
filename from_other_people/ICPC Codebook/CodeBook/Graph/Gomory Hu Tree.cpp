//0-based

struct PushRelabel {
	struct Edge {
		int dest, back;
		int f, c;
	};
	vector<vector<Edge>> g;
	vector<int> ec;
	vector<Edge*> cur;
	vector<vector<int>> hs; vector<int> H;
	PushRelabel(int n) : g(n), ec(n), cur(n), hs(2*n), H(n) {}

	void addEdge(int s, int t, int cap, int rcap=0) {
		if (s == t) return;
		g[s].push_back({t, (int)g[t].size(), 0, cap});
		g[t].push_back({s, (int)g[s].size()-1, 0, rcap});
	}

	void addFlow(Edge& e, int f) {
		Edge &back = g[e.dest][e.back];
		if (!ec[e.dest] && f) hs[H[e.dest]].push_back(e.dest);
		e.f += f; e.c -= f; ec[e.dest] += f;
		back.f -= f; back.c += f; ec[back.dest] -= f;
	}
	int calc(int s, int t) {
		int v = g.size(); H[s] = v; ec[t] = 1;
		vector<int> co(2*v); co[0] = v-1;
		for(int i = 0; i < v; ++i) cur[i] = g[i].data();
		for (Edge& e : g[s]) addFlow(e, e.c);

		for (int hi = 0;;) {
			while (hs[hi].empty()) if (!hi--) return -ec[s];
			int u = hs[hi].back(); hs[hi].pop_back();
			while (ec[u] > 0)  // discharge u
				if (cur[u] == g[u].data() + g[u].size()) {
					H[u] = 1e9;
					for (Edge& e : g[u]) if (e.c && H[u] > H[e.dest]+1)
						H[u] = H[e.dest]+1, cur[u] = &e;
					if (++co[H[u]], !--co[hi] && hi < v)
						for(int i = 0; i < v; ++i) if (hi < H[i] && H[i] < v)
							--co[H[i]], H[i] = v + 1;
					hi = H[u];
				} else if (cur[u]->c && H[u] == H[cur[u]->dest]+1)
					addFlow(*cur[u], min(ec[u], cur[u]->c));
				else ++cur[u];
		}
	}
	bool leftOfMinCut(int a) { return H[a] >= (int)g.size(); }
};

vector<array<int, 3>> gomoryHu(int N, vector<array<int, 3>> ed) {
	vector<array<int, 3>> tree;
	vector<int> par(N);
	for(int i = 0; i < N; ++i) {
		PushRelabel D(N); // Dinic also works
		for (array<int, 3> t : ed) D.addEdge(t[0], t[1], t[2], t[2]);

        int flow = D.calc(i, par[i]);
		if (flow != -1) tree.push_back({i, par[i], flow});
		for(int j = i + 1; j < N; ++j)
			if (par[j] == par[i] && D.leftOfMinCut(j)) par[j] = i;
	}
	return tree;
}