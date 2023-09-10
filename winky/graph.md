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
## Dijkstra (Shortest path)
```

```