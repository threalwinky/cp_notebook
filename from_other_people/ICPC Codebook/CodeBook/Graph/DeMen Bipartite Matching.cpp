vector <int> a[N];
int mr[N], cttme;
bool ml[N];
char f[N];

bool dfs(int u){
    if (f[u] == cttme) return false;
    f[u] = cttme;
    for(int i : a[u]){
        if (!mr[i] || dfs(mr[i])){
            mr[i] = u;
            return true;
        }
    }
    return false;
}

int maximum_matching(){
    int cnt = 0;
    for(bool run = true; run;){
        cttme++;
        run = false;
        for(int i = 1; i <= n; ++i){
            if (ml[i]) continue;
            if (dfs(i)){
                ml[i] = run = true;
                ++cnt;
            }
        }
    }
    return cnt;
}