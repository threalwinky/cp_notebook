string s;
vector <int> p(400007), c(400007), lcp(400007);
int n,k;

void build(int n){
    vector<pair<int,int>> a(n);
    for(int i = 0; i < n; ++i) a[i] = {s[i], i};
    sort(a.begin(), a.end());
    for(int i = 0; i < n; ++i) p[i] = a[i].y;
    c[p[0]] = 0;
    for(int i = 1; i < n; ++i){
        c[p[i]] = c[p[i-1]];
        if (a[i].x != a[i - 1].x) c[p[i]] += 1;
    }
    k = 0;
    while ((1 << k) < n){
        vector<pair<pair<int,int>,int>> a(n);
        for(int i = 0; i < n; ++i){
            a[i] = {{c[i], c[(i + (1 << k)) % n]}, i};
        }
        //Radix sort
        vector <int> cnt(n);
        for(auto i : a){
            cnt[i.x.y]++;
        }
        vector <pair<pair<int,int>,int>> b(n);
        vector <int> pos(n);
        pos[0] = 0;
        for(int i = 1; i < n; ++i) pos[i] = pos[i-1] + cnt[i-1];
        for(auto i : a){
            b[pos[i.x.y]] = i;
            pos[i.x.y]++;
        }
        a=b;
        ////////////////////////////////////
        vector <int> cnt2(n);
        for(auto i : a){
            cnt2[i.x.x]++;
        }
        vector <pair<pair<int,int>,int>> f(n);
        vector <int> pos2(n);
        pos2[0] = 0;
        for(int i = 1; i < n; ++i) pos2[i] = pos2[i-1] + cnt2[i-1];
        for(auto i : a){
            f[pos2[i.x.x]] = i;
            pos2[i.x.x]++;
        }
        a = f;
        //////////////
        for(int i = 0; i < n; ++i) p[i] = a[i].y;
        c[p[0]] = 0;
        for(int i = 1; i < n; ++i){
            c[p[i]] = c[p[i-1]];
            if (a[i].x != a[i-1].x) c[p[i]]++;
        }
        k++;
    }
}

void buildlcp(int n){
    k=0;
    for(i = 0; i < n - 1; ++i){
        k=max(0,k - 1);
        lcp[c[i]] = k;
        int s1 = i, s2 = p[c[i]-1];
        for(int j = k; j <= n - i + 1; ++j){
            if (s[s1 + j] == s[s2 + j]){
                k++;
                lcp[c[i]] = k;
            } else break;
        }
    }
}
