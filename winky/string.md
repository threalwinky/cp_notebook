# String
## Knuth-Morris-Pratt
```cpp
vector<int> prefix_func(string s){
    int n = s.length();
    vector<int> pi(n);
    for (int i=1; i<n; i++){
        int j = pi[i-1];
        while (j>0 && s[i] != s[j]) j=pi[j-1];
        pi[i] = j + (s[i] == s[j]);
    }
    return pi;
}
vector<int> match(string org, string pat){
    string t = pat + '\0' + org;
    int tsz = t.length(), psz = pat.length();
    vector<int> pi = prefix_func(t), res;
    for (int i=psz; i<tsz; i++){
        if (pi[i] == psz){
            res.push_back(i - 2*psz + 1);
        }
    }
    return res;
}
vector<string> split(string org, string pat){
    string p = pat + '\0' + org, t;
    int n = p.length(), psz = pat.length();
    vector<int> pf = prefix_func(p);
    vector<string> res;
    int r = psz + 1;
    for (int i=psz+1; i<=n; i++){
        if (pf[i] == psz || i==n){
            t.clear();
            for (int j=r; j<=i-psz; j++){
                t+=p[j];
            }
            res.push_back(t);
            r = i + 1;
        }
    }
    return res;
}
```