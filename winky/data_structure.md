# Data Structures
## Prefix Sum

```cpp
struct PS/*0-based*/{
    vector<int> p;
    PS(vector<int> a){
        p.resize(a.size() + 1);
        for (int i=1; i<=a.size(); i++){
            p[i] = p[i-1] + a[i];
        }
    }
    int sum(int l, int r){
        return p[r] - p[l-1];
    }
};
```

## Segment Tree

```cpp
struct ST/*0-based*/{
    using T = int;
    vector<T> st, org;
    T orz, cns = INT_MIN;
    /*Replace with any associative function*/
    T cmp(T a, T b){ return a > b ? a : b; }
    /*Ex : T cmp(T a, T b){ return a + b; }*/
    ST(vector<T> v){
        org.push_back(0);
        for(auto it : v) org.push_back(it);
        orz = v.size();
        st.resize(4 * v.size());
        build(1,1,v.size());
    }
    void build(T id, T l, T r){
        if (l == r){ st[id] = org[l]; return; }
        T m = (l + r) >> 1;
        build(id*2, l, m);
        build(id*2+1, m+1, r);
        st[id] = cmp(st[id*2], st[id*2+1]);
    }
    void update(int pos, int val){ call_update(1, 1, orz, pos, val); }
    void call_update(T id, T l, T r, T pos, T val){
        if (l > pos || r < pos){ return; }
        if (l == r){ st[id] = val; return; }
        T m = (l + r) >> 1;
        call_update(id*2, l, m, pos, val);
        call_update(id*2+1, m+1, r, pos, val);
        st[id] = cmp(st[id*2], st[id*2+1]);
    }
    T query(T u, T v){ return call_query(1, 1, orz, u, v); }
    T call_query(T id, T l, T r, T u, T v){
        if (l > v || r < u) return cns;
        if (l >= u && r <= v) return st[id];
        T m = (l + r) >> 1;
        return cmp(call_query(id*2, l, m, u, v), call_query(id*2+1, m+1, r, u, v));
    }
};


```


## Segment Tree(Optimized space complexity)

```



```