# Data Structures
## Prefix Sum

+ Initialize with a 0-based vector
+ PS.sum : get sum of numbers from index l to r in vector

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

+ Description: Segment tree with ability to add or set values of large intervals, and compute max of intervals.
+ Initialize with a 0-based vector
+ ST.update : update one element of vector to any value
+ ST.query : get range queries such as max query, sum query
+ Time complexity : O(log n)
+ 
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
# Lazy Propagation

+ Description: Segment tree with ability to add or set values of large intervals, and compute max of intervals.
+ Initialize with a 0-based vector
+ ST.update : update one element of vector to any value
+ ST.query : get range queries such as max query, sum query
+ Time complexity : O(log n)

```cpp
struct ST_w_lz/*0-based*/{
    using T = int;
    vector<T> st, org, lz;
    T orz, cns = INT_MIN;
    /*Replace with any associative function*/
    T cmp(T a, T b){ return a > b ? a : b; }
    /*Ex : T cmp(T a, T b){ return a + b; }*/
    ST(vector<T> v){
        org.push_back(0);
        for(auto it : v) org.push_back(it);
        orz = v.size();
        st.resize(4 * v.size());
        lz.resize(4 * v.size());
        build(1,1,v.size());
    }
    void build(T id, T l, T r){
        if (l == r){ st[id] = org[l]; return; }
        T m = (l + r) >> 1;
        build(id*2, l, m);
        build(id*2+1, m+1, r);
        st[id] = cmp(st[id*2], st[id*2+1]);
    }
    void push_down(T id, T l, T r){
        if (!lz[id]) return;
        st[id] += lz[id];
        if (l != r){
            lz[id*2] += lz[id];
            lz[id*2+1] += lz[id];
        }
        lz[id] = 0;
    }
    void update_lz(T u, T v, T val){
        call_update_lz(1, 1, orz, u, v, val);
    }
    void call_update_lz(T id, T l, T r, T u, T v, T val){
        push_down(id, l, r);
        if (l > v || r < u){ return; }
        if (l >= u && r <= v){
            lz[id] += val;
            push_down(id, l, r);
            return;
        }
        T m = (l + r) >> 1;
        call_update_lz(id*2, l, m, u, v, val);
        call_update_lz(id*2+1, m+1, r, u, v, val);
        st[id] = cmp(st[id*2], st[id*2+1]);
    }
    T query_lz(T u, T v){
        return call_query_lz(1, 1, orz, u, v);
    }
    T call_query_lz(T id, T l, T r, T u, T v){
        push_down(id, l, r);
        if (l > v || r < u) return cns;
        if (l >= u && r <= v) return st[id];
        T m = (l + r) >> 1;
        return cmp(call_query_lz(id*2, l, m, u, v), call_query_lz(id*2+1, m+1, r, u, v));
    }
};
```

## Segment Tree(Optimized space complexity)

```



```

## Fenwick Tree