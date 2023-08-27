# Data Structures
## Prefix Sum

+ Initialize with a 0-based vector
+ PS.sum : get sum of numbers from index l to r in vector

```cpp
struct PS/*0-based*/{
    using T = int;
    vector<T> p;
    PS(vector<T> a){
        p.resize(a.size() + 1, 0);
        for (T i=1; i<=a.size(); i++){
            p[i] = p[i-1] + a[i];
        }
    }
    T sum(T l, T r){
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
+ ST.update_lz : update all elements in range from l to r of vector to any value
+ ST.query_lz : get range queries such as max query, sum query
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

+ Description: Segment tree with ability to add or set values of large intervals, and compute max of intervals.
+ Initialize with a 0-based vector
+ ST.update : update one element of vector to any value
+ ST.query : get range queries such as max query, sum query
+ Time complexity : O(log n)
+ Space complexity : O(2*n)
```cpp
struct STO{
    using T = int;
    vector<T> st, org;
    T orz, cns = INT_MIN;
    T f(T a, T b){ return a > b ? a : b; }
    STO(vector<T> v){
        org = v;
        orz = org.size();
        st.resize(2*orz);
        for (int i=1; i<=orz; i++){
            update(i, org[i-1]);
        }
    }
    void update(T pos, T val){
        pos--;
        for (st[pos += orz] = val; pos/=2;)
            st[pos] = f(st[pos << 1], st[pos << 1 | 1]);
    }
    T query(T u, T v){
        u--;
        T res = cns;
        for (u += orz, v += orz; u < v; u >>= 1, v >>= 1){
            if (u & 1) res = f(res, st[u++]);
            if (v & 1) res = f(res, st[--v]);
        }
        return res;
    }
};

```

## Fenwick Tree

+ Description: Computes partial sums a[0] + a[1] + ... + a[pos - 1], and updates single elements a[i],
taking the difference between the old and new value.
+ Initialize with a 0-based vector
+ FT.update : update all elements in range from l to r of vector to any value
+ FT.query : get partial sums a[0] + a[1] + ... + a[pos - 1]
+ FT.query_range : get partial sums a[l] + a[1] + ... + a[r]
+ Time complexity : O(log n)

```cpp
struct FT/*0-based*/{
    using T = int;
    vector<T> ft;
    vector<T> org;
    T orz;
    FT(vector<T> v){
        ft.resize(v.size());
        org = v; orz = org.size();
        for (int i=0; i<org.size(); i++){
            update(i, org[i]);
        }
    }
    void update(int pos, int val){
        for (; pos < orz; pos|=pos+1) ft[pos] += val;
    }
    T query(T pos){
        T sum = 0;
        for (; pos > 0; pos&=pos-1){
                sum += ft[pos-1];
        }
        return sum;
    }
    T query_range(T l, T r){
        return query(r) - query(l-1);
    }
};
```