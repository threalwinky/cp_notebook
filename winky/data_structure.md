# Data Structure
## Prefix Sum

```cpp
struct PS{
    vector<int> p;
    PS(vector<int> a){
        p[0] = 0;
        for (int i=1; i<=a.size(); i++){
            p[i] = p[i-1] + a[i];
        }
    }
    int sum(int l, int r){
        return f[r] - f[l-1];
    }
};
```

## Segment Tree