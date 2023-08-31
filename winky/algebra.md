# Algebra
## Quadratic equation
```cpp
#define vec vector
using ll = long long;
using ldb = long double;
vec<ldb> Qe(ldb a, ldb b, ldb c){
    ldb dt = b*b - 4*a*c;
    vec<ldb> res;
    if (dt < 0) return res;
    if (dt == 0) { res.push_back(-b/(2*a));return res; }
    if (dt > 0){
        res.push_back((-b+sqrt(dt))/(2*a));
        res.push_back((-b-sqrt(dt))/(2*a));
        return res;
    }
}
```
## Cubic equation
![Alt text](./img/image.png)