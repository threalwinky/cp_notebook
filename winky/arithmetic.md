# Arithmetic
## Modulo
```cpp
using ll = long long;
const int mod = 1e9 + 7;
struct T{
    ll x;
    T (ll x):x(x){}
    T operator + (const T o){ return (x%mod+o.x%mod)%mod; }
    T operator - (const T o){ return (x%mod-o.x%mod+mod)%mod; }
    T operator * (const T o){ return (x%mod*o.x%mod)%mod; }
    T operator / (const T o){ return x/o.x; }
};
T pow(T a, T b){
    a.x%=mod;
    T res(1);
    while (b.x > 0){
        if (b.x & 1) res=res*a;
        a=a*a;
        b.x >>= 1;
    }
    return res;
}
ll modpow(ll a, ll b){
    if (!b) return 1;
    ll tmp = modpow(a%mod, b/2);
    if (b&1) return (tmp%mod*tmp%mod*a%mod)%mod;
    return (tmp%mod*tmp%mod)%mod;
}
ll modpownr(ll a, ll b){
    a %= mod;
    long long res = 1;
    while (b > 0){
        if (b & 1) res = res * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return res;
}
```

## Prime number
### The sieve of Eratosthenes
+ Description: Prime sieve for generating all primes smaller than LIM.
```cpp
const int maxP = 5000000;
bool isP[maxP];
vector<int> eratosthenesSieve(int lim){
    for (int i=2; i*i<lim; i++){
        if (!isP[i]){
            for (int j=2*i; j<lim; j+=i){
                isP[j] = 1;
            }
        }
    }
    vector<int> res;
    for (int i=2; i<lim; i++){
        if (!isP[i]) res.push_back(i);
    }
    return res;
}
```
### The sieve of Eratosthenes(using bitset)
```cpp
const int maxP = 5000000;
bitset<maxP> isP;
vector<int> eratosthenesSieve(int lim){
    isP.set();
    for (int i=4; i<lim; i+=2) isP[i] = 0;
    for (int i=3; i*i<lim; i+=2){
        if (isP[i]){
            for (int j=i*i; j<lim; j+=i*2){
                isP[j] = 0;
            }
        }
    }
    vector<int> res;
    for (int i=2; i<lim; i++){
        if (isP[i]) res.push_back(i);
    }
    return res;
}
```