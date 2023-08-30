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

## Euclidean algorithm
```cpp
using ll = long long;
ll gcd(ll a, ll b){ return b?gcd(b,a%b):a; }
inline ll lcm(ll a, ll b){ return (a*b)/gcd(a, b); }
ll dio(ll a, ll b, ll &x, ll &y){
    if (b == 0){
        x = 1;
        y = 0;
        return a;
    }
    ll x1, y1;
    ll d = dio(b, a%b, x1, y1);
    x = y1;
    y = x1 - y1*(a/b);
    return d;
}
```

## Prime number

### Normal check (O($\sqrt{n}$))
```cpp
bool isP(int n){
    if (n == 2 || n == 3) return 1;
    if (n < 3 || n % 2 == 0 || n % 3 == 0) return 0;
    for (int i=5; i*i <= n; i++){
        if (n % i ==0 || n % (i + 2) == 0) return 0;
    }
    return 1;
}

```
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
### Fermat's little theorem
```cpp
using ll = unsigned long long;
ll mulMod(ll a, ll b, ll m){
    a%=m;
    ll res = 0;
    while (b > 0){
        if (b & 1) res = (res%m + a%m) % m;
        a = (a%m + a%m) % m;
        b >>= 1;
    }
    return res%m;
}

ll powMod(ll a, ll b, ll m){
    a%=m;
    ll res = 1;
    while (b > 0){
        if (b & 1) res = mulMod(res,a,m);
        a = mulMod(a,a,m);
        b >>= 1;
    }
    return res;
}

bool millerRabin(ll n){
    const ll psz = 12, p[]={2,3,5,7,11,13,17,19,23,29,31,37};
    for (int i=0; i<psz; i++){
        if (n % p[i] == 0) return n == p[i];
    }
    if (n < p[psz - 1]) return 0;
    ll res = 1, s = 0, t;
    for (t = n-1; ~t&1; t>>=1, s++);
    for (ll i =0; i<psz && res; i++){
        ll a = powMod(p[i], t, n);
        if (a != 1){
            for (int b=s; b-- && (res = a + 1 != n);)
                a = mulMod(a, a, n);
            res = !res;
        }
    }
    return res;
}
```