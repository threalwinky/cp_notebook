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

## Modular Inverse
```cpp
/*using Extended Euclidean*/

ll gcd(ll a, ll b, ll &x, ll &y){
    if (b == 0){
        x = 1;
        y = 0;
        return a;
    }
    ll x1, y1;
    ll d = gcd(b, a%b, x1, y1);
    x = y1;
    y = x1 - (a/b)*x;
    return d;
}

ll mod_inv(ll a, ll m){
    ll x, y;
    ll g = gcd(a, m, x, y);
    assert(g == 1);
    return (x%m + m)%m;
}

/*using fast power*/
ll supow(ll a, ll b, ll m){
    ll res = 1;
    while (b > 0){
        if (b & 1) res = (res%m*a%m)%m;
        a = (a%m*a%m)%m;
        b >>= 1;
    }
    return res%m;
}
/*replace m-2 with phi(m) - 1 if m is not a prime number*/
ll mod_inv(ll a, ll m){
    return supow(a, m-2, m);
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
        if (!iEuclideansP[i]){
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
### Miller-rabin algorithm
```cpp
ll supow(ll a, ll b, ll m){
    a %= m;
    ll res = 1;
    while (b != 0){
        if (b & 1) res = (res%m*a%m)%m;
        a = (a%m*a%m)%m;
        b >>= 1;
    }
    return res;
}

ll test(ll s, ll d, ll n, ll a){
    if (n == a) return 1;
    ll p = supow(a, d, n);
    if (p == 1) return 1;
    for (;s>0;s--){
        if (p == n-1) return 1;
        p = (p%n*p%n)%n;
    }
    return 0;
}

ll prime(ll n){
    if (n < 2) return 0;
    if (!(n&1)) return n == 2;
    ll s = __builtin_ctz(n-1);
    ll d = (n-1) >> s;
    return test(s, d, n, 2) && test(s, d, n, 3);
}
```

## Matrix
```cpp
#define vec vector
#define sz(x) x.size()
using ll = long long;
const int md = 111539786;
struct Matrix{
    vec<vec<ll>> org;
    int n, m;
    Matrix(){}
    Matrix(vec<vec<ll>> v){org=v;n=sz(v);m=sz(v[0]);}
    Matrix(int _n, int _m){org.assign(n=_n,vec<ll>(m=_m));}
    Matrix unit(int s){
        Matrix mat(s, s);
        for (int i=0; i<s; i++){ mat.org[i][i]=1; }
        return mat;
    }
    Matrix operator * (const Matrix o){
        assert(m == o.n);
        Matrix mat(n, o.m);
        for (int i=0; i<n; i++){
            for (int j=0; j<o.m; j++){
                for (int k=0; k<m; k++){
                    mat.org[i][j] += (org[i][k]%md*o.org[k][j]%md)%md;
                }
            }
        }
        return mat;
    }
    Matrix operator ^ (int b){
        if (!b) return unit(n);
        assert(n == m);
        Matrix tmp = org;
        Matrix res = unit(n);
        while (b > 0){
            if (b & 1) res = res * tmp;
            tmp = tmp * tmp;
            b >>= 1;
        }
        return res;
    }
    void print(){
        for (auto x : org){
            for (auto y : x){
                cout << y << ' ';
            }
            cout << '\n';
        }
    }
};
```

## Euler's totient function
```cpp
int phi(int n){
    int res = n;
    for (int i=2; i<=sqrt(n); i++){
        if (n % i == 0){
            while (n % i == 0){
                n /= i;
            }
            res -= res/i;
        }
    }
    if (n != 1) res -= res/n;
    return res;
}

/*using the sieve of eratosthenes*/
int f[N];
void sieve(){
    for (int i=1; i<=N; i++) f[i] = i/(i%2?1:2);
    for (int i=3; i<=N; i++){
        if (f[i] == i){
            for (int j=i; j<=N; j+=i){
                f[j] = f[j]/i*(i-1);
            }
        }
    }
}
```