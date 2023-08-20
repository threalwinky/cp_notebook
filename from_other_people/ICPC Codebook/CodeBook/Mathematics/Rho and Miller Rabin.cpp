mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
vector <int> b;
ll binpower(ll base, ll e, ll mod) {
    ll result = 1;
    base %= mod;
    while (e){
        if (e & 1)result = (ll)result * base % mod;
        base = (ll)base * base % mod;e >>= 1;
    }
    return result;
}
bool check_composite(ll n, ll a, ll d, int s){
    ll x = binpower(a, d, n);
    if (x == 1 or x == n - 1)return false;
    for (int r = 1; r < s; r++) {
        x = (ll)x * x % n;
        if (x == n - 1)return false;
    }return true;
}

bool MillerRabin_checkprime(ll n) {
    if (n < 2)return false;
    int r = 0;
    ll d = n - 1;
    while ((d & 1) == 0) {
        d >>= 1;
        r++;
    }
    for (int a : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
        if (n == a)return true;
        if (check_composite(n, a, d, r))return false;
    }
    return true;
}

int mult(int a, int b, int mod) {
    return (ll)a * b % mod;
}

int F(int x, int c, int mod) {
    return (mult(x, x, mod) + c) % mod;
}
int rho(int n, int x0=2, int c=1) {
    int x = x0;
    int y = x0;
    int g = 1;
    if (n % 2 == 0) return 2;
    while (g == 1) {
        x = F(x, c, n);
        y = F(y, c, n);
        y = F(y, c, n);
        g = gcd(abs(x - y), n);
    }
    return g;
}

void Rho_factorization(int n){
    set <int> s;
    if (n == 1) {return;}
    while (!MillerRabin_checkprime(n)){
        ll k;
        while (1){
            int p = (rng()%(n-2))+2, q = (rng()%(n-1))+1;
            k = rho(n,p,q);
            if (MillerRabin_checkprime(k)) break;
        }
        s.insert(k);
        n/=k;
    }
    s.insert(n);
    fora(i,s) b.pb(i);
}
