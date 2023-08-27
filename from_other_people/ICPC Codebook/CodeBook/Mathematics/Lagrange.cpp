long long n,k,a[maxn],fac[maxn],ifac[maxn],prf[maxn],suf[maxn];
void build()
{
    fac[0] = ifac[0] = 1;
    for (int i = 1; i < maxn; i++)
    {
        fac[i] = fac[i - 1] * i % MOD;
        ifac[i] = binPow(fac[i], MOD - 2);
    }
}
    //Calculate P(x) of degree k - 1, k values form 1 to k
    //P(i) = a[i]
long long calc(long long x, long long k)
{
        if(x <= k)
        {
            return a[x];
        }
        prf[0] = suf[k + 1] = 1;
        for (long long i = 1; i <= k; i++) {
            prf[i] = prf[i - 1] * (x - i + MOD) % MOD;
        }
        for (long long i = k; i >= 1; i--) {
            suf[i] = suf[i + 1] * (x - i + MOD) % MOD;
        }
        long long res = 0;
        for (long long i = 1; i <= k; i++) {
            if (!((k - i) & 1)) {
                res = (res +  prf[i - 1] * suf[i + 1] % MOD
                        * ifac[i - 1] % MOD * ifac[k - i] % MOD * a[i]) % MOD;
            }
            else {
                res = (res -  prf[i - 1] * suf[i + 1] % MOD
                        * ifac[i - 1] % MOD * ifac[k - i] % MOD * a[i] % MOD + MOD) % MOD;
            }
        }
        return res;
}
void solve()
{
    cin >> n >> k;
    build();
    for(int i = 1; i <= k+2; i++)
        a[i] = (a[i-1]+binPow(i,k))%MOD;
    cout << calc(n,k+2);
}

