bool check(int r, int p)
{
    if (p == 1) return false;
    swap(m, r);
    matrix q; q.build1();
    matrix dv = q;
    q = bpow(q, p + 1);
    for(int i = 0; i < 2; ++i){
        for(int j = 0; j < 2; ++j){
            if (q(i, j) != dv(i, j)) {swap(r, m); return false;}
        }
    }
    swap(r, m);
    return true;
}

int pisano(int m)
{
    int p = 1;
    int tmp = m;
    for (int i : prime_factors(m)) {
        int expo = 0, fact = 1;
        for (; tmp % i == 0; ++expo, fact *= i)
            tmp /= i;

        int q = 1;
        if (i == 2) {
            q = (int) fact / 2 * 3;
        }
        else if (i == 5) {
            q = (int) fact * 4;
        }
        else {
            vector<int> cands;
            if (i % 10 == 1 || i % 10 == 9)
                cands = factorize(i - 1);
            else
                cands = factorize(2 * (i + 1));

            for (int x : cands)
                if (check(i, x)) {
                    q = x;
                    break;
                }
            q = (int) fact / i * q;
        }
        p = lcm(p, q);
    }
    return p;
}
