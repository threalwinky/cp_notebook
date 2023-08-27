//Given m1, m2, r1, r2, find x :
//- x mod m1 = r1
//- x mod m2 = r2

int cal_crt(int r1, int r2, int m1, int m2)){
    ii ans = ExEuclid(m1, m2);
    bool f = false;
    int g = __gcd(m1, m2);
    if ((r2 - r1) % g) return 1e18; //No solution

    int k = ans.x * ((r2 - r1) / g);
    k %= (m2 / g);
    return (r1 + k * m1) % lc;
    //all_ans = {ans + k*LCM(m1, m2)}
}
