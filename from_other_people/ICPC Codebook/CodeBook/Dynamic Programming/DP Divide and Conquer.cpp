void divide(int i, int L, int R, int optL, int optR) {
    if (L > R) return;
    int mid = (L+R) / 2, cut = optL;
    f[i][mid] = INF;
    for (int k = optL; k <= min(mid, optR); k++) {
        long long cur = f[i - 1][k] + Cost(k+1,mid);
        if (f[i][mid] > cur) {
            f[i][mid] = cur;
            cut = k;
        }
    }
    divide(i, L, mid - 1, optL, cut);
    divide(i, mid + 1, R, cut, optR);
}