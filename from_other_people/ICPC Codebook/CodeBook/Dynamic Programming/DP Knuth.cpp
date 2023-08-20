/*
* Complexity: O(N^2)
* f[i][j] = min(f[i][k] + f[k][j] + c[i][j], i < k < j)
* a[i][j] = min(k | i < k < j && f[i][j] = f[i][k] + f[k][j] + c[i][j])
* Sufficient condition: a[i][j - 1] <= a[i][j] <= a[i + 1][j] or
* c[x][z] + c[y][t] <= c[x][t] + c[y][z] (quadrangle inequality) and c[y][z] <= c[x][t] (monotonicity), x <= y <= z <= t
*/

void knuth() {
    for (int i = 1; i <= n; i++) {
        f[i][i] = 0;
        a[i][i] = i;
    }
    for(int len = 1; len <= n-1;len++)
            for(int i = 1; i <= n-len;i++)
        {
            int j = i + len;
            f[i][j] = INF;
            for (int k = a[i][j - 1]; k <= a[i + 1][j]; k++)
            {
                if (f[i][j] > f[i][k-1] + f[k][j] + c[i][j])
                {
                    f[i][j] = f[i][k-1] + f[k][j] + c[i][j];
                    a[i][j] = k;
                }
            }
        }
    cout << f[1][n] << '\n';
}
