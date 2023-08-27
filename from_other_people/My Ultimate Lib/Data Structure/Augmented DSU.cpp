/*
Author : DeMen100ns (a.k.a Vo Khac Trieu)
School : VNU-HCM High school for the Gifted
fuck you adhoc
*/

#include <bits/stdc++.h>
#define endl '\n'

using namespace std;

const int N = 2e5 + 5;
const long long INF = 1e18 + 7;
const int MAXA = 1e9;
const int B = sqrt(N) + 5;

int p[N], val[N];

int root(int u){
    if (u == p[u]) return u;
    
    int ru = root(p[u]);
    val[u] += val[p[u]];
    val[u] %= 3;
    p[u] = ru;

    return ru;
}

bool Union(int u, int v, int d){
    int ru = root(u), rv = root(v);
    if (ru == rv && (val[u] - val[v] + 3) % 3 != d) return true;

    p[ru] = rv;
    val[ru] = (d + val[v] - val[u] + 3) % 3;

    return false;
}

void solve()
{
    int n, q; cin >> n >> q;
    iota(p + 1, p + n + 1, 1);
    fill(val + 1, val + n + 1, 0);

    int ct = 0;

    for(int i = 1; i <= q; ++i){
        int type, x, y, d; cin >> type >> x >> y;
        if (max(x, y) > n){
            ++ct;
            continue;
        }

        if (type == 1){
            d = 0;   
        } else d = 2;

        ct += Union(x, y, d);
    }
    cout << ct << endl;
}

signed main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    // freopen("codeforces.inp","r",stdin);
    // freopen("codeforces.out","w",stdout);

    int t = 1; cin >> t;
    while (t--)
    {
        solve();
    }
}