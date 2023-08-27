/*
Author : DeMen100ns (a.k.a Vo Khac Trieu)
School : VNU-HCM High school for the Gifted
fuck you adhoc
*/

#include <bits/stdc++.h>
#define int long long
//#define endl '\n'

using namespace std;

const int N = 1e3 + 5;
const long long INF = 1e18 + 7;
const int MAXA = 1e9;
const int B = sqrt(N) + 5;
const string ch[20] = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"};
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

int randint(int l, int r){
    return uniform_int_distribution <int> (l, r) (rng);
}

double randdouble(double l, double r){
    return uniform_real_distribution <double> (l, r) (rng);
}

inline double tempFunc(int prevScore, int newScore, double temp) {
  if (newScore < prevScore) {
    return 2.0;
  }
  return exp(1.0 * -abs(prevScore - newScore) / temp);
}

pair<int, int> f(vector <int> p){
    //sth
}

vector <int> neighbour(vector <int> p, int tmp){
    //sth
}

void solve(int t)
{
    //sth

    double tme = 0, last = 0;
    double tme_per_test = 15;
    for(int tr = 1; tr <= 5; ++tr){
        last = tme;

        tme += tme_per_test;
        double T_start = 1e7;
        double T_end = 10;

        //sort(p.begin(), p.end(), cmp);

        shuffle(p.begin(), p.end(), rng);
        int tmp = m;

        int ct = 0;
        double T = T_start;
        for(int ite = 0;; ite = (ite + 1) & 128, ct++){
            if (!ite){
                double tme_now = 1.0 * clock() / CLOCKS_PER_SEC; 
                if (tme_now > tme) break;

                double tme_happen = tme_now - last;
                double frac = tme_happen / tme_per_test;

                T = T_start * pow(T_end / T_start, frac);
            }
            vector <int> p_try = neighbour(p, tmp);
            pair<int, int> fp = f(p), fptry = f(p_try);

            /*{cout << f(p_try) << endl; for(int i = 1; i <= m; ++i) cout << p_try[i];
            cout << endl;}*/

            if (tempFunc(fp.first, fptry.first, T) >= randdouble(0.0, 1.0)){
                p = p_try;
                tmp = fptry.second;
            }

            if (fp.first < f(p_opt).first) p_opt = p;
        }

        cout << "? " << 1.0 * clock() / CLOCKS_PER_SEC << " " << f(p).first << endl;
    }
}

signed main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    solve(2);
}