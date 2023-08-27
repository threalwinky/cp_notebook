#include<bits/stdc++.h>
using namespace std;

#define int long long

const int N = 2e5 + 5;

struct ST {
  struct Int { // arithmetic progression a, a + d, a + 2 * d, ...
    int a = 0, d = 0;
    Int() {};
    Int(int _a, int _d) {a = _a, d = _d;}
    int at(int n) {return (a + (n - 1) * d);}
    Int shift(int n) {return Int((a + (n - 1) * d), d);}
    int get_sum(int n) {return (((a * 2) + (n - 1) * d) * n) / 2;}
    const Int operator + (const Int &b) const {
      return Int(a + b.a, d + b.d);
    }
  };
  int t[N << 2];
  Int lazy[N << 2];
  ST() {
    memset(t, 0, sizeof t);
    memset(lazy, 0, sizeof lazy);
  }
  void push(int n, int b, int e) {
    if (lazy[n].a == 0 && lazy[n].d == 0) return;
    t[n] += lazy[n].get_sum(e - b + 1);
    if (b != e) {
      lazy[n << 1] = lazy[n << 1] + lazy[n];
      lazy[n << 1 | 1] = lazy[n << 1 | 1] + lazy[n].shift(((b + e) >> 1) + 2 - b);
    }
    lazy[n] = Int(0, 0);
  }
  void build(int n, int b, int e) {
    lazy[n] = Int(0, 0);
    if (b == e) {
      t[n] = 0;
      return;
    }
    int m = b + e >> 1, l = n << 1, r = n << 1 | 1;
    build(l, b, m);
    build(r, m + 1, e);
    t[n] = t[l] + t[r];
  }
  void upd(int n, int b, int e, int i, int j, pair<int, int> v) {
    push(n, b, e);
    if (b > j || e < i) return;
    if (i <= b && e <= j) {
      Int temp(v.first, v.second);
      lazy[n] = lazy[n] + temp.shift(b - i + 1);
      push(n, b, e);
      return;
    }
    int m = b + e >> 1, l = n << 1, r = n << 1 | 1;
    upd(l, b, m, i, j, v);
    upd(r, m + 1, e, i, j, v);
    t[n] = t[l] + t[r];
  }
  int query(int n, int b, int e, int i, int j) {
    push(n, b, e);
    if (e < i || b > j) return 0;
    if (i <= b && e <= j) return t[n];
    int m = b + e >> 1, l = n << 1, r = n << 1 | 1;
    return query(l, b, m, i, j) + query(r, m + 1, e, i, j);
  }
};

int n;

ST t;
int32_t main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  int q; cin >> n >> q;
  vector<int> a(n + 1);
  for (int i = 1; i <= n; i++) {
    cin >> a[i];
    t.upd(1, 1, n, i, i, {a[i], 0});
  }

  while (q--){
    int type, a, b; cin >> type >> a >> b;
    if (type == 1){
        t.upd(1, 1, n, a, b, {1, 1});
    } else {
        cout << t.query(1, 1, n, a, b) << endl;
    }
  }
  
  return 0;
}
