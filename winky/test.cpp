#include                    <bits/stdc++.h>
using                       namespace std;
//#define int               long long
#define fi                  first
#define se                  second
#define pb(x)               push_back(x)
#define endl                cout<<"\n";
typedef pair<int, int>      ii;
typedef vector<int>         vi;
typedef long double         ldb;
const string F =            "sample";
const string IF =           F + ".inp";
const string OF =           F + ".out";
const ldb PI =              3.14159265358979;
const int maxN =            1e6;
const int mod  =            1e9 + 7;
void opf(bool c){           if (c == 1){
freopen(IF.c_str(),         "r", stdin);
freopen(OF.c_str(),         "w", stdout);}}
//------------------------------------------------------------------------------------
//Code here
vector<int> prefix_func(string s){
    int n = s.length();
    vector<int> pi(n);
    for (int i=1; i<s.length(); i++){
        int j = pi[i-1];
        while (j>0 && s[i] != s[j]) j--;
        if (s[i] == s[j]) j++;
        pi[i] = j;
    }
    return pi;
}
//------------------------------------------------------------------------------------
signed main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    opf(0);
    string n, m;
    cin >> n >> m;
    string p = m+'\0'+n;
    vector<int> pi = prefix_func(p);
    for (auto it : pi) cout << it << " ";
    for (int i=m.length()+1; i<p.length(); i++){
        cout << pi[i] << " ";
    }
}
