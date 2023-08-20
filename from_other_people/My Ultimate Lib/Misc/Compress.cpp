vector <int> v;

void cmp(){
    for(int i = 1; i <= n; ++i){
        v.push_back(a[i]);
    }
    v.push_back(-1);
    sort(v.begin(), v.end());
    v.resize(unique(v.begin(), v.end()) - v.begin());

    for(int i = 1; i <= n; ++i){
        a[i] = lower_bound(v.begin(), v.end(), a[i]) - v.begin();
    }
}

int decmp(int p){
    return v[p];
}