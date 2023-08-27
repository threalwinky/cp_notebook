struct node{
    node *l, *r;
    int sz, pri;
    int val, sum;
    bool rev;
 
    node(){}
    node(int c){
        l = r = nullptr;
        sz = 1;
        rev = false;
        pri = rng();
        val = c;
        sum = c;
    }
};
 
void push(node *&treap){
    if (treap == nullptr) return;
    if (!(treap -> rev)) return;
    swap(treap -> l, treap -> r);
    if (treap -> l){
        (treap -> l) -> rev ^= 1;
    }
    if (treap -> r){
        (treap -> r) -> rev ^= 1;
    }
    treap -> rev = false;
}
 
int get_sz(node *treap){
    if (treap == nullptr) return 0;
    else return treap -> sz;
}
 
int get_sum(node *treap){
    if (treap == nullptr) return 0;
    else return treap -> sum;
}
 
void split(node *treap, node *&l, node *&r, int k){ //[1, k] [k + 1, sz]
    if (treap == nullptr){
        l = r = nullptr;
        return;
    }
    push(treap);
    if (get_sz(treap -> l) < k){
        split(treap -> r, treap -> r, r, k - get_sz(treap -> l) - 1);
        l = treap;
    } else {
        split(treap -> l, l, treap -> l, k);
        r = treap;
    }
    treap -> sz = get_sz(treap -> l) + get_sz(treap -> r) + 1;
    treap -> sum = get_sum(treap -> l) + get_sum(treap -> r) + treap -> val;
}
 
void merge(node *&treap, node *l, node *r){
    if (l == nullptr){
        treap = r;
        return;
    }
    if (r == nullptr){
        treap = l;
        return;
    }
    push(l); push(r);
    if (l -> pri < r -> pri){
        merge(l -> r, l -> r, r);
        treap = l;
    } else {
        merge(r -> l, l, r -> l);
        treap = r;
    }
    treap -> sz = get_sz(treap -> l) + get_sz(treap -> r) + 1;
    treap -> sum = get_sum(treap -> l) + get_sum(treap -> r) + treap -> val;
}
 
void print(node *treap){
    if (treap == nullptr) return;
    push(treap);
    print(treap -> l);
    cout << treap -> val;
    print(treap -> r);
}