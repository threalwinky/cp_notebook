struct Node{
    int mi;
    Node *l, *r;
 
    Node(int x){
        mi = x; l = nullptr; r = nullptr;
    }
    Node(Node *u, Node *v){
        l = u; r = v;
        mi = MAXA;
        if (l) mi = l -> mi;
        if (r) mi = min(mi, r -> mi);
    }
    Node(Node *u){
        mi = u -> mi;
        l = u -> l;
        r = u -> r;
    }
};
 
int a[N], last[N];
pair<int, int> r[N];
vector <int> v;
Node* root[N];
 
Node* build(int l, int r){
    if (l == r) return new Node(MAXA);
 
    int mid = (l + r) >> 1;
    return new Node(build(l, mid), build(mid + 1, r));
}
 
Node* upd(Node* node, int l, int r, int pos, int val){
    if (l == r) return new Node(val);
 
    int mid = (l + r) >> 1;
    if (pos > mid) return new Node(node -> l, upd(node -> r, mid + 1, r, pos, val));
    else return new Node(upd(node -> l, l, mid, pos, val), node -> r);
}
 
int get(Node* node, int l, int r, int u, int v){
    if (node == nullptr || l > v || r < u) return MAXA;
    if (l >= u && r <= v) return node -> mi;
 
    int mid = (l + r) >> 1;
    int v1 = get(node -> l, l, mid, u, v);
    int v2 = get(node -> r, mid + 1, r, u, v);
    return min(v1, v2);
}