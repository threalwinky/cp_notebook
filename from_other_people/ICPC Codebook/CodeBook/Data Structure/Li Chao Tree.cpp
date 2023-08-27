const int N = 2e5 + 5;

struct line {
    ll a, b;
    ll val(ll x) { return a * x + b; }
};

line seg[4 * N];

void upd(int id, int l, int r, line lin) {
    if (l == r) {
        if (lin.val(l) > seg[id].val(l))
            seg[id] = lin;
        return;
    }
    int mid = (l + r) >> 1;
    if (seg[id].a > lin.a)
        swap(seg[id], lin);
    // sap xep lai do doc duong thang de tong quat hoa bai toan
    if (lin.val(mid) > seg[id].val(mid)) {
        swap(seg[id], lin);
        upd(id << 1, l, mid, lin);
    } else
        upd(id << 1 | 1, mid + 1, r, lin);
}

void upd_range(int id, int l, int r, int u, int v, line lin) {
    if (l > v || r < u)
        return;
    if (l >= u && r <= v) {
        upd(id, l, r, lin);
        return;
    }
    int mid = (l + r) >> 1;
    upd_range(id << 1, l, mid, u, v, lin);
    upd_range(id << 1 | 1, mid + 1, r, u, v, lin);
}

ll get(int id, int l, int r, int pos) {
    if (l == pos && l == r) {
        return seg[id].val(pos);
    }
    int mid = (l + r) >> 1;
    if (mid >= pos)
        return max(seg[id].val(pos), get(id << 1, l, mid, pos));
    else
        return max(seg[id].val(pos), get(id << 1 | 1, mid + 1, r, pos));
}