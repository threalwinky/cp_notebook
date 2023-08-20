void build_y(int vx, int lx, int rx, int vy, int ly, int ry) {
    if (ly == ry) {
        if (lx == rx)
            segtree[vx][vy] = 0;
        else
            segtree[vx][vy] = max(segtree[vx*2][vy],segtree[vx*2+1][vy]);
    } else {
        int my = (ly + ry) / 2;
        build_y(vx, lx, rx, vy*2, ly, my);
        build_y(vx, lx, rx, vy*2+1, my+1, ry);
        segtree[vx][vy] = max(segtree[vx][vy*2],segtree[vx][vy*2+1]);
    }
}

void build_x(int vx, int lx, int rx) {
    if (lx != rx) {
        int mx = (lx + rx) / 2;
        build_x(vx*2, lx, mx);
        build_x(vx*2+1, mx+1, rx);
    }
    build_y(vx, lx, rx, 1, 0, m - 1);
}
int query_y(int vx, int vy, int tly, int try_, int ly, int ry) {
    if (ly > ry)
        return 0;
    if (ly == tly && try_ == ry)
        return segtree[vx][vy];
    int tmy = (tly + try_) / 2;
    return max(query_y(vx, vy*2, tly, tmy, ly, min(ry, tmy))
        ,query_y(vx, vy*2+1, tmy+1, try_, max(ly, tmy+1), ry));
}

int query_x(int vx, int tlx, int trx, int lx, int rx, int ly, int ry) {
    if (lx > rx)
        return 0;
    if (lx == tlx && trx == rx)
        return query_y(vx, 1, 0, m-1, ly, ry);
    int tmx = (tlx + trx) / 2;
    return max(query_x(vx*2, tlx, tmx, lx, min(rx, tmx), ly, ry)
         ,query_x(vx*2+1, tmx+1, trx, max(lx, tmx+1), rx, ly, ry));
}
void update_y(int vx, int lx, int rx, int vy, int ly, int ry, int x, int y, int new_val) {
    if (ly == ry) {
        if (lx == rx)
            segtree[vx][vy] = new_val;
        else
            segtree[vx][vy] = max(segtree[vx*2][vy], segtree[vx*2+1][vy]);
    } else {
        int my = (ly + ry) / 2;
        if (y <= my)
            update_y(vx, lx, rx, vy*2, ly, my, x, y, new_val);
        else
            update_y(vx, lx, rx, vy*2+1, my+1, ry, x, y, new_val);
        segtree[vx][vy] = max(segtree[vx][vy*2],segtree[vx][vy*2+1]);
    }
}

void update_x(int vx, int lx, int rx, int x, int y, int new_val) {
    if (lx != rx) {
        int mx = (lx + rx) / 2;
        if (x <= mx)
            update_x(vx*2, lx, mx, x, y, new_val);
        else
            update_x(vx*2+1, mx+1, rx, x, y, new_val);
    }
    update_y(vx, lx, rx, 1, 0, m-1, x, y, new_val);
}