ii ExEuclid(int a, int b){
    int x0 = 1, y0 = 0;
    int x1 = 0, y1 = 1;
    int x2, y2;
    while (b){
        int q = a / b;
        int r = a % b;
        a = b; b = r;
        x2 = x0 - q * x1;
        y2 = y0 - q * y1;
        x0 = x1; y0 = y1;
        x1 = x2; y1 = y2;
    }
    return {x0, y0};
}