# Geometry

## Point
```cpp
const ldb PI = 3.14159265358979;
const ldb eps = 1e-9;
bool eq(ldb a, ldb b){ return fabs(a-b) < eps; }
struct Point{
    ldb x=0, y=0;
    Point(){}
    Point(ldb x, ldb y):x(x), y(y){}
    ldb Edist(const Point o){
        return sqrt(pow((x-o.x),2)+pow((y-o.y),2));
    }
    ldb Mdist(const Point o){
        return abs(x-o.x)+abs(y-o.y);
    }
    Point Mpoint(const Point o){
        return Point((x+o.x)/2, (y+o.y)/2);
    }
    bool operator == (const Point o){
        return eq(x,o.x) & eq(y,o.y);
    }
    friend ostream& operator << (ostream &os, Point a){
        os << "Point( " << a.x << ", " << a.y << " )";
        return os;
    }
};

```
## Vector
```cpp
struct Vector{
    ldb a=0, b=0;
    Vector(){}
    Vector(Point x, Point y):a(y.x-x.x), b(y.y-x.y){}
    Vector(ldb a, ldb b): a(a), b(b){}
    ldb len(){ return sqrt(a*a+b*b); }
    Vector operator + (const Vector o){
        return Vector(a+o.a, b+o.b);}
    Vector operator - (const Vector o){
        return *this + Vector(-o.a, -o.b); }
    ldb dotProduct(const Vector o){
    /*also equal to |u|*|v|*cos */
        return a*o.a+b*o.b; }
    ldb crossProduct(const Vector o){
    /*also equal to |u|*|v|*sin */
    /*also equal to vec(n)*|u|*|v|*sin in 3D space */
        return a*o.b-o.a*b; }
    ldb angle(Vector o){
        return dotProduct(o)/(len()*o.len()); }
    bool operator == (const Vector o){
        return eq(a,o.a)&eq(b,o.b); }
    friend ostream& operator << (ostream &os, Vector a){
        os << "Vector( " << a.a << ", " << a.b << " )";
        return os;
    }
};
```

## Line
```cpp
struct Line{
    ldb a=0, b=0, c=0;
    Line(){}
    Line(ldb a, ldb b, ldb c):a(a), b(b), c(c){}
    Line(Point x, Point y){
        a = y.y - x.y;
        b = x.x - y.x;
        c = -a*x.x-b*x.y;
    }
    Line(Point x, Vector y){
        a = y.a;
        b = y.b;
        c = -y.a*x.x - y.b*x.y;
    }
    ldb distP(const Point x){
        return abs(a*x.x+b*x.y+c)/sqrt(a*a+b*b); }
    ldb have(const Point x){
        return distP(x) == 0; }
    friend ostream& operator << (ostream &os, Line a){
        os << "Line( " << a.a << ", " << a.b << ", " << a.c << " )";
        return os;
    }
};
```
## Ray
+ Same as line but distP have difference
```cpp
ldb distP(ldb xa, ya, xb, yb, xc, yc){
    Point a(xa, ya);
    Point b(xb, yb);
    Point c(xc, yc);
    Vector ba(b, a);
    Vector bc(b, c);
    Line lbc(b, c);
    if (ba.dotProduct(bc) < 0){
        return a.Edist(b);
    }
    return lbc.distP(a);
}
```
## Segment
+ Same as line but distP have difference
```cpp
ldb distP(ldb xa, ya, xb, yb, xc, yc){
    Point a(xa, ya);
    Point b(xb, yb);
    Point c(xc, yc);
    Vector ba(b, a);
    Vector bc(b, c);
    Vector ca(c, a);
    Vector cb(c, b);
    Line lbc(b, c);
    if (ba.dotProduct(bc) && ca.dotProduct(cb)){
        return lbc.distP(a);
    }
    else{
        if (ba.dotProduct(bc) < 0){
            return b.Edist(a);
        }
            return c.Edist(a);
    }
    return 0;
}
```