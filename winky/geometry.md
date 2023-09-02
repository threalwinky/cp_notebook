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