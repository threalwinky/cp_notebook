# Geometry
## Point
```cpp
struct Point{
    ldb x, y;
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
        return x == o.x && y == o.y;
    }
};
```