#ifndef DATA_LOADER_INCLUDED
#define DATA_LOADER_INCLUDED

#include "includes.h"

template <class T>
struct Point3D_t_
{
    T x;
    T y;
    T z;
    
    double Len ();
};

typedef Point3D_t_<double> Point3D_t;

struct InputData_t
{
    Point3D_t sourcePos_;
    double w_;
    double c_;
    double alpha_;
    Point3D_t anomalyPos_;
    Point3D_t anomalySize_;
    double V_;
    std::vector<Point3D_t_<double> > recievers_;
    
    void LoadData ();
};

#endif
