#ifndef DATA_LOADER_INCLUDED
#define DATA_LOADER_INCLUDED

#include "includes.h"

struct Point3D_t
{
    int x;
    int y;
    int z;
    
    double Len ();
};

struct InputData_t
{
    Point3D_t size_;
    Point3D_t sourcePos_;
    double w_;
    double c_;
    double alpha_;
    Point3D_t anomalyPos_;
    Point3D_t anomalySize_;
    std::vector<Point3D_t> receivers_;
    
    void LoadData ();
};

#endif
