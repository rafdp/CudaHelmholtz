#ifndef DATA_LOADER_INCLUDED
#define DATA_LOADER_INCLUDED

#include "includes.h"

struct Point3D_t
{
    double x;
    double y;
    double z;
    
    double Len ();
};

Point3D_t ToPhysical (Point3D_t p);
Point3D_t ToDiscrete (Point3D_t p);

struct InputData_t
{
    Point3D_t sourcePos_;
    double w_;
    double c_;
    double alpha_;
    Point3D_t anomalyPos_;
    Point3D_t anomalySize_;
    Point3D_t block_size_;
    std::vector<Point3D_t> recievers_;
    
    void LoadData ();
};

#endif
