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

Point3D_t ToPhysicalCenter (Point3D_t p);
Point3D_t ToPhysical (Point3D_t p);

Point3D_t ToDiscrete (Point3D_t p);

//---------------------------------------------------------------------
/*
File format: binary
double3             source              x, y, z     in physical space
double              frequency                       in Hz
double3             anomalyBegin        x, y, z     in physical space
double3             anomalySize         x, y, z     in physical space
int3                discretizationSize  x, y, z
double[dx][dy][dz]  ds^2
int                 NReceivers
double3[Nreceivers] receiverCoords      x, y, z     in physical space
*/
//---------------------------------------------------------------------

struct InputData_t
{
    Point3D_t   sourcePos_;
    double      f_;
    double      c_;
    Point3D_t   anomalyPos_;
    Point3D_t   anomalySize_;
    int         discretizationSize_[3];
    int         discreteBlockSize_[3];
    double*     ds2_;
    int         Nreceivers_;
    Point3D_t*  receivers_;
    
    void LoadData ();

    ~InputData_t ();
};

#endif
