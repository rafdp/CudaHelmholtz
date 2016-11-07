#ifndef BORN_CALC_INCLUDED
#define BORN_CALC_INCLUDED
//=================================================================

#include "includes.h"

//-----------------------------------------------------------------

const complex<double> I_  = complex<double> (0.0, 1.0);
const double         PI_ = 3.14159      ;

//-----------------------------------------------------------------


complex<double> BornForPoint (Point3D_t r, Point3D_t rj);

complex<double> GreenFunction (Point3D_t r);

complex<double> PressureI_ (Point3D_t r);

double SoundSpeed    (Point3D_t r);
double d_Slowness    (Point3D_t r);


//=================================================================

#endif
