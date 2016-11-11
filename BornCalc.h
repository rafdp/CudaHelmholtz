#ifndef BORN_CALC_INCLUDED
#define BORN_CALC_INCLUDED
//=================================================================


// This file is not needed anymore, about to delete


#include "includes.h"

//-----------------------------------------------------------------

const complex<double> I_  = complex<double> (0.0, 1.0);
const double         PI_ = 3.14159      ;

//-----------------------------------------------------------------


complex<double> BornForPoint  (Point3D_t rEmitter, Point3D_t rReceiver);

complex<double> GreenFunction (Point3D_t rEmitter, Point3D_t rReceiver);

complex<double> PressureI_    (Point3D_t rReceiver);

double          SoundSpeed    (Point3D_t rEmitter);

double          d_Slowness    (Point3D_t r);


//=================================================================

#endif
