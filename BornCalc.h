#ifndef BORN_CALC_INCLUDED
#define BORN_CALC_INCLUDED
//=================================================================


// This file is not needed anymore, about to delete


#include "includes.h"

//-----------------------------------------------------------------

const complex<double> I_  = complex<double> (0.0, 1.0);
const double         PI_ = 3.14159      ;

//-----------------------------------------------------------------


complex<double> BornForPoint  (Point3DDevice_t
 rEmitter, Point3DDevice_t
 rReceiver);

complex<double> GreenFunction (Point3DDevice_t
 rEmitter, Point3DDevice_t
 rReceiver);

complex<double> PressureI_    (Point3DDevice_t
 rReceiver);

double          SoundSpeed    (Point3DDevice_t
 rEmitter);

double          d_Slowness    (Point3DDevice_t
 r);


//=================================================================

#endif
