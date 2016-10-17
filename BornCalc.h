
//=================================================================

#include "includes.h"

//-----------------------------------------------------------------

const complex<double> I_  = (0.0, 1.0);
const double         PI_ = 3.14159      ;

//-----------------------------------------------------------------

complex<double> BornForPoint  (Point3D_t r, Point3D_t rj);
complex<double> GreenFunction (Point3D_t r, double c  );
complex<double> PressureI_    (Point3D_t r            );
double         c             (Point3D_t r            );

//-----------------------------------------------------------------

complex<double> BornForPoint (Point3D_t r, Point3D_t rj);

complex<double> GreenFunction (Point3D_t r, double c);

complex<double> PressureI_ (Point3D_t r);

double c (Point3D_t r);


//=================================================================
