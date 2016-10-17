
//=================================================================

#include "Builder.h"

//-----------------------------------------------------------------

const complex<double> I_  = 0.0 + 1.0 * I;
const double         PI_ = 3.14159      ;

//-----------------------------------------------------------------

complex<double> BornForPoint  (Point3D_t r, Point3D_t rj);
complex<double> GreenFunction (Point3D_t r, double c  );
complex<double> PressureI_    (Point3D_t r            );
double         c             (Point3D_t r            );

//-----------------------------------------------------------------

complex<double> BornForPoint (Point3D_t r, Point3D_t rj)
{
    Point3D_t dr = {rj.x - r.x, rj.y - r.y, rj.z - r.z};
    return (DATA.w_ * DATA.w_) * PressureI_ (r) * ((1 / c (r)) * (1 / c (r))) * GreenFunction (dr, c(r));
}

complex<double> GreenFunction (Point3D_t r, double c)
{
    double k = DATA.w_ / c (r);
    if (r.Len() == 0.0) return 0.0 + 0.0 * I;
    else return exp (I_ * r.Len() * k) / (4 * PI_ * r.Len());
}

complex<double> PressureI_ (Point3D_t r)
{
   return  GreenFunction (r, c (r));
}

double c (Point3D_t r)
{
    if (r.x >= anomalyPos_.x && r.x <= anomalyPos_.x + anomalySixe_.x &&
        r.y >= anomalyPos_.y && r.y <= anomalyPos_.y + anomalySixe_.y &&
        r.x >= anomalyPos_.z && r.z <= anomalyPos_.z + anomalySixe_.z)
        return DATA.c_ * DATA.alpha_;
    else
        return DATA.c_;
}


//=================================================================
