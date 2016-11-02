
//=================================================================

#include "Builder.h"

//-----------------------------------------------------------------

complex<double> BornForPoint  (Point3D_t r, Point3D_t rj);
complex<double> GreenFunction (Point3D_t r, double c  );
complex<double> PressureI_    (Point3D_t r            );
double          SoundSpeed    (Point3D_t r            );

//-----------------------------------------------------------------

complex<double> BornForPoint (Point3D_t r, Point3D_t rj)
{
    Point3D_t dr = {rj.x - r.x, rj.y - r.y, rj.z - r.z};
    return (INPUT_DATA_PTR->w_ * INPUT_DATA_PTR->w_) * PressureI_ (r) * ((1 / SoundSpeed (r)) * (1 / SoundSpeed (r))) * GreenFunction (dr, c(r));
}

complex<double> GreenFunction (Point3D_t r, double c)
{
    double k = INPUT_DATA_PTR->w_ / SoundSpeed (r);
    if (r.Len() == 0.0) return {};
    else return exp (I_ * r.Len() * k) / (4 * PI_ * r.Len());
}

complex<double> PressureI_ (Point3D_t r)
{
   return  GreenFunction (r, c (r));
}

double SoundSpeed (Point3D_t r)
{
    if (r.x >= INPUT_DATA_PTR->anomalyPos_.x && 
        r.x <= INPUT_DATA_PTR->anomalyPos_.x + INPUT_DATA_PTR->anomalySize_.x &&
        r.y >= INPUT_DATA_PTR->anomalyPos_.y && 
        r.y <= INPUT_DATA_PTR->anomalyPos_.y + INPUT_DATA_PTR->anomalySize_.y &&
        r.x >= INPUT_DATA_PTR->anomalyPos_.z && 
        r.z <= INPUT_DATA_PTR->anomalyPos_.z + INPUT_DATA_PTR->anomalySize_.z)
        return INPUT_DATA_PTR->c_ * INPUT_DATA_PTR->alpha_;
    else
        return INPUT_DATA_PTR->c_;
}


//=================================================================
