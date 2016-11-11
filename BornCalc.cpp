
//=================================================================

#include "Builder.h"

//-----------------------------------------------------------------

complex<double> BornForPoint  (Point3D_t rEmitter, Point3D_t rReceiver);

complex<double> GreenFunction (Point3D_t rEmitter, Point3D_t rReceiver);

complex<double> PressureI_    (Point3D_t rReceiver);

double          SoundSpeed    (Point3D_t rEmitter);

double          d_Slowness    (Point3D_t r);

//-----------------------------------------------------------------

complex<double> BornForPoint (Point3D_t rEmitter, Point3D_t rReceiver)
{
    rEmitter = ToPhysicalCenter (rEmitter);
    rReceiver = ToPhysical (rReceiver);
    
    static const double K = INPUT_DATA_PTR->block_size_.x * INPUT_DATA_PTR->block_size_.y * INPUT_DATA_PTR->block_size_.z * INPUT_DATA_PTR->w_ * INPUT_DATA_PTR->w_;
    
    
    return K * PressureI_ (rEmitter) * d_Slowness (rEmitter) * GreenFunction (rEmitter,rReceiver);
}

complex<double> GreenFunction (Point3D_t rEmitter, Point3D_t rReceiver)
{
    double k = INPUT_DATA_PTR->w_ / SoundSpeed (rEmitter);
    Point3D_t dr = {rReceiver.x - rEmitter.x, rReceiver.y - rEmitter.y, rReceiver.z - rEmitter.z};
    double len = dr.Len ();
    return exp (I_ * len * k) / (4 * PI_ * len);
}

complex<double> PressureI_ (Point3D_t rReceiver)
{
    static const Point3D_t sourcePhysical = ToPhysical (INPUT_DATA_PTR->sourcePos_);
    return GreenFunction (sourcePhysical, rReceiver);
}

double SoundSpeed (Point3D_t r)
{
    r = ToDiscrete (r);
    
    
    if (r.x >= INPUT_DATA_PTR->anomalyPos_.x &&
        r.x <= INPUT_DATA_PTR->anomalyPos_.x + INPUT_DATA_PTR->anomalySize_.x &&
        r.y >= INPUT_DATA_PTR->anomalyPos_.y &&
        r.y <= INPUT_DATA_PTR->anomalyPos_.y + INPUT_DATA_PTR->anomalySize_.y &&
        r.x >= INPUT_DATA_PTR->anomalyPos_.z &&
        r.z <= INPUT_DATA_PTR->anomalyPos_.z + INPUT_DATA_PTR->anomalySize_.z)
        return INPUT_DATA_PTR->c_ * (1.0+INPUT_DATA_PTR->alpha_);
    else
        return INPUT_DATA_PTR->c_;
}

double d_Slowness    (Point3D_t rEmitter)
{
    return (1 / (SoundSpeed (rEmitter) * SoundSpeed (rEmitter))) - (1 / (INPUT_DATA_PTR -> c_ * INPUT_DATA_PTR -> c_));
}

//=================================================================
