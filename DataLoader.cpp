
#include "Builder.h"

double Point3D_t::Len ()
{
    return sqrt(x*x*1.0 + y*y + z*z);
}

void InputData_t::LoadData ()
{
    FILE* load = fopen (INPUT_FILE, "rb");
    
    fread (&sourcePos_, sizeof (Point3D_t), 1, load);
    
    double params[3] = {};
    fread (params, sizeof (double), 3, load);
    w_ = params[0];
    c_ = params[1];
    alpha_ = params[2];
    fread (&anomalyPos_, sizeof (Point3D_t), 1, load);
    printf ("Anomaly pos %g %g %g", anomalyPos_.x, anomalyPos_.y, anomalyPos_.z);
    fread (&anomalySize_, sizeof (Point3D_t), 1, load);
    fread (&block_size_, sizeof (Point3D_t), 1, load);
    printf ("block_size_ %g %g %g\n", block_size_.x, block_size_.y, block_size_.z);
    int N = 0;
    fread (&N, sizeof (int), 1, load);
    recievers_.resize (N);
    fread (recievers_.data (), sizeof (Point3D_t), N, load);
}

Point3D_t ToPhysicalCenter (Point3D_t p)
{
    return Point3D_t {
        p.x*INPUT_DATA_PTR->block_size_.x + INPUT_DATA_PTR->block_size_.x / 2,
        p.y*INPUT_DATA_PTR->block_size_.y + INPUT_DATA_PTR->block_size_.y / 2,
        p.z*INPUT_DATA_PTR->block_size_.z + INPUT_DATA_PTR->block_size_.z / 2};
    
}

Point3D_t ToPhysical (Point3D_t p)
{
    return Point3D_t {
        p.x*INPUT_DATA_PTR->block_size_.x,
        p.y*INPUT_DATA_PTR->block_size_.y,
        p.z*INPUT_DATA_PTR->block_size_.z};
    
}

Point3D_t ToDiscrete (Point3D_t p)
{
    return Point3D_t {int(p.x/INPUT_DATA_PTR->block_size_.x)*1.0,
                      int(p.y/INPUT_DATA_PTR->block_size_.y)*1.0,
                      int(p.z/INPUT_DATA_PTR->block_size_.z)*1.0};

}



