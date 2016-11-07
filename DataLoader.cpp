
#include "Builder.h"

template <>
double Point3D_t_<double>::Len ()
{
    return sqrt(x*x*1.0 + y*y + z*z);
}

template <>
double Point3D_t_<int>::Len ()
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
    fread (&anomalySize_, sizeof (Point3D_t), 1, load);
    fread (&V_, sizeof (double), 1, load);
    printf ("----reading V: %g\n", V_);
    int N = 0;
    fread (&N, sizeof (int), 1, load);
    recievers_.resize (N);
    fread (recievers_.data (), sizeof (Point3D_t_<double>), N, load);
}




