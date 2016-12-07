
#include "Builder.h"

double Point3D_t::Len ()
{
    return sqrt(x*x*1.0 + y*y + z*z);
}


#define print(x) ;

void InputData_t::LoadData ()
{
    FILE* load = fopen (INPUT_FILE, "rb");

    fread (&sourcePos_, sizeof (Point3D_t), 1, load);
    print(("SourcePos %g %g %g\n",
            sourcePos_.x,
            sourcePos_.y,
            sourcePos_.z));

    double data[2] = {};
    fread (data, sizeof (double), 2, load);
    f_ = data[0];
    c_ = data[1];
    print(("f c %g %g\n",
            f_, c_));

    fread (&anomalyPos_, sizeof (Point3D_t), 1, load);
    print(("anomalyPos_ %g %g %g\n",
            anomalyPos_.x,
            anomalyPos_.y,
            anomalyPos_.z));
    fread (&anomalySize_, sizeof (Point3D_t), 1, load);
    print(("anomalySize_ %g %g %g\n",
            anomalySize_.x,
            anomalySize_.y,
            anomalySize_.z));
    fread (&discretizationSize_, sizeof (int), 3, load);
    print(("discretizationSize_ %d %d %d\n",
            discretizationSize_[0],
            discretizationSize_[1],
            discretizationSize_[2]));

    discreteBlockSize_[0] = int(anomalySize_.x / discretizationSize_[0]);
    discreteBlockSize_[1] = int(anomalySize_.y / discretizationSize_[1]);
    discreteBlockSize_[2] = int(anomalySize_.z / discretizationSize_[2]);
    print(("discreteBlockSize_ %d %d %d\n",
            discreteBlockSize_[0],
            discreteBlockSize_[1],
            discreteBlockSize_[2]));

    ds2_ = new double [discretizationSize_[0]*
                       discretizationSize_[1]*
                       discretizationSize_[2]];
    print(("ds2_ size %d\n",
            discretizationSize_[0]*
            discretizationSize_[1]*
            discretizationSize_[2]));

    fread (ds2_, sizeof (double),
            discretizationSize_[0]*
            discretizationSize_[1]*
            discretizationSize_[2],
            load);

    print(("ds2 %g\n", ds2_[411]));

    fread (&Nreceivers_, sizeof (int), 1, load);
    print(("Nreceivers_ %d\n",
            Nreceivers_));
    receivers_ = new Point3D_t [Nreceivers_];
    fread (receivers_, sizeof (Point3D_t), Nreceivers_, load);
}



#undef print

InputData_t::~InputData_t ()
{
    if (receivers_)
        delete [] receivers_;
    receivers_ = nullptr;

    if (ds2_)
        delete [] ds2_;
    ds2_ = nullptr;
}

Point3D_t ToPhysicalCenter (Point3D_t p)
{
    return Point3D_t {(p.x + 0.5)*INPUT_DATA_PTR->discreteBlockSize_[0],
                      (p.y + 0.5)*INPUT_DATA_PTR->discreteBlockSize_[1],
                      (p.z + 0.5)*INPUT_DATA_PTR->discreteBlockSize_[2]};

}

Point3D_t ToPhysical (Point3D_t p)
{
    return Point3D_t {
        p.x*INPUT_DATA_PTR->discreteBlockSize_[0],
        p.y*INPUT_DATA_PTR->discreteBlockSize_[1],
        p.z*INPUT_DATA_PTR->discreteBlockSize_[2]};

}

Point3D_t ToDiscrete (Point3D_t p)
{
    return Point3D_t {int(p.x/INPUT_DATA_PTR->discreteBlockSize_[0])*1.0,
                      int(p.y/INPUT_DATA_PTR->discreteBlockSize_[1])*1.0,
                      int(p.z/INPUT_DATA_PTR->discreteBlockSize_[2])*1.0};

}



