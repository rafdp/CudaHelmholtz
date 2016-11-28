#ifndef CUDA_CALC_INCLUDED
#define CUDA_CALC_INCLUDED
//=================================================================

#include "includes.h"
#include "DataLoader.h"

//-----------------------------------------------------------------

const int BLOCK_SIZE_ = 10;
const int GRID_SIZE_  = 50;

//-----------------------------------------------------------------

extern "C"
void ExternalKernelCaller (InputData_t* INPUT_DATA_PTR);

template<typename T>
struct Point3DDevice_t
{
    T x, y, z;
#ifdef __CUDACC__
    __host__ __device__
#endif
    float len ();
};

struct InputDataOnDevice;

//=================================================================
#endif
