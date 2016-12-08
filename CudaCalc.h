#ifndef CUDA_CALC_INCLUDED
#define CUDA_CALC_INCLUDED
//=================================================================

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/transform_reduce.h>
#include <thrust/memory.h>
#include <thrust/complex.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <thrust/functional.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

#include "includes.h"
#include "DataLoader.h"

//-----------------------------------------------------------------

const int BLOCK_SIZE_ = 10;
const int GRID_SIZE_  = 50;

//-----------------------------------------------------------------

extern "C"
void ExternalKernelCaller (InputData_t* INPUT_DATA_PTR, std::vector<std::complex<float> >* retData);

template<typename T>
struct Point3DDevice_t
{
    T x, y, z;

    template <typename T1>
#ifdef __CUDACC__
    __host__ __device__
#endif
    Point3DDevice_t (T1 tx, T1 ty, T1 tz);


#ifdef __CUDACC__
    __host__ __device__
#endif
    Point3DDevice_t (T* init);

#ifdef __CUDACC__
    __host__ __device__
#endif
    Point3DDevice_t (const Point3D_t& p);

#ifdef __CUDACC__
    __host__ __device__
#endif
    Point3DDevice_t ();

#ifdef __CUDACC__
    __host__ __device__
#endif
    float len ();
};


struct InputDataOnDevice;

#ifdef __CUDACC__
struct InputDataOnDevice
{
    Point3DDevice_t<float> sourcePos_;
    thrust::complex<float> uiCoeff_;
    Point3DDevice_t<float> anomalyPos_;
    Point3DDevice_t<float> anomalySize_;
    Point3DDevice_t<int>   discretizationSize_;
    Point3DDevice_t<int>   discreteBlockSize_;
    int                    size3_;
    int                    size2_;
    int                    size1_;
    float                  w2h3_;
};



__global__ void BornForRecieversKernel (int * P_recv, InputData_t* INPUT_DATA_PTR);

__global__ void DevicePrintData (InputDataOnDevice * inputDataPtr);

__global__ void DevicePrint ();


#endif


//=================================================================
#endif
