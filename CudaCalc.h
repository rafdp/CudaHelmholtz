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

#ifdef __CUDACC__
template <typename T>
__host__ __device__
float Point3DDevice_t<T>::len ()
{
    return sqrtf (x*x + y*y + z*z);
}
#endif

#ifdef __CUDACC__
template <typename T>
__host__ __device__
Point3DDevice_t<T>::Point3DDevice_t (const Point3D_t& p) :
    x (p.x),
    y (p.y),
    z (p.z)
{}
#endif

#ifdef __CUDACC__
template <typename T>
__host__ __device__
Point3DDevice_t<T>::Point3DDevice_t () :
    x (0.0f),
    y (0.0f),
    z (0.0f)
{}
#endif

#ifdef __CUDACC__
template <typename T>
template <typename T1>
__host__ __device__
Point3DDevice_t<T>::Point3DDevice_t (T1 tx, T1 ty, T1 tz) :
    x (tx),
    y (ty),
    z (tz)
{}
#endif

#ifdef __CUDACC__
template <typename T>
__host__ __device__
Point3DDevice_t<T>::Point3DDevice_t (T* init) :
    x (init[0]),
    y (init[1]),
    z (init[2])
{}
#endif


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


__global__ void DevicePrintData (InputDataOnDevice * inputDataPtr);

const char * cublasGetErrorString (cublasStatus_t error);
const char * cusolverGetErrorString (cusolverStatus_t error);

typedef thrust::complex<float> complex_t;
typedef Point3DDevice_t<float> point_t;

#define CC(op) \
cudaStat = (op); \
if (cudaStat != cudaSuccess) \
{ \
    printf ("-----------------\n    Error occurred (cuda)\n   line %d: %s\n    Error text:\"%s\"\n-----------------", __LINE__, #op, cudaGetErrorString(cudaStat)); \
    exit (1); \
}

#define CB(op) \
cublas_status = (op); \
if (cublas_status != CUBLAS_STATUS_SUCCESS) \
{ \
    printf ("-----------------\n    Error occurred (cublas)\n   line %d: %s\n    Error text:\"%s\"\n-----------------", __LINE__, #op, cublasGetErrorString(cublas_status)); \
    exit (1); \
}

#define CS(op) \
cusolver_status = (op); \
if (cusolver_status != CUSOLVER_STATUS_SUCCESS) \
{ \
    CC(cudaMemcpy(&devInfoHost, devInfo, sizeof(int), cudaMemcpyDeviceToHost));\
    printf ("-----------------\n    Error occurred (cusolver, devinfo %d)\n   line %d: %s\n    Error text:\"%s\"\n-----------------", devInfoHost, __LINE__, #op, cusolverGetErrorString(cusolver_status)); \
    exit (1); \
}

#define LL printf ("_%d_\n", __LINE__);


#include "BiCGStabCuda.h"


#endif


//=================================================================
#endif
