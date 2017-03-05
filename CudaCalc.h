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
#include <thrust/complex.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#endif
#include "Builder.h"

//-----------------------------------------------------------------

extern "C"
void ExternalKernelCaller (InputData_t* INPUT_DATA_PTR, std::vector<std::complex<float> >* retData);


struct InputDataOnDevice;

template <typename T>
struct Point3DDevice_t
{
    T x, y, z;

    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Point3DDevice_t (const Point3D_t &copy);


    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Point3DDevice_t ();


    template <typename T1>
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Point3DDevice_t (T1 tx, T1 ty, T1 tz);


    #ifdef __CUDACC__
    __host__ __device__
    #endif
    Point3DDevice_t (T* begin);


    #ifdef __CUDACC__
    __host__ __device__
    #endif
    T len () const;
};
    #ifdef __CUDACC__
    struct InputDataOnDevice
    {
        Point3DDevice_t<float> sourcePos_;
        float w_; //DROP
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
    


    #endif


//=================================================================
#endif
