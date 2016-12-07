
//=================================================================

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


#include "CudaCalc.h"

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


__global__ void BornForRecieversKernel (int * P_recv, InputData_t* INPUT_DATA_PTR);

__global__ void DevicePrintData (InputDataOnDevice * inputDataPtr);

__global__ void DevicePrint ();


template <typename T>
__host__ __device__
float Point3DDevice_t<T>::len ()
{
    return sqrtf (x*x + y*y + z*z);
}

//__device__ thrust::complex<float> * deviceKMatrixPtr;
//__device__ Point3DDevice_t<float> * deviceIndexesPtr;
__device__ InputDataOnDevice * inputDataPtr;

struct ModifyKMatrix
{
__device__
    thrust::complex<float> operator() (thrust::complex<float>& k, Point3DDevice_t<float>& pos)
    {
        Point3DDevice_t<float> dr = {inputDataPtr->sourcePos_.x - pos.x,
                                     inputDataPtr->sourcePos_.y - pos.y,
                                     inputDataPtr->sourcePos_.z - pos.z};
        float len = dr.len ();
        return inputDataPtr->w2h3_ * thrust::exp (inputDataPtr->uiCoeff_ * len) / (4 * 3.141592f * len) * k;
    }
};

/*
w2h3ds2ui *
exp (Gcoeff * len) / (4 * PI_ * len)
*/
struct SetAMatrix
{
    thrust::complex<float> * deviceKMatrixPtr;
    Point3DDevice_t<float> * deviceIndexesPtr;

    SetAMatrix (thrust::complex<float> * deviceKMatrixPtr_, Point3DDevice_t<float> * deviceIndexesPtr_) :
        deviceKMatrixPtr (deviceKMatrixPtr_),
        deviceIndexesPtr (deviceIndexesPtr_)
    {}

__device__
    thrust::complex<float> operator() (int idx)
    {
        int idx1 = idx % inputDataPtr->size3_; // receiver
        int idx2 = idx / inputDataPtr->size3_; // emitter
        if (idx1 == idx2) return thrust::complex <float> (0.0f, 0.0f);

        Point3DDevice_t<float> pos1 = *(deviceIndexesPtr + idx1);
        Point3DDevice_t<float> pos2 = *(deviceIndexesPtr + idx2);
        Point3DDevice_t<float> dr = {pos1.x-pos2.x,
                                     pos1.y-pos2.y,
                                     pos1.z-pos2.z};
        float len = dr.len ();

//--------------------------------------------------------------------+
// using ui in point   idx1   , maybe will need to tune               |
// if row-major order is used:                                        |
//                                  00 10 20                          |
//                                  01 11 21                          |
//                                  02 12 22                          |
//                                  03 13 23                          |
//                                  04 14 24                          |
//                                  05 15 25                          |
//                                  06 16 26                          |
//                                  07 17 27                          |
//                                  08 18 28                          |
//                                  09 19 29                          |
// every column contains all the points for a single receiver         |
// when converting to column-major:                                   |
// sequential receiver storage                                        |
//                                  00 01 02 03 04 05 06 07 08 09     |
//                                  10 11 12 13 14 15 16 17 18 19     |
//                                  20 21 22 23 24 25 26 27 28 29 ... |
//--------------------------------------------------------------------+


        return (*(deviceKMatrixPtr + idx2)) * thrust::exp (inputDataPtr->uiCoeff_ * len) / (4 * 3.141592f * len);
    }
};

//Aii = - ui
struct ModifyAMatrix
{
    thrust::complex<float> * deviceAMatrixPtr;
    Point3DDevice_t<float> * deviceIndexesPtr;

    ModifyAMatrix (thrust::complex<float> * deviceAMatrixPtr_, Point3DDevice_t<float> * deviceIndexesPtr_) :
        deviceAMatrixPtr (deviceAMatrixPtr_),
        deviceIndexesPtr (deviceIndexesPtr_)
    {}

__device__
    thrust::complex<float> operator() (int idx)
    {
        Point3DDevice_t<float> pos = *(deviceIndexesPtr + idx);
        Point3DDevice_t<float> dr = {inputDataPtr->sourcePos_.x - pos.x,
                                     inputDataPtr->sourcePos_.y - pos.y,
                                     inputDataPtr->sourcePos_.z - pos.z};
        float len = dr.len ();
        if (len < 0.0000001 && len > 0.0000001) return thrust::complex<float> (1.0f, 0.0f);
        *(deviceAMatrixPtr + idx*(inputDataPtr->size3_+1)) = -thrust::exp (inputDataPtr->uiCoeff_ * len) / (4 * 3.141592f * len);
        return thrust::complex<float> (1.0f, 0.0f);
    }
};


struct IndexFromSequence
{
    __device__
    Point3DDevice_t<float> operator() (int idx) const
    {

        Point3DDevice_t<float> point = { 1.0f * (idx % inputDataPtr->size2_),
                                         1.0f * ((idx / inputDataPtr->size1_) % inputDataPtr->discretizationSize_.y),
                                         1.0f * (idx / inputDataPtr->size2_)};
        point = {(float) (point.x*inputDataPtr->discreteBlockSize_.x*1.0f +
                 inputDataPtr->anomalyPos_.x +
                 inputDataPtr->discreteBlockSize_.x / 2.0),
                 (float) (point.y*inputDataPtr->discreteBlockSize_.y*1.0f +
                 inputDataPtr->anomalyPos_.y +
                 inputDataPtr->discreteBlockSize_.y / 2.0),
                 (float) (point.z*inputDataPtr->discreteBlockSize_.z*1.0f +
                 inputDataPtr->anomalyPos_.z +
                 inputDataPtr->discreteBlockSize_.z / 2.0)};
        return point;
    }
};

extern "C"
void ExternalKernelCaller (InputData_t* inputDataPtr_)
{

	InputData_t& inputData = *inputDataPtr_;

	InputDataOnDevice* deviceInputData = nullptr;

	printf ("ERROR: %s\n", cudaGetErrorString(cudaMalloc ((void**) &deviceInputData, sizeof (InputDataOnDevice))));

    printf ("ERROR: %s\n", cudaGetErrorString(cudaMemcpyToSymbol(inputDataPtr,
                                                                 &deviceInputData,
                                                                 sizeof(InputDataOnDevice*))));

    int size3 = inputData.discretizationSize_[0] *
                inputData.discretizationSize_[1] *
                inputData.discretizationSize_[2];

    #define PointConversion(var, type)\
    (Point3DDevice_t<type>) \
    {(type)(inputData.var.x),  \
     (type)(inputData.var.y), \
     (type)(inputData.var.z)}

    InputDataOnDevice hostDataCopy = {PointConversion (sourcePos_, float),
                                      (float) (2*3.141592f*inputData.f_),
                                      thrust::complex<float> (0, (float) (2*3.141592f*inputData.f_/inputData.c_)),
                                      PointConversion (anomalyPos_, float),
                                      PointConversion (anomalySize_, float),
                                      (Point3DDevice_t<int>){inputData.discretizationSize_[0],
                                       inputData.discretizationSize_[1],
                                       inputData.discretizationSize_[2]},
                                      (Point3DDevice_t<int>){inputData.discreteBlockSize_[0],
                                       inputData.discreteBlockSize_[1],
                                       inputData.discreteBlockSize_[2]},
                                      size3,
                                      inputData.discretizationSize_[0] *
                                      inputData.discretizationSize_[1],
                                      inputData.discretizationSize_[0],
                                      (float)(4*3.141592f*3.141592f*inputData.f_*inputData.f_*
                                      inputData.discreteBlockSize_[0]*inputData.discreteBlockSize_[1]*inputData.discreteBlockSize_[2])};

    #undef PointConversion

    cudaMemcpy (deviceInputData, &hostDataCopy, sizeof (InputDataOnDevice), cudaMemcpyHostToDevice);

    printf ("About to call kernel\n");
    DevicePrintData<<<1, 1>>> (deviceInputData);
    cudaDeviceSynchronize ();
    printf ("Kernel returned\n");

    thrust::host_vector<thrust::complex<float> > hostDs2Matrix (size3);

    for (int x = 0; x < inputData.discretizationSize_[0]; x++)
    {
        for (int y = 0; y < inputData.discretizationSize_[1]; y++)
        {
            for (int z = 0; z < inputData.discretizationSize_[2]; z++)
            {
                int currentIndex = (x + y*inputData.discretizationSize_[0] + z*inputData.discretizationSize_[0]*inputData.discretizationSize_[1]);
                hostDs2Matrix[currentIndex] = thrust::complex<float> (float (inputData.ds2_[currentIndex]), 0.0);
            }
        }
    }

    thrust::device_vector<thrust::complex<float> > deviceKMatrix (hostDs2Matrix);

    printf ("%d\n", __LINE__);

    /*void * tempPtr = deviceKMatrix.data ().get ();
    printf ("%d\n", __LINE__);

    cudaMemcpyToSymbol(deviceKMatrixPtr,
                       &tempPtr,
                       sizeof(void*));
    printf ("%d\n", __LINE__);*/

    thrust::device_vector<Point3DDevice_t<float> > indexes (size3);
    printf ("%d\n", __LINE__);

    /*tempPtr = indexes.data ().get ();
    printf ("%d\n", __LINE__);

    cudaMemcpyToSymbol(deviceIndexesPtr,
                       &tempPtr,
                       sizeof(void*));
    printf ("%d\n", __LINE__);*/

    thrust::tabulate (indexes.begin(), indexes.end(), IndexFromSequence ());
    printf ("%d\n", __LINE__);

    thrust::transform (deviceKMatrix.begin (), deviceKMatrix.end (), indexes.begin (), deviceKMatrix.begin (), ModifyKMatrix ());
    printf ("%d\n", __LINE__);

    thrust::device_vector<thrust::complex<float> > deviceAMatrix (size3*size3);
    printf ("%d\n", __LINE__);

    SetAMatrix sMatrixSetter (deviceKMatrix.data ().get (), indexes.data ().get ());

    thrust::tabulate (deviceAMatrix.begin (), deviceAMatrix.end (), sMatrixSetter);

    printf ("%d\n", __LINE__);

    thrust::complex <float> b = deviceKMatrix[17];

    printf ("b = %g %g\n", b.real(), b.imag ());

    cudaFree (deviceInputData);
    printf ("%d\n", __LINE__);


    /// ////////////////////////////////////
    /// solution part (linear system, not fft)
    /// ////////////////////////////////////


    /// 1. Creating handles

    cublasHandle_t cublasH = nullptr;
    cublasCreate(&cublasH);

    cusolverDnHandle_t cudenseH = nullptr;
    cusolverDnCreate(&cudenseH);
    printf ("%d\n", __LINE__);

    /// 1. Setting up data

    thrust::device_vector<thrust::complex<float> > ones (size3, thrust::complex<float> (1.0f, 0.0f));
    thrust::device_vector<thrust::complex<float> > reductedA (size3, 0.0f);

    thrust::complex<float> alpha (1.0f, 1.0f);
    thrust::complex<float> beta (0.0f, 0.0f);

    /// reductedA = alpha*A*ones+beta*reductedA = A*ones
    cublasCgemv (cublasH, CUBLAS_OP_N, size3, size3,
                 reinterpret_cast <cuComplex*> (&alpha),
                 reinterpret_cast <cuComplex*> (deviceAMatrix.data ().get ()),
                 size3,
                 reinterpret_cast <cuComplex*> (ones.data ().get ()), 1,
                 reinterpret_cast <cuComplex*> (&beta),
                 reinterpret_cast <cuComplex*> (reductedA.data ().get ()), 1);




    /// need to subtract ui from every diagonal element of A
    /// strategy1: run tabulate on something of size size3 and modify A alongside
    /// strategy2: run for_each on a sequence, but need to create sequence of size size3

    ///using strategy1
    ModifyAMatrix modificatorA (deviceAMatrix.data ().get (), indexes.data ().get ());
    thrust::tabulate (ones.begin(), ones.end(), modificatorA);


    printf ("%d\n", __LINE__);


}


//=================================================================
