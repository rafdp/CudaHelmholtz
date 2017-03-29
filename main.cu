
#include "Builder.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/complex.h>
#include <thrust/transform_reduce.h>
#include <thrust/plus.h>
#include "cublas_v2.h"


const dim3 blockSize = (30, 30, 30);
const dim3 gridSize  = (30, 30, 30);

__global__ d_I_  = I_;
__global__ d_PI_ = PI_;

struct advDS
{
	int idx;
	complex <double> val;
};

InputData_t* INPUT_DATA_PTR = nullptr;

__global__ complex <double> d_PressureI     (Point3D_t r);
__global__ complex <double> d_GreenFunction (Point3D_t r, Point3D_t rj);

struct BornCalculation 
{
	__host__ __device__
		 operator()(const advDS x) const 
		{ return x.val * d_PressureI (d_indexToPoint (x.idx) * d_GreenFunction (rj, d_indexToPoint (x.idx) * (INPUT_DATA_PTR -> w_) * (INPUT_DATA_PTR -> w_); }
};

Point3D_t   anomalyPos_;
Point3D_t   anomalySize_;
int         discretizationSize_[3];
int         discreteBlockSize_[3];
double*     ds2_;
int         Nreceivers_;
Point3D_t*  receivers_;

template<typename T>
struct Point3DDevice_t
{
    T x, y, z;

    __host__ __device__
    float len ()
    {
        return thrust::sqrt (x*x + y*y + z*z);
    }
};

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

__device__ thrust::device_ptr<thrust::complex<float> > deviceKMatrixPtr;
__device__ thrust::device_ptr<Point3DDevice_t<float> > deviceIndexesPtr;
__device__ thrust::device_ptr<InputDataOnDevice> inputDataPtr;

struct ModifyKMatrix
{
__host__ __device__
    thrust::complex<float> operator() (Point3DDevice_t<float>& pos, thrust::complex<float>& k)
    {
        Point3DDevice_t<float> dr = {inputDataPtr->sourcePos_.x - pos.x,
                                     inputDataPtr->sourcePos_.y - pos.y,
                                     inputDataPtr->sourcePos_.z - pos.z};
        float len = dr.len ();
        return k*inputDataPtr->w2h3_ * thrust::exp (inputDataPtr->uiCoeff_ * len) / (4 * 3.141592f * len);
    }
};

/*
w2h3ds2ui *
exp (Gcoeff * len) / (4 * PI_ * len)
*/
struct SetAMatrix
{
__host__ __device__
    thrust::complex<float> operator() (int idx)
    {
        int idx1 = idx % inputDataPtr->size3;
        int idx2 = idx / inputDataPtr->size3;
        Point3DDevice_t<float> pos1 = *(inputIndexesPtr + idx1);
        Point3DDevice_t<float> pos2 = *(inputIndexesPtr + idx2);
        Point3DDevice_t<float> dr = {pos1.x-pos2.x,
                                     pos1.y-pos2.y,
                                     pos1.z-pos2.z};
        float len = dr.len ();

//--------------------------------------------------------------------+
// using ui in point   idx2   , maybe will need to tune               |
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
// row = receiver                                                     |
//                                  00 01 02 03 04 05 06 07 08 09     |
//                                  10 11 12 13 14 15 16 17 18 19     |
//                                  20 21 22 23 24 25 26 27 28 29 ... |
//--------------------------------------------------------------------+


        return *(deviceKMatrixPtr + idx2) * thrust::exp (inputDataPtr->uiCoeff_ * len) / (4 * 3.141592f * len);
    }
};


struct IndexFromSequence
{
    __host__ __device__
    Point3DDevice_t<float> operator() (int idx) const
    {
        Point3DDevice_t<float> point = { idx % inputDataPtr->size2_,
                                        (idx / inputDataPtr->size1_) % inputDataPtr->discretizationSize_[1],
                                         idx / inputDataPtr->size2_};
        point = {point.x*inputDataPtr->discreteBlockSize_.x*1.0f + inputDataPtr->anomalyPos_.x + inputDataPtr->discreteBlockSize_.x / 2.0,
                 point.y*inputDataPtr->discreteBlockSize_.y*1.0f + inputDataPtr->anomalyPos_.y + inputDataPtr->discreteBlockSize_.y / 2.0,
                 point.z*inputDataPtr->discreteBlockSize_.z*1.0f + inputDataPtr->anomalyPos_.z + inputDataPtr->discreteBlockSize_.z / 2.0};
        return point;
    }
};


int main()
{
	InputData_t inputData = {};
	inputData.LoadData();
	INPUT_DATA_PTR = &inputData;
	int recvNum = inputData.Nreceivers_;


    inputDataPtr = thrust::new<InputDataOnDevice> (1);


    int size3 = inputData->discretizationSize_[0] *
                inputData->discretizationSize_[1] *
                inputData->discretizationSize_[2];

    InputDataOnDevice hostDataCopy = {inputData.sourcePos,
                                      2*3.141592f*inputData.f_,
                                     (2*3.141592f*inputData.f_/inputData.c_ * thrust::complex<float> (0, 1.0f)),
                                      inputData.anomalyPos_,
                                      inputData.anomalySize_,
                                      {inputData.discretizationSize_[0],
                                       inputData.discretizationSize_[1],
                                       inputData.discretizationSize_[2]},
                                      {inputData.discreteBlockSize_[0],
                                       inputData.discreteBlockSize_[1],
                                       inputData.discreteBlockSize_[2]},
                                      size3,
                                      inputData->discretizationSize_[0] *
                                      inputData->discretizationSize_[1],
                                      inputData->discretizationSize_[0],
                                      4*3.141592f*3.141592f*inputData.f_*inputData.f_*
                                      inputData.discreteBlockSize_[0]*inputData.discreteBlockSize_[1]*inputData.discreteBlockSize_[2]};

    cudaMemcpy (inputDataPtr, &hostDataCopy, sizeof (InputDataOnDevice), cudaMemcpyHostToDevice);

    cublasHandle_t handle = 0;
    cublasCreate(&handle);

    thrust::host_vector<float> hostDs2Matrix (size3);

    for (int x = 0; x < inputData->discretizationSize_[0]; x++)
    {
        for (int y = 0; y < inputData->discretizationSize_[1]; y++)
        {
            for (int z = 0; z < inputData->discretizationSize_[2]; z++)
            {
                int currentIndex = (x + y*inputData->discretizationSize_[0] + z*inputData->discretizationSize_[0]*inputData->discretizationSize_[1]);
                hostDs2Matrix[currentIndex] = thrust::complex<float> (1.0, 1.0) * inputData.ds2[currentIndex];
            }
        }
    }

    thrust::device_vector<thrust::complex<float> > deviceKMatrix (hostDs2Matrix);

    deviceKMatrixPtr = thrust::raw_pointer_cast(deviceKMatrix.data ());

    thrust::device_vector<Point3DDevice_t<float> > indexes (size3);

    deviceIndexesPtr = thrust::raw_pointer_cast(indexes.data ());

    thrust::tabulate (indexes.begin(), indexes.end(), IndexFromSequence ());

    thrust::transform (indexes.begin (), indexes.end (), deviceKMatrix.begin (), deviceKMatrix.begin (), ModifyKMatrix ());

    thrust::device_vector<thrust::complex<float> > deviceAMatrix (size3*size3);

    thrust::tabulate (deviceAMatrix.begin (), deviceAMatrix.end (), SetAMatrix ());

    


}

__global__ complex <double> d_PressureI      (Point3D_t r)
{
	return  exp (d_inputData.f_ * 2 * d_PI_ / d_inputData.c_ * d_I_ * r .Len()) / (4 * PI_ * r .Len());
}

__global__ complex <double> d_GreenFunction (Point3D_t r, Point3D_t rj)
{
	return  exp (d_inputData.f_ * 2 * d_PI_ / d_inputData.c_ * d_I_ * rj.Len()) / (4 * PI_ * rj.Len());
}

