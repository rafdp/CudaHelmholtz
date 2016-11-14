
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


int main()
{
	InputData_t inputData = {};
	inputData.LoadData();
	INPUT_DATA_PTR = &inputData;
	int recvNum = inputData.Nreceivers_;

	const int matrixSize = inputData.anomalySize_.x * inputData.anomalySize_.y * inputData.anomalySize_.z;
	
	thrust::device_vector <advDS> AdvancedDS (matrixSize);

	thrust::sequence(AdvancedDS.begin().idx, AdvancedDS.end().idx);
	for (int i = 0; i < matrixSize; i++) AdvancedDS.val [i] = inputData.ds2_ [i];

	thrust::device_vector d_outputData(recvNum);
	thrust::fill(d_outputData.begin(), d_outputData.end(), 0);

	__global__ InputData_t d_inputData = intputData;

	for (int i = 0; i < recvNum; i++)
	{
		__global__ Point3D_t rj = inputData.receivers_[i];
		BornCalculation unary_op;
		double init = 0;
		thrust::plus <double> binary_op;
		d_outputData = thrust::transform_reduce(AdvancedDS.begin(), AdvancedDS.end(), unary_op, init, binary_op);
	}


}

__global__ complex <double> d_PressureI      (Point3D_t r)
{
	return  exp (d_inputData.f_ * 2 * d_PI_ / d_inputData.c_ * d_I_ * r .Len()) / (4 * PI_ * r .Len());
}

__global__ complex <double> d_GreenFunction (Point3D_t r, Point3D_t rj)
{
	return  exp (d_inputData.f_ * 2 * d_PI_ / d_inputData.c_ * d_I_ * rj.Len()) / (4 * PI_ * rj.Len());
}

