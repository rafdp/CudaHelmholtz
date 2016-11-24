
#include "Builder.h"
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

__constant__ d_I_  = I_ ;
__constant__ d_PI_ = PI_;


InputData_t* INPUT_DATA_PTR   = nullptr;
__device__ InputData_t* D_INPUT_DATA_PTR = nullptr;

__device__ thrust::device_vector <complex <double>> * Ui;
__device__ thrust::device_vector <Point3DDevice_t>  * Points;

struct Point3DDevice_t
{
    T x, y, z;

    __host__ __device__
    float len ()
    {
        return thrust::sqrt (x*x + y*y + z*z);
    }
};


struct BornCalculation 
{
	__host__ __device__
	double operator()(const complex <float> S) const
	{
		const Point3DDevice_t <float> rj;
		BornCalculation(Point3D_t _rj) : rj(_rj) {}

		InputData_t* d_inputData = D_INPUT_DATA_PTR;

		const Point3DDevice_t <float> r = (*Points) [S];

		Point3DDevice_t <float> dr = {r.x - rj.x, r.y - rj.y, r.z - rj.z);
						            
		return (*Ui) [S] * thrust::exp(d_inputData -> f_ * 2 * d_PI_ / d_inputData -> c_ * d_I_ * dr.len()) / (4 * PI_ * dr.len());
	}
};

struct UiMultiply
{
	__host__ __device__
	complex <float>()(const complex<float>& ds, const Point3DDevice_t& r) const 
	{
		InputData_t* d_inputData = D_INPUT_DATA_PTR;
		return d_inputData -> w_ * d_inputData -> w_ * ds * thrust::exp(d_inputData -> f_ * 2 * d_PI_ / d_inputData -> c_ * d_I_ * r.len()) / (4 * PI_ * r.len());
	}

};

struct IndexFromSequence
{
	__host__ __device__
		Point3DDevice_t <float> operator() (int idx) const
	{
		Point3DDevice_t<float> point = { idx % inputDataPtr->size2_,
			(idx / inputDataPtr->size1_) % inputDataPtr->discretizationSize_[1],
			idx / inputDataPtr->size2_ };
		point = { point.x*inputDataPtr->discreteBlockSize_.x*1.0f + inputDataPtr->anomalyPos_.x + inputDataPtr->discreteBlockSize_.x / 2.0,
			      point.y*inputDataPtr->discreteBlockSize_.y*1.0f + inputDataPtr->anomalyPos_.y + inputDataPtr->discreteBlockSize_.y / 2.0,
			      point.z*inputDataPtr->discreteBlockSize_.z*1.0f + inputDataPtr->anomalyPos_.z + inputDataPtr->discreteBlockSize_.z / 2.0 };
		return point;
	}
};


int main()
{
	InputData_t inputData = {};
	inputData.LoadData();
	INPUT_DATA_PTR = &inputData;
	__constant__ InputData_t d_inputData = inputData;
	D_INPUT_DATA_PTR = &d_inputData;
	int recvNum = inputData.Nreceivers_;
	const int matrixSize = inputData.discretizationSize.x * inputData.discretizationSize.y * inputData.discretizationSize.z;

	thrust::device_vector <complex <double>> dS (matrixSize);
	dS = inputData -> ds2_;


	CudaMalloc((void**)Ui, matrixSize);
	thrust::sequence(Ui.begin(), Ui.end());
    CudaMalloc((void**)Points, matrixSize);
	thrust::transform(Points.begin(), Points.end(), Ui.Begin(), IndexFromSequence());

	thrust::transform(Ui.begin(), Ui.end(), dS.begin(), Points.begin(), UiMultiply());

	thrust::device_vector <thrust::complex <double>> d_output(recvNum);
	thrust::host_vector   <thrust::complex <double>> h_output(recvNum);

	for (int i = 0; i < recvNum; i ++)
	{
		Point3D_t rj = inputData.receivers_[i];

		thrust::device_vector <complex <double>> BornForReciever(matrixSize);
		thrust::sequence(BornForReciever.begin(), BornForReciever().end());

		double init = 0;
		thrust::plus <double> binary_op;

		d_output [i] = thrust::transform_reduce(BornForReciever.begin(), BornForReciever.end(), BornCalculation (rj), init, binary_op);
	}

	h_output = d_output;

	for (int i = 0; i < recvNum; i++)
	{
		printf("%f + %fi\n", h_output.real(), h_output.imag());
	}

	return 0;
}

