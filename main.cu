
#include "Builder.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/complex.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

__global__ d_I_  = I_ ;
__global__ d_PI_ = PI_;


InputData_t* INPUT_DATA_PTR   = nullptr;
InputData_t* D_INPUT_DATA_PTR = nullptr;

thrust::device_vector <complex <double>> * Ui;


struct BornCalculation 
{
	__host__ __device__
	double operator()(const complex<double>& S) const
	{
		const Point3D_t rj;
		BornCalculation(Point3D_t _rj) : rj(_rj) {}

		InputData_t* d_inputData = D_INPUT_DATA_PTR;

		Point3D_t r = { (int)S %  (d_inputData -> discretizationSize).x                                        - rj.x,
					   ((int)(S / (d_inputData -> discretizationSize).x)) % (d_inputData -> discretizationSize).y   - rj.y,
						(int)S / ((d_inputData-> discretizationSize) .x * (d_inputData -> discretizationSize).y)    - rz.z};
		return (*Ui) [S] * exp(d_inputData -> f_ * 2 * d_PI_ / d_inputData -> c_ * d_I_ * dr.Len()) / (4 * PI_ * dr.Len());
	}
};

struct UiMultiply
{
	__host__ __device__
	double operator()(const complex<float>& ds, const complex<double>& S) const 
	{
		InputData_t* d_inputData = D_INPUT_DATA_PTR;
		Point3D_t r = {(int) S % (d_inputData -> discretizationSize).x,
		              ((int) S / (d_inputData -> discretizationSize).x) % x % (d_inputData -> discretizationSize).y,
		               (int) S / ((d_inputData -> discretizationSize).x * (d_inputData -> discretizationSize).y)};

		return d_w_ * _d_w_ * ds * exp(d_inputData -> f_ * 2 * d_PI_ / d_inputData -> c_ * d_I_ * r.Len()) / (4 * PI_ * r.Len());
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

	thrust::transform(Ui.begin(), Ui.end(), Ui.begin(), Ui.begin(), UiMultiply());

	thrust::device_vector <thrust::complex <double>> d_output(recvNum);
	thrust::host_vector   <thrust::complex <double>> h_output(recvNum);

	for (int i = 0; i < recvNum; i++)
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
		printf("%f + %fi\n", h_ouutput.real(), h_output.imag());
	}


}

