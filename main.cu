
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
#include "Builder.h"

#define _PI 3.1415926f
__device__ thrust::complex<float> * UiPtr;
InputData_t* INPUT_DATA_PTR = nullptr;

__device__ Point3DDevice_t <float> * PointsPtr;


__device__ InputDataOnDevice * inputDataPtr;

struct BornCalculation  // rewrite inputdata use
{
	const Point3DDevice_t <float> rj;
	BornCalculation(Point3D_t _rj) : rj(_rj) {}

    __device__
	thrust::complex<float> operator()(int idx) const
	{

		InputDataOnDevice* d_inputData = inputDataPtr;
	
		Point3DDevice_t <float> r = *(PointsPtr + idx);

		Point3DDevice_t <float> dr = {r.x - rj.x, r.y - rj.y, r.z - rj.z};
						            
		return *(UiPtr + idx)  * thrust::exp(d_inputData -> uiCoeff_ * dr.len()) / (4 * _PI * dr.len());
	}
};

struct complexPlus
{
    __host__ __device__
    thrust::complex <float> operator () (const thrust::complex<float> &z1, const thrust::complex<float> &z2) const
    {
        return z1 + z2;
    }
};

struct UiMultiply
{
	__device__
	thrust::complex <float> operator()(const thrust::complex<float>& ds, const Point3DDevice_t<float>& r) const 
	{
		InputDataOnDevice* d_inputData = inputDataPtr;
		return d_inputData -> w2h3_ * ds * thrust::exp(d_inputData -> uiCoeff_ * r.len()) / (4 * _PI * r.len());
	}

};

struct IndexFromSequence
{
	__device__
		Point3DDevice_t <float> operator() (int idx) const
	{
		Point3DDevice_t<float> point = { idx % inputDataPtr->size2_,
			(idx / inputDataPtr->size1_) % inputDataPtr->discretizationSize_.y,
			idx / inputDataPtr->size2_ };
		point = { point.x*inputDataPtr->discreteBlockSize_.x*1.0f + inputDataPtr->anomalyPos_.x + inputDataPtr->discreteBlockSize_.x / 2.0,
			      point.y*inputDataPtr->discreteBlockSize_.y*1.0f + inputDataPtr->anomalyPos_.y + inputDataPtr->discreteBlockSize_.y / 2.0,
			      point.z*inputDataPtr->discreteBlockSize_.z*1.0f + inputDataPtr->anomalyPos_.z + inputDataPtr->discreteBlockSize_.z / 2.0 };
		return point;
	}
};

__global__

void DevicePrint ()
{
    printf ("--------------------------------------------------------------\n");
    printf ("threadIdx.x: %d\n", threadIdx.x);
    printf ("--------------------------------------------------------------\n");
}


__global__ void DevicePrintData (InputDataOnDevice * inputDataPtr)
{
    /*    Point3DDevice_t<float> sourcePos_;
    float w_; //DROP
    thrust::complex<float> uiCoeff_;
    Point3DDevice_t<float> anomalyPos_;
    Point3DDevice_t<float> anomalySize_;
    Point3DDevice_t<int>   discretizationSize_;
    Point3DDevice_t<int>   discreteBlockSize_;
    int                    size3_;
    int                    size2_;
    int                    size1_;
    float                  w2h3_;*/

    printf ("device address %x\n", inputDataPtr);

    InputDataOnDevice& handle = *(inputDataPtr);

    printf ("--------------------------------------------------------------\n");
    printf ("Printing from device:\n");
    printf ("   sourcePos_: %f %f %f\n",
            inputDataPtr->sourcePos_.x,
            handle.sourcePos_.y,
            handle.sourcePos_.z);
    printf ("   w: %f\n",
            handle.w_);
    printf ("   uiCoeff: %f i*%f\n",
            handle.uiCoeff_.real (),
            handle.uiCoeff_.imag ());
    printf ("   anomalySize_: %f %f %f\n",
            handle.anomalySize_.x,
            handle.anomalySize_.y,
            handle.anomalySize_.z);
    printf ("   discretizationSize_: %d %d %d\n",
            handle.discretizationSize_.x,
            handle.discretizationSize_.y,
            handle.discretizationSize_.z);
    printf ("   discreteBlockSize_: %d %d %d\n",
            handle.discreteBlockSize_.x,
            handle.discreteBlockSize_.y,
            handle.discreteBlockSize_.z);
    printf ("   size3_: %d %d %d\n", handle.size3_);
    printf ("   size2_: %d %d %d\n", handle.size2_);
    printf ("   size1_: %d %d %d\n", handle.size1_);
    printf ("   w2h3_: %f\n", handle.w2h3_);
    printf ("End print from device\n");
    printf ("--------------------------------------------------------------\n");
}

extern "C"
int main ()
{

	InputData_t inputData = {};
    //inputData.LoadData ();

	printf ("ERROR: %s\n", cudaGetErrorString(cudaMalloc ((void**) &deviceInputData, sizeof (InputDataOnDevice))));

    	printf ("ERROR: %s\n", cudaGetErrorString(cudaMemcpyToSymbol(&inputData,
                                                                 &deviceInputData,
                                                                 sizeof(InputDataOnDevice*))));

	int recvNum = inputData.Nreceivers_;
	


    	int size3 = inputData.discretizationSize_[0] *
                inputData.discretizationSize_[1] *
                inputData.discretizationSize_[2];



    	InputDataOnDevice hostDataCopy = {(inputData.sourcePos_),
                                      (float) (2*3.141592f*inputData.f_),
                                      thrust::complex<float> (0, (float) (2*3.141592f*inputData.f_/inputData.c_)),
                                     (inputData.anomalyPos_),
                                      (inputData.anomalySize_),
                                      inputData.discretizationSize_,
                                      inputData.discreteBlockSize_,
                                      size3,
                                      inputData.discretizationSize_[0] *
                                      inputData.discretizationSize_[1],
                                      inputData.discretizationSize_[0],
                                      (float)(4*3.141592f*3.141592f*inputData.f_*inputData.f_*
                                      inputData.discreteBlockSize_[0]*inputData.discreteBlockSize_[1]*inputData.discreteBlockSize_[2])};


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

    thrust::device_vector<thrust::complex<float> > dS (hostDs2Matrix);


	thrust::device_vector <thrust::complex <float> > Ui (size3);
	 void * tempPtr = Ui.data ().get ();
    cudaMemcpyToSymbol(UiPtr,
                       &tempPtr,
                       sizeof(void*));


thrust::device_vector <Point3DDevice_t <float> > Points (size3);
	tempPtr = Points.data ().get ();
    cudaMemcpyToSymbol(PointsPtr,
                       &tempPtr,
                       sizeof(void*));
	
	

	//thrust::device_vector <Point3DDevice_t <float> > Points (size3); // remove ds or ui
	
	thrust::tabulate(Points.begin(), Points.end(), IndexFromSequence()); // filling Point with coordinates

	thrust::transform(dS.begin(), dS.end(), Points.begin(), Ui.begin(), UiMultiply()); // filling Ui array with w^2 * G(r) * ds^2 * h^3

	thrust::device_vector <thrust::complex <float> > d_output(recvNum);
	

	for (int i = 0; i < recvNum; i ++)
	{
		Point3D_t rj = inputData.receivers_[i];

		thrust::device_vector <int> BornForReciever(size3);
		thrust::complex <float> init = {};
		thrust::sequence(BornForReciever.begin(), BornForReciever.end()); 

		//float init = 0; //ui to global
		complexPlus binary_op;

		d_output [i] = thrust::transform_reduce(BornForReciever.begin(), BornForReciever.end(), BornCalculation (rj), init, binary_op); //born calc to global ui
	}

	thrust::host_vector<thrust::complex <float> > h_output(d_output);
	//*retData = h_output;

	for (int i = 0; i < recvNum; i++)
	{
		printf("%f + %fi\n", h_output[i].real(), h_output[i].imag());
	}
	return 0;
}

