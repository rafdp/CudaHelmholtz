

#include "CudaCalc.h"

#define _PI 3.1415926f

__global__ void BornForRecieversKernel (int * P_recv, InputData_t* INPUT_DATA_PTR);

__global__ void DevicePrintData (InputDataOnDevice * inputDataPtr);

__global__ void DevicePrint ();

__device__ thrust::complex <float> * UiPtr;
__device__ thrust::complex <float> * UbPtr;
__device__ thrust::complex <float> * dSPtr;
__device__ Point3DDevice_t <float> * PointsPtr;

__device__ InputDataOnDevice * inputDataPtr;


	template <typename T>
	__host__ __device__	
	Point3DDevice_t<T>::Point3DDevice_t (const Point3D_t &copy)
    {
        x = (T) copy.x;
        y = (T) copy.y;
        z = (T) copy.z;
    }
   
	template <typename T>
    __host__ __device__
    Point3DDevice_t<T>::Point3DDevice_t (): x(0), y(0), z (0){}

	template <typename T>    
	template <typename T1>
    __host__ __device__
    Point3DDevice_t<T>::Point3DDevice_t (T1 tx, T1 ty, T1 tz) : 
        x (tx), y (ty), z (tz){}

	template <typename T>
    __host__ __device__
    Point3DDevice_t<T>::Point3DDevice_t (T* begin) : 
        x (begin [0]), y (begin [1]), z (begin [2]){}

    template <typename T>
    __host__ __device__
    T Point3DDevice_t<T>::len () const
    {
        return (T) sqrtf (x*x + y*y + z*z);
    }

struct GreenOperatorBorn
{
	const Point3DDevice_t <float> rj;
	GreenOperatorBorn(Point3DDevice_t <float> _rj) : rj(_rj) {}

    __device__
	thrust::complex<float> operator()(thrust::complex<float> idxz) const
	{

		int idx = roundf(idxz.real());		
		InputDataOnDevice* d_inputData = inputDataPtr;
	
		Point3DDevice_t <float> r  = *(PointsPtr + idx);
		Point3DDevice_t <float> dr = {r.x - rj.x, r.y - rj.y, r.z - rj.z};

		thrust::complex <float> out = *(UiPtr + idx)  * thrust::exp(d_inputData -> uiCoeff_ * dr.len()) / (4 * _PI * dr.len());
	}
};

struct GreenOperatorQA
{
	const Point3DDevice_t <float> rj;
	GreenOperatorQA (Point3D_t _rj) : rj(_rj) {}

    __device__
	thrust::complex<float> operator()(thrust::complex<float> idxz) const
	{

		int idx = roundf(idxz.real());		
		InputDataOnDevice* d_inputData = inputDataPtr;
		Point3DDevice_t <float> r  = *(PointsPtr + idx);
		Point3DDevice_t <float> dr = {r.x - rj.x, r.y - rj.y, r.z - rj.z};
						            
		if (abs (r.len()) > 0.0000001)
			return (*(UiPtr + idx)/((*(UiPtr + idx) - *(UbPtr + idx)) / *(UiPtr + idx)))* 
				   inputDataPtr -> w2h3_ * (*(dSPtr + idx)) * thrust::exp(d_inputData -> uiCoeff_ * dr.len()) / (4 * _PI * dr.len());
		else return thrust::complex <float> (0.0, 0.0);
	}
};

struct PrintComplexVector
{
  __device__
  void operator () (thrust::complex<float> val)
  {     
	printf ("%e + %ei\n", val.real(), val.imag());
  }
};

struct PrintPointsVector
{
  __device__
  void operator () (const Point3DDevice_t <float> &val)
  {    
	printf ("(%f, %f, %f)\n", val.x, val.y, val.z);
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

struct ComplexIndex
{
	__host__ __device__
	thrust::complex <float> operator () (int index) const
	{
		return thrust::complex <float> (index * 1.0f, 0.0f);
	}
};

struct GreenForPoints
{
	__device__
	thrust::complex <float> operator()(const Point3DDevice_t<float>& r) const 
	{
		if (abs (r.len()) > 0.0000001) return thrust::exp(inputDataPtr -> uiCoeff_ * r.len()) / (4 * _PI * r.len());
		else return thrust::complex <float> (0.0, 0.0);
	}

};

struct ReflectGreen
{
	__device__
	thrust::complex <float> operator()(int idx) const 
	{
		if (idx <= inputDataPtr->size2_ * 2)
		{
			if ((idx / inputDataPtr->size1_) % 2 == 0) 
				return *(Ui + (idx/(4*inputDataPtr->size2_)*size2 + 
						inputDataPtr->size1_*(idx - 2*inputDataPtr->size2_)/(2 * inputDataPtr->size1_) - idx % 2*inputDataPtr->size1_));
			else
				return *(Ui + (idx/(4*inputDataPtr->size2_)*size2 + 
						inputDataPtr->size1_*(idx - 2*inputDataPtr->size2_)/(2 * inputDataPtr->size1_)   - idx % 2*inputDataPtr->size1_));
		}
		else
		{
			if ((idx / inputDataPtr->size1_) % 2 == 0) 
				return *(Ui + (idx/(4*inputDataPtr->size2_)*size2 + 
						inputDataPtr->size1_(inputDataPtr->size1_ - idx/(2 * inputDataPtr->size1_))   - idx % 2*inputDataPtr->size1_));
			else
				return *(Ui + (idx/(4*inputDataPtr->size2_)*size2 + 
						inputDataPtr->size1_(inputDataPtr->size1_ - idx/(2 * inputDataPtr->size1_))   + idx % 2*inputDataPtr->size1_));
		}
	}

};

struct ExtendBorn
{
	__device__
	thrust::complex <float> operator()(int idx) const 
	{
		if (idx <= inputDataPtr->size2_ * 2)
		{
			if ((idx / inputDataPtr->size1_) % 2 == 0) 
				return thrust::complex <float> (0.0, 0.0);
			else
				return *(Ui + (idx/(4*inputDataPtr->size2_)*size2 + 
						inputDataPtr->size1_*(idx - 2*inputDataPtr->size2_)/(2 * inputDataPtr->size1_) - idx % 2*inputDataPtr->size1_));
		}
		else
		{
			return thrust::complex <float> (0.0, 0.0);
		}
	}

};

struct IndexFromSequence
{
    __device__
    Point3DDevice_t<float> operator() (int idx) const
    {

        Point3DDevice_t<float> point = { 1.0f * (idx % inputDataPtr->size1_),
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

__global__ void DevicePrint ()
{
    printf ("--------------------------------------------------------------\n");
    printf ("threadIdx.x: %d\n", threadIdx.x);
    printf ("--------------------------------------------------------------\n");
}


__global__ void DevicePrintData ()
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
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void ExternalKernelCaller (InputData_t* inputDataPtr_, std::vector<std::complex<float> >* retData)
{

	

	InputData_t& inputData = *inputDataPtr_;
	InputDataOnDevice* deviceInputData = nullptr;

//	printf ("ERROR: %s\n", cudaGetErrorString(cudaMalloc ((void**) &deviceInputData, sizeof (InputDataOnDevice))));
//                                                                       +-------inputDataPtr
//                                                                       |
//                                                                       v
//    	printf ("ERROR: %s\n", cudaGetErrorString(cudaMemcpyToSymbol(&inputDataPtr,
  //                                                               &deviceInputData,
    //                                                             sizeof(InputDataOnDevice*))));
    printf("ERROR: %s\n", cudaMalloc ((void**) &deviceInputData, sizeof (InputDataOnDevice)));

    printf("ERROR: %s\n", cudaMemcpyToSymbol(inputDataPtr, &deviceInputData, sizeof(InputDataOnDevice*)));

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
                                      (float)(4*3.1415926f*3.1415926f*inputData.f_*inputData.f_*
                                      inputData.discreteBlockSize_[0]*inputData.discreteBlockSize_[1]*inputData.discreteBlockSize_[2])};


    cudaMemcpy (deviceInputData, &hostDataCopy, sizeof (InputDataOnDevice), cudaMemcpyHostToDevice);

    printf ("About to call kernel\n");
    DevicePrintData<<<1, 1>>> ();
    cudaDeviceSynchronize ();
    printf ("Kernel returned\n");
	


    thrust::host_vector<thrust::complex<float> > hostDs2Matrix (size3);
	for (int i = 0; i < size3; i++) hostDs2Matrix[i] = thrust::complex <float> (float (inputData.ds2_[i]), 0.0);
	printf ("hostDs2 done\n");


    thrust::device_vector<thrust::complex<float> > dS (hostDs2Matrix);
	void * tempPtr = dS.data ().get ();
    cudaMemcpyToSymbol(dSPtr,
                       &tempPtr,
                       sizeof(void*));
	printf ("ds2 sent to device\n");

	thrust::device_vector <thrust::complex <float>> Ui (size3);
	 tempPtr = Ui.data ().get ();
   cudaMemcpyToSymbol(UiPtr,
                       &tempPtr,
                      sizeof(void*));


	thrust::device_vector <Point3DDevice_t <float>> Points (size3);
	tempPtr = Points.data ().get ();
    cudaMemcpyToSymbol(PointsPtr,
                       &tempPtr,
                       sizeof(void*));
	printf ("arrays copied\n");
	
	
	thrust::tabulate(Points.begin(), Points.end(), IndexFromSequence()); // filling Point with coordinates


	////////////////////////
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudaEventCreate(&stop);
    ////////////////////////


	thrust::transform (Points.begin(), Points.end(), Ui.begin(), GreenForPoints()); // filling Ui array with G(r)
    cudaDeviceSynchronize();
	thrust::device_vector <Point3DDevice_t <float>> GreenSymmetry(4*size3);			
	thrust::tabulate (GreenSymmetry.begin(), GreenSymmetry.end(), ReflectGreen());



	thrust::device_vector <thrust::complex <float>> Ub(size3);
	tempPtr = Ub.data ().get ();
    	cudaMemcpyToSymbol(UbPtr,
                     	  &tempPtr,														// Ub alloc
                     	  sizeof(void*));

	thrust::device_vector <thrust::complex<float>> BornForReciever(size3);
	thrust::complex <float> init = (0.0f, 0.0f);complexPlus binary_op;

	thrust::transform (Points.begin(), Points.end(), Ub.begin(), GreenForPoints()); // filling Ub
    cudaDeviceSynchronize();
	thrust::device_vector <Point3DDevice_t <float>> BornQuad(4*size3);			
	thrust::tabulate (BornQuad.begin(), BornQuad.end(), ExtendBorn());
	

	cufftHandle plan;
	cufftPlan3d(&plan, 2*size1_, 2*size1_, size1_, CUFFT_C2C);
	cufftExecC2C(plan, Ui      , Ui      , CUFFT_FORWARD);	
	cufftExecC2C(plan, BornQuad, BornQuad, CUFFT_FORWARD);

	thrust::transform(BornQuad.begin(), BornQuad.end(), Ui.begin(), BornQuad.begin(),  thrust::multiplies<trust::complex<float>>())

	cufftExecC2C(plan, BornQuad, BornQuad, CUFFT_INVERSE);

	cudaDeviceSynchronize ();

	thrust::device_vector <Point3DDevice_t <float>> BornForReciever(size3);	
	thrust::tabulate(BornForReciever.begin(), BornForReciever.end(), ShrinkBorn());


	for (int i = 0; i < recvNum; i ++) //computing QA values
	{
		Point3D_t rj = inputData.receivers_[i];
		thrust::tabulate(BornForReciever.begin(), BornForReciever.end(), ComplexIndex()); 
		(*retData) [i] = thrust::transform_reduce(BornForReciever.begin(), BornForReciever.end(), GreenOperatorQA (rj), init, binary_op); //born calc to global ui
	}

	
	//////////////////////////////////////////////
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);
	//////////////////////////////////////////////

}

