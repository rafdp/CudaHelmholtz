

#include "CudaCalc.h"

#include <thrust/execution_policy.h>


#define _PI 3.1415926f

__global__ void BornForRecieversKernel (int * P_recv, InputData_t* INPUT_DATA_PTR);

__global__ void DevicePrintData (InputDataOnDevice * inputDataPtr);

__global__ void DevicePrint ();

__device__ thrust::complex<float> * UiPtr;
__device__ thrust::complex<float> * UbPtr;
__device__ thrust::complex<float> * dSPtr;
__device__ thrust::complex<float> * BigBornPtr;
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


#define SIZE1_ inputDataPtr->size1_
#define SIZE2_ inputDataPtr->size2_
#define SIZE3_ inputDataPtr->size3_
struct ReflectBlq
{
	const int l , q;
	ReflectBlq (int _l, int _q) : l(_l), q(_q) {}

    __device__
	thrust::complex <float> operator()(int id) const 
	{
        Point3DDevice_t <float> PointL (*(PointsPtr + SIZE2_*l));
        Point3DDevice_t <float> PointQ;
        id -= 2*SIZE1_;
        if (id < 0) return (0.0f, 0.0f);
        else
        {	
            if (id < 2*SIZE2_)
            {
                if      (id % (2*SIZE1_) == 0) return (0.0f, 0.0f);			
                else if (id % (2*SIZE1_) <= SIZE1_) 
		        PointQ = 
                *(PointsPtr + q*SIZE2_ + (SIZE2_ - 1) - ((id - 1) / (2*SIZE1_))*SIZE1_ -                 (id - 1) % (2*SIZE1_));
	            else
		        PointQ =
                *(PointsPtr + q*SIZE2_ + (SIZE2_ - 1) - ((id - 1) / (2*SIZE1_))*SIZE1_ - (2*SIZE1_ - 2 - (id - 1) % (2*SIZE1_)));
	        }  
	        else
	        {
	            if      (id % (2*SIZE1_) == 0) return (0.0f, 0.0f);
                else if (id % (2*SIZE1_) <= SIZE1_) 
		        PointQ = 
                *(PointsPtr + q*SIZE2_ +  (SIZE1_ - 1) + ((id - 2*(SIZE2_ - SIZE1_) - 1) / (2*SIZE1_))*SIZE1_ -                 (id - 1) % (2*SIZE1_));
	            else
		        PointQ = 
                *(PointsPtr + q*SIZE2_ +  (SIZE1_ - 1) + ((id - 2*(SIZE2_ - SIZE1_) - 1) / (2*SIZE1_))*SIZE1_ - (2*SIZE1_ - 2 - (id - 1) % (2*SIZE1_)));
	        }
        }

        Point3DDevice_t <float> dr = {PointL.x - PointQ.x, PointL.y - PointQ.y, PointL.z - PointQ.z};
	    if (dr.len() == 0.0f) return (0.0f, 0.0f);
        else                  return (inputDataPtr -> w2h3_) * thrust::exp(inputDataPtr -> uiCoeff_ * dr.len()) / (4 * _PI * dr.len()); 
	}

};

struct GreenCorner
{

    const int q;
	GreenCorner (int _q) : q(_q) {}

	__device__
	thrust::complex <float> operator()(int idx) const 
	{
	    int id = idx - 2*SIZE1_;
       if (idx >= 2*SIZE2_ + SIZE1_ && idx % (2*SIZE1_) >= SIZE1_) 
            return *(UiPtr + q*SIZE2_ +  (SIZE1_ - 1) + ((id - 2*(SIZE2_ - SIZE1_) - 1) / (2*SIZE1_))*SIZE1_ - (2*SIZE1_ - 2 - (id - 1) % (2*SIZE1_)));
                     
       else return (0.0f, 0.0f);

	}

};

struct ShrinkBorn
{
    

    __device__
	thrust::complex <float> operator()(int idx) const 
	{
	    return  *(dSPtr + idx) *
                   (*(BigBornPtr + (idx / SIZE1_) * (2*SIZE1_) + idx % (SIZE1_))) / thrust::complex<float> (4*SIZE2_*1.0f, 0);
	}
};

#undef SIZE1_
#undef SIZE2_
#undef SIZE3_

struct GreenOperatorBorn


{
	const Point3DDevice_t <float> rj;
	GreenOperatorBorn(Point3DDevice_t <float> _rj) : rj(_rj) {}

    __device__
	thrust::complex<float> operator()(thrust::complex<float> idxz) const
	{

		int idx = roundf(idxz.real());		
		InputDataOnDevice* d_inputData = inputDataPtr;
	
		Point3DDevice_t <float> r = *(PointsPtr + idx);

		Point3DDevice_t <float> dr = {r.x - rj.x, r.y - rj.y, r.z - rj.z};

		//printf ("(%e, %e, %e)\n", dr.x, dr.y, dr.z);
	
		thrust::complex <float> out = (*(dSPtr + idx)) * inputDataPtr -> w2h3_ * *(UiPtr + idx)  
                                      * thrust::exp(d_inputData -> uiCoeff_ * dr.len()) / (4 * _PI * dr.len());
						            
		if (abs(out.real()) > 1) return thrust::complex<float> (0.0, 0.0);
        else return out;
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
	
		Point3DDevice_t <float> r = *(PointsPtr + idx);

		Point3DDevice_t <float> dr = {r.x - rj.x, r.y - rj.y, r.z - rj.z};
						            
		if (abs (r.len()) > 0.0000001) return (*(UiPtr + idx)/((*(UiPtr + idx) - *(UbPtr + idx)) / *(UiPtr + idx)))* 
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

typedef thrust::complex <float> complex_t;

__global__ void PrintGrid (complex_t* data, int size, int div = 1)
{
    printf ("Printing grid 2d (%X)\n", data);
    for (int i = 0; i < size; i++)
    { 
	printf ("  ");
        for (int j = 0; j < size; j++)
	{
            printf ("%.2e ", data[i*size + j].real ()/div/*, data[i*size + j].imag ()/div*/);
	}
	printf ("\n");
    }
    printf ("PrintGrid2D ended (%X)\n", data);
}

struct PrintComplexMartix
{
  __device__
  void operator () (thrust::complex<float> val)
  {     
	printf ("%e + %e \n", val.real(), val.imag());
    
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

struct tabMultiply
{
    complex_t* current, *bigborn;

    __host__
    tabMultiply (complex_t * _current, complex_t * _bigborn): current (_current), bigborn (_bigborn) {}

    __device__
    complex_t operator () (int idx)
    {
        printf ("idx is %d\n", idx);
        return *(current + idx) * *(bigborn + idx);
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

struct UiMultiply
{
	__device__
	thrust::complex <float> operator()(const Point3DDevice_t<float>& r) const 
	{
		if (abs (r.len()) > 0.0000001) return thrust::exp(inputDataPtr -> uiCoeff_ * r.len()) / (4 * _PI * r.len());
		else return thrust::complex <float> (0.0, 0.0);
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
	
    const int SIZE1_ = inputData.discretizationSize_[0];
    const int SIZE2_ = inputData.discretizationSize_[0] * inputData.discretizationSize_[1];

    thrust::host_vector<thrust::complex<float> > hostDs2Matrix (size3);

    for (int i = 0; i < size3; i++)
    {
    	hostDs2Matrix[i] = thrust::complex <float> (float (inputData.ds2_[i]), 0.0);
	//printf ("%e + 0\n", hostDs2Matrix[i].real());

    }

	printf ("hostDs2 done\n");
    thrust::device_vector<thrust::complex<float> > dS (hostDs2Matrix);
	void * tempPtr = dS.data ().get ();
    cudaMemcpyToSymbol(dSPtr,
                       &tempPtr,
                       sizeof(void*));
	//thrust::for_each (dS.begin(), dS.begin() + 20, PrintComplexVector());
	printf ("ds2 sent to device\n");

	thrust::device_vector <thrust::complex <float> > Ui (size3);
	 tempPtr = Ui.data ().get ();
   cudaMemcpyToSymbol(UiPtr,
                       &tempPtr,
                      sizeof(void*));


	thrust::device_vector <Point3DDevice_t <float> > Points (size3);
	tempPtr = Points.data ().get ();
    cudaMemcpyToSymbol(PointsPtr,
                       &tempPtr,
                       sizeof(void*));
	printf ("arrays copied\n");
	
	
	
	thrust::tabulate(Points.begin(), Points.end(), IndexFromSequence()); // filling Point with coordinates
//cudaDeviceSynchronize ();
	printf ("tabulated\n");
	//PrintPointsVector printP;
	//thrust::for_each (Points.begin(), Points.end(), printP);
	cudaDeviceSynchronize ();
	//printf ("afte forech\n");

	////////////////////////
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudaEventCreate(&stop);
        ////////////////////////

	thrust::transform (Points.begin(), Points.end(), Ui.begin(), UiMultiply()); // filling Ui array with G(r)
    //thrust::for_each (Ui.begin(), Ui.end(), PrintComplexVector());
    //cudaDeviceSynchronize();
    printf ("transformed\n");

	thrust::device_vector <thrust::complex <float> > Ub (size3);
	tempPtr = Ub.data ().get ();
    	cudaMemcpyToSymbol(UbPtr,
                     	  &tempPtr,
                     	  sizeof(void*));

	//thrust::transform (Points.begin(), Points.end(), Ub.begin(), UbMultiply());
//cudaDeviceSynchronize ();
//printf ("UB copied\n");
    thrust::device_vector <thrust::complex <float> > BigBorn   (SIZE2_ * 4);
//printf ("Bigborn ready\n");
//cudaDeviceSynchronize ();
    thrust::device_vector <thrust::complex <float> > BigUi     (SIZE2_ * 4);
//cudaDeviceSynchronize ();
//cudaDeviceSynchronize ();
    thrust::device_vector <thrust::complex <float> > CurrentLayer (SIZE2_ * 4);
//printf ("\n===size2*4 is %d===\n", SIZE2_*4);
//printf ("vectors done\n");
	tempPtr = BigBorn.data ().get ();
    	cudaMemcpyToSymbol(BigBornPtr,
                     	  &tempPtr,
                     	  sizeof(void*));
//printf ("\nffstarted\n");
cufftHandle plan = {};
cufftPlan2d(&plan, 2*SIZE1_, 2*SIZE1_, CUFFT_C2C);
    for (int l = 0; l < SIZE1_; l ++)
    {
        CurrentLayer.assign(4*SIZE2_, complex_t (0.0f, 0.0f));
        for (int q = 0; q < SIZE1_; q ++)
        {
            //printf ("1 im here, %d %d\n", q, l);
            thrust::tabulate (BigBorn.begin(), BigBorn.end(), ReflectBlq  (l, q)); // returns g (r - r`)
            //printf ("2 im there, %d %d\n", q, l);
            thrust::tabulate (BigUi  .begin(), BigUi  .end(), GreenCorner (q)   ); // returns Ui
            //printf ("3 im there, %d %d\n", q, l);
	        cufftExecC2C(plan, reinterpret_cast<cufftComplex*> (BigUi.data ().get ()), 
                        reinterpret_cast<cufftComplex*> (BigUi.data ().get ()), CUFFT_FORWARD);	
            //printf ("4 im there, %d %d\n", q, l);
	        cufftExecC2C(plan, reinterpret_cast<cufftComplex*> (BigBorn.data ().get ()), 
                        reinterpret_cast<cufftComplex*> (BigBorn.data ().get ()), CUFFT_FORWARD);
            
            //printf ("5 im there, %d %d\n", q, l);
            thrust::transform(BigBorn.begin(), BigBorn.end(),
                              BigUi  .begin(), BigBorn.begin(),
                              thrust::multiplies<thrust::complex<float>>());
            //printf ("6 im there, %d %d\n", q, l);
            //cudaDeviceSynchronize ();

            thrust::transform (CurrentLayer.begin(), CurrentLayer.end(), BigBorn.begin(), 
                                CurrentLayer.begin(), thrust::plus<thrust::complex<float>>());
          
            //printf ("im there, %d %d\n", q, l);


         }

        cufftExecC2C(plan, reinterpret_cast<cufftComplex*> (CurrentLayer.data ().get ()), 
                        reinterpret_cast<cufftComplex*> (BigBorn.data ().get ()), CUFFT_INVERSE);
        //cudaDeviceSynchronize();

        //cudaDeiceSynchronize (); 
        //thrust::for_each (BigBorn.begin(), BigBorn.end(), PrintComplexVector());
        //cudaDeviceSynchronize ();
        //printf ("///\n");
        thrust::tabulate(Ub.begin() + SIZE2_*l, Ub.begin() + SIZE2_*l + SIZE2_, ShrinkBorn()); // returns w2 * ds * res/4size2
        //cudaDeviceSynchronize ();     
    }

//	thrust::for_each (Ub.begin(), Ub.end(), PrintComplexVector());
//    thrust::for_each (BigBorn.begin(), BigBorn.end(), PrintComplexVector());
    thrust::complex <float> init = (0.0f, 0.0f);complexPlus binary_op;
   // getchar();
	for (int i = 0; i < recvNum; i ++)
	{
		Point3D_t rj = inputData.receivers_[i];

		thrust::device_vector <thrust::complex <float> > BornForReciever (size3);
		thrust::tabulate(BornForReciever.begin(), BornForReciever.end(), ComplexIndex()); 
        

		(*retData) [i] = 
                thrust::transform_reduce(BornForReciever.begin(), BornForReciever.end(), GreenOperatorQA (rj), init, binary_op); 
	}

	
	//////////////////////////////////////////////
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);
	//////////////////////////////////////////////

}

