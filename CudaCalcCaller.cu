
//=================================================================

#include "CudaCalc.h"


static const char * cublasGetErrorString (cublasStatus_t error);
static const char * cusolverGetErrorString (cusolverStatus_t error);


template <typename T>
__host__ __device__
float Point3DDevice_t<T>::len ()
{
    return sqrtf (x*x + y*y + z*z);
}

template <typename T>
__host__ __device__
Point3DDevice_t<T>::Point3DDevice_t (const Point3D_t& p) :
    x (p.x),
    y (p.y),
    z (p.z)
{}

template <typename T>
__host__ __device__
Point3DDevice_t<T>::Point3DDevice_t () :
    x (0.0f),
    y (0.0f),
    z (0.0f)
{}

template <typename T>
template <typename T1>
__host__ __device__
Point3DDevice_t<T>::Point3DDevice_t (T1 tx, T1 ty, T1 tz) :
    x (tx),
    y (ty),
    z (tz)
{}

template <typename T>
__host__ __device__
Point3DDevice_t<T>::Point3DDevice_t (T* init) :
    x (init[0]),
    y (init[1]),
    z (init[2])
{}

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
    void operator() (int idx)
    {
        Point3DDevice_t<float> pos = *(deviceIndexesPtr + idx);
        Point3DDevice_t<float> dr = {inputDataPtr->sourcePos_.x - pos.x,
                                     inputDataPtr->sourcePos_.y - pos.y,
                                     inputDataPtr->sourcePos_.z - pos.z};
        float len = dr.len ();
        if (len < 0.0000001 && len > 0.0000001) return;
        *(deviceAMatrixPtr + idx*(inputDataPtr->size3_+1)) = -thrust::exp (inputDataPtr->uiCoeff_ * len) / (4 * 3.141592f * len);

    }
};


struct QLReduction
{
	const Point3DDevice_t <float> rj;
    thrust::complex<float> * deviceLambdaPtr;
    Point3DDevice_t<float> * deviceIndexesPtr;

    __host__
	QLReduction (Point3DDevice_t <float> _rj, thrust::complex<float> * deviceLambdaPtr_, Point3DDevice_t<float> * deviceIndexesPtr_) :
        rj (_rj),
        deviceLambdaPtr (deviceLambdaPtr_),
        deviceIndexesPtr (deviceIndexesPtr_)
    {}

    __device__
	thrust::complex<float> operator()(int idx) const
	{
		Point3DDevice_t <float>& r = *(deviceIndexesPtr + idx);

		Point3DDevice_t <float> dr = {r.x - rj.x, r.y - rj.y, r.z - rj.z};

		return *(deviceLambdaPtr + idx)  * thrust::exp(inputDataPtr -> uiCoeff_ * dr.len()) / (4 * 3.141592f * dr.len());
	}
};

struct ComplexAddition
{
    __host__ __device__
	thrust::complex<float> operator()(const thrust::complex<float>& a, const thrust::complex<float>& b) const
	{
		return a + b;
	}
};

struct ReceiverConvolution
{
    thrust::complex<float> * deviceLambdaPtr;
    Point3DDevice_t<float> * deviceIndexesPtr;
    thrust::device_vector<int> * seqPtr;

	ReceiverConvolution (thrust::complex<float> * deviceLambdaPtr_,
                         Point3DDevice_t<float> * deviceIndexesPtr_,
                         thrust::device_vector<int> * seqPtr_) :
        deviceLambdaPtr (deviceLambdaPtr_),
        deviceIndexesPtr (deviceIndexesPtr_),
        seqPtr (seqPtr_)
    {}

    __host__
	std::complex<float> operator() (Point3DDevice_t <float> receiver) const
	{
        QLReduction qlRed (receiver, deviceLambdaPtr, deviceIndexesPtr);
        thrust::complex<float> init (0.0f, 0.0f);
        ComplexAddition complexSum;
        return thrust::transform_reduce (seqPtr->begin(), seqPtr->end(), qlRed, init, complexSum);
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
void ExternalKernelCaller (InputData_t* inputDataPtr_, std::vector<std::complex<float> >* retData)
{

	InputData_t& inputData = *inputDataPtr_;

	InputDataOnDevice* deviceInputData = nullptr;

	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;
    int* devInfo = nullptr;
    int devInfoHost = 0;


    #define CC(op) \
    cudaStat = (op); \
	if (cudaStat != cudaSuccess) \
    { \
        printf ("-----------------\n    Error occurred (cuda)\n   line %d: %s\n    Error text:\"%s\"\n-----------------", __LINE__, #op, cudaGetErrorString(cudaStat)); \
        return; \
    }

    #define CB(op) \
    cublas_status = (op); \
	if (cublas_status != CUBLAS_STATUS_SUCCESS) \
    { \
        printf ("-----------------\n    Error occurred (cublas)\n   line %d: %s\n    Error text:\"%s\"\n-----------------", __LINE__, #op, cublasGetErrorString(cublas_status)); \
        return; \
    }

    #define CS(op) \
    cusolver_status = (op); \
	if (cusolver_status != CUSOLVER_STATUS_SUCCESS) \
    { \
        CC(cudaMemcpy(&devInfoHost, devInfo, sizeof(int), cudaMemcpyDeviceToHost));\
        printf ("-----------------\n    Error occurred (cusolver, devinfo %d)\n   line %d: %s\n    Error text:\"%s\"\n-----------------", devInfoHost, __LINE__, #op, cusolverGetErrorString(cusolver_status)); \
        return; \
    }

    #define LL printf ("_%d_\n", __LINE__);

    CC(cudaMalloc ((void**) &deviceInputData, sizeof (InputDataOnDevice)));

    CC(cudaMemcpyToSymbol(inputDataPtr, &deviceInputData, sizeof(InputDataOnDevice*)));

    int size3 = inputData.discretizationSize_[0] *
                inputData.discretizationSize_[1] *
                inputData.discretizationSize_[2];

    #define PointConversion(var, type)\
    (Point3DDevice_t<type>) \
    {(type)(inputData.var.x),  \
     (type)(inputData.var.y), \
     (type)(inputData.var.z)}

    InputDataOnDevice hostDataCopy = {inputData.sourcePos_,
                                      thrust::complex<float> (0, (float) (2*3.141592f*inputData.f_/inputData.c_)),
                                      inputData.anomalyPos_,
                                      inputData.anomalySize_,
                                      inputData.discretizationSize_,
                                      inputData.discreteBlockSize_,
                                      size3,
                                      inputData.discretizationSize_[0] *
                                      inputData.discretizationSize_[1],
                                      inputData.discretizationSize_[0],
                                      (float)(4*3.141592f*3.141592f*inputData.f_*inputData.f_*
                                      inputData.discreteBlockSize_[0]*inputData.discreteBlockSize_[1]*inputData.discreteBlockSize_[2])};

    #undef PointConversion

    CC(cudaMemcpy (deviceInputData, &hostDataCopy, sizeof (InputDataOnDevice), cudaMemcpyHostToDevice));

    /*printf ("About to call kernel\n");
    DevicePrintData<<<1, 1>>> (deviceInputData);
    CC(cudaDeviceSynchronize ());
    printf ("Kernel returned\n");*/

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

    LL

    thrust::device_vector<Point3DDevice_t<float> > indexes (size3);
    LL

    thrust::tabulate (indexes.begin(), indexes.end(), IndexFromSequence ());
    LL

    thrust::transform (deviceKMatrix.begin (), deviceKMatrix.end (), indexes.begin (), deviceKMatrix.begin (), ModifyKMatrix ());
    LL

    thrust::device_vector<thrust::complex<float> > deviceAMatrix (size3*size3);
    LL

    SetAMatrix sMatrixSetter (deviceKMatrix.data ().get (), indexes.data ().get ());

    thrust::tabulate (deviceAMatrix.begin (), deviceAMatrix.end (), sMatrixSetter);

    LL

    /// ////////////////////////////////////
    /// solution part (linear system, not fft)
    /// ////////////////////////////////////


    /// 1. Creating handles

    cublasHandle_t cublasH = nullptr;
    CB(cublasCreate(&cublasH));

    cusolverDnHandle_t cudenseH = nullptr;
    CS(cusolverDnCreate(&cudenseH));
    LL

    /// 2. Setting up data

    thrust::device_vector<thrust::complex<float> > ones (size3, thrust::complex<float> (-1.0f, -1.0f)); // is it -1 or -1 - i ?
    thrust::device_vector<thrust::complex<float> > reductedA_solution (size3, 0.0f);

    thrust::complex<float> alpha (1.0f, 1.0f);
    thrust::complex<float> beta (0.0f, 0.0f);

    /// reductedA_solution = alpha*A*ones+beta*reductedA_solution = A*ones
    CB(cublasCgemv (cublasH, CUBLAS_OP_N, size3, size3,
                    reinterpret_cast <cuComplex*> (&alpha),
                    reinterpret_cast <cuComplex*> (deviceAMatrix.data ().get ()),
                    size3,
                    reinterpret_cast <cuComplex*> (ones.data ().get ()), 1,
                    reinterpret_cast <cuComplex*> (&beta),
                    reinterpret_cast <cuComplex*> (reductedA_solution.data ().get ()), 1));




    /// need to subtract ui from every diagonal element of A
    /// strategy1: run tabulate on something of size size3 and modify A alongside
    /// strategy2: run for_each on a sequence, but need to create sequence of size size3

    ///using strategy2

    thrust::device_vector<int> seq (size3);
    thrust::sequence (seq.begin (), seq.end ());
    ModifyAMatrix modificatorA (deviceAMatrix.data ().get (), indexes.data ().get ());
    thrust::for_each (seq.begin(), seq.end(), modificatorA);

    LL

    /// 3. Querying workspace for cusolverDn

    //thrust::device_vector<thrust::complex<float> > tau (size3);

    int workspaceSize = 0;

    CS(cusolverDnCgeqrf_bufferSize(cudenseH,
                                   size3,
                                   size3,
                                   reinterpret_cast <cuComplex*> (deviceAMatrix.data ().get ()),
                                   size3,
                                   &workspaceSize));

    thrust::device_vector<thrust::complex<float> > workspace (workspaceSize);

    LL

    /// 4. Computing QR decomposition

    thrust::device_vector<thrust::complex<float> > tau (size3);

    CC(cudaMalloc ((void**)&devInfo, sizeof(int)));

    LL

    CS(cusolverDnCgeqrf(cudenseH,
                        size3,
                        size3,
                        reinterpret_cast <cuComplex*> (deviceAMatrix.data ().get ()),
                        size3,
                        reinterpret_cast <cuComplex*> (tau.data ().get ()),
                        reinterpret_cast <cuComplex*> (workspace.data ().get ()),
                        workspaceSize,
                        devInfo));
    CC(cudaDeviceSynchronize());

    LL

    /// 5. compute Q^H*B
    CS(cusolverDnCunmqr(cudenseH,
                        CUBLAS_SIDE_LEFT,
                        CUBLAS_OP_C,
                        size3,
                        1,
                        size3, //k 	host 	input 	number of elementary relfections
                        reinterpret_cast <cuComplex*> (deviceAMatrix.data ().get ()),
                        size3,
                        reinterpret_cast <cuComplex*> (tau.data ().get ()),
                        reinterpret_cast <cuComplex*> (reductedA_solution.data ().get ()),
                        size3,
                        reinterpret_cast <cuComplex*> (workspace.data ().get ()),
                        workspaceSize,
                        devInfo));




    CC(cudaDeviceSynchronize());

    LL

    /// 6. solve Rx = Q^H*B
    CB(cublasCtrsm(cublasH,
                   CUBLAS_SIDE_LEFT,
                   CUBLAS_FILL_MODE_UPPER,
                   CUBLAS_OP_N,
                   CUBLAS_DIAG_NON_UNIT,
                   size3,
                   1,
                   reinterpret_cast <cuComplex*> (&alpha),
                   reinterpret_cast <cuComplex*> (deviceAMatrix.data ().get ()),
                   size3,
                   reinterpret_cast <cuComplex*> (reductedA_solution.data ().get ()),
                   size3));

    CC(cudaDeviceSynchronize());


    /// 7. receiver convolution

    ReceiverConvolution recConv (reductedA_solution.data().get(),
                                 indexes.data ().get (),
                                 &seq);

    thrust::host_vector<Point3DDevice_t <float> > deviceReceivers (inputData.receivers_,
                                                                   inputData.receivers_ + inputData.Nreceivers_);

    thrust::transform (deviceReceivers.begin (),
                       deviceReceivers.end (),
                       retData->begin (),
                       recConv);

    CC(cudaFree (deviceInputData));
    CC(cudaFree (devInfo));
    LL

    #undef CC
    #undef CB
    #undef CS
    #undef PointConversion

}


static const char * cublasGetErrorString (cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

static const char * cusolverGetErrorString (cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "The operation completed successfully";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "The library was not initialized";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "Invalid parameters were passed";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "The device only supports compute capability 2.0 and above";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}


//=================================================================
