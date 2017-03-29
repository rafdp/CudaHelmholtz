
//=================================================================

#include "CudaCalc.h"

__device__ InputDataOnDevice * inputDataPtr;

struct ModifyKMatrix
{
__device__
    complex_t operator() (complex_t& k, point_t& pos)
    {
        point_t dr = {inputDataPtr->sourcePos_.x - pos.x,
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
    complex_t * deviceKMatrixPtr;
    point_t * deviceIndexesPtr;

    SetAMatrix (complex_t * deviceKMatrixPtr_, point_t * deviceIndexesPtr_) :
        deviceKMatrixPtr (deviceKMatrixPtr_),
        deviceIndexesPtr (deviceIndexesPtr_)
    {}

__device__
    complex_t operator() (int idx)
    {
        int idx1 = idx % inputDataPtr->size3_; // receiver
        int idx2 = idx / inputDataPtr->size3_; // emitter
        if (idx1 == idx2) return thrust::complex <float> (0.0f, 0.0f);

        point_t pos1 = *(deviceIndexesPtr + idx1);
        point_t pos2 = *(deviceIndexesPtr + idx2);
        point_t dr = {pos1.x-pos2.x,
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
    complex_t * deviceAMatrixPtr;
    point_t * deviceIndexesPtr;

    ModifyAMatrix (complex_t * deviceAMatrixPtr_, point_t * deviceIndexesPtr_) :
        deviceAMatrixPtr (deviceAMatrixPtr_),
        deviceIndexesPtr (deviceIndexesPtr_)
    {}

__device__
    void operator() (int idx)
    {
        point_t pos = *(deviceIndexesPtr + idx);
        point_t dr = {inputDataPtr->sourcePos_.x - pos.x,
                                     inputDataPtr->sourcePos_.y - pos.y,
                                     inputDataPtr->sourcePos_.z - pos.z};
        float len = dr.len ();
        if (len < 0.0000001 && len > 0.0000001) return;
        *(deviceAMatrixPtr + idx*(inputDataPtr->size3_+1)) = -thrust::exp (inputDataPtr->uiCoeff_ * len) / (4 * 3.141592f * len);

    }
};


struct QLReduction
{
	const point_t receiver;
    complex_t * deviceLambdaPtr;
    point_t * deviceIndexesPtr;
    complex_t * deviceKMatrixPtr;

    __host__
	QLReduction (point_t receiver_,
                 complex_t * deviceLambdaPtr_,
                 point_t * deviceIndexesPtr_,
                 complex_t * deviceKMatrixPtr_) :
        receiver (receiver_),
        deviceLambdaPtr (deviceLambdaPtr_),
        deviceIndexesPtr (deviceIndexesPtr_),
        deviceKMatrixPtr (deviceKMatrixPtr_)
    {}

    __device__
	complex_t operator()(int idx) const
	{
		point_t& r = *(deviceIndexesPtr + idx);

		point_t dr = {r.x - receiver.x +
                                      inputDataPtr->discreteBlockSize_.x / 2.0,
                                      r.y - receiver.y +
                                      inputDataPtr->discreteBlockSize_.y / 2.0,
                                      r.z - receiver.z +
                                      inputDataPtr->discreteBlockSize_.z / 2.0};

		float len = dr.len ();

		return (*(deviceKMatrixPtr + idx)) * (complex_t (1.0f, 0.0f) + *(deviceLambdaPtr + idx)) * //is it 1 or 1+i?
                thrust::exp(inputDataPtr -> uiCoeff_ * len) / (4 * 3.141592f * len);
	}
};

struct ComplexAddition
{
    __host__ __device__
	complex_t operator()(const complex_t& a, const complex_t& b) const
	{
		return a + b;
	}
};

struct IndexFromSequence
{
    __device__
    point_t operator() (int idx) const
    {

        point_t point = { 1.0f * (idx % inputDataPtr->size1_),
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
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    LL

    cublasHandle_t cublasH = nullptr;
    CB(cublasCreate(&cublasH));

    thrust::host_vector <complex_t> A_ (9);
    A_[0] = complex_t (1.0f, 0.0f);
    A_[3] = complex_t (1.0f, 0.0f);
    A_[6] = complex_t (1.0f, 0.0f);
    A_[1] = complex_t (1.0f, 0.0f);
    A_[4] = complex_t (-4.0f, 0.0f);
    A_[7] = complex_t (1.0f, 0.0f);
    A_[2] = complex_t (1.0f, 0.0f);
    A_[5] = complex_t (3.0f, 0.0f);
    A_[8] = complex_t (4.0f, 0.0f);
    thrust::host_vector <complex_t> b_ (3);

    LL
    b_[0] = complex_t (-5.0f, 0.0f);
    b_[1] = complex_t (35.0f, 0.0f);
    b_[2] = complex_t (-18.0f, 0.0f);

    LL
    thrust::device_vector <complex_t> x (3);
    thrust::device_vector <complex_t> A (A_);
    thrust::device_vector <complex_t> b (b_);

    LL
    BiCGStabCudaSolver solver (3, b.data().get (), A.data().get (), &cublasH);

    for (int i = 5; i < 10; i++)
    {
        LL
        solver.solve (x.data().get (), i);
        LL
        thrust::host_vector <complex_t> x_ (x);

        printf ("After %d iterations: %g+(%g)i  %g+(%g)i  %g+(%g)i\n\n", i,
                x_[0].real (), x_[0].imag (),
                x_[1].real (), x_[1].imag (),
                x_[2].real (), x_[2].imag ());
    }

	/*InputData_t& inputData = *inputDataPtr_;

	InputDataOnDevice* deviceInputData = nullptr;

	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;
    int* devInfo = nullptr;
    int devInfoHost = 0;

    CC(cudaMalloc ((void**) &deviceInputData, sizeof (InputDataOnDevice)));

    CC(cudaMemcpyToSymbol(inputDataPtr, &deviceInputData, sizeof(InputDataOnDevice*)));

    int size3 = inputData.discretizationSize_[0] *
                inputData.discretizationSize_[1] *
                inputData.discretizationSize_[2];

    InputDataOnDevice hostDataCopy = {inputData.sourcePos_,
                                      complex_t (0.0f, (float) (2*3.141592f*inputData.f_/inputData.c_)),
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

    CC(cudaMemcpy (deviceInputData, &hostDataCopy, sizeof (InputDataOnDevice), cudaMemcpyHostToDevice));

    printf ("About to call kernel\n");
    DevicePrintData<<<1, 1>>> (deviceInputData);
    CC(cudaDeviceSynchronize ());
    printf ("Kernel returned\n");

    thrust::host_vector<complex_t > hostDs2Matrix (size3);

    for (int x = 0; x < inputData.discretizationSize_[0]; x++)
    {
        for (int y = 0; y < inputData.discretizationSize_[1]; y++)
        {
            for (int z = 0; z < inputData.discretizationSize_[2]; z++)
            {
                int currentIndex = (x + y*inputData.discretizationSize_[0] + z*inputData.discretizationSize_[0]*inputData.discretizationSize_[1]);
                hostDs2Matrix[currentIndex] = complex_t (float (inputData.ds2_[currentIndex]), 0.0);
            }
        }
    }

    thrust::device_vector<complex_t > deviceKMatrix (hostDs2Matrix);

    LL

    thrust::device_vector<point_t > indexes (size3);
    LL

    thrust::tabulate (indexes.begin(), indexes.end(), IndexFromSequence ());
    LL

    thrust::transform (deviceKMatrix.begin (), deviceKMatrix.end (), indexes.begin (), deviceKMatrix.begin (), ModifyKMatrix ());
    LL

    thrust::device_vector<complex_t > deviceAMatrix (size3*size3);
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

    thrust::device_vector<complex_t> ones (size3, complex_t (-1.0f, 0.0f)); // is it -1 or -1 - i ?
    thrust::device_vector<complex_t> reductedA_solution (size3, 0.0f);

    complex_t alpha (1.0f, 0.0f);
    complex_t beta (0.0f, 0.0f);

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

    int workspaceSize = 0;

    CS(cusolverDnCgeqrf_bufferSize(cudenseH,
                                   size3,
                                   size3,
                                   reinterpret_cast <cuComplex*> (deviceAMatrix.data ().get ()),
                                   size3,
                                   &workspaceSize));

    thrust::device_vector<complex_t> workspace (workspaceSize);

    LL

    /// 4. Computing QR decomposition

    thrust::device_vector<complex_t> tau (size3);

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

    for (int i = 0; i < inputData.Nreceivers_; i++)
    {
        QLReduction qlRed (inputData.receivers_[i], reductedA_solution.data().get(), indexes.data ().get (), deviceKMatrix.data ().get ());
        complex_t init (0.0f, 0.0f);
        ComplexAddition complexSum;
        thrust::transform (seq.begin (), seq.end (), ones.begin(), qlRed);
        (*retData)[i] = thrust::reduce (ones.begin(), ones.end(), init, complexSum);
    }



    CC(cudaFree (deviceInputData));
    CC(cudaFree (devInfo));
    LL*/


}


//=================================================================
