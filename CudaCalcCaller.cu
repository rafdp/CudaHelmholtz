
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

struct MatVecFunctor : MatVecFunctorBase
{
    cublasHandle_t cublasH;
    cuComplex* device_A_;
    size_t size_;
    

    __host__
    MatVecFunctor (cublasHandle_t cH,
                   complex_t * deviceAMatrixPtr,
                   size_t size) :
        cublasH   (cH),
        device_A_ (reinterpret_cast<cuComplex*> (deviceAMatrixPtr)),
        size_     (size)
    {}

    __host__
    void operator()(cuComplex* source, cuComplex* destination) const
    {
        cuComplex one = {1.0f, 0.0f};
        cuComplex zero = {0.0f, 0.0f};
        cublasCgemv (cublasH, CUBLAS_OP_N, size_, size_, &one,
                    device_A_, size_, source, 1, &zero, destination, 1);
    }
};
/*
struct MatVecFunctorFFT : MatVecFunctorBase
{
    cublasHandle_t cublasH;
    cuComplex* device_A_;
    point_t size_;
    

    __host__
    MatVecFunctor (cublasHandle_t cH,
                   complex_t * deviceAMatrixPtr,
                   point_t size) :
        cublasH   (cH),
        device_A_ (reinterpret_cast<cuComplex*> (deviceAMatrixPtr)),
        size_     (size)
    {}

    __host__
    void operator()(cuComplex* source, cuComplex* destination) const
    {
        size_t size3 = size_.x*size_.y*size_.z;
        size_t size2 = size_.x*size_.y;
        size_t size1 = size_.x;
        const int center = 2*size2 + size1;
        for (int l = 0; l < sizeZ; l++)
        {
            thrust::device_vector Q_lq (size2*size2*4, complex_t (0.0f, 0.0f));
            for (int q = 0; q < sizeZ; q++)
            {
                for (int iter2dRec = 0; iter2dRec < size2; iter2dRec ++)
                {
                    CB (cublasCcopy (cublasH, size1, Q_lq + center + iter2dRec*size1, 1, 
                                    device_A_ + (iter2dRec+l)*size3 + q*size2, 1)); // ++
                    
                    CB (cublasCcopy (cublasH, size1, Q_lq + center + (iter2dRec+1)*size1, -1, 
                                    device_A_ + (iter2dRec+l)*size3 + q*size2, 1)); // -+
                    
                    CB (cublasCcopy (cublasH, size1, Q_lq + center + (size2 - iter2dRec - 1)*size1, +1, 
                                    device_A_ + (iter2dRec+l)*size3 + q*size2, 1)); // +-
                }
            
            }
        }
    }
};*/


extern "C"
void ExternalKernelCaller (InputData_t* inputDataPtr_, std::vector<std::complex<float> >* retData)
{
    /*cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    
    cublasHandle_t cublasH = nullptr;*/
    //CB(cublasCreate(&cublasH));

    /*thrust::host_vector <complex_t> A_ (25);
    A_[0] = complex_t (7.0f, 0.0f);
    A_[1] = complex_t (0.0f, 0.0f);
    A_[2] = complex_t (17.0f, 0.0f);
    A_[3] = complex_t (8.0f, 0.0f);
    A_[4] = complex_t (-2.0f, 0.0f);
    A_[5] = complex_t (9.0f, 0.0f);
    A_[6] = complex_t (3.0f, 0.0f);
    A_[7] = complex_t (2.0f, 0.0f);
    A_[8] = complex_t (0.0f, 0.0f);
    A_[9] = complex_t (-3.0f, 0.0f);
    A_[10] = complex_t (-1.0f, 0.0f);
    A_[11] = complex_t (4.0f, 0.0f);
    A_[12] = complex_t (1.0f, 0.0f);
    A_[13] = complex_t (3.0f, 0.0f);
    A_[14] = complex_t (10.0f, 0.0f);
    A_[15] = complex_t (15.0f, 0.0f);
    A_[16] = complex_t (-5.0f, 0.0f);
    A_[17] = complex_t (-17.0f, 0.0f);
    A_[18] = complex_t (11.0f, 0.0f);
    A_[19] = complex_t (-10.0f, 0.0f);
    A_[20] = complex_t (2.0f, 0.0f);
    A_[21] = complex_t (-9.0f, 0.0f);
    A_[22] = complex_t (6.0f, 0.0f);
    A_[23] = complex_t (5.0f, 0.0f);
    A_[24] = complex_t (-2.0f, 0.0f);
    thrust::host_vector <complex_t> b_ (5);

    b_[0] = complex_t (92.0f, 0.0f);
    b_[1] = complex_t (-47.0f, 0.0f);
    b_[2] = complex_t (-14.0f, 0.0f);
    b_[3] = complex_t (86.0f, 0.0f);
    b_[4] = complex_t (-28.0f, 0.0f);

    thrust::host_vector <complex_t> x_0 (5);
    x_0[0] = complex_t (0.0f, 0.0f);
    x_0[1] = complex_t (0.0f, 0.0f);
    x_0[2] = complex_t (0.0f, 0.0f);
    x_0[3] = complex_t (0.0f, 0.0f);
    x_0[4] = complex_t (0.0f, 0.0f);
    thrust::device_vector <complex_t> x (x_0);
    thrust::device_vector <complex_t> A (A_);
    thrust::device_vector <complex_t> b (b_);

    BiCGStabCudaSolver solver (5, b.data().get (), A.data().get ());

    int nIter = 0;
    printf ("enter nIter\n");
    scanf ("%d", &nIter);
    solver.solve (x.data().get (), nIter);
    thrust::host_vector <complex_t> x_ (x);

    printf ("After %d iterations: \n%f+(%f)i  \n%f+(%f)i  \n%f+(%f)i\n%f+(%f)i  \n%f+(%f)i  \n\n",
            nIter,
            x_[0].real (), x_[0].imag (),
            x_[1].real (), x_[1].imag (),
            x_[2].real (), x_[2].imag (),
            x_[3].real (), x_[3].imag (),
            x_[4].real (), x_[4].imag ());
    return;*/
    

	InputData_t& inputData = *inputDataPtr_;

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

    
    thrust::device_vector<point_t > indexes (size3);
    
    thrust::tabulate (indexes.begin(), indexes.end(), IndexFromSequence ());
    
    thrust::transform (deviceKMatrix.begin (), deviceKMatrix.end (), indexes.begin (), deviceKMatrix.begin (), ModifyKMatrix ());
    
    thrust::device_vector<complex_t > deviceAMatrix (size3*size3);
    
    SetAMatrix sMatrixSetter (deviceKMatrix.data ().get (), indexes.data ().get ());

    thrust::tabulate (deviceAMatrix.begin (), deviceAMatrix.end (), sMatrixSetter);


    /// ////////////////////////////////////
    /// solution part (linear system, not fft)
    /// ////////////////////////////////////


    /// 1. Creating handles

    cublasHandle_t cublasH = nullptr;
    CB(cublasCreate(&cublasH));

    cusolverDnHandle_t cudenseH = nullptr;
    CS(cusolverDnCreate(&cudenseH));
    
    /// 2. Setting up data

    thrust::device_vector<complex_t> ones (size3, complex_t (-1.0f, 0.0f)); // is it -1 or -1 - i ?
    thrust::device_vector<complex_t> reductedA_solution (size3, 0.0f);

    complex_t alpha (1.0f, 0.0f);
    complex_t beta (0.0f, 0.0f);
    
    /*for (int i = 0; i < size3; i ++)
    {
        deviceAMatrix[i] *= alpha;
    }*/

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


    /// 3. Querying workspace for cusolverDn

    /*int workspaceSize = 0;

    CS(cusolverDnCgeqrf_bufferSize(cudenseH,
                                   size3,
                                   size3,
                                   reinterpret_cast <cuComplex*> (deviceAMatrix.data ().get ()),
                                   size3,
                                   &workspaceSize));

    thrust::device_vector<complex_t> workspace (workspaceSize);


    /// 4. Computing QR decomposition

    thrust::device_vector<complex_t> tau (size3);

    CC(cudaMalloc ((void**)&devInfo, sizeof(int)));


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
    //3-6. Bicgstab solution*/

    thrust::host_vector <complex_t> x_0 (size3, complex_t (1.0f, 0.0f));
    thrust::device_vector <complex_t> x (x_0);

    MatVecFunctor matvecf (cublasH, deviceAMatrix.data().get (), size3);
    BiCGStabCudaSolver solver (size3, reductedA_solution.data().get (), x.data().get ());

    solver.solve (&matvecf);

    CC(cudaDeviceSynchronize());

    CB (cublasCcopy (cublasH, size3,
                         (reinterpret_cast <cuComplex*> (x.data().get ())), 1,
                         (reinterpret_cast <cuComplex*> (reductedA_solution.data().get ())), 1));
    
    alpha = complex_t (1.0f, 0.0f);
    
    CB(cublasCscal(cublasH, size3,
                    reinterpret_cast <cuComplex*> (&alpha),
                    reinterpret_cast <cuComplex*> (reductedA_solution.data ().get ()), 1));





    /// 7. receiver convolution

    for (int i = 0; i < inputData.Nreceivers_; i++)
    {
        QLReduction qlRed (inputData.receivers_[i], reductedA_solution.data().get(), indexes.data ().get (), deviceKMatrix.data ().get ());
        complex_t init (0.0f, 0.0f);
        ComplexAddition complexSum;
        thrust::transform (seq.begin (), seq.end (), ones.begin(), qlRed);
        (*retData)[i] = thrust::reduce (ones.begin(), ones.end(), init, complexSum);
    }


    CB(cublasDestroy (cublasH));
    CC(cudaFree (deviceInputData));
    CC(cudaFree (devInfo));
    
    printf ("Cuda part ended\n");


}


//=================================================================
