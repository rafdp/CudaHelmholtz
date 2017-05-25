
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

struct FillRadialQ_lq
{
    complex_t* deviceKMatrixPtr;
    point_t * deviceIndexesPtr;
    complex_t* Q_lq;
    pointInt_t size;
    int l;
    int q;
    
    __host__
    FillRadialQ_lq (complex_t * deviceKMatrixPtr_,
                    point_t * deviceIndexesPtr_,
                    complex_t * Q_lq_,
                    pointInt_t size_,
                    int l_,
                    int q_) :
        deviceKMatrixPtr (deviceKMatrixPtr_),
        deviceIndexesPtr (deviceIndexesPtr_),
        Q_lq             (Q_lq_),
        size             (size_),
        l                (l_),
        q                (q_)
    {}

    __device__
    void operator()(int idx) const
    {
        int idxx = idx % (2*size.x);
        int idxy = idx / (2*size.x);
        if (idxx >= size.x) idxx -= size.x;
        else idxx = size.x - idxx - 1;
        if (idxy >= size.y) idxy -= size.y;
        else idxy = size.y - idxy - 1;
        
        int emIdx = l*size.x*size.y + idxy * size.x + idxx;
        int recIdx = q*size.x*size.y + idxx * size.x + idxy;
        if (emIdx == recIdx)
        {
            point_t pos = *(deviceIndexesPtr + emIdx);
            point_t dr = {inputDataPtr->sourcePos_.x - pos.x,
                          inputDataPtr->sourcePos_.y - pos.y,
                          inputDataPtr->sourcePos_.z - pos.z};
            float len = dr.len ();
            *(Q_lq + idx) = -thrust::exp (inputDataPtr->uiCoeff_ * len) / 
                            (4 * 3.141592f * len);
            return;
        }
        
        point_t rec = *(deviceIndexesPtr + 
                        q*size.x*size.y + 
                        idxx * size.x + 
                        idxy);
        
        point_t em = *(deviceIndexesPtr + 
                       l*size.x*size.y + 
                       idxy * size.x + 
                       idxx);
        
        point_t dr = {rec.x - em.x,
                      rec.y - em.y,
                      rec.z - em.z};
                      
        float len = dr.len ();
        
        *(Q_lq + idx) = *(deviceKMatrixPtr+emIdx) * thrust::exp (inputDataPtr->uiCoeff_ * len) / (4 * (3.141592f) * len);
    }
};

struct FillV_q
{
    complex_t* source;
    complex_t* V_q;
    pointInt_t size;
    int q;
    
    __host__
    FillV_q (complex_t * source_,
             complex_t * V_q_,
             pointInt_t size_,
             int q_) :
        source (source_),
        V_q (V_q_),
        size     (size_),
        q (q_)
    {}

    __device__
    void operator()(int idx) const
    {
        int idxx = idx % (2*size.x);
        int idxy = idx / (2*size.x);
        
        *(V_q + (size.x*size.y)*2+size.x+size.x*2*idxy + idxx) = 
        *(source + q*size.x*size.y + idxy * size.x + idxx);
    }
};

struct ElementwiseMultiplier_SumFFT
{
    complex_t* V_q;
    complex_t* Q_lq;
    complex_t* Acc;
    __host__
    ElementwiseMultiplier_SumFFT (complex_t * Q_lq_,
                                  complex_t * V_q_,
                                  complex_t* Acc_) :
        Q_lq (Q_lq_),
        V_q (V_q_),
        Acc (Acc_)
    {}

    __device__
    void operator()(int idx) const
    {
        *(Acc + idx) += *(Q_lq + idx) * *(V_q + idx);
    }
};

struct FillS_l
{
    complex_t* destination;
    complex_t* acc;
    pointInt_t size;
    int l;
    
    __host__
    FillS_l (complex_t * destination_,
             complex_t * acc_,
             pointInt_t size_,
             int l_) :
        destination (destination_),
        acc (acc_),
        size     (size_),
        l (l_)
    {}

    __device__
    void operator()(int idx) const
    {
        int idxx = idx % (size.x);
        int idxy = idx / (size.x);
        
        *(destination + l*size.x*size.y + idxy*size.x + idxx) = 
        *(acc + size.x*size.y*2+size.x+size.x*2*idxy + idxx);
    }
};


struct MatVecFunctorFFT : MatVecFunctorBase
{
    complex_t* deviceKMatrixPtr;
    point_t * deviceIndexesPtr;
    int* seq;
    pointInt_t size;
    
    __host__
    MatVecFunctorFFT (complex_t * deviceKMatrixPtr_,
                      point_t *   deviceIndexesPtr_,
                      int* seq_,
                      pointInt_t size_) :
        deviceKMatrixPtr (deviceKMatrixPtr_),
        deviceIndexesPtr (deviceIndexesPtr_),
        seq      (seq_),
        size     (size_)
    {}

    __host__
    void operator()(cuComplex* source, cuComplex* destination) const
    {
        LL
        const int size2 = size.x*size.y;
        thrust::device_vector <complex_t> Q_lq (size2*size2*4,
                                                complex_t(0.0f, 0.0f));
        
        thrust::device_vector <complex_t> V_q (Q_lq);
        
        thrust::device_vector <complex_t> acc (Q_lq);
        
        LL
        cufftHandle plan;
        cufftCreate(&plan);
        cufftPlan2d(&plan, 2*size.x, 2*size.y, CUFFT_C2C);
        LL
        printf ("About to loop over l\n");
        for (int l = 0; l < size.z; l++)
        {
        printf ("About to loop over q\n");
            for (int q = 0; q < size.z; q++)
            {        
                printf ("l = %d q = %d\n", l, q);
                FillRadialQ_lq fr (deviceKMatrixPtr,
                                   deviceIndexesPtr,
                                   Q_lq.data ().get (),
                                   size, l, q);
                LL 
                thrust::for_each (thrust::device, seq, seq + 4*size2, fr);
                LL
                cufftExecC2C(plan, 
                             reinterpret_cast<cufftComplex*> (Q_lq.data ().get ()),
                             reinterpret_cast<cufftComplex*> (Q_lq.data ().get ()), CUFFT_FORWARD);
                LL
                
                FillV_q fv (reinterpret_cast<complex_t*> (source),
                            V_q.data ().get (),
                            size, q);
                thrust::for_each (thrust::device, seq, seq + size2, fv);
                LL
                cufftExecC2C(plan, 
                             reinterpret_cast<cufftComplex*> (V_q.data ().get ()),
                             reinterpret_cast<cufftComplex*> (V_q.data ().get ()), CUFFT_FORWARD);
                LL
                ElementwiseMultiplier_SumFFT ems (Q_lq.data ().get (), 
                                                  V_q.data ().get (), 
                                                  acc.data ().get ());
                thrust::for_each (thrust::device, seq, seq + size2*4, fv);
                
            }
            
            cufftExecC2C(plan, 
                         reinterpret_cast<cufftComplex*> (acc.data ().get ()),
                         reinterpret_cast<cufftComplex*> (acc.data ().get ()),
                         CUFFT_INVERSE);
            
            FillS_l fs (reinterpret_cast<complex_t*> (destination),
                        acc.data().get(), size, l);
        }
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
    
    thrust::device_vector<int> seq (size3);
    thrust::sequence (seq.begin (), seq.end ());
    
    ReduceEmittersToReceiver 
    <<<inputData.discretizationSize_[0]*
       inputData.discretizationSize_[1], 
       inputData.discretizationSize_[2]>>> 
        (deviceInputData,
         deviceKMatrix.data ().get (),
         reductedA_solution.data ().get (),
         seq.data().get (),
         indexes.data ().get ());




    /// need to subtract ui from every diagonal element of A
    /// strategy1: run tabulate on something of size size3 and modify A alongside
    /// strategy2: run for_each on a sequence, but need to create sequence of size size3

    ///using strategy2

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
    thrust::device_vector <complex_t> t0 (x_0);
    thrust::device_vector <complex_t> t1 (x_0);

    MatVecFunctor matvecf_ (cublasH, deviceAMatrix.data().get (), size3);
    
    
    
    MatVecFunctorFFT matvecf (deviceKMatrix.data().get (), indexes.data (). get (), seq.data ().get (), inputData.discretizationSize_);
    
    alpha = complex_t (-1.0f, 0.0f);
    
    matvecf (reinterpret_cast<cuComplex*> (x.data().get ()),
             reinterpret_cast<cuComplex*> (t1.data().get ()));
    
    matvecf_ (reinterpret_cast<cuComplex*> (x.data().get ()),
              reinterpret_cast<cuComplex*> (t0.data().get ()));
    
    CB (cublasCaxpy(cublasH, size3, reinterpret_cast<cuComplex*> (&alpha), 
                    reinterpret_cast<cuComplex*> (t1.data().get ()), 1, 
                    reinterpret_cast<cuComplex*> (t0.data().get ()), 1));
    
    float norm = 0.0f;
    
    CB (cublasScnrm2 (cublasH, size3, reinterpret_cast<cuComplex*> (t0.data().get ()), 1, &norm));
    
    printf ("norm = %e\n", norm);
    
    return;
    
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
