
//=================================================================

#include "CudaCalc.h"
#define FFT_FUNCTOR
//#define MATRIX_FUNCTOR
//#define QR_SOLUTION 
#define TIME_TESTING

#ifdef TIME_TESTING
#define TT(x) x
#else
#define TT(x) 
#endif

#include "MatVecAlgorithms.cu"
extern "C"
void ExternalKernelCaller (InputData_t* inputDataPtr_, std::vector<std::complex<float> >* retData)
{ 
    TT(timespec initStart = {};)
    TT(timespec initEnd = {};)
    TT(timespec algEnd = {};)

    TT(clock_gettime(CLOCK_REALTIME, &initStart);)
    InputData_t& inputData = *inputDataPtr_;

    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;

    int size3 = inputData.discretizationSize_[0] *
                inputData.discretizationSize_[1] *
                inputData.discretizationSize_[2];
    complex_t uiCoeff = complex_t (0.0f, (float) (2*3.141592f*inputData.f_/inputData.c_));
    float w2h3 = 4*3.141592f*3.141592f*inputData.f_*inputData.f_*
                                       inputData.discreteBlockSize_[0]*
		 		       inputData.discreteBlockSize_[1]*
				       inputData.discreteBlockSize_[2];



    thrust::host_vector<complex_t > hostDs2Matrix (inputData.ds2_, inputData.ds2_ + size3);
    thrust::device_vector<complex_t> deviceKMatrix   (hostDs2Matrix);
    thrust::device_vector<complex_t> deviceDS2Matrix (hostDs2Matrix);
    
    thrust::device_vector<point_t > indexes (size3);
    IndexFromSequence index_filler (inputData.discretizationSize_,
		                    inputData.discreteBlockSize_,
				    inputData.anomalyPos_); 
    thrust::tabulate (indexes.begin(), indexes.end(), index_filler);
    
    thrust::device_vector<int> seq (size3 * 4*inputData.discretizationSize_[2]);
    thrust::sequence (seq.begin (), seq.end ());

    ModifyKMatrix k_matrix_modifier (inputData.sourcePos_, uiCoeff, w2h3);
    thrust::transform (deviceKMatrix.begin (), 
		       deviceKMatrix.end (), 
		       indexes.begin (), 
		       deviceKMatrix.begin (), 
		       k_matrix_modifier);

    thrust::device_vector <complex_t> reductedA_solution (size3, 0.0f);
    cublasHandle_t cublasH = nullptr;
    CB(cublasCreate(&cublasH));
#ifdef FFT_FUNCTOR


    thrust::device_vector <complex_t> x                 (size3, complex_t (1.0f, 0.0f));

    
    ReduceEmittersToReceiver 
    <<<inputData.discretizationSize_[0]*
       inputData.discretizationSize_[1], 
       inputData.discretizationSize_[2]>>> 
        (deviceKMatrix.data ().get (),
         reductedA_solution.data ().get (),
         seq.data().get (),
         indexes.data ().get (),
	 uiCoeff,
	 size3);

    MatVecFunctorFFT matvecf (deviceDS2Matrix.data().get (), 
		              indexes.data (). get (), 
			      seq.data ().get (), 
			      inputData.discretizationSize_,
			      w2h3,
			      uiCoeff,
			      inputData.sourcePos_);
    BiCGStabCudaSolver solver (size3, reductedA_solution.data().get (), x.data().get ());
    
    TT(clock_gettime(CLOCK_REALTIME, &initEnd);)
    size_t usedSize = solver.solve (&matvecf);

    TT(clock_gettime(CLOCK_REALTIME, &algEnd);)
    CB (cublasCcopy (cublasH, size3,
                     (reinterpret_cast <cuComplex*> (x.data().get ())), 1,
                     (reinterpret_cast <cuComplex*> (reductedA_solution.data().get ())), 1));
    cudaDeviceSynchronize ();
#endif 
#ifdef MATRIX_FUNCTOR

    
    thrust::device_vector<complex_t > deviceAMatrix (size3*size3);
    
    SetAMatrix aMatrixSetter (deviceKMatrix.data ().get (), 
                              indexes.data ().get (), 
			      inputData.sourcePos_, 
			      uiCoeff,
			      size3);

    thrust::tabulate (deviceAMatrix.begin (), deviceAMatrix.end (), aMatrixSetter);
    ReduceEmittersToReceiver 
    <<<inputData.discretizationSize_[0]*
       inputData.discretizationSize_[1], 
       inputData.discretizationSize_[2]>>> 
        (deviceKMatrix.data ().get (),
         reductedA_solution.data ().get (),
         seq.data().get (),
         indexes.data ().get (),
	 uiCoeff,
	 size3);
    
    ModifyAMatrix modificatorA (deviceAMatrix.data ().get (), 
                                indexes.data ().get (),
                                inputData.sourcePos_,
                                size3,
                                uiCoeff);
    
    PrintGrid<<<1, 1>>> (deviceAMatrix.data().get (), 4);
    thrust::device_vector <complex_t> x (size3, complex_t (1.0f, 0.0f));
    
    MatVecFunctor matvecf (cublasH, deviceAMatrix.data().get (), size3);
    cudaDeviceSynchronize ();
    BiCGStabCudaSolver solver (size3, reductedA_solution.data().get (), x.data().get ());

    TT(clock_gettime(CLOCK_REALTIME, &initEnd);)
    size_t usedSize = solver.solve (&matvecf);
    TT(clock_gettime(CLOCK_REALTIME, &algEnd);)
    CB (cublasCcopy (cublasH, size3,
                     (reinterpret_cast <cuComplex*> (x.data().get ())), 1,
                     (reinterpret_cast <cuComplex*> (reductedA_solution.data().get ())), 1));
    cudaDeviceSynchronize ();
#endif
#ifdef QR_SOLUTION

    cudaError_t cudaStat = cudaSuccess;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    int* devInfo = nullptr;
    int devInfoHost = 0;
    cusolverDnHandle_t cudenseH = nullptr;
    CS(cusolverDnCreate(&cudenseH));
    
    thrust::device_vector<complex_t > deviceAMatrix (size3*size3);
    
    SetAMatrix aMatrixSetter (deviceKMatrix.data ().get (), 
                              indexes.data ().get (), 
			      inputData.sourcePos_, 
			      uiCoeff,
			      size3);

    thrust::tabulate (deviceAMatrix.begin (), deviceAMatrix.end (), aMatrixSetter);
    ReduceEmittersToReceiver 
    <<<inputData.discretizationSize_[0]*
       inputData.discretizationSize_[1], 
       inputData.discretizationSize_[2]>>> 
        (deviceKMatrix.data ().get (),
         reductedA_solution.data ().get (),
         seq.data().get (),
         indexes.data ().get (),
	 uiCoeff,
	 size3);

    ModifyAMatrix modificatorA (deviceAMatrix.data ().get (), 
                                indexes.data ().get (),
                                inputData.sourcePos_,
                                size3,
                                uiCoeff);
    thrust::for_each (seq.begin(), seq.begin() + size3, modificatorA);
    int workspaceSize = 0;

    CS(cusolverDnCgeqrf_bufferSize(cudenseH,
                                   size3,
                                   size3,
                                   reinterpret_cast <cuComplex*> (deviceAMatrix.data ().get ()),
                                   size3,
                                   &workspaceSize));

    thrust::device_vector<complex_t> workspace (workspaceSize);


    TT(clock_gettime(CLOCK_REALTIME, &initEnd);)
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

    
    complex_t complex_one (1.0f, 0.0f); 
    /// 6. solve Rx = Q^H*B
    CB(cublasCtrsm(cublasH,
                   CUBLAS_SIDE_LEFT,
                   CUBLAS_FILL_MODE_UPPER,
                   CUBLAS_OP_N,
                   CUBLAS_DIAG_NON_UNIT,
                   size3,
                   1,
                   reinterpret_cast <cuComplex*> (&complex_one),
                   reinterpret_cast <cuComplex*> (deviceAMatrix.data ().get ()),
                   size3,
                   reinterpret_cast <cuComplex*> (reductedA_solution.data ().get ()),
                   size3));
    TT(clock_gettime(CLOCK_REALTIME, &algEnd);)
    
    size_t free_byte = 0;
    size_t total_byte = 0;

    cudaMemGetInfo( &free_byte, &total_byte );
    size_t usedSize = total_byte-free_byte;
    CC(cudaFree (devInfo));
#endif

#ifdef TIME_TESTING
    unsigned long long timeInit = (initEnd.tv_sec - initStart.tv_sec)*1000000000 + initEnd.tv_nsec-initStart.tv_nsec; 
    unsigned long long timeAlg =  (algEnd.tv_sec - initEnd.tv_sec)*1000000000 +    algEnd.tv_nsec-initEnd.tv_nsec; 

#ifdef FFT_FUNCTOR
#define FILE_ "fft_"
#elif defined MATRIX_FUNCTOR
#define FILE_ "matrix_"
#elif defined QR_SOLUTION
#define FILE_ "qr_"
#endif

#define WRITE_FILE(suffix, data, coeff) \
    FILE* fft_##suffix = fopen (FILE_ #suffix ".txt", "a"); \
    if (!fft_##suffix) return; \
    float value##suffix = (data); \
    fprintf (fft_##suffix, "%d %f\n", inputData.discretizationSize_[0], value##suffix/coeff); \
    fclose (fft_##suffix);
    WRITE_FILE (time_init, timeInit, 1e9f)
    WRITE_FILE (time_alg, timeAlg, 1e9f)
    WRITE_FILE (time_full, timeInit + timeAlg, 1e9f)
    WRITE_FILE (size, usedSize, (1024.0f*1024.0f))

#undef WRITE_FILE
#undef FILE_

#else
    thrust::device_vector<complex_t> ones (size3, complex_t (-1.0f, 0.0f)); 
    for (int i = 0; i < inputData.Nreceivers_; i++)
    {
        QLReduction qlRed (inputData.receivers_[i], 
			   reductedA_solution.data().get(), 
			   indexes.data ().get (), 
			   deviceKMatrix.data ().get (),
			   inputData.discreteBlockSize_,
			   uiCoeff);
        complex_t init (0.0f, 0.0f);
        ComplexAddition complexSum;
        thrust::transform (seq.begin (), seq.begin () + size3, ones.begin(), qlRed);
        (*retData)[i] = thrust::reduce (ones.begin(), ones.end(), init, complexSum);
    }

    CB(cublasDestroy (cublasH));
#ifdef QR_SOLUTION
    CS(cusolverDnDestroy (cudenseH));
#endif
#endif

    printf ("Cuda part ended\n");
    /// ////////////////////////////////////
    /// solution part (linear system, not fft)
    /// ////////////////////////////////////

    /// 1. Creating handles
    /*cublasHandle_t cublasH = nullptr;
    CB(cublasCreate(&cublasH));
#ifdef QR_SOLUTION
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    int* devInfo = nullptr;
    int devInfoHost = 0;
    cusolverDnHandle_t cudenseH = nullptr;
    CS(cusolverDnCreate(&cudenseH));
#endif

 /// 2. Setting up data
	    
    thrust::device_vector<complex_t> reductedA_solution (size3, 0.0f);
    ReduceEmittersToReceiver 
    <<<inputData.discretizationSize_[0]*
       inputData.discretizationSize_[1], 
       inputData.discretizationSize_[2]>>> 
        (deviceKMatrix.data ().get (),
         reductedA_solution.data ().get (),
         seq.data().get (),
         indexes.data ().get (),
	 uiCoeff,
	 size3);
    

    //begin comment here

    /// need to subtract ui from every diagonal element of A
    /// strategy1: run tabulate on something of size size3 and modify A alongside
    /// strategy2: run for_each on a sequence, but need to create sequence of size size3

    ///using strategy2
#ifndef FFT_FUNCTOR    
    ModifyAMatrix modificatorA (deviceAMatrix.data ().get (), 
                                indexes.data ().get (),
                                inputData.sourcePos_,
                                size3,
                                uiCoeff);
    thrust::for_each (seq.begin(), seq.begin() + size3, modificatorA);
    clock_gettime(CLOCK_REALTIME, &ts01);

    //end comment here


    /// 3. Querying workspace for cusolverDn
#ifdef QR_SOLUTION
    int workspaceSize = 0;

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

    
    complex_t complex_one (1.0f, 0.0f); 
    /// 6. solve Rx = Q^H*B
    CB(cublasCtrsm(cublasH,
                   CUBLAS_SIDE_LEFT,
                   CUBLAS_FILL_MODE_UPPER,
                   CUBLAS_OP_N,
                   CUBLAS_DIAG_NON_UNIT,
                   size3,
                   1,
                   reinterpret_cast <cuComplex*> (&complex_one),
                   reinterpret_cast <cuComplex*> (deviceAMatrix.data ().get ()),
                   size3,
                   reinterpret_cast <cuComplex*> (reductedA_solution.data ().get ()),
                   size3));
#endif
    //3-6. Bicgstab solution

    thrust::host_vector <complex_t> x_0 (size3, complex_t (1.0f, 0.0f));
    thrust::device_vector <complex_t> x (x_0);
    thrust::device_vector <complex_t> t0 (x_0);
    thrust::device_vector <complex_t> t1 (x_0);

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MATRIX_FUNCTOR    
    MatVecFunctor matvecf (cublasH, deviceAMatrix.data().get (), size3);
#elif defined FFT_FUNCTOR
    timespec ts10 = {}, ts11 = {};
    clock_gettime(CLOCK_REALTIME, &ts10); // Works on Linux
    MatVecFunctorFFT matvecf (deviceDS2Matrix.data().get (), 
		              indexes.data (). get (), 
			      seq.data ().get (), 
			      inputData.discretizationSize_,
			      w2h3,
			      uiCoeff,
			      inputData.sourcePos_);
    
    clock_gettime(CLOCK_REALTIME, &ts11); // Works on Linux
#endif
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifndef QR_SOLUTION

    BiCGStabCudaSolver solver (size3, reductedA_solution.data().get (), x.data().get ());

    timespec ts0 = {}, ts1 = {};
    /*clock_gettime(CLOCK_REALTIME, &ts0); // Works on Linux
    size_t usedSize = solver.solve (&matvecf_);
    clock_gettime(CLOCK_REALTIME, &ts1);
    unsigned long long time0 = (ts01.tv_sec - ts00.tv_sec)*1000000000 + ts01.tv_nsec-ts00.tv_nsec; 
    unsigned long long time1 = (ts1.tv_sec - ts0.tv_sec)*1000000000 + ts1.tv_nsec-ts0.tv_nsec; 
    printf ("MATRIX took %d %f ms\n", size1, (time1 + time0)/1000000.0f);
    FILE* matrix_data = fopen ("matrix_time.txt", "a");
    if (!matrix_data) return;
    fprintf (matrix_data, "%d %f\n", size1, (time1 + time0)/1000000000.0f);
    fclose (matrix_data);
    FILE* matrix_size = fopen ("matrix_size.txt", "a");
    if (!matrix_size) return;
    fprintf (matrix_size, "%d %f\n", size1, usedSize/1048576.0f);*
    ts0 = {};
    ts1 = {};
    clock_gettime(CLOCK_REALTIME, &ts0); // Works on Linux
    size_t usedSize = solver.solve (&matvecf);
    clock_gettime(CLOCK_REALTIME, &ts1);
    cudaDeviceSynchronize ();
    unsigned long long time2 = (ts11.tv_sec - ts10.tv_sec)*1000000000 + ts11.tv_nsec-ts10.tv_nsec; 
    unsigned long long time3 = (ts1.tv_sec - ts0.tv_sec)*1000000000 + ts1.tv_nsec-ts0.tv_nsec; 
    printf ("FFT took %f ms\n", (time2+time3)/1000000.0f);

    /*FILE* fft_data = fopen ("fft_time.txt", "a");
    if (!fft_data) return;
    fprintf (fft_data, "%d %f\n", size1, (time2 + time3)/1000000000.0f);
    fclose (fft_data);
    FILE* fft_size = fopen ("fft_size.txt", "a");
    if (!fft_size) return;
    fprintf (fft_size, "%d %f\n", size1, usedSize/1048576.0f);
    //printf ("Matrix is %.5f times faster than FFT (%d^3)\n", (1.0*time2 + time3)/(time1 + time0), inputData.discretizationSize_[0]);


    CC(cudaDeviceSynchronize());*

    CB (cublasCcopy (cublasH, size3,
                         (reinterpret_cast <cuComplex*> (x.data().get ())), 1,
                         (reinterpret_cast <cuComplex*> (reductedA_solution.data().get ())), 1));
    
#endif
    /// 7. receiver convolution

    thrust::device_vector<complex_t> ones (size3, complex_t (-1.0f, 0.0f)); 
    for (int i = 0; i < inputData.Nreceivers_; i++)
    {
        QLReduction qlRed (inputData.receivers_[i], 
			   reductedA_solution.data().get(), 
			   indexes.data ().get (), 
			   deviceKMatrix.data ().get (),
			   inputData.discreteBlockSize_,
			   uiCoeff);
        complex_t init (0.0f, 0.0f);
        ComplexAddition complexSum;
        thrust::transform (seq.begin (), seq.begin () + size3, ones.begin(), qlRed);
        (*retData)[i] = thrust::reduce (ones.begin(), ones.end(), init, complexSum);
    }

    CB(cublasDestroy (cublasH));
    printf ("Cuda part ended\n");

*/
}
#undef TT

//=================================================================
