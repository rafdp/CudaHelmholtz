

#include "CudaCalc.h"


BiCGStabCudaSolver::BiCGStabCudaSolver (int n, 
                                        complex_t* device_b, 
                                        complex_t* device_workspace) :
    n_ (n),
    device_b_ (device_b),
    device_x_ (device_workspace)
{
    if (device_x_ == nullptr)
    {
        printf ("----------------------------------------------------------\n");
        printf ("    ERROR: BiCGStabCudaSolver (...) device_x = nullptr\n");
        printf ("----------------------------------------------------------\n");
        exit (1);
    }

    if (device_b_ == nullptr)
    {
        printf ("----------------------------------------------------------\n");
        printf ("    ERROR: BiCGStabCudaSolver (...) device_A = nullptr\n");
        printf ("----------------------------------------------------------\n");
        exit (1);
    }

    if (n < 1)
    {
        printf ("----------------------------------------------------------\n");
        printf ("    ERROR: BiCGStabCudaSolver (...) invalid n = %d\n", n_);
        printf ("----------------------------------------------------------\n");
        exit (1);
    }

}

void BiCGStabCudaSolver::solve (MatVecFunctorBase* matVec, size_t nIter, float tol)
{
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    if (!nIter)
    {
        printf ("----------------------------------------------------------\n");
        printf ("    ERROR: solve: invalid nIter = 0\n");
        printf ("----------------------------------------------------------\n");
        exit (1);
    }
    
    cublasHandle_t cublasHandle_ = nullptr;
    CB(cublasCreate(&cublasHandle_));

    /// Initializing scalars:
    complex_t rho    = complex_t(1.0f, 0.0f),
              rhoOld = complex_t(1.0f, 0.0f),
              alpha  = complex_t(1.0f, 0.0f),
              omega  = complex_t(1.0f, 0.0f),
              omegaT = complex_t(0.0f, 0.0f),
              beta   = complex_t(0.0f, 0.0f),
              h      = complex_t(0.0f, 0.0f);
    float norm = 0.0f;

    complex_t minusOne = complex_t(-1.0f, 0.0f);
    complex_t one      = complex_t(1.0f, 0.0f);
    complex_t zero     = complex_t(0.0f, 0.0f);
    complex_t normC    = complex_t(0.0f, 0.0f);

    #define cc(x) (reinterpret_cast <cuComplex*> (x))
    
    cuComplex* x = cc(device_x_);
    cuComplex* b = cc(device_b_);

    /// Initializing vectors:
    thrust::device_vector<complex_t> r_ (n_, complex_t(0.0f, 0.0f));
    
    cuComplex* r = cc(r_.data().get());
    (*matVec) (x, r);
    CB (cublasCscal(cublasHandle_, n_, cc(&minusOne), r, 1));
    CB (cublasCaxpy(cublasHandle_, n_, cc(&one), b, 1, r, 1));

    thrust::device_vector<complex_t> r0_ (r_);
    
    cuComplex* r0 = cc(r0_.data().get());

    thrust::device_vector<complex_t> v_ (n_, complex_t(0.0f, 0.0f));
    
    cuComplex* v = cc(v_.data().get());

    thrust::device_vector<complex_t> p_ (v_);
    
    cuComplex* p = cc(p_.data().get());

    thrust::device_vector<complex_t> t_ (v_);
    
    cuComplex* t = cc(t_.data().get());

    thrust::device_vector<complex_t> s_ (v_);
    
    cuComplex* s = cc(s_.data().get());

    #define cn(x) if (x.real() != x.real () || x.imag() != x.imag ()) { break;}
    
    int restarts = 0;
    float r0Norm = 0.0f;
    CB (cublasScnrm2 (cublasHandle_, n_, r0, 1, &r0Norm));
    float bNorm = 0.0f;
    CB (cublasScnrm2 (cublasHandle_, n_, b, 1, &bNorm));
    int iOld = 0;
    int consecutiveRestarts = 0;
    
    const float tolerance = tol;
    const float zeroNorm = tolerance*tolerance;
    const int Nrestarts = 10;
    
    for (int i = 0; i < nIter; i++)
    {
        /// 1. rho = r0 dot r
        rhoOld = rho;
        CB (cublasCdotc (cublasHandle_, n_,
                         r0, 1,
                         r, 1,
                         cc(&rho)));
        cn (rho) 
        
        if (i != 0 && (thrust::abs (rho) < tolerance*r0Norm || 
                       thrust::abs (rhoOld) < zeroNorm || 
                       thrust::abs (omega) < zeroNorm || 
                       thrust::abs (alpha) < zeroNorm))
        {
            /// 2.1. r = b - mat * x
            (*matVec) (x, r);
            CB (cublasCscal(cublasHandle_, n_, cc(&minusOne), r, 1));
            CB (cublasCaxpy(cublasHandle_, n_, cc(&one), b, 1, r, 1));
            
            CB (cublasCcopy (cublasHandle_, n_,
                         r, 1,
                         r0, 1));
    
            CB (cublasScnrm2 (cublasHandle_, n_, r0, 1, &r0Norm));
            
            CB (cublasCdotc (cublasHandle_, n_,
                         r0, 1,
                         r, 1,
                         cc(&rho)));
            if (restarts == 0) i = 0;
            restarts++;
            if (iOld == i-1) consecutiveRestarts++;
            iOld = i;
            if (consecutiveRestarts >= Nrestarts) break;
            
            alpha  = complex_t(1.0f, 0.0f);
            omega  = complex_t(1.0f, 0.0f);
        }
        
        if (i == 0 && thrust::abs (rho) < zeroNorm) break;
        
        
        /// 2.0.0. beta = rho*alpha/(rhoOld*omega))
        beta = (rho/rhoOld) * (alpha/omega);
        cn(beta)
        
        /// 2.0.1. p -= omega*v
        omega = -omega;
        CB (cublasCaxpy(cublasHandle_, n_, cc(&omega), v, 1, p, 1));
        omega = -omega;

        /// 2.0.2. p *= beta
        CB (cublasCscal(cublasHandle_, n_, cc(&beta), p, 1));

        /// 2.0.3. p += r
        CB (cublasCaxpy(cublasHandle_, n_, cc(&one), r, 1, p, 1));


        /// 3. v = A*p
        
        (*matVec) (p, v);
        
        /// 4. alpha = rho/(r0, v)
        CB (cublasCdotc (cublasHandle_, n_, r0, 1, v, 1, cc(&alpha)));
        alpha = rho/alpha;
        if ( thrust::abs (alpha) < zeroNorm) break;
        cn(alpha)
        
        /// 4. s = r - alpha*v
        CB (cublasCcopy (cublasHandle_, n_,
                         r, 1,
                         s, 1));
        alpha = -alpha;
        CB (cublasCaxpy(cublasHandle_, n_, cc(&alpha), v, 1, s, 1));
        alpha = -alpha;

        /// 5. t = A*s
        
        (*matVec) (s, t);
        
        /// 6. omega = (t, s)/(t, t)
        
        CB (cublasCdotc (cublasHandle_, n_, t, 1, t, 1, cc(&normC)));
        if (norm < zeroNorm) omega = zero;
        else
        {
            CB (cublasCdotc (cublasHandle_, n_, t, 1, s, 1, cc(&omega)));
            omega /= normC;
            cn(omega)
        }

        /// 6. x += alpha*p + omega*s

        CB (cublasCaxpy(cublasHandle_, n_, cc(&alpha), p, 1, x, 1));
        CB (cublasCaxpy(cublasHandle_, n_, cc(&omega), s, 1, x, 1));

        /// 7. r = s - omega*t
        CB (cublasCcopy (cublasHandle_, n_, s, 1, r, 1));
        omega = -omega;
        CB (cublasCaxpy(cublasHandle_, n_, cc(&omega), t, 1, r, 1));
        omega = -omega;

        CB (cublasScnrm2 (cublasHandle_, n_, r, 1, &norm));
        if (norm < tolerance * bNorm) break;
    }
    CB(cublasDestroy(cublasHandle_));
#undef cn
}



