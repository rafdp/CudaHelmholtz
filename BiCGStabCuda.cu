

#include "CudaCalc.h"



BiCGStabCudaSolver::BiCGStabCudaSolver (int n, complex_t* device_b, complex_t* device_A) :
    n_ (n),
    device_b_ (device_b),
    device_A_ (device_A)
{
    if (device_A_ == nullptr)
    {
        printf ("----------------------------------------------------------\n");
        printf ("    ERROR: BiCGStabCudaSolver (...) device_A = nullptr\n");
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

void BiCGStabCudaSolver::solve (complex_t* device_workspace, size_t nIter)
{
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    if (device_workspace == nullptr)
    {
        printf ("----------------------------------------------------------\n");
        printf ("    ERROR: solve: device_workspace = nullptr\n");
        printf ("----------------------------------------------------------\n");
        exit (1);
    }
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

    #define cc(x) (reinterpret_cast <cuComplex*> (x))
    
    cuComplex* x = cc(device_workspace);
    cuComplex* b = cc(device_b_);
    cuComplex* A = cc(device_A_);

    /// Initializing vectors:
    thrust::device_vector<complex_t> r_ (device_b_, device_b_ + n_);
    LL
    cuComplex* r = cc(r_.data().get());
    CB (cublasCgemv(cublasHandle_, CUBLAS_OP_N, n_, n_, cc(&minusOne), 
                    A, n_, x, 1, cc(&one), r, 1));

    thrust::device_vector<complex_t> r0_ (r_);
    LL
    cuComplex* r0 = cc(r_.data().get());

    thrust::device_vector<complex_t> v_ (n_, complex_t(0.0f, 0.0f));
    LL
    cuComplex* v = cc(v_.data().get());

    thrust::device_vector<complex_t> p_ (v_);
    LL
    cuComplex* p = cc(p_.data().get());

    thrust::device_vector<complex_t> t_ (v_);
    LL
    cuComplex* t = cc(t_.data().get());

    thrust::device_vector<complex_t> s_ (v_);
    LL
    cuComplex* s = cc(s_.data().get());

    LL
    //CB (cublasCscal(cublasHandle_, n_, cc(&zero), x, 1));
    /*#define pp(x) printf ("%s = \n(%e, %e)\n\n", #x, x.real (), x.imag ());
    #define cn(x) if (x.real() != x.real ()) {printf ("break on nan %s\n", #x); break;}
    #define pp_(a) printf ("%s = \n(%f, %f)\n(%f, %f)\n(%f, %f)\n\n", #a, \
    a[0].x, a[0].y, \
    a[1].x, a[1].y, \
    a[2].x, a[2].y);*/
    #define pp(x)
    #define cn(x) if (x.real() != x.real ()) {printf ("break on nan %s\n", #x); break;}
    #define pp_(a) 
    
    int restarts = 0;
    float r0Norm = 0.0f;
    CB (cublasScnrm2 (cublasHandle_, n_, r0, 1, &r0Norm));
    for (int i = 0; i < nIter; i++)
    {
        CB (cublasScnrm2 (cublasHandle_, n_, r, 1, &norm));
        printf ("Start iter %d\n", i);
        //printf ("|rho| = %e\n", thrust::abs (rho));
        
        /// 1. rho = r0 dot r
        
        rhoOld = rho;
        CB (cublasCdotc (cublasHandle_, n_,
                         r0, 1,
                         r, 1,
                         cc(&rho)));
        pp (rho)
        cn(rho)
        
        if (i != 0 && (thrust::abs (rho) < 1e-5*r0Norm || thrust::abs (rhoOld) < 1e-12 || thrust::abs (omega) < 1e-12 || thrust::abs (alpha) < 1e-12))
        {
            /// 2.1. r = b - mat * x
            CB (cublasCcopy (cublasHandle_, n_,
                         b, 1,
                         r, 1));
            CB (cublasCgemv(cublasHandle_, CUBLAS_OP_N, n_, n_, cc(&minusOne),
                        A, n_, x, 1, cc(&one), r, 1));
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
            
            alpha  = complex_t(1.0f, 0.0f);
            omega  = complex_t(1.0f, 0.0f);
            printf ("restart!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        }
        
        
            /// 2.0.0. beta = rho*alpha/(rhoOld*omega))
            beta = (rho/rhoOld) * (alpha/omega);
            pp (beta)
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
        CB (cublasCgemv(cublasHandle_, CUBLAS_OP_N, n_, n_, cc(&one),
                        A, n_, p, 1, cc(&zero), v, 1));

        /// 4. alpha = rho/(r0, v)
        CB (cublasCdotc (cublasHandle_, n_, r0, 1, v, 1, cc(&alpha)));
        alpha = rho/alpha;
        
        pp (alpha)
        cn(alpha)
        
        /// 4. s = r - alpha*v
        CB (cublasCcopy (cublasHandle_, n_,
                         r, 1,
                         s, 1));
        alpha = -alpha;
        CB (cublasCaxpy(cublasHandle_, n_, cc(&alpha), v, 1, s, 1));
        alpha = -alpha;

        /// 5. t = A*s
        CB (cublasCgemv(cublasHandle_, CUBLAS_OP_N, n_, n_, cc(&one),
                        A, n_, s, 1, cc(&zero), t, 1));
        
        /// 6. omega = (t, s)/(t, t)
        
        CB (cublasScnrm2 (cublasHandle_, n_, t, 1, &norm));
        if (norm < 1e-18) omega = complex_t(0.0f, 0.0f);
        else
        {
            CB (cublasCdotc (cublasHandle_, n_, t, 1, s, 1, cc(&omega)));
	        pp (omega) pp (omegaT)
            omega /= (norm*norm);
        }
        cn(omega)

        /// 6. x += alpha*p + omega*s

        CB (cublasCaxpy(cublasHandle_, n_, cc(&alpha), p, 1, x, 1));
        CB (cublasCaxpy(cublasHandle_, n_, cc(&omega), s, 1, x, 1));

        

        /// 7. r = s - omega*t
        CB (cublasCcopy (cublasHandle_, n_, s, 1, r, 1));
        omega = -omega;
        CB (cublasCaxpy(cublasHandle_, n_, cc(&omega), t, 1, r, 1));
        omega = -omega;

        CB (cublasScnrm2 (cublasHandle_, n_, r, 1, &norm));
        printf ("LAST NORM %e %e %e\n", norm, r0Norm, thrust::abs(rho));
        if (norm < 1e-9) {printf ("break on %d iterations\n", i+1); break;}
    }
    CB(cublasDestroy(cublasHandle_));
}

