#ifndef BICGSTAB_CUDA_INCLUDED
#define BICGSTAB_CUDA_INCLUDED

#include "includes.h"


class MatVecFunctorBase
{   public:
    __host__
    virtual void operator()(cuComplex* source, cuComplex* destination) = 0;
};

/// Fortran order
class BiCGStabCudaSolver
{
    private:
    complex_t* device_b_;
    complex_t* device_x_;
    int    n_;

    public:
    BiCGStabCudaSolver (int n, complex_t* device_b, complex_t* device_workspace);

    
    size_t solve (MatVecFunctorBase* matVec, size_t nIter = 5000, float tol = 1e-6f);
};



#endif
