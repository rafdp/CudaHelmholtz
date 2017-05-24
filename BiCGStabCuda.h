#ifndef BICGSTAB_CUDA_INCLUDED
#define BICGSTAB_CUDA_INCLUDED

#include "includes.h"


/// Fortran order
class BiCGStabCudaSolver
{
    private:
    complex_t* device_b_;
    complex_t* device_A_;
    int    n_;

    public:

    BiCGStabCudaSolver (int n, complex_t* device_b, complex_t* device_A);

    void solve (complex_t* device_workspace, size_t nIter = 5000, float tol = 1e-6f);
};

#endif
