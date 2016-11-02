
//=================================================================

#include "CudaCalc.h"

//-----------------------------------------------------------------

__global__ void BornForRecieversKernel (complex<double> * P_recv, InputData_t* INPUT_DATA_PTR)
//cannot pass host data pointer to kernel, need to copy to device memory
{
    /*Point3D_t r = {static_cast<int> (threadIdx.x + blockIdx.x * BLOCK_SIZE_),
                   static_cast<int> (threadIdx.y + blockIdx.y * BLOCK_SIZE_),
                   static_cast<int> (threadIdx.z + blockIdx.z * BLOCK_SIZE_)};
//static cast needed for unsigned int -> int warning

    int recv_num = INPUT_DATA_PTR->recievers_.size ();
    for (int i = 0; i < recv_num; i ++)
    {
        P_recv [i] += BornForPoint (r, INPUT_DATA_PTR->recievers_ [i]);
    }*/
//There is a major problem with this kernel code: CUDA cannot call functions that are implemented on host (e.g. BornForPoint). 
//You need to rewrite them on the gpu via cuda

}

//=================================================================
