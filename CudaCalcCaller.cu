
//=================================================================

#include "CudaCalc.h"

__global__ void BornForRecieversKernel (int * P_recv, InputData_t* INPUT_DATA_PTR);

#include "CudaCalc.cu"

//-----------------------------------------------------------------
extern "C"
void ExternalKernelCaller (InputData_t* INPUT_DATA_PTR)
//this function is executed on HOST
{
    printf ("Hello from CudaCalcCaller.o\n");
    //Do stuff
    //allocate memory on device
    //run kernel BornForRecieversKernel<<< ... >>> (...);
    // or use library to do the same without calling a kernel
    //move data from device to host
    //free memory on device

    //int recv_num = INPUT_DATA_PTR->recievers_.size ();
    //double

    //double complex h_P_recv [recv_num] = {};
    //needs to be dynamic

    //double complex * d_P_recv;
    //is it complex<double>?

    //CudaMalloc ((void(**) d_P_recv, recv_num);
    //malloc frees recv_num bytes, not recv_num complex numbers

    //dim3 blockSize (BLOCK_SIZE_, BLOCK_SIZE_, BLOCK_SIZE_);
    //dim3 gridSize  (GRID_SIZE_ , GRID_SIZE_ , GRID_SIZE_ );
    //BornForRecieversKernel <<<gridSize, blockSize>>> (d_P_recv, INPUT_DATA_PTR);
    //cannot pass host data pointer to device kernel, need to memcpy into device memory

    //CudaMemcpy (h_P_recv, d_P_recv, recv_num, cudaMemcpyDeviceToHost);

    //for (int i = 0; i < recv_num, i ++)
    //{
    //    printf ("%f + %fi\n", creal (h_P_recv), cimag (h_P_recv));
    //}

    //return 0;
}


//=================================================================
