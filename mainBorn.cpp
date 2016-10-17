
//=================================================================

#include "CudaCalc.h"

//-----------------------------------------------------------------

int main ()
{
    int recv_num = sizeof DATA.recievers_;
    //double
    double complex h_P_recv [recv_num] = {};
    double complex * d_P_recv;

    CudaMalloc ((void(**) d_P_recv, recv_num);

    dim3 blockSize (BLOCK_SIZE_, BLOCK_SIZE_, BLOCK_SIZE_);
    dim3 gridSize  (GRID_SIZE_ , GRID_SIZE_ , GRID_SIZE_ );
    BornForRecievers <<<gridSize, blockSize>>> (d_P_recv);

    CudaMemcpy (h_P_recv, d_P_recv, recv_num, cudaMemcpyDeviceToHost);

    for (int i = 0; i < recv_num, i ++)
    {
        printf ("%f + %fi\n", creal (h_P_recv), cimag (h_P_recv));
    }

    return 0;
}

//=================================================================
