
//=================================================================

#include "BornCalc.h"

//-----------------------------------------------------------------

const int BLOCK_SIZE_ = 10;
const int GRID_SIZE_  = 50;

//-----------------------------------------------------------------

__global__ void BornForRecievers (int * P_recv);

//-----------------------------------------------------------------

__global__ void BornForRecievers (int * P_recv)
{
    Point3D r = {threadIdx.x + blockIdx.x * BLOCK_SIZE_;
                 threadIdx.y + blockIdx.y * BLOCK_SIZE_;
                 threadIdx.z + blockIdx.z * BLOCK_SIZE_}

    int recv_num = sizeof (P_recv);
    for (int i = 0; i < recv_num; i ++)
    {
        P_recv [i] += BornForPoint (r, DATA.recievers_ [i]);
    }
}

//=================================================================
