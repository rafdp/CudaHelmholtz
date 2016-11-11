#include <stdio.h>
#include <stdlib.h>
#include "DataLoader.h"
#include "SharedDeclarations.h"

const int SOURCE_POS_WORLD_X = 0,
          SOURCE_POS_WORLD_Y = 0,
          SOURCE_POS_WORLD_Z = 0;

const double ANOMALY_POS_WORLD_X = 500,
             ANOMALY_POS_WORLD_Y = -150,
             ANOMALY_POS_WORLD_Z = 300;

const double ANOMALY_SIZE_WORLD_X = 300,
             ANOMALY_SIZE_WORLD_Y = 300,
             ANOMALY_SIZE_WORLD_Z = 300;

const double DISCRETIZATION_NX = 10,
          DISCRETIZATION_NY = 10,
          DISCRETIZATION_NZ = 10;

const double ANOMALY_POS_X = (ANOMALY_POS_WORLD_X * DISCRETIZATION_NX) / ANOMALY_SIZE_WORLD_X,
          ANOMALY_POS_Y = (ANOMALY_POS_WORLD_Y * DISCRETIZATION_NY) / ANOMALY_SIZE_WORLD_Y,
          ANOMALY_POS_Z = (ANOMALY_POS_WORLD_Z * DISCRETIZATION_NZ) / ANOMALY_SIZE_WORLD_Z;

const double SOURCE_POS_X = (SOURCE_POS_WORLD_X * DISCRETIZATION_NX) / ANOMALY_SIZE_WORLD_X,
          SOURCE_POS_Y = (SOURCE_POS_WORLD_Y * DISCRETIZATION_NY) / ANOMALY_SIZE_WORLD_Y,
          SOURCE_POS_Z = (SOURCE_POS_WORLD_Z * DISCRETIZATION_NZ) / ANOMALY_SIZE_WORLD_Z;

const double RECIEVER_X_BEGIN_WORLD = 500,
             RECIEVER_X_END_WORLD   = 1500,
             RECIEVER_Y_WORLD       = 0,
             RECIEVER_Z_WORLD       = 0,
             N_RECIEVERS            = 1000;

const double OMEGA = 5 * 2 * 3.141592;
const double SOUND_SPEED = 3000;
const double ALPHA = 0.5;

int main ()
{
    FILE* inputData = fopen (INPUT_FILE, "wb");
    
    double coords[3] = {SOURCE_POS_X, SOURCE_POS_Y, SOURCE_POS_Z};
    fwrite (coords, sizeof (double), 3, inputData);
    
    double params[3] = {OMEGA, SOUND_SPEED, ALPHA};
    fwrite (params, sizeof (double), 3, inputData);

    
    double anomalyData[3] = {ANOMALY_POS_X, ANOMALY_POS_Y, ANOMALY_POS_Z};
    fwrite (anomalyData, sizeof (double), 3, inputData);
    
    double anomalySize[3] = {DISCRETIZATION_NX, DISCRETIZATION_NY, DISCRETIZATION_NZ};
    fwrite (anomalySize, sizeof (double), 3, inputData);
    
    double block_size[3] = {ANOMALY_SIZE_WORLD_X / (DISCRETIZATION_NX*1.0), ANOMALY_SIZE_WORLD_Y / (DISCRETIZATION_NY*1.0), ANOMALY_SIZE_WORLD_Z / (DISCRETIZATION_NZ*1.0)};
    
    fwrite (&block_size, sizeof (double), 3, inputData);
    
    
    int Nreceivers = N_RECIEVERS;
    
    fwrite (&Nreceivers, sizeof (int), 1, inputData);
    
    double d = (RECIEVER_X_END_WORLD - RECIEVER_X_BEGIN_WORLD)/(Nreceivers*1.0);

    for (int i = 0; i < Nreceivers; i++)
    {
        double coords_[3] = {(d*i + RECIEVER_X_BEGIN_WORLD) / block_size[0],
                             RECIEVER_Y_WORLD / block_size[1],
                             RECIEVER_Z_WORLD / block_size[2]};
        fwrite (coords_, sizeof (double), 3, inputData);

    }
    
    fclose (inputData);
    return 0;
    
}
