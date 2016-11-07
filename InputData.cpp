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

const int DISCRETIZATION_NX = 10,
          DISCRETIZATION_NY = 10,
          DISCRETIZATION_NZ = 10;

const int ANOMALY_POS_X = (ANOMALY_POS_WORLD_X * ANOMALY_SIZE_WORLD_X) / DISCRETIZATION_NX,
          ANOMALY_POS_Y = (ANOMALY_POS_WORLD_Y * ANOMALY_SIZE_WORLD_Y) / DISCRETIZATION_NY,
          ANOMALY_POS_Z = (ANOMALY_POS_WORLD_Z * ANOMALY_SIZE_WORLD_Z) / DISCRETIZATION_NZ;

const int SOURCE_POS_X = (SOURCE_POS_WORLD_X * ANOMALY_SIZE_WORLD_X) / DISCRETIZATION_NX,
          SOURCE_POS_Y = (SOURCE_POS_WORLD_Y * ANOMALY_SIZE_WORLD_Y) / DISCRETIZATION_NY,
          SOURCE_POS_Z = (SOURCE_POS_WORLD_Z * ANOMALY_SIZE_WORLD_Z) / DISCRETIZATION_NZ;

const double RECIEVER_X_BEGIN_WORLD = 500,
             RECIEVER_X_END_WORLD   = 1500,
             RECIEVER_Y_WORLD       = 0,
             RECIEVER_Z_WORLD       = 0,
             N_RECIEVERS            = 100;

const double OMEGA = 5 * 2 * 3.141592;
const double SOUND_SPEED = 3000;
const double ALPHA = 0.5;

int main ()
{
    FILE* inputData = fopen (INPUT_FILE, "wb");
    
    int coords[3] = {SOURCE_POS_X, SOURCE_POS_Y, SOURCE_POS_Z};
    fwrite (coords, sizeof (int), 3, inputData);
    
    double params[3] = {OMEGA, SOUND_SPEED, ALPHA};
    fwrite (params, sizeof (double), 3, inputData);

    
    int anomalyData[3] = {ANOMALY_POS_X, ANOMALY_POS_Y, ANOMALY_POS_Z};
    fwrite (anomalyData, sizeof (int), 3, inputData);
    
    int anomalySize[3] = {DISCRETIZATION_NX, DISCRETIZATION_NY, DISCRETIZATION_NZ};
    fwrite (anomalySize, sizeof (int), 3, inputData);
    
    double V = (ANOMALY_SIZE_WORLD_X * ANOMALY_SIZE_WORLD_Y * ANOMALY_SIZE_WORLD_Z) / (DISCRETIZATION_NX * DISCRETIZATION_NY * DISCRETIZATION_NZ);
    
    fwrite (&V, sizeof (double), 1, inputData);
    
    
    int Nreceivers = N_RECIEVERS;
    
    fwrite (&Nreceivers, sizeof (int), 1, inputData);
    
    double d = (RECIEVER_X_END_WORLD - RECIEVER_X_BEGIN_WORLD)/Nreceivers;

    for (int i = 0; i < Nreceivers; i++)
    {
        int coords_[3] = {d*i + RECIEVER_X_BEGIN_WORLD,
                          RECIEVER_Y_WORLD,
                          RECIEVER_Z_WORLD};
        fwrite (coords_, sizeof (int), 3, inputData);

    }
    
    fclose (inputData);
    return 0;
    
}