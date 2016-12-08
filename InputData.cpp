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

const double RECEIVER_X_BEGIN_WORLD = 500,
             RECEIVER_X_END_WORLD   = 1500,
             RECEIVER_Y_WORLD       = 0,
             RECEIVER_Z_WORLD       = 0,
             N_RECEIVERS            = 100;

const double FREQUENCY = 5;
const double SOUND_SPEED = 3000;
const double ALPHA = 0.5;

//---------------------------------------------------------------------
/*
File format: binary
double3             source              x, y, z     in physical space
double              frequency                       in Hz
double              c
double3             anomalyBegin        x, y, z     in physical space
double3             anomalySize         x, y, z     in physical space
int3                discretizationSize  x, y, z
double[dx][dy][dz]  ds^2
int                 NReceivers
double3[Nreceivers] receiverCoords      x, y, z     in physical space
*/
//---------------------------------------------------------------------

#define print(x) ;

int main ()
{
    FILE* inputData = fopen (INPUT_FILE, "wb");

    double coords[3] = {SOURCE_POS_WORLD_X,
                        SOURCE_POS_WORLD_Y,
                        SOURCE_POS_WORLD_Z};
    fwrite (coords, sizeof (double), 3, inputData);
    print (("coords %g %g %g\n",
            coords[0],
            coords[1],
            coords[2]));

    double data[2] = {FREQUENCY, SOUND_SPEED};
    fwrite (data, sizeof (double), 2, inputData);
    print(("data %g %g\n",
            data[0],
            data[1]));

    double anomalyData[3] = {ANOMALY_POS_WORLD_X,
                             ANOMALY_POS_WORLD_Y,
                             ANOMALY_POS_WORLD_Z};
    fwrite (anomalyData, sizeof (double), 3, inputData);
    print(("anomalyData %g %g %g\n",
            anomalyData[0],
            anomalyData[1],
            anomalyData[2]));

    double anomalySize[3] = {ANOMALY_SIZE_WORLD_X,
                             ANOMALY_SIZE_WORLD_Y,
                             ANOMALY_SIZE_WORLD_Z};
    fwrite (anomalySize, sizeof (double), 3, inputData);
    print(("anomalySize %g %g %g\n",
            anomalySize[0],
            anomalySize[1],
            anomalySize[2]));

    int discretizationRatio[3] = {DISCRETIZATION_NX,
                                  DISCRETIZATION_NY,
                                  DISCRETIZATION_NZ};

    fwrite (&discretizationRatio, sizeof (int), 3, inputData);
    print(("discretizationRatio %d %d %d\n",
            discretizationRatio[0],
            discretizationRatio[1],
            discretizationRatio[2]));

    int discrete_size[3] = {ANOMALY_SIZE_WORLD_X/DISCRETIZATION_NX,
                            ANOMALY_SIZE_WORLD_Y/DISCRETIZATION_NY,
                            ANOMALY_SIZE_WORLD_Z/DISCRETIZATION_NZ};

    print(("discrete_size %d %d %d\n",
            discrete_size[0],
            discrete_size[1],
            discrete_size[2]));

    double* ds2 = new double [discretizationRatio[0]*discretizationRatio[1]*discretizationRatio[2]];

    double deltaS2 = 1.0/(SOUND_SPEED*SOUND_SPEED)*
                    (1.0/((1+ALPHA)*(1+ALPHA)) - 1.0);
    for (int x = 0; x < discretizationRatio[0]; x++)
    {
        for (int y = 0; y < discretizationRatio[1]; y++)
        {
            for (int z = 0; z < discretizationRatio[2]; z++)
            {
                ds2[x +
                    y*discretizationRatio[1] +
                    z*discretizationRatio[1]*discretizationRatio[2]] = deltaS2;
            }
        }
    }

    print(("ds2 %g\n", ds2[411]));

    fwrite (ds2, sizeof (double), discretizationRatio[0]*discretizationRatio[1]*discretizationRatio[2], inputData);


    int Nreceivers = N_RECEIVERS;

    fwrite (&Nreceivers, sizeof (int), 1, inputData);
    print(("Nreceivers %d\n",
            Nreceivers));

    double shift = (RECEIVER_X_END_WORLD - RECEIVER_X_BEGIN_WORLD)/(Nreceivers*1.0);

    for (int i = 0; i < Nreceivers; i++)
    {
        double coords_[3] = {(shift*i + RECEIVER_X_BEGIN_WORLD),
                             RECEIVER_Y_WORLD,
                             RECEIVER_Z_WORLD};
        fwrite (coords_, sizeof (double), 3, inputData);

    }

    fclose (inputData);

    delete [] ds2;

    return 0;

}

#undef print
