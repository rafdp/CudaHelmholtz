#include <stdio.h>
#include <stdlib.h>
#include "DataLoader.h"
#include "SharedDeclarations.h"

const int SIZE_X = 500;
const int SIZE_Y = 500;
const int SIZE_Z = 500;

const int SOURCE_POS_X = SIZE_X / 2,
SOURCE_POS_Y = SIZE_Y / 2,
SOURCE_POS_Z = 0;

const int ANOMALY_POS_X = SIZE_X / 4,
ANOMALY_POS_Y = SIZE_Y / 5,
ANOMALY_POS_Z = SIZE_Z / 6;

const int ANOMALY_SIZE_X = SIZE_X / 7,
ANOMALY_SIZE_Y = SIZE_Y / 6,
ANOMALY_SIZE_Z = SIZE_Z / 5;

const int CELL_FREQ = 10;

const double OMEGA = 62.8;
const double SOUND_SPEED = 3000;
const double ALPHA = 0.1;

int main ()
{
    FILE* inputData = fopen (INPUT_FILE, "wb");
    
    Point3D_t size = {SIZE_X, SIZE_Y, SIZE_Z};
    fwrite (&size, sizeof (int), 3, inputData);
    
    int coords[3] = {SOURCE_POS_X, SOURCE_POS_Y, SOURCE_POS_Z};
    fwrite (coords, sizeof (int), 3, inputData);
    
    double params[3] = {OMEGA, SOUND_SPEED, ALPHA};
    fwrite (params, sizeof (double), 3, inputData);

    
    int anomalyData[3] = {ANOMALY_POS_X, ANOMALY_POS_Y, ANOMALY_POS_Z};
    fwrite (anomalyData, sizeof (int), 3, inputData);
    
    int anomalySize[3] = {ANOMALY_SIZE_X, ANOMALY_SIZE_Y, ANOMALY_SIZE_Z};
    fwrite (anomalySize, sizeof (int), 3, inputData);
    
    
    int Nreceivers = CELL_FREQ*CELL_FREQ;
    
    fwrite (&Nreceivers, sizeof (int), 1, inputData);

    for (int x = 0; x < SIZE_X; x += SIZE_X / CELL_FREQ)
    {
        for (int y = 0; y < SIZE_Y; y += SIZE_Y / CELL_FREQ)
        {
            int coords_[3] = {x, y, 0};
            fwrite (coords_, sizeof (int), 3, inputData);
        }
    }
    
    fclose (inputData);
    return 0;
    
}