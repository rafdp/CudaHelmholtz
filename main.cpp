
#include "Builder.h"

int main ()
{
    InputData_t inputData;
    inputData.LoadData ();
    printf ("%g\n", inputData.w_);
    return 0;
}