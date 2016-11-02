
#include "Builder.h"

InputData_t* INPUT_DATA_PTR = nullptr;

int main ()
{
    InputData_t inputData = {};
    inputData.LoadData ();
    INPUT_DATA_PTR = &inputData;
    printf ("%g\n", inputData.w_);
    return 0;
}
