#include "Builder.h"

InputData_t* INPUT_DATA_PTR = nullptr;


int main ()
{
	InputData_t inputData = {};
    inputData.LoadData ();

    std::vector<std::complex<float> > result (inputData.Nreceivers_ + 1);

    ExternalKernelCaller (&inputData, &result);

    FILE * output1 = fopen ("output.txt", "wb");

    for (int i = 0; i < inputData.Nreceivers_; i ++)
    {
        fprintf (output1, "%g %e %e\r\n", inputData.receivers_[i].x, std::real (result[i]), std::imag (result[i]));
    }
	
	return 0;
	
}
