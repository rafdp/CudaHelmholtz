#include "Builder.h"
#include <thread>

#include <time.h>

InputData_t* INPUT_DATA_PTR = nullptr;


struct ThreadDataBorn_t
{
    int recv_numBegin, recv_numEnd;
    complex<double>* ui;
    std::pair<double, complex<double> >* write;
};

void ThreadBorn_ (ThreadDataBorn_t td);

struct ThreadDataUi_t
{
    int xBegin, xEnd;
    complex<double>* ui;
};
void ThreadUi_ (ThreadDataUi_t td);

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

    /*
    INPUT_DATA_PTR = &inputData;
    int recv_num = inputData.Nreceivers_;

    const unsigned char Nthreads_ = std::thread::hardware_concurrency();

    const unsigned char Nthreads = Nthreads_ ? Nthreads_ : 2;

    FILE * output1 = fopen ("output.txt", "wb");

    std::pair<double, std::complex<double> > * data = new std::pair<double, std::complex<double> > [inputData.Nreceivers_];

    complex<double>* ui = new complex<double>
                            [inputData.discretizationSize_[0] *
                             inputData.discretizationSize_[1] *
                             inputData.discretizationSize_[2]];

    timespec spec0 = {};
    timespec spec1 = {};
    clock_gettime(CLOCK_MONOTONIC, &spec0);



    if (Nthreads != 1)
    {
        ThreadDataUi_t uData[Nthreads] = {};

        for (unsigned char i = 0; i < Nthreads; i++)
            uData[i] = {(i*inputData.discretizationSize_[0])/Nthreads,
                        ((i+1)*inputData.discretizationSize_[0])/Nthreads, ui};
        std::thread* threads[Nthreads - 1] = {};

        for (unsigned char i = 0; i < Nthreads - 1; i++)
            threads[i] = new std::thread (ThreadUi_, uData[i]);

        ThreadUi_ (uData[Nthreads - 1]);

        for (unsigned char i = 0; i < Nthreads - 1; i++)
        {
            threads[i]->join();
            delete threads[i];
            threads[i] = nullptr;
        }

        ThreadDataBorn_t bData[Nthreads] = {};

        for (unsigned char i = 0; i < Nthreads; i++)
            bData[i] = { i*recv_num/Nthreads,
                        (i+1)*recv_num/Nthreads,
                        ui, data};

        for (unsigned char i = 0; i < Nthreads - 1; i++)
            threads[i] = new std::thread (ThreadBorn_, bData[i]);

        ThreadBorn_ (bData[Nthreads - 1]);


        for (unsigned char i = 0; i < Nthreads - 1; i++)
        {
            threads[i]->join();
            delete threads[i];
            threads[i] = nullptr;
        }
    }

    else
    {
        ThreadDataUi_t uData = {0, inputData.discretizationSize_[0], ui};
        ThreadUi_ (uData);
        ThreadDataBorn_t bData = { 0, recv_num, ui, data};
        ThreadBorn_ (bData);
    }


    clock_gettime(CLOCK_MONOTONIC, &spec1);

    for (int i = 0; i < recv_num; i ++)
    {
        fprintf (output1, "%g %e %e\r\n", data[i].first, std::real (data[i].second), std::imag (data[i].second));
    }

    printf ("%g\n", (spec1.tv_sec - spec0.tv_sec)*1000.0 + (spec1.tv_nsec - spec0.tv_nsec)/1000000.0);


    fclose (output1);

    delete [] ui;
*/
    return 0;
}


#define DISCRETE_TO_PHYSICAL_CENTER(var, ind) \
var*inputData.discreteBlockSize_[ind]*1.0  + \
inputData.anomalyPos_.var + \
inputData.discreteBlockSize_[ind] / 2.0


void ThreadBorn_ (ThreadDataBorn_t td)
{
    InputData_t& inputData = *INPUT_DATA_PTR;

    static const double w = inputData.f_*2*3.141592;

    const double K = inputData.discreteBlockSize_[0] *
                     inputData.discreteBlockSize_[1] *
                     inputData.discreteBlockSize_[2] *
                     w*w;

    complex<double> Gcoeff = w/inputData.c_ * std::complex<double>(0.0, 1.0);

    for (int n = td.recv_numBegin;
         n < td.recv_numEnd;
         n ++)
    {
        complex <double> result = 0;

	    for (int x = 0; x < inputData.discretizationSize_[0]; x++)
        {
            for (int y = 0; y < inputData.discretizationSize_[1]; y++)
            {
                for (int z = 0; z < inputData.discretizationSize_[2]; z++)
                {
                    Point3D_t p = {DISCRETE_TO_PHYSICAL_CENTER (x, 0) -
                                   inputData.receivers_[n].x,
                                   DISCRETE_TO_PHYSICAL_CENTER (y, 1) -
                                   inputData.receivers_[n].y,
                                   DISCRETE_TO_PHYSICAL_CENTER (z, 2) -
                                   inputData.receivers_[n].z};


                    double len = p.Len ();

                    int index = x +
                                y*inputData.discretizationSize_[1] +
                                z*inputData.discretizationSize_[1] *
                                  inputData.discretizationSize_[2];

                    result += K *
                              td.ui[index] *
                              inputData.ds2_[index] *
                              exp (Gcoeff * len) / (4 * 3.141592 * len);

                }
            }
        }

        *(td.write + n) = {inputData.receivers_[n].x, result};
        printf ("Receiver %d\n", n);
    }

}


void ThreadUi_ (ThreadDataUi_t td)
{
    InputData_t& inputData = *INPUT_DATA_PTR;

    complex<double> Gcoeff = inputData.f_*2*3.141592/inputData.c_*std::complex<double>(0.0, 1.0);

    for (int x = td.xBegin; x < td.xEnd; x++)
    {
        for (int y = 0; y < inputData.discretizationSize_[1]; y++)
        {
            for (int z = 0; z < inputData.discretizationSize_[2]; z++)
            {
                Point3D_t p = {DISCRETE_TO_PHYSICAL_CENTER (x, 0) -
                               inputData.sourcePos_.x,
                               DISCRETE_TO_PHYSICAL_CENTER (y, 1) -
                               inputData.sourcePos_.y,
                               DISCRETE_TO_PHYSICAL_CENTER (z, 2) -
                               inputData.sourcePos_.z};

                double len = p.Len ();

                int index = x +
                            y*inputData.discretizationSize_[1] +
                            z*inputData.discretizationSize_[1] *
                              inputData.discretizationSize_[2];
                td.ui[index] = exp (Gcoeff * len) / (4 * 3.141592 * len);
            }
        }
    }

}

#undef DISCRETE_TO_PHYSICAL_CENTER

