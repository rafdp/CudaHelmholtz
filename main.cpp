#include "Builder.h"
#include <thread>

//using namespace std;
            	
InputData_t* INPUT_DATA_PTR = nullptr;

struct ThreadData_t
{
    int recv_num;
    std::pair<double, complex<double> >* write;
};

void Thread_ (ThreadData_t* threadData)
{
    ThreadData_t& td = *(threadData);
    
    InputData_t& inputData = *INPUT_DATA_PTR;
    
    Point3D_t* recv_array = inputData.recievers_.data();

    
    for (int n = td.recv_num;
         n < td.recv_num + INPUT_DATA_PTR->recievers_.size () / 4;
         n ++)
    {
        complex <double> result = 0;
        for (double i = inputData.anomalyPos_.x; i < inputData.anomalySize_.x + inputData.anomalyPos_.x + 0.001; i ++)
            for (double j = inputData.anomalyPos_.y; j < inputData.anomalySize_.y + inputData.anomalyPos_.y + 0.001; j ++)
                for (double k = inputData.anomalyPos_.z; k < inputData.anomalySize_.z + inputData.anomalyPos_.z + 0.001; k ++)
                {
                    Point3D_t p = {i*1.0, j*1.0, k*1.0};
                    printf ("%g %g %g\n",
                            ToPhysical (p).x,
                            ToPhysical (p).y,
                            ToPhysical (p).z);
                    result += BornForPoint (p, *(recv_array + n));
                }
        
        *(td.write + n) = {ToPhysical (*(recv_array + n)).x, result};
        printf ("Reciever %d\n", n);
    }

}

int main ()
{

    InputData_t inputData = {};
    inputData.LoadData ();
    INPUT_DATA_PTR = &inputData;
    int recv_num = inputData.recievers_.size ();

    FILE * output1 = fopen ("output.txt", "wb");
    
    std::pair<double, std::complex<double> > * data = new std::pair<double, std::complex<double> > [inputData.recievers_.size ()];

    ThreadData_t p1 = {0 , data};
    ThreadData_t p2 = {recv_num/4, data};
    ThreadData_t p3 = {recv_num/2, data};
    ThreadData_t p4 = {3*recv_num/4, data};

    std::thread t1(Thread_, &p1);
    std::thread t2(Thread_, &p2);
    std::thread t3(Thread_, &p3);
    std::thread t4(Thread_, &p4);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    for (int i = 0; i < recv_num; i ++)
    {
        fprintf (output1, "%g %e %e\r\n", data[i].first, std::real (data[i].second), std::imag (data[i].second));
    }
    
    
    /*for (int n = 0; n < recv_num; n ++)
    {
            complex <double> result = 0;
            for (int i = inputData.anomalyPos_.x; i < inputData.anomalySize_.x + inputData.anomalyPos_.x; i ++)
                for (int j = inputData.anomalyPos_.y; j < inputData.anomalySize_.y + inputData.anomalyPos_.y; j ++)
                    for (int k = inputData.anomalyPos_.z; k < inputData.anomalySize_.z + inputData.anomalyPos_.z; k ++)
                    {
                        result += BornForPoint ({i*1.0, j*1.0, k*1.0}, *(recv_array + n));
                    }

        fprintf (output1, "%g %e %e\r\n", ToPhysical (*(recv_array + n)).x, std::real (result), std::imag (result));
             printf ("Reciever (%.05f, %g, %g): %e + %e i\n",
                     ToPhysical (*(recv_array + n)).x ,
                                                          (recv_array + n) -> y ,
                                                          (recv_array + n) -> z ,
                                                          std::real (result), std::imag(result));
        }*/
    
        fclose (output1);

    return 0;
}


