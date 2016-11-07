 #include "BornThreadCPU.h"
#include <thread>

//using namespace std;

InputData_t* INPUT_DATA_PTR = nullptr;

int main ()
{

    InputData_t inputData = {};
    inputData.LoadData ();
    INPUT_DATA_PTR = &inputData;
    int recv_num = inputData.recievers_.size ();

    printf ("%g\n", inputData.w_);

    Point3D_t* recv_array = inputData.recievers_.data();

    FILE * output1 = fopen ("output_1.txt", "wb");
    /*FILE * output2 = fopen ("output_2.txt", "wb");
    FILE * output3 = fopen ("output_3.txt", "wb");
    FILE * output4 = fopen ("output_4.txt", "wb");

    params p1 = {0 , &inputData, recv_num, output1};
    params p2 = {25, &inputData, recv_num, output2};
    params p3 = {50, &inputData, recv_num, output3};
    params p4 = {75, &inputData, recv_num, output4};

    std::thread t1(BornForReciever, p1);
    std::thread t2(BornForReciever, p2);
    std::thread t3(BornForReciever, p3);
    std::thread t4(BornForReciever, p4);

    t1.join();
    t2.join();
    t3.join();
    t4.join();*/

    for (int n; n < recv_num; n ++)
        {
            complex <double> result = 0;
            for (int i = inputData.anomalyPos_.x; i < inputData.anomalySize_.x + inputData.anomalyPos_.x; i ++)
                for (int j = inputData.anomalyPos_.y; j < inputData.anomalySize_.y + inputData.anomalyPos_.y; j ++)
                    for (int k = inputData.anomalyPos_.z; k < inputData.anomalySize_.z + inputData.anomalyPos_.z; k ++)
                    {
                        Point3D_t r = {i, j, k};
                        result += BornForPoint (r, *(recv_array + n));
                    }

             fprintf (output1, "Reciever (%d, %d, %d): %e + %e i\r\n", (recv_array + n) -> x ,
                                                                     (recv_array + n) -> y ,
                                                                     (recv_array + n) -> z ,
                                                                     std::real (result), std::imag(result));
             printf ("Reciever (%d, %d, %d): %e + %e i\n", (recv_array + n) -> x ,
                                                          (recv_array + n) -> y ,
                                                          (recv_array + n) -> z ,
                                                          std::real (result), std::imag(result));
        }

    fclose (output1);
    /*fclose (output2);
    fclose (output3);
    fclose (output4);*/

    return 0;
}


