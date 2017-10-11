
//=================================================================

#include "CudaCalc.h"

__global__ void PrintGrid (complex_t* data, int size)
{
    printf ("Printing grid 2d (%X)\n", data);
    for (int i = 0; i < size; i++)
    { 
	printf ("  ");
        for (int j = 0; j < size; j++)
	{
            printf ("(%.2e %.2e) ", data[i*size + j].real (), data[i*size + j].imag ());
	}
	printf ("\n");
    }
    printf ("PrintGrid2D ended (%X)\n", data);
}

__global__ void PrintGrid3 (complex_t* data, int size)
{
    printf ("Printing grid 3d (%X)\n", data);
    for (int i = 0; i < size; i++)
    {
	printf ("Layer %d\n", i);
        for (int j = 0; j < size; j++)
	{
	    printf ("  ");
            for (int k = 0; k < size; k++)
	    {
		int idx = i*size*size + j*size + k;
		printf ("(%.2e %.2e) ", 
			data[idx].real (), 
			data[idx].imag ());
	    }
	    printf ("\n");
	}
	printf ("\n");
    }
    printf ("PrintGrid3D ended (%X)\n", data);
}

__global__ void DevicePrintData (InputDataOnDevice * inputDataPtr)
{
    /*    Point3DDevice_t<float> sourcePos_;
    float w_; //DROP
    thrust::complex<float> uiCoeff_;
    Point3DDevice_t<float> anomalyPos_;
    Point3DDevice_t<float> anomalySize_;
    Point3DDevice_t<int>   discretizationSize_;
    Point3DDevice_t<int>   discreteBlockSize_;
    int                    size3_;
    int                    size2_;
    int                    size1_;
    float                  w2h3_;*/

    printf ("device address %x\n", inputDataPtr);

    InputDataOnDevice& handle = *(inputDataPtr);

    printf ("--------------------------------------------------------------\n");
    printf ("Printing from device:\n");
    printf ("   sourcePos_: %f %f %f\n",
            inputDataPtr->sourcePos_.x,
            handle.sourcePos_.y,
            handle.sourcePos_.z);
    printf ("   uiCoeff: %f i*%f\n",
            handle.uiCoeff_.real (),
            handle.uiCoeff_.imag ());
    printf ("   anomalySize_: %f %f %f\n",
            handle.anomalySize_.x,
            handle.anomalySize_.y,
            handle.anomalySize_.z);
    printf ("   discretizationSize_: %d %d %d\n",
            handle.discretizationSize_.x,
            handle.discretizationSize_.y,
            handle.discretizationSize_.z);
    printf ("   discreteBlockSize_: %d %d %d\n",
            handle.discreteBlockSize_.x,
            handle.discreteBlockSize_.y,
            handle.discreteBlockSize_.z);
    printf ("   size3_: %d\n", handle.size3_);
    printf ("   size2_: %d\n", handle.size2_);
    printf ("   size1_: %d\n", handle.size1_);
    printf ("   w2h3_: %f\n", handle.w2h3_);
    printf ("End print from device\n");
    printf ("--------------------------------------------------------------\n");
}

__global__ void ReduceEmittersToReceiver (InputDataOnDevice * inputDataPtr,
                                          complex_t* deviceKMatrixPtr,
                                          complex_t* reductedA_solution,
                                          int* sequence,
                                          point_t* indexesPtr)
{
    int receiver = blockIdx.x * blockDim.x + threadIdx.x;
    BTransformReduceUnary tempFunctor (deviceKMatrixPtr,
                                       indexesPtr, 
                                       receiver,
                                       inputDataPtr);
    ComplexAddition complexSum;
    *(reductedA_solution + receiver) = 
    thrust::transform_reduce (thrust::device, 
                              sequence,
                              sequence + inputDataPtr->size3_,
                              tempFunctor,
                              complex_t (0.0f, 0.0f),
                              complexSum);
}

const char * cublasGetErrorString (cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

const char * cusolverGetErrorString (cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "The operation completed successfully";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "The library was not initialized";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "Invalid parameters were passed";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "The device only supports compute capability 2.0 and above";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

//=================================================================
