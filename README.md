# CudaHelmholtz
This project is a CUDA adaptation of various solutions of acoustic integral equations. 
A source, some receivers and an acoustic anomaly are placed in halfspace. The forward problem needs to be solved: anomaly location and properties are known, and the wavefield at the receivers is to be found. 
The main goal was to make computations substantially faster compared to a CPU version. 

## Mathematical model
The scalar wavefield is described by Helmholtz's equation, which can be transformed into an integral equation. 
The equation is recurrent, so different methods of approximation are used to ease to solving.

## Approximations
We use Born, quasilinear and quasianalytical approximations. 

Read here on approximations: Zhdanov M.S. Geophysical Inverse Theory and Regularization Problems. Utah: Elsevier, 2002

## CUDA 
The computations needed repeated multiplication of simmetrical matrices and vectors, 
which can be done faster on GPU as it can be done in number of rows independent threads.
The process can be further sped up through FFT using **cuFFT** as the matrices are simmetrical.
For general memory usage **thrust** was used.
For the implementation of BiCGSTAB, which involved numerous matrix-vector and vector-vector multiplications, **cuBLAS** was used.
**cuSolver** was also utilized for comparison with BiCGSTAB (QR solution).

## Results
GPU vs CPU: ~60 times faster  
QA with FFT vs QA with basic matvec multiplication (still on GPU and with reduction): ~500 times faster  
QL with FFT vs QL with basic matvec multiplication: ~40 times faster



Developed in contribution with @higheroplane

[Thesises](https://abitu.net/public/admin/mipt-conference/FPMI.pdf) p.69

[Presentation](https://www.dropbox.com/s/9wp4vxnkdam3bc6/CudaHelmholtz.pdf?dl=0)
