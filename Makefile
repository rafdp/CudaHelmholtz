
# to compile and run: make r
# to just compile: make
# to clean: make c
# to compile, run and clean: make rc
# to compile, run, plot and clean: make rcp


.DEFAULT_GOAL := main

CC = g++
CU = nvcc
CUFLAGS = -Wno-deprecated-gpu-targets -arch=sm_20
CFLAGS = -std=c++11 -Wall


main: DataLoader.o main.o CudaCalcCaller.o CudaCalc.o BiCGStabCuda.o
	$(CU) $(CUFLAGS) -o main CudaCalcCaller.o BiCGStabCuda.o CudaCalc.o  DataLoader.o main.o -lcuda -lcudart -lcublas -lcusolver -lcufft

DataLoader.o: DataLoader.cpp Builder.h
	$(CC) $(CFLAGS) -c DataLoader.cpp

CudaCalcCaller.o: CudaCalcCaller.cu CudaCalc.h
	$(CU) $(CUFLAGS) -std=c++11 -c CudaCalcCaller.cu

CudaCalc.o: CudaCalc.cu CudaCalc.h
	$(CU) $(CUFLAGS) -std=c++11 -c CudaCalc.cu

BiCGStabCuda.o: BiCGStabCuda.cu BiCGStabCuda.h CudaCalc.h
	$(CU) $(CUFLAGS) -std=c++11 -c BiCGStabCuda.cu

main.o: main.cpp Builder.h
	$(CC) $(CFLAGS) -c main.cpp

c:
	rm -rf main *.o InputDataExec input.data

r: main
	g++ -o InputDataExec InputData.cpp
	./InputDataExec
	@echo
	@echo "------Execution begins: " ||:
	@./main ||:
	@echo "------Execution ended" ||:
	@echo

rc: main
	make r
	make c

p: main
	gnuplot plot.p

rcp: main
	make r
	make p
	make c


