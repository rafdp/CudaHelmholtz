
# to compile and run: make r
# to just compile: make
# to clean: make c
# to compile, run and clean: make rc
# to compile, run, plot and clean: make rcp


.DEFAULT_GOAL := main

CC = g++ 
CU = nvcc
CUFLAGS = -Wno-deprecated-gpu-targets
CFLAGS = -std=c++11 -Wall


main: DataLoader.o main.o cufftQACaller.o 
	$(CU) $(CUFLAGS) -o main cufftQACaller.o  DataLoader.o main.o -lpthread 

DataLoader.o: DataLoader.cpp Builder.h
	$(CC) $(CFLAGS) -c DataLoader.cpp

BornCalc.o: BornCalc.cpp BornCalc.h
	$(CC) $(CFLAGS) -c BornCalc.cpp	

cufftQACaller.o: cufftQACaller.cu CudaCalc.h
	$(CU) $(CUFLAGS) -std=c++11 -c cufftQACaller.cu

main.o: main.cpp Builder.h
	$(CC) $(CFLAGS) -c main.cpp

cuda:	DataLoader.o # cufftQACaller.o
	nvcc -c -std=c++11 main.cu
	$(CU) $(CUFLAGS) -o main main.o DataLoader.o # cufftQACaller.o
	./main

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
	g++ -o InputDataExec InputData.cpp
	./InputDataExec
	@echo
	@echo "------Execution begins: " ||:
	@./main ||:
	@echo "------Execution ended" ||:
	@echo
	make c

rcp: main
	g++ -o InputDataExec InputData.cpp
	./InputDataExec
	@echo
	@echo "------Execution begins: " ||:
	@./main ||:
	@echo "------Execution ended" ||:
	@echo
	gnuplot plot.p
	make c

	
