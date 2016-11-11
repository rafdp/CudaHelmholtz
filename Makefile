
# to compile and run: make r
# to just compile: make
# to clean: make c
# to compile, run and clean: make rc
# to compile, run, plot and clean: make rcp


.DEFAULT_GOAL := main

CC = g++ 
CU = nvcc
CFLAGS = -std=c++11 -Wall


main: DataLoader.o main.o 
#CudaCalcCaller.o BornCalc.o 
	$(CC) -o main DataLoader.o main.o -lpthread
# CudaCalcCaller.o 

DataLoader.o: DataLoader.cpp Builder.h
	$(CC) $(CFLAGS) -c DataLoader.cpp

BornCalc.o: BornCalc.cpp BornCalc.h
	$(CC) $(CFLAGS) -c BornCalc.cpp	

CudaCalcCaller.o: CudaCalcCaller.cu CudaCalc.h
	$(CU) -std=c++11 -c CudaCalcCaller.cu

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

	
