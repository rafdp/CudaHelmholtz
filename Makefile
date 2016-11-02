.DEFAULT_GOAL := main

CC = g++ 
CU = nvcc
CFLAGS = -std=c++11 -Wall


main: DataLoader.o BornCalc.o CudaCalcCaller.o main.o 
	$(CU) -o main DataLoader.o BornCalc.o CudaCalcCaller.o main.o 

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
	make clean

# to compile and run: make r
# to just compile: make
# to clean: make c
# to compile, run and clean: make rc


	
