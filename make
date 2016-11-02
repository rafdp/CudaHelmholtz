CC = g++ 
CU = nvcc
CFLAGS = -std=c++11 -Wall


main: main.o DataLoader.o CudaCalc.o
	$(CC) $(CFLAGS) -o main main.o DataLoader.o CudaCalc.o

DataLoader.o: DataLoader.cpp Builder.h
	$(CC) $(CFLAGS) -c DataLoader.cpp
BornCalc.o: BornCalc.cpp BornCalc.h
	$(CC) $(CFLAGS) -c BornCalc.cpp	
CudaCalc.o: CudaCalc.cu CudaCalc.h
	$(CU) -std=c++11 -c BornCalc.cpp
main: main.o DataLoader.o
	$(CC) $(CFLAGS) -o main main.o DataLoader.o
main.o: main.cpp Builder.h
	$(CC) $(CFLAGS) -c main.cpp
	
