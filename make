CC = g++
CU = nvcc
CFLAGS = -Wall




main.o: main.cpp Builder.h
	$(CC) $(CFLAGS) -c main.cpp

DataLoader.o: DataLoader.cpp Builder.h
	$(CC) $(CFLAGS) -c DataLoader.cpp

BornCalc.o: BornCalc.cpp BornCalc.h
	$(CC) $(CFLAGS) -c BornCalc.cpp
	
CudaCalc.o: CudaCalc.cu CudaCalc.h
	$(CU) $(CFLAGS) -c BornCalc.cpp

main: main.o DataLoader.o
	$(CC) $(CFLAGS) -o main main.o DataLoader.o
	