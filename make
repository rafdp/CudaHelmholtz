
CC = g++
CU = nvcc
CFLAGS = -Wall




main.o: main.cpp Builder.h
	$(CC) $(CFLAGS) -c main.cpp

DataLoader.o: DataLoader.cpp Builder.h
	$(CC) $(CFLAGS) -c DataLoader.cpp

BornCalc.o: BornCalc.cpp BornCalc.h
	$(CC) $(CFLAGS) -c BpenCalc.cpp

main: main.o DataLoader.o
	$(CC) $(CFLAGS) -o main main.o DataLoader.o
	
