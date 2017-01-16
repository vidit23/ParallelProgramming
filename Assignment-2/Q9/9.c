#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>

#define MATRIX_SIZE 10

int main(int argc, char* argv[])
{
	int size, MyRank, i, j, a[MATRIX_SIZE][MATRIX_SIZE];
	int blocklen[MATRIX_SIZE],displacement[MATRIX_SIZE];
	MPI_Datatype upper;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

	for(i = 0; i < MATRIX_SIZE; i++)
	{
		blocklen[i] = MATRIX_SIZE - i;
		displacement[i] = i * MATRIX_SIZE + i;
	}

	MPI_Type_indexed(MATRIX_SIZE, blocklen, displacement, MPI_INT, &upper);
	MPI_Type_commit(&upper);
	if(MyRank == 0)
	{
		for(i = 0; i < MATRIX_SIZE; i++)
			for(j = 0; j < MATRIX_SIZE; j++)
				a[i][j] = rand() % 10;
		MPI_Send(a, 1, upper, 1, 0, MPI_COMM_WORLD);
	}
	else if(MyRank == 1)
	{
		for(i = 0; i < MATRIX_SIZE; i++)
			for(j = 0; j < MATRIX_SIZE; j++)
				a[i][j] = 0;
		MPI_Recv(a, 1, upper, 0, 0, MPI_COMM_WORLD, &status);
		for(i = 0; i < MATRIX_SIZE; i++)
		{
			for(j = 0; j < MATRIX_SIZE; j++)
				printf("%d ",a[i][j]);
			printf("\n");
		}
	}
	MPI_Type_free(&upper);
	MPI_Finalize();

}
