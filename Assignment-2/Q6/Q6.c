#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<math.h>
#include <mpi.h>

int main(int argc, char** argv)
{
	int size, MyRank, i;
	float *arr, *newarr, recvbuf;

	srand(time(NULL));

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (MyRank == 0)
	{
		arr = (float *)malloc(sizeof(float) * size);
		newarr = (float*)malloc(sizeof(float) * size);
		printf("The old array: \n");
		for (i = 0; i < size; i++)
		{
			arr[i] = rand() % 100;
			printf("%f ", arr[i]);
		}

	}
	int *displs = (int *)malloc(sizeof(int) * size);
	int *sendcounts = (int *)malloc(sizeof(int) * size);
	for (i = 0; i < size; i++)
	{
		sendcounts[i] = 1;
		displs[i] = i;
	}

	MPI_Scatterv(arr, sendcounts, displs, MPI_FLOAT, &recvbuf, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	//printf("\nthe process%d\n has %f ", MyRank, recvbuf);

	float root = sqrt(recvbuf);

	MPI_Gatherv(&root, 1, MPI_FLOAT, newarr, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if (MyRank == 0)
	{
		printf("\nThe new array:\n");
		for(i=0;i<size;i++)
			printf("%f ", newarr[i]);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}
