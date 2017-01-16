/*
   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0
#define BUF 20
#define TAG 42

int main(int argc, char* argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size<2)
    {
        printf("Less than 2 processes\n");
        MPI_Finalize();
        exit(0);
    }

    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(rank == MASTER)
    {
        printf("Master with rank %d waiting for messages\n", MASTER);
        char str[BUF];
        for (int i = 0; i < size-1; i++)
        {
            MPI_Recv(str, BUF, MPI_CHAR, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);
            int source = status.MPI_SOURCE;
            printf("Received message \"%s\" from process %d\n", str, source);
        }
    }
    else
    {
        char str[] = "Hello World!";
        MPI_Send(str, BUF, MPI_CHAR, MASTER, TAG, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
