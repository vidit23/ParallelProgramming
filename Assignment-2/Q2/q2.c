/*
   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define LEN 65336

int X[LEN], Y[LEN], Xd[LEN], Yd[LEN];
int a = 2;

void fillArrays()
{
    for (int i = 0; i < LEN; i++)
    {
        X[i] = rand()%5;
        Xd[i] = X[i];
        Y[i] = rand()%5;
        Yd[i] = Y[i];
    }
}

int main(int argc, char* argv[])
{
    int rank, numP;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numP);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) printf("No. of processes: %d\n", numP);
    double share = (double)LEN/numP;
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    for (int i = rank*share; i < (rank+1)*share; i++)
    {
        X[i] = a*X[i] + Y[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime() - t1;
    //printf("Rank: %d out of %d Processes\n", rank, numP);

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    if(rank == 0)
    {
        for (int i = 0; i < LEN; i++)
        {
            Xd[i] = a*Xd[i] + Yd[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime() - t2;

    if (rank == 0)
    {
        printf("Time taken for multiprocessor calculation: %lf\n", t1);
        printf("Time taken for uniprocessor implementation: %lf\n", t2);
        printf("%lf times faster\n", t2/t1);
    }    

    MPI_Finalize();

    return 0;
}