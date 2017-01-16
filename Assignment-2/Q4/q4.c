/*
   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0
#define STEPS 100000

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double step = 1.0/(double)STEPS, share;

    if (rank == MASTER)
    {
        share = STEPS/size;
    }

    //MPI_Bcast(&step, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&share, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    double x, sum=0;
    for (int i = rank*share; i < (rank+1)*share; i++)
    {
        x = (i+0.5)*step;
        sum += 4.0/(1.0+x*x);
    }
    double pi = step * sum;
    
    double pisum;
    MPI_Reduce(&pi, &pisum, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

    if(rank == MASTER) printf("Calculated value of pi: %lf\n", pisum);

    MPI_Finalize();
    return 0;
}