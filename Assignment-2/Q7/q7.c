/*
   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0
#define TAG 42

struct dd
{
    char c;
    int i[2];
    float f[4];
}s, send1, send2;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    ///CREATION OF STRUCT TYPE
    int numBlocks = 3;
    int arrayOfBlockLengths[] = {1, 2, 5};
    
    //Calculating Displacements
    MPI_Aint displacements[3];
    displacements[0] = (MPI_Aint)0;
    int block1Addr, block2Addr, block3Addr;
    MPI_Get_address(&s.c, &block1Addr);
    MPI_Get_address(s.i, &block2Addr);
    MPI_Get_address(s.f, &block3Addr);
    displacements[1] = block2Addr - block1Addr;
    displacements[2] = block3Addr - block2Addr;

    MPI_Datatype types[] = {MPI_CHAR, MPI_INT, MPI_FLOAT};
    MPI_Datatype newtype;

    MPI_Type_create_struct(numBlocks, arrayOfBlockLengths, displacements, types, &newtype);
    MPI_Type_commit(&newtype); //Committing our new type


    ////////BROADCAST COMMUNICATION////////////
    //struct send1 filled in master
    if (rank == MASTER)
    {
        send1.c = 'j';
        send1.i[0] = 1;
        send1.i[1] = 0;
        send1.f[0] = 6.3;
        send1.f[1] = 9.2;
        send1.f[2] = 8.1;
        send1.f[3] = 0.9;
        printf("\nMASTER filling send1 as follows: ");
        printf("c: %c; i: {%d, %d}; f: {%f, %f, %f, %f};\n", send1.c, send1.i[0], 
            send1.i[1], send1.f[0], send1.f[1], send1.f[2], send1.f[3]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&send1, 1, newtype, MASTER, MPI_COMM_WORLD);
    printf("\nBCAST: %d received send1 as follows: c: %c; i: {%d, %d}; f: {%f, %f, %f, %f};\n", rank, send1.c, send1.i[0], 
        send1.i[1], send1.f[0], send1.f[1], send1.f[2], send1.f[3]);


    ///////////P2P COMMUNICATION/////////
    //struct send 2 filled in master
    if (rank == MASTER)
    {
        send2.c = 'X';
        send2.i[0] = 22;
        send2.i[1] = 51;
        send2.f[0] = 3.3;
        send2.f[1] = 1.5;
        send2.f[2] = 2.1;
        send2.f[3] = 4.4;
        printf("\nMASTER filling send2 as follows: ");
        printf("c: %c; i: {%d, %d}; f: {%f, %f, %f, %f};\n", send2.c, send2.i[0], 
            send2.i[1], send2.f[0], send2.f[1], send2.f[2], send2.f[3]);
    }

    for (int i = 1; i < size; i++)
    {
        if (rank == MASTER)
        {
            MPI_Send(&send2, 1, newtype, i, TAG, MPI_COMM_WORLD);
        }
        if (rank == i)
        {
            MPI_Recv(&send2, 1, newtype, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("\nP2P: %d received send2 as follows: c: %c; i: {%d, %d}; f: {%f, %f, %f, %f};\n", rank, send2.c, send2.i[0], 
                send2.i[1], send2.f[0], send2.f[1], send2.f[2], send2.f[3]);
        }
    }

    MPI_Finalize();
    return 0;
}
