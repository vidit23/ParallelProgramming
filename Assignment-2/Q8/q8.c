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
    
    char buff1[200], buff2[200];
    int position;

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
        
        position = 0;
        MPI_Pack(&send1.c, 1, MPI_CHAR, buff1, 200, &position, MPI_COMM_WORLD);
        MPI_Pack(send1.i, 2, MPI_INT, buff1, 200, &position, MPI_COMM_WORLD);
        MPI_Pack(send1.f, 5, MPI_FLOAT, buff1, 200, &position, MPI_COMM_WORLD);
    }

    MPI_Bcast(buff1, 200, MPI_CHAR, MASTER, MPI_COMM_WORLD);
    position = 0;
    MPI_Unpack(buff1, 200, &position, &send1.c, 1, MPI_CHAR, MPI_COMM_WORLD);
    MPI_Unpack(buff1, 200, &position, send1.i, 2, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buff1, 200, &position, send1.f, 4, MPI_FLOAT, MPI_COMM_WORLD);
    
    printf("\nBCAST: %d received buff1 and unpacked send1 as follows: c: %c; i: {%d, %d}; f: {%f, %f, %f, %f};\n", rank, send1.c, send1.i[0], 
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

        position = 0;
        MPI_Pack(&send2.c, 1, MPI_CHAR, buff2, 200, &position, MPI_COMM_WORLD);
        MPI_Pack(send2.i, 2, MPI_INT, buff2, 200, &position, MPI_COMM_WORLD);
        MPI_Pack(send2.f, 5, MPI_FLOAT, buff2, 200, &position, MPI_COMM_WORLD);
    }

    for (int i = 1; i < size; i++)
    {
        if (rank == MASTER)
        {
            MPI_Send(&buff2, 200, MPI_CHAR, i, TAG, MPI_COMM_WORLD);
        }
        if (rank == i)
        {
            MPI_Recv(&buff2, 200, MPI_CHAR, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            position = 0;
            MPI_Unpack(buff2, 200, &position, &send2.c, 1, MPI_CHAR, MPI_COMM_WORLD);
            MPI_Unpack(buff2, 200, &position, send2.i, 2, MPI_INT, MPI_COMM_WORLD);
            MPI_Unpack(buff2, 200, &position, send2.f, 4, MPI_FLOAT, MPI_COMM_WORLD);
            printf("\nP2P: %d received buff2 and unpacked send2 as follows: c: %c; i: {%d, %d}; f: {%f, %f, %f, %f};\n", rank, send2.c, send2.i[0], 
                send2.i[1], send2.f[0], send2.f[1], send2.f[2], send2.f[3]);
        }
    }

    MPI_Finalize();
    return 0;
}