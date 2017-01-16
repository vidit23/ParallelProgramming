#include <stdio.h>
#include <math.h>
#include <string.h>
#include<stdlib.h>
#include "mpi.h"

#define NO_OF_PROCESSES 4
#define MESH_SIZE 2
#define ndims 2
#define MATRIX_SIZE 8  // Has to be a multiple of 4

typedef struct
{
	int N; /* The number of processors in a row (column). */
	int size; /* Number of processors. (Size = N*N) */
	int Row; /* This processor’s row number. */
	int Col; /* This processor’s column number. */
	int MyRank; /* This processor’s unique identifier. */
	MPI_Comm Comm; /* Communicator for all processors in the grid.*/
	MPI_Comm Row_comm; /* All processors in this processor’s row . */
	MPI_Comm Col_comm; /* All processors in this processor’s column. */
} grid_info;

void SetUp_Mesh(grid_info *grid)
{

	// Number of processes in each dimension
	int dims[] = {4, 4};

	// Whether the grid is periodic or not
	int periods[] = {1, 1};

	int Coordinates[2];  /* processor Row and Column identification */
	int Remain_dims[2] = {0, 1};      /* For row and column communicators */


	/* MPI rank and MPI size */
	MPI_Comm_size(MPI_COMM_WORLD, &(grid->size));
	MPI_Comm_rank(MPI_COMM_WORLD, &(grid->MyRank));

	/* For square mesh */
	grid->N = 4;

	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &(grid->Comm));
	MPI_Cart_coords(grid->Comm, grid->MyRank, ndims, Coordinates);

	grid->Row = Coordinates[0];
	grid->Col = Coordinates[1];

	MPI_Cart_sub(grid->Comm, Remain_dims, &(grid->Row_comm));

	Remain_dims[0] = 1;
	Remain_dims[1] = 0;

	MPI_Cart_sub(grid->Comm, Remain_dims, &(grid->Col_comm));
}

int main (int argc, char *argv[])
{

	int istage,x,y,index,Proc_Id,Root =0,i,j,k,l,block_size;
	int matrix_block;
	int lindex, gr, gc;
	int src, dest, send_tag, recv_tag, Bcast_root;

	int A[MATRIX_SIZE][MATRIX_SIZE], B[MATRIX_SIZE][MATRIX_SIZE], C[MATRIX_SIZE][MATRIX_SIZE];
	int *A_block, *B_block, *C_block, *Temp_BufferA;

	int *A_array, *B_array, *C_array;

	grid_info grid;
	MPI_Status status;

	/* Initialising */
	MPI_Init (&argc, &argv);

	/* Set up the MPI_COMM_WORLD and CARTESIAN TOPOLOGY */
	SetUp_Mesh(&grid);


	/* Rndomizing Values Input */
	if (grid.MyRank == Root)
	{
		for(i = 0; i < MATRIX_SIZE; i++)
			for(j = 0; j < MATRIX_SIZE;j++)
			{
				A[i][j] = rand() % 5;
				B[i][j] = rand() % 5;
			}

	} /* MyRank == Root */

	/*  Send Matrix Size to all processors  */
	MPI_Barrier(grid.Comm);

	block_size = MATRIX_SIZE / grid.N;

	matrix_block = block_size * block_size;

	/* Memory allocating for Bloc Matrices */
	A_block = (int *) malloc (matrix_block * sizeof(int));
	B_block = (int *) malloc (matrix_block * sizeof(int));

	/* memory for arrangmeent of the data in one dim. arrays before MPI_SCATTER */
	A_array =(int *)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
	B_array =(int *)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);

	/*Rearrange the input matrices in one dim arrays by approriate order*/
	if (grid.MyRank == Root)
	{

		/* Rearranging Matrix A*/
		for (x = 0; x < grid.N; x++)
		{
			for (y = 0; y < grid.N; y++)
			{
				Proc_Id = x * grid.N + y;
				for (i = 0; i < block_size; i++)
				{
					gr = x * block_size + i;
					for (j = 0; j < block_size; j++)
					{
						lindex  = (Proc_Id * matrix_block) + (i * block_size) + j;
						gc = y * block_size + j;
						A_array[lindex] = A[gr][gc];
					}
				}
			}
		}

		/* Rearranging Matrix B*/
		for (x = 0; x < grid.N; x++)
		{
			for (y = 0; y < grid.N; y++)
			{
				Proc_Id = x * grid.N + y;
				for (i = 0; i < block_size; i++)
				{
					gr = x * block_size + i;
					for (j = 0; j < block_size; j++)
					{
						lindex = (Proc_Id * matrix_block) + (i * block_size) + j;
						gc = y * block_size + j;
						B_array[lindex] = B[gr][gc];
					}
				}
			}
		}

	} /* if loop ends here */


	MPI_Barrier(grid.Comm);

	/* Scatter the Data  to all processes by MPI_SCATTER */
	MPI_Scatter (A_array, matrix_block, MPI_FLOAT, A_block , matrix_block , MPI_FLOAT, 0, grid.Comm);

	MPI_Scatter (B_array, matrix_block, MPI_FLOAT, B_block, matrix_block, MPI_FLOAT, 0, grid.Comm);


	/* Do initial arrangement of Matrices */

	if(grid.Row !=0)
	{
		src   = (grid.Col + grid.Row) % grid.N;
		dest = (grid.Col + grid.N - grid.Row) % grid.N;
		recv_tag =0;
		send_tag = 0;
		MPI_Sendrecv_replace(A_block, matrix_block, MPI_FLOAT, dest, send_tag, src, recv_tag, grid.Row_comm, &status);
	}
	if(grid.Col !=0)
	{
		src   = (grid.Row + grid.Col) % grid.N;
		dest = (grid.Row + grid.N - grid.Col) % grid.N;
		recv_tag =0;
		send_tag = 0;
		MPI_Sendrecv_replace(B_block, matrix_block, MPI_FLOAT, dest,send_tag, src, recv_tag, grid.Col_comm, &status);
	}

	/* Allocate Memory for Bloc C Array */
	C_block = (int *) malloc (block_size * block_size * sizeof(int));
	for(index=0; index<block_size*block_size; index++)
		C_block[index] = 0;

	/* The main loop */

	send_tag = 0;
	recv_tag = 0;
	for(istage=0; istage<grid.N; istage++)
	{
		index=0;
		for(i=0; i<block_size; i++)
		{
			for(j=0; j<block_size; j++)
			{
				for(l=0; l<block_size; l++)
				{
					C_block[index] += A_block[i*block_size + l] * B_block[l*block_size + j];
				}
				index++;
			}
		}
		/* Move Bloc of Matrix A by one position left with wraparound */
		src   = (grid.Col + 1) % grid.N;
		dest = (grid.Col + grid.N - 1) % grid.N;
		MPI_Sendrecv_replace(A_block, matrix_block, MPI_FLOAT, dest,send_tag, src, recv_tag, grid.Row_comm, &status);

		/* Move Bloc of Matrix B by one position upwards with wraparound */
		src   = (grid.Row + 1) % grid.N;
		dest = (grid.Row + grid.N - 1) % grid.N;
		MPI_Sendrecv_replace(B_block, matrix_block, MPI_FLOAT, dest, send_tag, src, recv_tag, grid.Col_comm, &status);
	}


	/* Memory for output global matrix in the form of array  */
	if(grid.MyRank == Root)
		C_array = (int *) malloc (sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);

	MPI_Barrier(grid.Comm);

	/* Gather output block matrices at processor 0 */
	MPI_Gather (C_block, block_size * block_size, MPI_FLOAT, C_array,block_size*block_size, MPI_FLOAT, Root, grid.Comm);


	/* Rearranging the output matrix in a array by approriate order  */
	if (grid.MyRank == Root)
	{
		for (x = 0; x < grid.N; x++)
		{
			for (y = 0; y < grid.N; y++)
			{
				Proc_Id = x * grid.N + y;
				for (i = 0; i < block_size; i++)
				{
					gr = x * block_size + i;
					for (j = 0; j < block_size; j++)
					{
						lindex = (Proc_Id * block_size * block_size) + (i * block_size) + j;
						gc = y * block_size + j;
						C[gr][gc] = C_array[lindex];
					}
				}
			}
		}
		printf("Matrix A :\n");
		for(i = 0; i < MATRIX_SIZE; i++)
		{
			for(j = 0; j < MATRIX_SIZE; j++)
				printf ("%d ", A[i][j]);
			printf ("\n");
		}
		printf("\n");

		printf("Matrix B : \n");
		for(i = 0; i < MATRIX_SIZE; i++)
		{
			for(j = 0; j < MATRIX_SIZE; j++)
				printf("%d ", B[i][j]);
			printf("\n");
		}
		printf("\n");

		printf("Matrix C :\n");
		for(i = 0; i < MATRIX_SIZE; i++)
		{
			for(j = 0; j < MATRIX_SIZE; j++)
				printf("%d ",C[i][j]);
			printf("\n");
		}
	}
	MPI_Finalize();
}
