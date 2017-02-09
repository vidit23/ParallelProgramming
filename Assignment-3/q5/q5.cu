/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

__global__ void bf_2flags(int *Na, int *src, int *F1, int *F2, int *exists, int *Sa, int *Ea, int threadsPerBlock )
{
	
	int id = blockIdx.x * threadsPerBlock + threadIdx.x;
	
	if (exists[id]==1)
	{
		Na[id] = 65000; //MAX INT Value		
		F1[id] = 0;
		F2[id] = 0;
		
		if (id == *src)
		{	//Starting node conditions			
			Na[id] = 0;
			F1[id] = 1;
		}

		for (int i = 0; i < 103689; ++i)
		{			
			if (F1[Sa[id]] == 1)
			{
				if (Na[Ea[id]] > Na[Sa[id]] + 1)
				{
					//Relax
					// atomicAdd(&Na[Ea[id]], Na[Sa[id]] + 1 - Na[Ea[id]]);
					Na[Ea[id]] = Na[Sa[id]] + 1;
					F2[Ea[id]] = 1;					
				}
			}

			//Swap flags
			F1[id] = F2[id];
			F2[id] = 0;			
		}
	}
}

#define N 8300
#define ROWS 103689
#define min_threads 16
#define max_threads 1024

int main(void)
{

	int size = N * sizeof(int);
	int *F1 = (int *)malloc(size), *F2 = (int *)malloc(size), *Na = (int *)malloc(size);
	int *Sa = (int *)malloc(ROWS * sizeof(int)), *Ea = (int *)malloc(ROWS * sizeof(int));
	int *exists = (int*)malloc(size);
	
	//File Operations
	FILE *ptr_file;
	char *buf;
	size_t len = 0;

	ptr_file = fopen("Wiki-Vote.txt", "r");

	if (!ptr_file)
	{
		printf("File Open Failed\n");
		return 0;
	}

	int m_left, m_right;

	//Set up the adjacency matrix
	for (long int i = 0; i < ROWS; ++i)
	{
		buf = (char *)malloc(30);
		getline(&buf, &len, ptr_file);
		m_left = atoi(strsep(&buf, "\t"));
		m_right = atoi(strsep(&buf, "\n"));
		
		exists[m_left] = 1;
		exists[m_right] = 1;

		Sa[i] = m_left;
		Ea[i] = m_right;
	}

	
	int src;
	printf("Enter source node: ");
	scanf("%d", &src);
	int *d_src;
	cudaMalloc((void **)&d_src, sizeof(int));
	cudaMemcpy(d_src, &src, sizeof(int), cudaMemcpyHostToDevice);

	//Timing events
	cudaEvent_t start, stop;

	int numBlocks;
	int *d_F1, *d_F2, *d_Na, *d_Sa, *d_Ea, *d_exists;

	printf("Threads Per Block\tTime(ms)\n");
	for (int threadsPerBlock = min_threads; threadsPerBlock <= max_threads; threadsPerBlock*=2)
	{
		numBlocks = (N/1024 + 1) * 1024 / 16;

		//Allocate space for all inputs and copy them
		
		cudaMalloc((void **)&d_F1, size);
		cudaMalloc((void **)&d_F2, size);
		cudaMalloc((void **)&d_Na, size);
		cudaMalloc((void **)&d_Sa, ROWS*sizeof(int));
		cudaMalloc((void **)&d_Ea, ROWS*sizeof(int));
		cudaMalloc((void **)&d_exists, size);
		cudaMemcpy(d_F1, F1, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_F2, F2, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Na, Na, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Sa, Sa, ROWS*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Ea, Ea, ROWS*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_exists, exists, size, cudaMemcpyHostToDevice);

		//Run kernel
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		bf_2flags<<<numBlocks, threadsPerBlock>>>(d_Na, d_src, d_F1, d_F2, d_exists, d_Sa, d_Ea, threadsPerBlock);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float t = 0;
		cudaEventElapsedTime(&t, start, stop);
		printf("%d\t\t%f\n", threadsPerBlock, t);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	

	//Copy result back from device to host
	cudaMemcpy(Na, d_Na, N*sizeof(int), cudaMemcpyDeviceToHost);
	
	//Output display
	
	printf("Nodes reachable from %d\n", src);
	for (int i = 0; i < N; ++i)
	{
		if (exists[i]==0)
		{
			continue;
		}
		if (Na[i] == 65000)
		{
			//printf("U ");
		}
		else
		{printf("%d : %d\n", i, Na[i]);}
	}

	//Check for errors in Cuda
	cudaCheckError();

	//Cleanup
	free(F1); free(F2); free(exists); free(Na); free(Sa); free(Ea);
	cudaFree(d_F1); cudaFree(d_F2); cudaFree(d_Na); cudaFree(d_Sa); cudaFree(d_Ea);

	return 0;

}
