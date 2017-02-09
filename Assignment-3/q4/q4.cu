/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void rotateArray(int *c, int numThreads)
{
	int nextIndex = (threadIdx.x + 1)%numThreads;
	int val = c[nextIndex];

	__syncthreads();

	c[threadIdx.x] = val;
}

#define N 1024
int main(void)
{
	int *c, *res;
	int *d_c;
	int size = N * sizeof(int);

	//Allocate memory for array in GPU
	cudaMalloc((void **)&d_c, size);

	//Allocate memory on host
	c = (int *)malloc(size); 
	res = (int *)malloc(size);

	srand(time(NULL));
	//Populate array
	for (int i = 0; i < N; ++i)
	{
		c[i] = rand()%20;
	}

	//Copy input to device
	cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);
	
	//Launch rotateArray() kernel on GPU
	rotateArray<<<1,N>>>(d_c, N);

	//Copy result back to host
	cudaMemcpy(res, d_c, size, cudaMemcpyDeviceToHost);

	printf("First thirty elements are as follows:\n");
	printf("Original\tNew\n");
	for (int i = 0; i < 30; ++i)
	{
		printf("%d\t\t%d\n", c[i], res[i]);
	}

	//Cleanup
	free(c); free(res);
	cudaFree(d_c);
	return 0;
}
