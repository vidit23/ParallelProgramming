/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void square(long long int *c)
{
	c[threadIdx.x] = threadIdx.x * threadIdx.x;
}

#define N 1024
int main(void)
{
	long long int *c;
	long long int *d_c;
	int size = N * sizeof(long long int);

	//Allocate memory for array in GPU
	cudaMalloc((void **)&d_c, size);

	//Allocate memory on host
	c = (long long int *)malloc(size); 
	
	//Launch square() kernel on GPU
	square<<<1,1024>>>(d_c);

	//Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	printf("c[1023]= %d\n", c[1023]);
	printf("1023*1023 = %ld\n", 1023*1023);

	//Cleanup
	free(c);
	cudaFree(d_c);
	return 0;
}
