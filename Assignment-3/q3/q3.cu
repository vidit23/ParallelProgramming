/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ITERATIONS 4	//Repeat the experiment for greater accuracy

__global__ void add(int *a, int *b, int *c, int tpb)
{
	//Find the correct thread index in the grid
	int i = blockIdx.x * tpb + threadIdx.x;
	c[i] = a[i] + b[i];
	
}

#define N 1000000		//Array Size
#define min_threads 16
#define max_threads 1024


int main(void)
{
	int *a,*b,*c;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);

	a = (int *)malloc(size); 
	b = (int *)malloc(size); 
	c = (int *)malloc(size); 

	//Allocate on device
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	srand(time(NULL));
	//Populate a and b
	for (int i = 0; i < N; ++i)
	{
		a[i] = rand()%20;
		b[i] = rand()%37;
	}

	int numBlocks;
	cudaEvent_t start, copy, exec, result;	//Events for measuring time
	
	//To calculate average over a number of iterations
	float t1[7], t2[7], t3[7], total[7];
	for (int i = 0; i < 7; ++i)
	{
		t1[i]=0;
		t2[i]=0;
		t3[i]=0;
		total[i]=0;
	}

	printf("t1: time for copying arrays\n");
	printf("t2: time for kernel execution\n");
	printf("t3: time for copying result back\n\n");
	printf("All times in milliseconds\n");
	printf("TPB\t\tNB\t\tt1\t\tt2\t\tt3\t\ttotal\t\n");
	
	int count;
	for (int i = 0; i < ITERATIONS; ++i)
	{
		count=0;
		for (int threadsPerBlock = min_threads; threadsPerBlock <= max_threads; threadsPerBlock*=2)
		{
			numBlocks = (N + threadsPerBlock - 1)/threadsPerBlock;	



			cudaEventCreate(&start);
			cudaEventCreate(&copy);
			cudaEventCreate(&exec);
			cudaEventCreate(&result);

			cudaEventRecord(start);
			
			//Copy inputs to device
			cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);		
			cudaEventRecord(copy);
			cudaEventSynchronize(copy);

			//Launch add() kernel on GPU
			add<<<numBlocks,threadsPerBlock>>>(d_a, d_b, d_c, threadsPerBlock);
			cudaEventRecord(exec);
			cudaEventSynchronize(exec);
			
			//Copy result back to host
			cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
			cudaEventRecord(result);
			cudaEventSynchronize(result);
			

			float temp1=0, temp2=0, temp3=0, temptotal;
			cudaEventElapsedTime(&temp1, start, copy);
			cudaEventElapsedTime(&temp2, copy, exec);
			cudaEventElapsedTime(&temp3, exec, result);		
			cudaEventElapsedTime(&temptotal, start, result);

			t1[count] += temp1;
			t2[count] += temp2;
			t3[count] += temp3;
			total[count] += temptotal;

			
			cudaEventDestroy(start);
			cudaEventDestroy(copy);
			cudaEventDestroy(exec);
			cudaEventDestroy(result);
			count++;
		}
	}

	int threadsPerBlock = min_threads;

	for (int i = 0; i < 7; ++i)
	{
		numBlocks = (N + threadsPerBlock - 1)/threadsPerBlock;

		t1[i]/=(float)ITERATIONS;
		t2[i]/=(float)ITERATIONS;
		t3[i]/=(float)ITERATIONS;
		total[i]/=(float)ITERATIONS;
		printf("%d\t\t%d\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f\t\t\n", 
				threadsPerBlock, numBlocks, t1[i], t2[i], t3[i], total[i]);
		threadsPerBlock*=2;
	}

	//Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}
