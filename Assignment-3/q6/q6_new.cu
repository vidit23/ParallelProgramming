#include <cuda.h>
//#include <cutil_inline.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>

#define ASIZE 256
#define DATA_SIZE 1024

__device__ int shifts[ASIZE];
__device__ int results[DATA_SIZE];

__global__ void processPattern(char* x ,int m, int shifts[])
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if ( idx >= m ) return;

	char c = x[idx];
	for( int i = m - 1; i >= idx; --i )
	{
		if ( x[i] == c )
		{// match is found
			shifts[c] = m - i;
			return;
		}
	}
}

__global__ void search(char *x, int m, char* y, int n, int shifts[], int indx[], int results[])
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx > (n - m) )
		return;
	if ( indx[idx] != idx )
		return;

	unsigned int yes = 1;
	for( int i = 0; i < m; ++i )
	{
		// try to match the string
		if ( x[i] != y[idx + i] )
		{
			yes = 0;
			break;
		}
	}
	results[idx] = yes;
}

void precomputeShiftIndx(char* y, int n, int m, int shifts[], int indx[])
{
	int j = 0;
	int limit = n - m;

	while (j <= limit ) {
		j += shifts[ y[j + m] ];
		indx[j] = j;
	}
}


void display_results(int n, int  results[]) {
	int j = 0;
	int flag =0;
	for( int i =0; i < n; ++i )
		if ( results[i] == 1 )
		{
			printf("%d. Found match at %d\n",j++, i);
			flag=1;
		}
	if(flag==0)
		printf("Not found\n");
}

int main(int argc, char* argv[])
{
	srand(time(NULL));
	char values[] = "ACGT";
	int cuda_device = 0;
	int n = 10000000; // length of main string
	int m = 100; // length of substring


	char* mainString = (char*)malloc(n * sizeof(char));
	char* subString = (char*)malloc(m * sizeof(char));
	for(int i=0;i < n;i++)
	{
		mainString[i] = values[rand()%4];
	}
	for(int i=0;i < m;i++)
	{
		subString[i] = values[rand()%4];
	}

	//
	// Initialize the shift and index array
	//
	int* l_shifts = (int*)malloc( ASIZE * sizeof(int) );
	for( int i = 0; i < ASIZE; ++i )
		l_shifts[i] = m + 1;
	int* l_indx = (int*) malloc( n * sizeof(int) );
	for( int i = 0; i < n; ++i )
		l_indx[i] = -1;

	cudaError_t error;
	cudaEvent_t start_event, stop_event;
	float time;
	float time2;

	// initializing the GPU timers
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);


	//
	// Allocate global memory to host the pattern, text and other supporting data
	// structures
	//
	char* d_substr = 0;
	int* d_shifts = 0;
	int* d_indx = 0;
	char* d_text = 0;
	int *d_results = 0,*l_results=(int*) malloc( n * sizeof(int) );
	for( int i = 0; i < n; ++i )
		l_results[i] = 0;
	//cudaGetSymbolAddress((void**)&d_shifts, "shifts");
	cudaMalloc((void**)&d_results, n * sizeof(int)) ;

	cudaMalloc((void**)&d_shifts, sizeof(int) * ASIZE) ;
	//error = cudaGetLastError();
	//printf("Error1: %s\n", cudaGetErrorString(error));
	cudaMalloc((void**)&d_indx, n * sizeof(int)) ;
	//error = cudaGetLastError();
	//printf("Error2: %s\n", cudaGetErrorString(error));
	cudaMalloc((void**)&d_substr, (m + 1)*sizeof(char)) ;
	//error = cudaGetLastError();
	//printf("Error3: %s\n", cudaGetErrorString(error));
	cudaMalloc((void**)&d_text, (strlen(mainString)+1)*sizeof(char)) ;
	//error = cudaGetLastError();
	//printf("Error4: %s\n", cudaGetErrorString(error));
	cudaMemcpy(d_shifts, l_shifts, sizeof(int) * ASIZE, cudaMemcpyHostToDevice ) ;
	cudaMemcpy(d_results, l_results, sizeof(int) * n, cudaMemcpyHostToDevice ) ;
	//error = cudaGetLastError();
	//printf("Error5: %s\n", cudaGetErrorString(error));
	cudaMemcpy(d_text, mainString, sizeof(char)*(strlen(mainString)+1), cudaMemcpyHostToDevice ) ;
	//error = cudaGetLastError();
	//printf("Error6: %s\n", cudaGetErrorString(error));
	cudaMemcpy(d_substr, subString, sizeof(char)*(strlen(subString)+1), cudaMemcpyHostToDevice) ;
	//error = cudaGetLastError();
	//printf("Error7: %s\n", cudaGetErrorString(error));

	//
	// Pre-process the pattern to be matched
	//
	dim3 threadsPerBlocks(ASIZE, 1);
	int t = m / threadsPerBlocks.x;
	int t1 = m % threadsPerBlocks.x;
	if ( t1 != 0 ) t += 1;
	dim3 numBlocks(t, 1);

	printf("Launching kernel with blocks=%d, threadsperblock=%d\n", numBlocks.x, threadsPerBlocks.x);
	cudaEventRecord(start_event, 0);
	processPattern<<<numBlocks,threadsPerBlocks>>>(d_substr, m, d_shifts);
	cudaThreadSynchronize();

	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize( stop_event );
	cudaEventElapsedTime( &time, start_event, stop_event );

	cudaMemcpy(l_shifts, d_shifts, sizeof(int) * ASIZE, cudaMemcpyDeviceToHost ) ;
	//error = cudaGetLastError();
	//printf("Error8: %s\n", cudaGetErrorString(error));
	//
	// Transfer the pre-computed shift indexes from host to device memory
	//
	cudaMemcpy(l_shifts, d_shifts, ASIZE * sizeof(int), cudaMemcpyDeviceToHost) ;
	precomputeShiftIndx(mainString , n, m, l_shifts, l_indx);
	cudaMemcpy(d_shifts, l_shifts, ASIZE * sizeof(int), cudaMemcpyHostToDevice) ;
	cudaMemcpy(d_indx, l_indx, n * sizeof(int), cudaMemcpyHostToDevice) ;
	//error = cudaGetLastError();
	//printf("Error9: %s\n", cudaGetErrorString(error));
	//
	// Perform the actual search
	//
	t = n / threadsPerBlocks.x;
	t1 = n % threadsPerBlocks.x;
	if ( t1 != 0 ) t += 1;
	dim3 numBlocks2(t, 1);
	printf("Launching kernel with blocks=%d, threadsperblock=%d\n", numBlocks2.x, threadsPerBlocks.x);
	cudaEventRecord(start_event, 0);
	search<<<numBlocks2,threadsPerBlocks>>>(d_substr, m, d_text, n, d_shifts, d_indx,d_results);
	cudaThreadSynchronize();

	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize( stop_event );
	cudaEventElapsedTime( &time2, start_event, stop_event );

	cudaEventDestroy( start_event ); // cleanup
	cudaEventDestroy( stop_event ); // cleanup
	printf("done and it took: %f+%f=%f milliseconds\n",time, time2, time+time2);

	//cudaGetSymbolAddress((void**)&d_results, "results");
	//cudaMalloc((void**)&d_results, n * sizeof(int)) ;
	//int* l_results = (int*) malloc( n * sizeof(int) );
	cudaMemcpy(l_results, d_results, n * sizeof(int), cudaMemcpyDeviceToHost) ;
	display_results(n, l_results);
	//error = cudaGetLastError();
	//printf("Error10: %s\n", cudaGetErrorString(error));
	cudaFree(d_substr);
	cudaFree(d_shifts);
	cudaFree(d_indx);
	cudaFree(d_text);
	free(mainString);
	free(subString);
	free(l_indx);
	free(l_shifts);
	free(l_results);

	cudaThreadExit();

	return 0;
}
