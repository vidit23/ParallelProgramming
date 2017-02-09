/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>    

#define CUDA_ERROR_CHECK 1

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#define MAXDEPTH 24
#define INSORT_CRITERIA 48
#define N 100

void quicksortGPU(int *, int);

__device__ void selSort( int *data, int left, int right )
{
	for( int i = left ; i <= right ; ++i )
	{
		int min_val = data[i];
		int min_idx = i;

		// Find the smallest value in the range [left, right].
		for( int j = i+1 ; j <= right ; ++j )
		{
			int val_j = data[j];
			if( val_j < min_val )
			{
				min_idx = j;
				min_val = val_j;
			}
		}

		// Swap the values.
		if( i != min_idx )
		{
			data[min_idx] = data[i];
			data[i] = min_val;
		}
	}
}

__global__ void cdp_simple_quicksort(int *data, int left, int right, int depth ){
    //If we're too deep or there are few elements left, we use an insertion sort...
    if( depth >= MAXDEPTH || right-left <= INSORT_CRITERIA )
    {
        selSort( data, left, right );
        return;
    }

    cudaStream_t s,s1;
    int *lptr = data+left;
    int *rptr = data+right;
    int  pivot = data[(left+right)/2];

    int lval;
    int rval;

    int nright, nleft;

    // Do the partitioning.
    while (lptr <= rptr)
    {
        // Find the next left- and right-hand values to swap
        lval = *lptr;
        rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot && lptr < data+right)
        {
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot && rptr > data+left)
        {
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr)
        {
            *lptr = rval;
            *rptr = lval;
            lptr++;
            rptr--;
        }
    }

    // Now the recursive part
    nright = rptr - data;
    nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr-data))
    {
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right)
    {
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}

void quicksortGPU(int *data, int n)
{
    int* gpuData;
    int left = 0;
    int right = n-1;

    // Prepare CDP for the max depth 'MAXDEPTH'.
    CudaSafeCall(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAXDEPTH));

    // Allocate GPU memory.
    CudaSafeCall(cudaMalloc((void**)&gpuData,n*sizeof(int)));
    CudaSafeCall(cudaMemcpy(gpuData,data, n*sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t pstart, pstop, sstart, sstop;

	cudaEventCreate(&pstart);
	cudaEventCreate(&pstop);
	cudaEventCreate(&sstart);
	cudaEventCreate(&sstop);

	cudaEventRecord(pstart);

	// Launch on device
    cdp_simple_quicksort<<< 1, 1 >>>(gpuData, left, right, 0);

    cudaEventRecord(pstop);
	cudaEventSynchronize(pstop);

	float t = 0;
	cudaEventElapsedTime(&t, pstart, pstop);
	printf("Parallel execution, array size %d: %f milliseconds\n", N, t);

	cudaEventDestroy(pstart);
	cudaEventDestroy(pstop);

    CudaSafeCall(cudaDeviceSynchronize());

    // Copy back
    CudaSafeCall(cudaMemcpy(data,gpuData, n*sizeof(int), cudaMemcpyDeviceToHost));

    CudaSafeCall(cudaFree(gpuData));
    
    CudaSafeCall(cudaDeviceReset());
}

//Serial quicksort
void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}
 
int partition (int arr[], int low, int high)
{
    int pivot = arr[high];    // pivot
    int i = (low - 1);  // Index of smaller element
 
    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] <= pivot)
        {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}
 
void quickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        int pi = partition(arr, low, high);
 
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}




int main()
{
	int *arr = (int*)malloc(N*sizeof(int));
	int *sarr = (int*)malloc(N*sizeof(int));
	srand(time(NULL));
	// printf("Initial Array\n");
	for (int i = 0; i < N; i+=N/100)
	{
		arr[i] = rand()%200;
		sarr[i] = arr[i];
		// printf("%d ", arr[i]);
	}
	printf("\n");

	
	quicksortGPU(arr, N);
	


	// printf("After sorting\n");
	// for (int i = 0; i < N; i+=N/100)
	// {
	// 	printf("%d ", arr[i]);
	// }

	//Check for errors in Cuda
	CudaCheckError();
	cudaCheckError();

	// cudaEventRecord(sstart);
	double elapsedTime;
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	// high_resolution_clock::time_point t1 = high_resolution_clock::now();

	quickSort(sarr, 0, N-1);
 	gettimeofday(&t2, NULL);
	// high_resolution_clock::time_point t2 = high_resolution_clock::now();
 	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
	// Execution time measurement, that is the result
	// auto duration = duration_cast<milliseconds>(t2 - t1).count();
	// cudaEventRecord(sstop);
	// cudaEventSynchronize(sstop);

	// cudaEventElapsedTime(&t, sstart, sstop);
	printf("Serial execution, array size %d: %ld milliseconds\n", N, elapsedTime);

	// cudaEventDestroy(sstart);
	// cudaEventDestroy(sstop);

	free(arr); free(sarr);
	return 0;
}