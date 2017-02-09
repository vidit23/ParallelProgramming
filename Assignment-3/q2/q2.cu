/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

*/

#include <stdio.h>

int main()
{
    int count;
    cudaGetDeviceCount(&count);
    printf("Device Queries:\n");
    printf("There are %d CUDA devices.\n", count);

    for (int i = 0; i < count; ++i)
    {
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
		printf("\tDevice Identification:          %s\n",  devProp.name);
		printf("\tGlobal memory:                  %u\n",  devProp.totalGlobalMem);
		printf("\tShared memory per block:        %u\n",  devProp.sharedMemPerBlock);
		printf("\tNumber of registers per block:  %d\n",  devProp.regsPerBlock);
		printf("\tNumber of thread in warp:       %d\n",  devProp.warpSize);
		printf("\tMaximum threads per block:      %d\n",  devProp.maxThreadsPerBlock);
		for (int i = 0; i < 3; ++i)
			printf("\tMaximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
		for (int i = 0; i < 3; ++i)
			printf("\tMaximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
		printf("\tAvailable constant memory:      %u\n",  devProp.totalConstMem);
		printf("\tMajor revision number:          %d\n",  devProp.major);
		printf("\tMinor revision number:          %d\n",  devProp.minor);
		printf("\tNumber of multiprocessors:      %d\n",  devProp.multiProcessorCount);
		printf("\tClock rate:                     %d\n",  devProp.clockRate);
    }

    return 0;
}

