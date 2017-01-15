/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

   Producer-Consumer Program
*/

#include <iostream>
#include <stdlib.h>
#include <omp.h>

using namespace std;
#define NUM 100

// PRODUCER: initialize A with random data
void fill_rand(int nval, double *A)
{
	int i;
	for (i=0; i<nval; i++)
		A[i] = (double) rand()/1111111111;
}
// CONSUMER: Sum the data in A
double Sum_array(int nval, double *A)
{
	int i;
	double sum = 0.0;
	for (i=0; i<nval; i++)
		sum = sum + A[i];
	return sum;
}


int main()
{
	double *A, sum, runtime;
	int flag = 0;
	A = (double *)malloc(NUM*sizeof(double));
	runtime = omp_get_wtime();
	#pragma omp parallel shared(A,sum,flag)
	{
		#pragma omp sections
		{
			#pragma omp section
			{
				fill_rand(NUM,A);
				#pragma omp flush
				flag = 1;
				if(flag==1)
				{
					#pragma omp flush(flag)
				}
			}
			#pragma omp section
			{
				while (!flag)
				{
					#pragma omp flush(flag)
				}
				#pragma omp flush
				sum = Sum_array(NUM,A);
			}
	 	}
	}
	runtime = omp_get_wtime() - runtime;
	cout<<" In "<<runtime<<" seconds, The sum is "<<sum<<"\n";
	/*
	double *A, sum, runtime;
	int flag = 0;
	A = (double *)malloc(N*sizeof(double));
	runtime = omp_get_wtime();
	fill_rand(N, A); // Producer: fill an array of data
	sum = Sum_array(N, A); // Consumer: sum the array
	runtime = omp_get_wtime() - runtime;
	cout<<" In "<<runtime<<" seconds, The sum is "<<sum<<"\n";
	*/
}
