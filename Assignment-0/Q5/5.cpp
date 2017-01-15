/*
Write a program to compute dot product of two vectors and calculate its running sum.
An iteration in a sequential implementation would look like Sum = Sum + (X[i] * Y[i]); .
Use mutex variables in this program.
Arrays X and Y, and variable Sum are available to all threads through a globally accessible structure.
Each thread works on a different part of the data.
The main thread waits for all the threads to complete their computations, and then it prints the resulting sum.
*/

#include <pthread.h>
#include <iostream>
#include <stdlib.h>
#include<cstdlib>
using namespace std;

#define THREADNUM 4
#define VECLEN 1000000

struct DOT
 {
   int *a;
   int *b;
   long long int sum;
   long int veclen;
 };

DOT data;
pthread_t callThd[THREADNUM];
pthread_mutex_t mutexsum;

void *dotprod(void *arg)
{
   int i, start, end, len ;
   long offset;
   int threadsum, *x, *y;
   offset = (long)arg;

   len = data.veclen;
   start = offset * len;
   end   = start + len;
   x = data.a;
   y = data.b;

   threadsum = 0;
   // Common code for each thread, operates based on the thread number (= offset)
   for (i=start; i<end ; i++)
    {
      threadsum += (x[i] * y[i]);
    }
	// Occupy mutex lock because we are changing the value of shared sum
   pthread_mutex_lock (&mutexsum);
   data.sum += threadsum;
   cout<<"Thread "<<offset<<" did "<<start<<" to "<<end<<":  ThreadSum="<<threadsum<<" global sum="<<data.sum<<"\n";
   pthread_mutex_unlock (&mutexsum);

   pthread_exit((void*) 0);
}


int main (int argc, char *argv[])
{
	int i;
	int *a, *b;
	void *status;
	pthread_attr_t attr;
	// Intilizing all the values
	a = (int*) malloc (THREADNUM*VECLEN*sizeof(int));
	b = (int*) malloc (THREADNUM*VECLEN*sizeof(int));

	for (i=0; i<VECLEN*THREADNUM; i++)
	{
		  a[i]=rand() % 1 ;
		  b[i]=rand() % 1 ;
	}

	data.veclen = VECLEN;
	data.a = a;
	data.b = b;
	data.sum=0;

	pthread_mutex_init(&mutexsum, NULL);
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(i=0;i<THREADNUM;i++)
	{
		pthread_create(&callThd[i], &attr, dotprod, (void *)i);
	}

	pthread_attr_destroy(&attr);
	/* Wait on the other threads */

	for(i=0;i<THREADNUM;i++)
	{
		pthread_join(callThd[i], &status);
	}

	cout<<"Sum =  "<<data.sum<<" \n";
	free (a);
	free (b);
	pthread_mutex_destroy(&mutexsum);
	pthread_exit(NULL);
}
