/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

   This program considers two matrices A[M][N] and B[N][P] and computes their product 
   C[M][P]. This is done using the trivial O(n^3) mat mul algorithm.

   The operation is made multithreaded by dividing the rows between different threads.
   If there are n threads, each thread will be responsible for npt = (M/n) rows
   As all operations on A and B will be read operations, there is no worry about
   race condition or deadlocks occuring. 
   
   This is repeated with varying number of threads used. The execution times are recorded.

      If there are n threads, 
      Thread 1 will be responsible for rows i=0 to i=npt-1
      Thread 2 will be responsible for rows i=npt to i=2npt-1
      Thread 3 will be responsible for rows i=2npt to i=3npt-1 and so on

   The execution times are recorded using chrono functions and a number of iterations
   are performed to negate variable factors such as processor utilization, etc.

   The resulting output of Speedup vs No. of Threads has been compiled into a graph 
   by us in an attached image.
*/

#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;


#define TMAX   350      //Max no of threads
#define ITERATIONS 4    //No of iterations to negate random factors

#define M 250
#define N 250
#define P 250

int A[M][N], B[N][P], C[M][P];

struct limits
{/*contains the limits of the vectors that a particular thread will
have to perform operations on*/
   float start, end;
};

/*Function to allow threads to only work on certain rows in the 
multiplication process*/
void *partialOp(void* limit)
{
   struct limits *cur = (limits*)limit;
   float start = cur->start;
   float end = cur->end;

   for(int i=start;i<end;++i)
   {
      for(int j=0;j<P;++j)
      {
         C[i][j]=0;
         for(int k=0;k<N;++k)
            C[i][j]=C[i][j]+(A[i][k]*B[k][j]);
      }
   }

}

void operation(int n, struct limits arrayOfLimits[])
{//Creates the threads and waits for them to complete execution
   pthread_t tid[n+1];
   for (int i = 1; i <= n; i++) 
   {
      struct limits cur = arrayOfLimits[i];
      //limits are passed through the thread using cur
      pthread_create(&tid[i], NULL, partialOp, (void *)(&cur));
   }
   for (int i = 1; i <= n; i++) 
      pthread_join(tid[i], NULL);
   return;
}

int calcAvgTime(int n)
{
   //creation of the limits for each thread
   int npt = M / n;
   limits arrayOfLimits[n+1];
   arrayOfLimits[1].start=0;
   arrayOfLimits[1].end=npt-1;
   for (int i = 2; i <= n; ++i)
   {//Creating the boundary limits for each thread
      arrayOfLimits[i].start=arrayOfLimits[i-1].start+npt;
      arrayOfLimits[i].end=arrayOfLimits[i-1].end+npt;
   }

   //Performs the calculation for a number of iterations
   int sum_durations = 0;
   for (int i = 0; i < 1; ++i)
   {

      //clock function
      high_resolution_clock::time_point t1 = high_resolution_clock::now();
      
      // operation(); Create n threads, pass limits, commit operations, wait for execution
      operation(n,arrayOfLimits);

      high_resolution_clock::time_point t2 = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>( t2 - t1 ).count();
      sum_durations += duration;
   }
   int average_time = sum_durations/1;

   return average_time;
}

void populateMatrices()
{
   for (int i = 0; i < M; ++i)
   {
      for (int j = 0; j < N; ++j)
      {
         A[i][j] = rand() % 5;
      }
   }

   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < P; ++j)
      {
         B[i][j] = rand() % 5;
      }
   }
}

int main () {
   populateMatrices();
   int avg_times[TMAX+1];

   for (int i = 1; i <= TMAX; ++i)
   {//run once prior to negate any cache issues that effect exec time
      avg_times[i] = calcAvgTime(i);
   }
   for (int i = 1; i <= TMAX; ++i)
   {
      avg_times[i] = 0;
   }
   
   //run over a number of iterations
   for (int x = 0; x < ITERATIONS; ++x)
   {
      for (int i = 1; i <= TMAX; ++i)
      {
         avg_times[i] += calcAvgTime(i);
      }
   }

   for (int i = 1; i <= TMAX; ++i)
   {
      avg_times[i] /= ITERATIONS;
   }
   cout<<"Num_t\tExec Time (micro-s)"<<endl;
   for (int i = 1; i <= TMAX; ++i)
   {
      cout<<i<<"\t"<<avg_times[i]<<endl;
   }

   return 0;
}
