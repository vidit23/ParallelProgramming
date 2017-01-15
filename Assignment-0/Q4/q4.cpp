/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

   This program considers two vectors X and Y of length 2^16 and a scalar 'a'.
   The operation X[i] = a*X[i] + Y[i] is performed by the program.
   This is repeated with varying number of threads used. 

   Work is divided between the threads as follows:
      If there are n threads, each thread will traverse over npt = ((2^16)/n) elements
      Thread 1 will loop from i=0 to i=npt-1
      Thread 2 will loop from i=npt to i=2npt-1
      Thread 3 will loop from i=2npt to i=3npt-1 and so on

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


#define LEN    65536    //Length of vectors - 2^16
#define TMAX   20       //Max no. of threads
#define ITERATIONS 100  //No. of iterations to cancel out errors

struct limits
{//contains the limits of the vectors that a particular thread will
   //have to perform operations on
   float start, end;
};

vector<float> X (LEN, 1.1);
vector<float> Y (LEN, 2.1);
int a = 2;

//Operation performed over the entire vectors in unithread manner
void wholeOperation()
{
   for (int i = 0; i < LEN; ++i)
   {
      X[i] = a*X[i] + Y[i];
   }
}

//Function to allow threads to only work on a portion of vectors
void *partialOp(void* limit)
{
   struct limits *cur = (limits*)limit; //contains limits for thread
   float start = cur->start;
   float end = cur->end;
   for (int i = start; i < end; ++i)
   {
      X[i] = a*X[i] + Y[i];
   }
}

void operation(int n, struct limits arrayOfLimits[])
{//Creates the threads and waits for them to complete execution
   pthread_t tid[n+1];
   for (int i = 1; i <= n; i++) 
   {
      struct limits cur = arrayOfLimits[i];
      pthread_create(&tid[i], NULL, partialOp, (void *)(&cur));
   }
   for (int i = 1; i <= n; i++) 
      pthread_join(tid[i], NULL);
   return;
}

//Creates the limits and then performs execution over a number of iterations
int calcAvgTime(int n)
{
   int npt = LEN / n;
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
   for (int i = 0; i < ITERATIONS; ++i)
   {
      X = vector<float> (LEN, 1.1);
      Y = vector<float> (LEN, 2.1);

      //clock function
      high_resolution_clock::time_point t1 = high_resolution_clock::now();
      
      // operation(); Create n threads, pass limits, commit operations, wait for execution
      operation(n,arrayOfLimits);

      high_resolution_clock::time_point t2 = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>( t2 - t1 ).count();
      sum_durations += duration;
   }
   int average_time = sum_durations/ITERATIONS;

   return average_time;
}

int main () {

   int avg_times[TMAX+1];


   for (int i = 1; i <= TMAX; ++i)
   {
      avg_times[i] = calcAvgTime(i);
   }
   //run twice to negate array not being in cache the first time
   for (int i = 1; i <= TMAX; ++i)
   {
      avg_times[i] = calcAvgTime(i);
   }

   cout<<"Num_t\tExec Time (micro-s)"<<endl;

   for (int i = 1; i <= TMAX; ++i)
   {
      cout<<i<<"\t"<<avg_times[i]<<endl;
   }

   return 0;
}
