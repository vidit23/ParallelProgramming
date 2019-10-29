/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

   This program considers two vectors X and Y of length 2^16 and a scalar 'a'.
   The operation X[i] = a*X[i] + Y[i] is performed by the program.
   This is repeated with varying number of threads used. 

   The execution times are recorded using chrono functions and a number of iterations
   are performed to negate variable factors such as processor utilization, etc.

   The resulting output of Speedup vs No. of Threads has been compiled into a graph 
   by us in an attached image.
*/


#include <iostream>
#include "omp.h"
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;


#define LEN    65536    //Length of vectors - 2^16
#define TMAX   20       //Max no. of threads
#define ITERATIONS 100  //No. of iterations to cancel out errors

vector<float> X (LEN, 1.1);
vector<float> Y (LEN, 2.1);
int a = 2;

void operation(int num_t)
{
   omp_set_num_threads(num_t);
   //with pragma omp for
   #pragma omp parallel
   {
      #pragma omp for
      for (int i = 0; i < LEN; ++i)
      {
         X[i] = a*X[i] + Y[i];
      }
   }
   
   //without pragma omp for
   # pragma omp parallel
   {omp_set_num_threads(n);
      int id ;
      id = omp_get_thread_num();
      n= omp_get_num_threads ();
      for(int i = id; i<LEN; i=i+n)
      a[i] = a[i] * p + b[i];
   }
}

int calcAvgTime(int n)
{
   X = vector<float> (LEN, 1.1);
   Y = vector<float> (LEN, 2.1);

   //clock function
   high_resolution_clock::time_point t1 = high_resolution_clock::now();
   
   // operation(); Create n threads, pass limits, commit operations, wait for execution
   operation(n);

   high_resolution_clock::time_point t2 = high_resolution_clock::now();
   auto duration = duration_cast<microseconds>( t2 - t1 ).count();

   return (int)duration;
}

int main () {

   int avg_times[TMAX+1];

   for (int i = 1; i <= 1; ++i)
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
