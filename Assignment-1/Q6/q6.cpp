/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

   Calculation of Pi - Worksharing and Reduction
*/

#include <iostream>
#include <cstdlib>
#include "omp.h"
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;


#define TMAX   16     	//Max no of threads
#define ITERATIONS 4    //No of iterations to negate random factors

static long num_steps = 100000;
double step = 1.0/(double)num_steps;

void operation(int num_t)
{
	double x, pi, sum = 0.0;
	int i;
	omp_set_num_threads(num_t);
	#pragma omp parallel reduction(+:sum)//Reduction used on sum
	{
		#pragma omp for
		for (i=0; i<num_steps; i++) 
		{
			x = (i+0.5)*step;			
			sum += 4.0/(1.0+x*x); //Split between different threads, later added together
		}
	}
	pi = step * sum;
	// cout<<pi<<endl;
}

int calcAvgTime(int n)
{
	//Performs the calculation for a number of iterations
	//clock function
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// operation(); Pass the number of threads, wait for execution
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