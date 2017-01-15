/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

   Calculation of Pi - Monte Carlo Simulation
*/

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#define REPS 1500
#define NUMTHREADS 4

using namespace std;

static long MULTIPLIER  = 1366;
static long ADDEND      = 150889;
static long PMOD        = 714025;
long random_last = 0.0;
double random_low, random_hi;

double random1()
{
	long random_next;
	double ret_val;

	random_next = (MULTIPLIER  * random_last + ADDEND)% PMOD;
    random_last = random_next;

	ret_val = ((double)random_next/(double)PMOD)*(random_hi-random_low)+random_low;
	return ret_val;
}

void seed1(double low_in, double hi_in)
{
	if(low_in < hi_in)
	{ 
		random_low = low_in;
		random_hi  = hi_in;
	}
	else
	{
		random_low = hi_in;
		random_hi  = low_in;
	}
	//random_last = PMOD/ADDEND;  // just pick something
	random_last = chrono::duration_cast< chrono::milliseconds >(chrono::system_clock::now().time_since_epoch()).count();

}

int main()
{
	double x,y,pi,x2,y2;
	double in = 0,i;
	//srand(time(NULL));
	seed1(0,2);
	//omp_set_num_threads(NUMTHREADS);
	#pragma omp parallel firstprivate(x, y, x2, y2, i) reduction(+:in) num_threads(NUMTHREADS)
	{
		for(i = 1; i <= REPS; i++)
		{
			x = (double)(random1()-1);
			y = (double)(random1()-1);
			x2 = pow(x,2);
			//cout<<"x is "<<x;
			y2 = pow(y,2);
			//cout<<"y is "<<y;
			if(sqrt(x2+y2)<=1.0)
				in++;
		}
		pi = (in/REPS) * 4.0;
	}
	cout<<"Estimated value of pi, after "<<REPS<<" repitition is "<<pi<<"\n";
}
