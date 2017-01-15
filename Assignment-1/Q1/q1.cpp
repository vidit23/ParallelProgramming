#include <iostream>
#include "omp.h"
using namespace std;

int main()
{
	omp_set_num_threads(5);
	
	#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		cout<<"Hello: "<<ID<<endl;
		cout<<"World: "<<ID<<endl;
	}
	return 0;
}