#include <iostream>
#include "omp.h"
using namespace std;

void printHello(int ID)
{
	cout<<"Hello: "<<ID<<endl;
	cout<<"World: "<<ID<<endl;
}

int main()
{
	omp_set_num_threads(5);
	
	#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		printHello(ID);
	}
	return 0;
}