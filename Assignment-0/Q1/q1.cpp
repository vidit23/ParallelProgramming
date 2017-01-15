/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

   This program creates 5 threads that terminate after displaying a Hello World message
   Once all 5 threads have completed the program ends
*/

#include <iostream>
#include <cstdlib>
#include <pthread.h>
using namespace std;

#define NUM_T    5   //Number of threads

//void * function pointer that is used to display the message
void *HelloWorld(void*) {
   cout << "Hello World"<<endl;
   pthread_exit(NULL);
}

int main () {
   pthread_t threads[NUM_T];
   int rv; //to check return value of pthread_create for error
   int i;
	
   for( i=0; i < NUM_T; i++ ){
      cout << "main created thread " << i << endl;
      rv = pthread_create(&threads[i], NULL, HelloWorld, NULL);
		
      //error condition
      if (rv){
         cout << "Error:unable to create thread," << rv << endl;
         exit(-1);
      }
   }
	
   pthread_exit(NULL);
}
