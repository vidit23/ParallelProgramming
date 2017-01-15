/*
Implement the following behaviour in a pthreads program using mutexes and condition variables.
Create 2 increment- count threads and 1 watch-count thread.
Increment-count threads increment a count variable (shared by both) till a threshold is reached.
On reaching the threshold, a signal is sent to the watch-count thread (use pthread cond signal).
The watch-count thread locks the count variable, and waits for the signal (use pthread cond wait()) from one of the increment-count threads.
As signal arrives, the watch-count thread releases lock and exits.
The other two threads exit too.
*/

#include <pthread.h>
#include <iostream>
#include <stdlib.h>
#include<unistd.h>
using namespace std;

#define NUM_THREADS  3
#define TCOUNT 10
#define COUNT_LIMIT 12

int count1 = 0;
pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;

void *inc_count(void *t)
{
  int i;
  long my_id = (long)t;

  for (i=0; i < TCOUNT; i++) {
    pthread_mutex_lock(&count_mutex);
    count1++;

    /*
    Check the value of count and signal waiting thread when condition is
    reached.  Note that this occurs while mutex is locked.
    */
    if (count1 == COUNT_LIMIT) {
      cout<<"inc_count(): thread "<<my_id<<", count = "<<count1<<"  Threshold reached. ";
      pthread_cond_signal(&count_threshold_cv);
      cout<<"Just sent signal.\n";
    }
    cout<<"inc_count(): thread "<<my_id<<", count = "<<count1<<", unlocking mutex\n";
    pthread_mutex_unlock(&count_mutex);

    /* Do some work so threads can alternate on mutex lock */
    usleep(100);
    }
  pthread_exit(NULL);
}

void *watch_count(void *t)
{
  long my_id = (long)t;

  cout<<"Starting watch_count(): thread "<<my_id<<"\n";

  /*
  Lock mutex and wait for signal.  Note that the pthread_cond_wait routine
  will automatically and atomically unlock mutex while it waits.
  Also, note that if COUNT_LIMIT is reached before this routine is run by
  the waiting thread, the loop will be skipped to prevent pthread_cond_wait
  from never returning.
  */
  pthread_mutex_lock(&count_mutex);
  while (count1 < COUNT_LIMIT) {
    cout<<"watch_count(): thread "<<my_id<<" Count= "<<count1<<". Going into wait...\n";
    pthread_cond_wait(&count_threshold_cv, &count_mutex);
    cout<<"watch_count(): thread "<<my_id<<" Condition signal received. Count= "<<count1<<"\n";
    cout<<"watch_count(): thread "<<my_id<<" Updating the value of count...\n";
    count1 += 125;
    cout<<"watch_count(): thread "<<my_id<<" count now = "<<count1<<".\n";
    }
  cout<<"watch_count(): thread "<<my_id<<" Unlocking mutex.\n";
  pthread_mutex_unlock(&count_mutex);
  pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
  int i, rc;
  long t1=1, t2=2, t3=3;
  pthread_t threads[3];
  pthread_attr_t attr;

  /* Initialize mutex and condition variable objects */
  pthread_mutex_init(&count_mutex, NULL);
  pthread_cond_init (&count_threshold_cv, NULL);

  /* For portability, explicitly create threads in a joinable state */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_create(&threads[0], &attr, watch_count, (void *)t1);
  pthread_create(&threads[1], &attr, inc_count, (void *)t2);
  pthread_create(&threads[2], &attr, inc_count, (void *)t3);

  /* Wait for all threads to complete */
  for (i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
  cout<<"Main(): Waited and joined with "<<NUM_THREADS<<" threads. Final value of count = "<<count1<<". Done.\n";

  /* Clean up and exit */
  pthread_attr_destroy(&attr);
  pthread_mutex_destroy(&count_mutex);
  pthread_cond_destroy(&count_threshold_cv);
  pthread_exit (NULL);

}
