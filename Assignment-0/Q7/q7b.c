/*

   Jaitirth Jacob - 13CO125      Vidit Bhargava - 13CO151

   Simple MM program with matrices allocated to stack.
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define SIZE 830

int main(void)
{
    int A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE] = {0};
    int i, j, k;

    srand(time(NULL));

    for(i = 0; i < SIZE; i++)
    {
        for(j = 0; j < SIZE; j++)
        {
            A[i][j] = rand()%100;
            B[i][j] = rand()%100;
        }
    }

    clock_t begin, end;
    double time_spent;

    begin = clock();

    for(i = 0; i < SIZE; i++)
        for(j = 0; j < SIZE; j++)
            for(k = 0; k < SIZE; k++)
                C[i][j] += A[i][k] * B[k][j];

    end = clock();

    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("Elapsed time: %.2lf seconds.\n", time_spent);

    return 0;
}