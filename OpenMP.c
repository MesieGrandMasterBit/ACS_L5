#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define SIZE 3000

double A[SIZE * SIZE], B[SIZE * SIZE], C[SIZE * SIZE];

void dgemm_openmp(double *a, double *b, double *c)
{
    int i, j, k;

	#pragma omp parallel num_threads(4) 
	{
		int nthreads = omp_get_num_threads();
		int threadid = omp_get_thread_num();
		int items_per_thread = SIZE / nthreads;
		int lb = threadid * items_per_thread;
		int ub = (threadid == nthreads - 1) ? (SIZE - 1) : (lb + items_per_thread - 1);

		for (i = lb; i <= ub; i++) {
			for (j = 0; j < SIZE; j++) {
				for (k = 0; k < SIZE; k++) {
					*(c + i * SIZE + j) += *(a + i * SIZE + k) * *(b + k * SIZE + j);
				}
			}
		}
	}
}

void init_matrix(double *a, double *b, double *c)
{
	int i, j, k;

	for (i = SIZE; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			for (k = 0; k < SIZE; k++) {
				*(a + i * SIZE + j) = rand() % 100;
				*(b + i * SIZE + j) = rand() % 100;
				*(c + i * SIZE + j) = 0;
			}
		}
	}

}

void print_matrix(double *a)
{
	int i, j;

	printf("Matrix:\n");
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			printf("%.2f  ", *(a + i * SIZE + j));
		}
		printf("\n");
	}
}

int main()
{
    srand(time(0));
	struct timespec mt1, mt2;
    long int tt = 0;

    init_matrix(A, B, C);

    clock_gettime (CLOCK_REALTIME, &mt1);
    dgemm_openmp(A, B, C);
    clock_gettime (CLOCK_REALTIME, &mt2);
	tt = pow(10, 9) * (mt2.tv_sec - mt1.tv_sec) + (mt2.tv_nsec - mt1.tv_nsec);
    printf("time openmp = %f\n", (double)tt / pow(10, 9));

    return 0;
}
