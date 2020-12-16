#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <assert.h>
#include <math.h>

#define SIZE 3000
int NTHREADS;

double *A, *B, *C;

void *dgemm_posix(double *a, double *b, double *c, int tid)
{
    int lb = tid * SIZE / NTHREADS;
    int ub = (tid + 1) * (SIZE / NTHREADS) - 1;

    int i, j, k;
    for (i = lb; i <= ub; i++) {
        for (j = 0; j < SIZE; j++) {
            for (k = 0; k < SIZE; k++) {
                *(c + i * SIZE + j) += *(a + i * SIZE + k) * *(b + k * SIZE + j);
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

void *worker(void *arg)
{
    int tid = (int)arg;
    dgemm_posix(A, B, C, tid);
}

int main(int argc, char* argv[])
{
    srand(time(0));
	struct timespec mt1, mt2;
    long int tt = 0;
    int rc, i;

    NTHREADS = atoi(argv[1]);
    pthread_t* threads; /* идентификатор потока */
    threads = (pthread_t*)malloc(NTHREADS * sizeof(pthread_t));

    A = malloc(sizeof(*A) * SIZE * SIZE);
    B = malloc(sizeof(*B) * SIZE * SIZE);
    C = malloc(sizeof(*C) * SIZE * SIZE);

    for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
            A[i * SIZE + j] = rand() % 100;
            B[i * SIZE + j] = rand() % 100;
            C[i * SIZE + j] = 0;
		}
	}

    clock_gettime (CLOCK_REALTIME, &mt1);

    for(i = 0; i < NTHREADS; i++) {
        rc = pthread_create(&threads[i], NULL, worker, (void *)i);
        assert(rc == 0);
    }

    for(i = 0; i < NTHREADS; i++) {
        rc = pthread_join(threads[i], NULL);
        assert(rc == 0);
    } 

    clock_gettime (CLOCK_REALTIME, &mt2);
	tt = pow(10, 9) * (mt2.tv_sec - mt1.tv_sec) + (mt2.tv_nsec - mt1.tv_nsec);
    printf("time posix threads = %f\n", (double)tt / pow(10, 9));

    return 0;
}
