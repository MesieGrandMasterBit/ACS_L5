#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <math.h>

#define SIZE 1000

double *A, *B, *C;

void dgemm_mpi(double *a, double *b, double *c)
{
    int commsize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rows_per_proc = SIZE / commsize;
    int lb = rank * rows_per_proc;
    int ub = (rank == commsize - 1) ? (SIZE - 1) : (lb + rows_per_proc - 1);

    int i, j, k;

    for (i = lb; i <= ub; i++) {
        for (j = 0; j < SIZE; j++) {
            for (k = 0; k < SIZE; k++) {
                *(c + i * SIZE + j) += *(a + i * SIZE + k) * *(b + k * SIZE + j);
            }
        }
    }

    int *displs = malloc(sizeof(*displs) * commsize);
    int *rcounts = malloc(sizeof(*rcounts) * commsize);

    for (int i = 0; i < commsize; i++) {
        rcounts[i] = (i == commsize - 1) ? SIZE - i * rows_per_proc : rows_per_proc;
        displs[i] = (i > 0) ? displs[i - 1] + rcounts[i - 1]: 0;
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, c, rcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
}

void dgemm_default(double *a, double *b, double *c)
{
    int i, j, k;

    for (i = 0; i < SIZE; i++) {
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

int main(int argc, char* argv[])
{
    int commsize, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    A = malloc(sizeof(*A) * SIZE * SIZE);
    B = malloc(sizeof(*B) * SIZE * SIZE);
    C = malloc(sizeof(*C) * SIZE * SIZE);

    srand(time(0));

    for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
            A[i * SIZE + j] = rand() % 100;
            B[i * SIZE + j] = rand() % 100;
            C[i * SIZE + j] = 0;
		}
	}

    double t = MPI_Wtime();

    dgemm_mpi(A, B, C);

    t = MPI_Wtime() - t;
	//print_matrix(C);

    if (rank == 0) {
        printf("Elapsed time (%d procs): %.6f sec.\n", commsize, t);
    }
    MPI_Finalize();

    return 0;
}
