#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void multiply_by_rows(double *local_matrix, double *vector, double *local_result, int rows_per_proc, int n) {
    for (int i = 0; i < rows_per_proc; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < n; j++) {
            local_result[i] += local_matrix[i * n + j] * vector[j];
        }
    }
}

void multiply_by_columns(double *local_matrix, double *local_vector, double *local_result, int n, int cols_per_proc) {
    for (int i = 0; i < n; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < cols_per_proc; j++) {
            local_result[i] += local_matrix[i * cols_per_proc + j] * local_vector[j];
        }
    }
}

void multiply_by_blocks(double *local_matrix, double *local_vector, double *local_result, int block_size) {
    for (int i = 0; i < block_size; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < block_size; j++) {
            local_result[i] += local_matrix[i * block_size + j] * local_vector[j];
        }
    }
}

void executeTask(int argc, char *argv[]) {
    int rank, size, n, algorithm, rows_per_proc, cols_per_proc, block_size;
    double *matrix = NULL, *vector = NULL, *result = NULL;
    double *local_matrix = NULL, *local_vector = NULL, *local_result = NULL;
    double start_time, end_time;
    int *sendcounts = NULL, *displs = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        scanf("%d", &n);
        printf("Algorithm: 1 (rows), 2 (columns), 3 (blocks): ");
        scanf("%d", &algorithm);

        matrix = malloc(n * n * sizeof(double));
        vector = malloc(n * sizeof(double));
        result = malloc(n * sizeof(double));

        srand(time(0));
        for (int i = 0; i < n; i++) {
            vector[i] = rand() % 10;
            for (int j = 0; j < n; j++) {
                matrix[i * n + j] = rand() % 10;
            }
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&algorithm, 1, MPI_INT, 0, MPI_COMM_WORLD);

    switch (algorithm) {
        case 1: 
            rows_per_proc = n / size;
            local_matrix = malloc(rows_per_proc * n * sizeof(double));
            local_vector = malloc(rows_per_proc * sizeof(double));
            local_result = malloc(rows_per_proc * sizeof(double));

            MPI_Scatter(matrix, rows_per_proc * n, MPI_DOUBLE, local_matrix, rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatter(vector, rows_per_proc, MPI_DOUBLE, local_vector, rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            start_time = MPI_Wtime();
            multiply_by_rows(local_matrix, local_vector, local_result, rows_per_proc, n);
            end_time = MPI_Wtime();

            MPI_Gather(local_result, rows_per_proc, MPI_DOUBLE, result, rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            break;

        case 2: 
            cols_per_proc = n / size;
            local_matrix = malloc(n * cols_per_proc * sizeof(double));
            local_vector = malloc(cols_per_proc * sizeof(double));
            local_result = malloc(n * sizeof(double));

            MPI_Scatter(matrix, n * cols_per_proc, MPI_DOUBLE, local_matrix, n * cols_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatter(vector, cols_per_proc, MPI_DOUBLE, local_vector, cols_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            start_time = MPI_Wtime();
            multiply_by_columns(local_matrix, local_vector, local_result, n, cols_per_proc);
            end_time = MPI_Wtime();

            MPI_Reduce(local_result, result, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            break;

        case 3: 
            block_size = n / size;
            local_matrix = malloc(block_size * block_size * sizeof(double));
            local_vector = malloc(block_size * sizeof(double));
            local_result = malloc(block_size * sizeof(double));

            if (rank == 0) {
                sendcounts = malloc(size * sizeof(int));
                displs = malloc(size * sizeof(int));
                for (int i = 0; i < size; i++) {
                    sendcounts[i] = block_size * block_size;
                    displs[i] = i * block_size * block_size;
                }
            }

            MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE, local_matrix, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatter(vector, block_size, MPI_DOUBLE, local_vector, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            start_time = MPI_Wtime();
            multiply_by_blocks(local_matrix, local_vector, local_result, block_size);
            end_time = MPI_Wtime();

            MPI_Gather(local_result, block_size, MPI_DOUBLE, result, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            free(sendcounts);
            free(displs);
            break;
    }

    if (rank == 0) {
        printf("Time: %f seconds\n", end_time - start_time);
        free(matrix);
        free(vector);
        free(result);
    }

    free(local_matrix);
    free(local_result);
    free(local_vector);

    MPI_Finalize();
}

int main(int argc, char *argv[]) {
    executeTask(argc, argv);
    return 0;
}