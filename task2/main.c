#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void fill_matrix(int size, double *mat, int randomize) {
    for (int idx = 0; idx < size * size; idx++) {
        mat[idx] = randomize ? (rand() % 10 + 1) : 0; 
    }
}

int main(int argc, char **argv) {
    int proc_rank, proc_count;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    if (argc != 2) {
        if (proc_rank == 0) {
            fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int matrix_size = atoi(argv[1]); 
    int sqrt_proc_count = sqrt(proc_count);

    if (sqrt_proc_count * sqrt_proc_count != proc_count || matrix_size % sqrt_proc_count != 0) {
        if (proc_rank == 0) {
            fprintf(stderr, "Error: Number of processes must be a perfect square, and matrix size must be divisible by sqrt(processes).\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int block_dim = matrix_size / sqrt_proc_count;
    double *local_matrix_A = (double *)malloc(block_dim * block_dim * sizeof(double));
    double *local_matrix_B = (double *)malloc(block_dim * block_dim * sizeof(double));
    double *local_matrix_C = (double *)malloc(block_dim * block_dim * sizeof(double));
    
    if (!local_matrix_A || !local_matrix_B || !local_matrix_C) {
        fprintf(stderr, "Error: Not enough memory for local matrices.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int idx = 0; idx < block_dim * block_dim; idx++) {
        local_matrix_C[idx] = 0.0;
    }

    double *global_matrix_A = NULL;
    double *global_matrix_B = NULL;
    if (proc_rank == 0) {
        global_matrix_A = (double *)malloc(matrix_size * matrix_size * sizeof(double));
        global_matrix_B = (double *)malloc(matrix_size * matrix_size * sizeof(double));
        if (!global_matrix_A || !global_matrix_B) {
            fprintf(stderr, "Error: Not enough memory for global matrices.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fill_matrix(matrix_size, global_matrix_A, 1);
        fill_matrix(matrix_size, global_matrix_B, 1);
    }

    MPI_Datatype block_type;
    MPI_Type_vector(block_dim, block_dim, matrix_size, MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, block_dim * sizeof(double), &block_type);
    MPI_Type_commit(&block_type);

    int *displacements = NULL;
    int *send_counts = NULL;
    if (proc_rank == 0) {
        displacements = (int *)malloc(proc_count * sizeof(int));
        send_counts = (int *)malloc(proc_count * sizeof(int));
        for (int i = 0; i < sqrt_proc_count; i++) {
            for (int j = 0; j < sqrt_proc_count; j++) {
                displacements[i * sqrt_proc_count + j] = i * matrix_size * block_dim + j * block_dim;
                send_counts[i * sqrt_proc_count + j] = 1;
            }
        }
    }

    MPI_Scatterv(global_matrix_A, send_counts, displacements, block_type, local_matrix_A, block_dim * block_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_matrix_B, send_counts, displacements, block_type, local_matrix_B, block_dim * block_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (proc_rank == 0) {
        free(global_matrix_A);
        free(global_matrix_B);
        free(displacements);
        free(send_counts);
    }

    double start_time = MPI_Wtime();
    
    int coordinates[2], dimensions[2] = {sqrt_proc_count, sqrt_proc_count}, periodic[2] = {1, 1};
    MPI_Comm cartesian_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periodic, 1, &cartesian_comm);
    MPI_Cart_coords(cartesian_comm, proc_rank, 2, coordinates);

    int left_proc, right_proc, up_proc, down_proc;
    MPI_Cart_shift(cartesian_comm, 1, -1, &right_proc, &left_proc);
    MPI_Cart_shift(cartesian_comm, 0, -1, &down_proc, &up_proc);

    MPI_Sendrecv_replace(local_matrix_A, block_dim * block_dim, MPI_DOUBLE, left_proc, 0, right_proc, 0, cartesian_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(local_matrix_B, block_dim * block_dim, MPI_DOUBLE, up_proc, 0, down_proc, 0, cartesian_comm, MPI_STATUS_IGNORE);

    double end_time = MPI_Wtime();
    if (proc_rank == 0) {
        printf("Execution time: %f seconds\n", end_time - start_time);
    }

    free(local_matrix_A);
    free(local_matrix_B);
    free(local_matrix_C);

    MPI_Comm_free(&proc_rank);
    MPI_Type_free(&block_type);

    MPI_Finalize();
    return 0;
}