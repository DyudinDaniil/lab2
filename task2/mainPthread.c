#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define N 4 // Размер матриц (N x N)
#define P 2 // Количество потоков (должно быть делителем N)

typedef struct {
    int id;
    int size;
    int **A;
    int **B;
    int **C;
} ThreadData;

void *cannon(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int id = data->id;
    int size = data->size;
    int **A = data->A;
    int **B = data->B;
    int **C = data->C;

    // Разделение матриц на блоки
    int blockSize = size / P;
    for (int step = 0; step < P; step++) {
        // Смещение для текущего потока
        int rowOffset = (id + step) % P;
        int colOffset = (step + id) % P;

        // Умножение блоков
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                for (int k = 0; k < blockSize; k++) {
                    C[i + rowOffset * blockSize][j + colOffset * blockSize] +=
                        A[i + rowOffset * blockSize][k + colOffset * blockSize] *
                        B[k + colOffset * blockSize][j + colOffset * blockSize];
                }
            }
        }

        // Циклический сдвиг матрицы B
        int **temp = (int **)malloc(size * sizeof(int *));
        for (int i = 0; i < size; i++) {
            temp[i] = (int *)malloc(size * sizeof(int));
            for (int j = 0; j < size; j++) {
                temp[i][j] = B[i][(j + blockSize) % size];
            }
        }

        // Копирование обратно в B
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                B[i][j] = temp[i][j];
            }
        }
        free(temp);
    }

    return NULL;
}

void initialize_matrices(int **A, int **B, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i][j] = rand() % 10; // Заполнение случайными числами
            B[i][j] = rand() % 10; // Заполнение случайными числами
        }
    }
}

void print_matrix(int **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    srand(time(NULL));
    
    int **A = (int **)malloc(N * sizeof(int *));
    int **B = (int **)malloc(N * sizeof(int *));
    int **C = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        A[i] = (int *)malloc(N * sizeof(int));
        B[i] = (int *)malloc(N * sizeof(int));
        C[i] = (int *)malloc(N * sizeof(int));
    }

    initialize_matrices(A, B, N);

    pthread_t threads[P];
    ThreadData threadData[P];

    clock_t start = clock();

    for (int i = 0; i < P; i++) {
        threadData[i].id = i;
        threadData[i].size = N;
        threadData[i].A = A;
        threadData[i].B = B;
        threadData[i].C = C;
        pthread_create(&threads[i], NULL, cannon, (void *)&threadData[i]);
    }

    for (int i = 0; i < P; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Matrix A:\n");
    print_matrix(A, N);
    printf("Matrix B:\n");
    print_matrix(B, N);
    printf("Matrix C (Result):\n");
    print_matrix(C, N);
    printf("Time taken: %f seconds\n", time_spent);

    // Освобождение памяти
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    

    return 0;
}