#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Matrix dimensions for the computation
// M x K matrix (A) multiplied by K x N matrix (B) produces M x N matrix (C)
#define M 256  // Number of rows in matrix A and result matrix C
#define K 512  // Number of columns in A and rows in B (must match for valid multiplication)
#define N 256  // Number of columns in matrix B and result matrix C
#define BLOCK_SIZE 32  // Size of thread blocks (32x32 threads per block is common for good occupancy)

/* Matrix multiplication example to illustrate the computation:
   3x2 matrix A multiplied by 2x4 matrix B produces 3x4 matrix C
   
   A = [[1, 2],      B = [[7,  8,  9, 10],     C = A * B = [[29,  32,  35,  38],
        [3, 4],           [11, 12, 13, 14]]              [65,  72,  79,  86], 
        [5, 6]]                                          [101, 112, 123, 134]]

   Each element C[i,j] is computed as the dot product of row i from A and column j from B
*/

/**
 * CPU implementation of matrix multiplication
 * @param A Input matrix A
 * @param B Input matrix B 
 * @param C Output matrix C = A * B
 * @param m Number of rows in A
 * @param k Number of columns in A / rows in B
 * @param n Number of columns in B
 */
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            // Compute dot product of row i from A and column j from B
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/**
 * CUDA kernel for parallel matrix multiplication
 * Each thread computes one element of the output matrix C
 * @param A Input matrix A in device memory
 * @param B Input matrix B in device memory
 * @param C Output matrix C in device memory
 * @param m Number of rows in A
 * @param k Number of columns in A / rows in B
 * @param n Number of columns in B
 */
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    // Calculate global row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only compute if within matrix bounds
    if (row < m && col < n) {
        float sum = 0.0f;
        // Compute dot product of row from A and column from B
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}

/**
 * Initialize matrix with random float values between 0 and 1
 * @param mat Pointer to matrix to initialize
 * @param rows Number of rows
 * @param cols Number of columns
 */
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

/**
 * Get current time with nanosecond precision
 * @return Current time in seconds
 */
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // Host (CPU) and device (GPU) matrix pointers
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;  // Host matrices
    float *d_A, *d_B, *d_C;                 // Device matrices
    
    // Calculate required memory sizes
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Allocate host memory for matrices
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    // Initialize input matrices with random values
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Allocate device (GPU) memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Configure CUDA kernel execution parameters
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);  // 32x32 threads per block
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,   // Ceiling division to cover matrix
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Perform warm-up runs to stabilize GPU clock speeds
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation (average over 20 runs)
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation (average over 20 runs)
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print benchmark results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Free allocated memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
