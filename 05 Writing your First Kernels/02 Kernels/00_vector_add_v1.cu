#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/*
 * Vector size constant - 10 million elements
 * Choose a large enough size to demonstrate GPU parallelism benefits
 * But not too large to exceed GPU memory
 */
#define N 10000000

/*
 * CUDA thread block size
 * 256 is a common choice because:
 * - It's a multiple of 32 (warp size)
 * - Provides good occupancy on most GPUs
 * - Balances resource usage and parallelism
 */
#define BLOCK_SIZE 256

/*
 * CPU implementation of vector addition
 * Performs sequential addition of two vectors
 * Parameters:
 *   a, b - input vectors
 *   c - output vector
 *   n - vector size
 * Time complexity: O(n)
 */
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // Sequential addition
    }
}

/*
 * CUDA kernel for parallel vector addition
 * Each thread processes one element of the vectors
 * Parameters:
 *   a, b - input vectors in device memory
 *   c - output vector in device memory
 *   n - vector size
 * 
 * Thread organization:
 * - Multiple thread blocks, each with BLOCK_SIZE threads
 * - Each thread handles one array element
 * - Global thread ID = blockIdx.x * blockDim.x + threadIdx.x
 */
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to prevent buffer overflow
    if (i < n) {
        c[i] = a[i] + b[i];  // Parallel addition
    }
}

/*
 * Initialize vector with random float values between 0 and 1
 * Parameters:
 *   vec - vector to initialize
 *   n - vector size
 * Note: Uses rand(), which is not thread-safe
 */
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

/*
 * High-precision timer function
 * Returns:
 *   Current time in seconds with nanosecond precision
 * Uses CLOCK_MONOTONIC to avoid issues with system time changes
 */
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // Host (CPU) and Device (GPU) pointers
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;  // h_ prefix for host memory
    float *d_a, *d_b, *d_c;                 // d_ prefix for device memory
    size_t size = N * sizeof(float);

    // Allocate host (CPU) memory
    // Using malloc() for page-able memory
    // Consider using cudaHostAlloc() for pinned memory in production
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    // Initialize random number generator and vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate device (GPU) memory
    // cudaMalloc() allocates linear memory on the device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy input data from host to device
    // cudaMemcpy() is synchronous (blocking)
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    /*
     * Calculate grid dimensions
     * Formula ensures enough blocks to cover N elements:
     * - If N = 1000 and BLOCK_SIZE = 256:
     * - num_blocks = (1000 + 256 - 1) / 256 = 4
     * - This creates 4 blocks × 256 threads = 1024 threads total
     */
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Perform warm-up runs to stabilize GPU clock speeds
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();  // Wait for GPU to finish
    }

    // Benchmark CPU implementation (average of 20 runs)
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation (average of 20 runs)
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();  // Ensure GPU finished before stopping timer
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Display benchmark results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Verify results by comparing CPU and GPU outputs
    // Copy GPU results back to host for comparison
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        // Allow small floating-point differences (epsilon = 1e-5)
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Clean up: Free all allocated memory
    // Host memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    // Device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
