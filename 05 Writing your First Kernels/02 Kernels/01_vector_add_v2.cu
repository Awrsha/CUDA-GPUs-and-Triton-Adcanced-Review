/*
 * CUDA Vector Addition Benchmark
 * This program compares the performance of vector addition across three implementations:
 * 1. CPU (single-threaded)
 * 2. GPU 1D (using 1D thread blocks)
 * 3. GPU 3D (using 3D thread blocks)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

// Configuration constants
#define N 10000000  // Total vector size (10 million elements)
#define BLOCK_SIZE_1D 1024  // Number of threads per block for 1D implementation
// 3D block dimensions - total threads per block = 16 * 8 * 8 = 1024
#define BLOCK_SIZE_3D_X 16  // Threads per block in X dimension
#define BLOCK_SIZE_3D_Y 8   // Threads per block in Y dimension
#define BLOCK_SIZE_3D_Z 8   // Threads per block in Z dimension

/**
 * CPU implementation of vector addition
 * @param a First input vector
 * @param b Second input vector
 * @param c Output vector (a + b)
 * @param n Vector size
 */
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // Sequential addition
    }
}

/**
 * GPU kernel for 1D vector addition
 * Each thread processes one element of the vector
 * Memory access pattern: coalesced global memory access
 */
__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (i < n) {
        c[i] = a[i] + b[i];  // Parallel addition
    }
}

/**
 * GPU kernel for 3D vector addition
 * Demonstrates how to handle 3D data structures in CUDA
 * Memory access pattern: strided access due to 3D layout
 */
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz) {
    // Calculate 3D thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // X dimension
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Y dimension
    int k = blockIdx.z * blockDim.z + threadIdx.z;  // Z dimension
    
    // Boundary check for each dimension
    if (i < nx && j < ny && k < nz) {
        // Convert 3D index to linear memory index
        int idx = i + j * nx + k * nx * ny;
        // Additional bounds check for safety
        if (idx < nx * ny * nz) {
            c[idx] = a[idx] + b[idx];
        }
    }
}

/**
 * Initialize vector with random float values between 0 and 1
 * @param vec Vector to initialize
 * @param n Vector size
 */
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

/**
 * High-precision timer function
 * @return Current time in seconds with nanosecond precision
 */
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // Host pointers (h_) and device pointers (d_)
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;  // Host memory
    float *d_a, *d_b, *d_c_1d, *d_c_3d;                     // Device memory
    size_t size = N * sizeof(float);  // Total memory size in bytes

    // Allocate host memory with error checking
    if ((h_a = (float*)malloc(size)) == NULL ||
        (h_b = (float*)malloc(size)) == NULL ||
        (h_c_cpu = (float*)malloc(size)) == NULL ||
        (h_c_gpu_1d = (float*)malloc(size)) == NULL ||
        (h_c_gpu_3d = (float*)malloc(size)) == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(1);
    }

    // Initialize input vectors with random data
    srand(time(NULL));  // Seed random number generator
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_1d, size);
    cudaMalloc(&d_c_3d, size);

    // Transfer input data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Calculate grid dimensions for 1D kernel
    int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    // Calculate grid dimensions for 3D kernel
    // Decompose 1D problem into 3D structure
    int nx = 100, ny = 100, nz = 1000;  // nx * ny * nz = N
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z
    );

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 5; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 5.0;

    // Benchmark GPU 1D implementation
    printf("Benchmarking GPU 1D implementation...\n");
    double gpu_1d_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_c_1d, 0, size);  // Clear previous results
        double start_time = get_time();
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_1d_total_time += end_time - start_time;
    }
    double gpu_1d_avg_time = gpu_1d_total_time / 100.0;

    // Verify 1D results immediately
    cudaMemcpy(h_c_gpu_1d, d_c_1d, size, cudaMemcpyDeviceToHost);
    bool correct_1d = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu_1d[i]) > 1e-4) {
            correct_1d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_1d[i] << std::endl;
            break;
        }
    }
    printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");

    // Benchmark GPU 3D implementation
    printf("Benchmarking GPU 3D implementation...\n");
    double gpu_3d_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_c_3d, 0, size);  // Clear previous results
        double start_time = get_time();
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_3d_total_time += end_time - start_time;
    }
    double gpu_3d_avg_time = gpu_3d_total_time / 100.0;

    // Verify 3D results immediately
    cudaMemcpy(h_c_gpu_3d, d_c_3d, size, cudaMemcpyDeviceToHost);
    bool correct_3d = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-4) {
            correct_3d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_3d[i] << std::endl;
            break;
        }
    }
    printf("3D Results are %s\n", correct_3d ? "correct" : "incorrect");

    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU 1D average time: %f milliseconds\n", gpu_1d_avg_time * 1000);
    printf("GPU 3D average time: %f milliseconds\n", gpu_3d_avg_time * 1000);
    printf("Speedup (CPU vs GPU 1D): %fx\n", cpu_avg_time / gpu_1d_avg_time);
    printf("Speedup (CPU vs GPU 3D): %fx\n", cpu_avg_time / gpu_3d_avg_time);
    printf("Speedup (GPU 1D vs GPU 3D): %fx\n", gpu_1d_avg_time / gpu_3d_avg_time);

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1d);
    cudaFree(d_c_3d);

    return 0;
}
