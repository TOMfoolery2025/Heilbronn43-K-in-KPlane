/**
 * Phase 1: Hello World CUDA Kernel
 * Purpose: Verify Python -> C++ -> CUDA pipeline works
 * Test: test_phase1_pipeline.py
 */

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <string>

// CUDA kernel: runs on GPU
__global__ void add_vectors_kernel(const int* a, const int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function: called from C++/Python
std::vector<int> add_vectors_gpu(const std::vector<int>& a, const std::vector<int>& b) {
    int n = a.size();
    if (n != b.size()) {
        throw std::runtime_error("Vector sizes must match");
    }
    
    // Handle empty vectors
    if (n == 0) {
        return std::vector<int>();
    }

    // Allocate device memory
    int* d_a;
    int* d_b;
    int* d_c;
    size_t bytes = n * sizeof(int);
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy input data to device
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    add_vectors_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // Copy result back to host
    std::vector<int> c(n);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
