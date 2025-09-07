#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>

__global__ void generateUniqueRandomIntsAtomic(int* output, int N, int k, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= k) return;
    
    curandState state;
    curand_init(seed + tid, tid, 0, &state);
    
    __shared__ int generated_count;
    __shared__ bool slot_taken[1024];  // Assuming k <= 1024
    
    if (threadIdx.x == 0) {
        generated_count = 0;
        for (int i = 0; i < k; i++) {
            slot_taken[i] = false;
        }
    }
    __syncthreads();
    
    // Each thread tries to claim a unique number
    while (generated_count < k) {
        int candidate = curand(&state) % N;
        bool is_unique = true;
        
        // Check if already generated
        for (int i = 0; i < k; i++) {
            if (slot_taken[i] && output[i] == candidate) {
                is_unique = false;
                break;
            }
        }
        
        if (is_unique) {
            // Try to claim a slot
            for (int i = 0; i < k; i++) {
                if (!slot_taken[i]) {
                    // Atomic compare and swap
                    if (atomicCAS((int*)&slot_taken[i], 0, 1) == 0) {
                        output[i] = candidate;
                        atomicAdd(&generated_count, 1);
                        break;
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Host function to verify uniqueness (for testing)
bool verifyUniqueness(int* arr, int k) {
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < k; j++) {
            if (arr[i] == arr[j]) {
                printf("Duplicate found: arr[%d] = arr[%d] = %d\n", i, j, arr[i]);
                return false;
            }
        }
    }
    return true;
}

// Host function to verify range validity
bool verifyRange(int* arr, int k, int N) {
    for (int i = 0; i < k; i++) {
        if (arr[i] < 0 || arr[i] >= N) {
            printf("Out of range value found: arr[%d] = %d\n", i, arr[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // Default parameters
    int N = 1000000;  // Range [0, N)
    int k = 100;      // Number of unique integers to select
    
    // Parse command line arguments if provided
    if (argc >= 3) {
        N = atoi(argv[1]);
        k = atoi(argv[2]);
    }
    
    // Validate parameters
    if (k > N) {
        fprintf(stderr, "Error: k (%d) cannot be greater than N (%d)\n", k, N);
        return 1;
    }
    if (k > 1024) {
        fprintf(stderr, "Error: k (%d) cannot be greater than 1024 (shared memory limit)\n", k);
        return 1;
    }
    
    printf("========================================\n");
    printf("Generating %d unique random integers from range [0, %d)\n", k, N);
    printf("========================================\n");
    
    // Allocate device memory
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, k * sizeof(int)));
    
    // Calculate grid and block dimensions
    // Use only one block to ensure shared memory works correctly
    int blockSize = std::min(k, 1024);
    int gridSize = 1;
    
    printf("Grid size: %d, Block size: %d\n", gridSize, blockSize);
    
    // Generate seed based on current time
    unsigned long long seed = time(NULL);
    printf("Random seed: %llu\n", seed);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start time
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch kernel
    generateUniqueRandomIntsAtomic<<<gridSize, blockSize>>>(d_output, N, k, seed);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Record stop time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Allocate host memory
    int* h_output = new int[k];
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, k * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Verification
    printf("\n--- Verification ---\n");
    bool unique = verifyUniqueness(h_output, k);
    bool rangeValid = verifyRange(h_output, k, N);
    
    printf("Uniqueness check: %s\n", unique ? "PASSED ✓" : "FAILED ✗");
    printf("Range check: %s\n", rangeValid ? "PASSED ✓" : "FAILED ✗");
    
    // Performance statistics
    printf("\n--- Performance ---\n");
    printf("Time taken: %.3f ms\n", milliseconds);
    printf("Throughput: %.2f numbers/ms\n", k / milliseconds);
    
    // Display results
    printf("\n--- Results ---\n");
    int display_count = std::min(20, k);
    printf("First %d numbers:\n", display_count);
    for (int i = 0; i < display_count; i++) {
        printf("%6d", h_output[i]);
        if ((i + 1) % 10 == 0) printf("\n");
    }
    if (display_count % 10 != 0) printf("\n");
    
    // Statistical information
    if (k > 0) {
        int min_val = h_output[0];
        int max_val = h_output[0];
        long long sum = 0;
        
        for (int i = 0; i < k; i++) {
            min_val = std::min(min_val, h_output[i]);
            max_val = std::max(max_val, h_output[i]);
            sum += h_output[i];
        }
        
        printf("\n--- Statistics ---\n");
        printf("Min value: %d\n", min_val);
        printf("Max value: %d\n", max_val);
        printf("Average: %.2f\n", (double)sum / k);
        printf("Expected average: %.2f\n", (N - 1) / 2.0);
    }
    
    // Cleanup
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("\n========================================\n");
    printf("Program completed successfully!\n");
    
    return 0;
}