#include <iostream>
#include <cuda_runtime.h>

__global__
void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}


int main(){
    float *h_A,*h_B,*h_C,*d_A,*d_B,*d_C;
    int n=2;
    float h_a[2]={1.0,2.0};
    size_t size = n * sizeof(float);
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    for (int i=0 ; i<n;i++){
        h_A[i]=2.0;
        h_B[i]=1.0;
    }
    
    cudaMalloc((void**) &d_A,size);
    cudaMalloc((void**) &d_B,size);
    cudaMalloc((void**) &d_C,size);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
    vectorAdd <<< (n + 255) / 256, 256 >>> (d_A, d_B, d_C, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    for (int i=0 ; i<n;i++){
        std::cout<<h_C[i]<<std::endl;
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}