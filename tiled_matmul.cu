#include<iostream>
#include<cuda_runtime.h>
#define TILE_WIDTH 2
__global__
void matmul(float *A,float *B, float *C,int m , int n , int k){
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    __shared__ float P[TILE_WIDTH][TILE_WIDTH],Q[TILE_WIDTH][TILE_WIDTH];
    
    float Pval=0.0f;
    for (int i=0 ; i< std::ceil(n/(float)TILE_WIDTH);i++){
        if ((row<m) && (tx+i*TILE_WIDTH<n))
            P[ty][tx]=A[row*n+ tx+i*TILE_WIDTH];
        else
            P[ty][tx]=0.0f;
        if ((i*TILE_WIDTH+ty<n) && (col<k))
            Q[ty][tx]=B[(i*TILE_WIDTH+ty)*k+col];
        else
            Q[ty][tx]=0.0f;
        __syncthreads();
        if ((row<m) && (col<k)){
            for (int j=0;j<TILE_WIDTH;j++){
                Pval+=(P[ty][j]*Q[j][tx]);
            }
        }
        __syncthreads();
    }
    C[row*k+col]=Pval;

    
}

int main(){
    float hA[4][5] = {{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15},{16,17,18,19,20}};
    float hB[5][3] = {{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}};
    float hC[4][3]; 
    float hD[4][3] = {{135, 150, 165}, {310, 350, 390}, {485, 550, 615}, {660, 750, 840}}; 
    float *dA, *dB, *dC;
    

    int m = sizeof(hA)/sizeof(hA[0]);
    int k = sizeof(hA[0])/sizeof(hA[0][0]);
    int n = sizeof(hB[0])/sizeof(hB[0][0]);

    // std::cout<<m<<k<<n<<"\n";

    // for (int i=0; i<m; ++i){
    //     for (int j=0; j<k; ++j) hA[i][j] = 1;
    // }
    
    // for (int i=0; i<k; ++i){
    //     for (int j=0; j<n; ++j) hB[i][j] = 2;
    // }

    // for (int i=0; i<m; ++i){
    //     for (int j=0; j<n; ++j) hD[i][j] = k*hA[0][0]*hB[0][0];
    // }

    cudaMalloc((void**) &dA, sizeof(hA));
    cudaMalloc((void**) &dB, sizeof(hB));
    cudaMalloc((void**) &dC, sizeof(hC));

    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);

    dim3 grid (std::ceil(n/(float)TILE_WIDTH), std::ceil(m/(float)TILE_WIDTH));
    dim3 block (TILE_WIDTH, TILE_WIDTH);

    matmul<<<grid,block>>>(dA, dB, dC, m, k, n);

    cudaDeviceSynchronize();
    cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // std::cout<<hA[0][0]<<""<<hB[0][0]<<""<<k<<"="<<hC[2][3];

    for(int i=0; i<m; ++i){
        for(int j=0; j<n; ++j){
            printf("%d,%d = %d | %f | %f\n", i, j, hC[i][j]==hD[i][j], hC[i][j], hD[i][j]);
        }
    }

    return 0;
}