#include<iostream>
#include<cuda_runtime.h>
#include<math.h>
__global__
void matmul(float* P,float* Q,float* R,int width , int height,int c){
    int row= blockIdx.x*blockDim.x+threadIdx.x;
    int col= blockIdx.y*blockDim.y+threadIdx.y;
    
    if(row<height && col<width){
        float val=0;
        for(int k=0;k<c;k++){
            val+=(P[row*width+k]*Q[k*width+col]);
        }
        
        R[row*width+col]=val;
        
    }
    return;
}

int main(){
    float hP[20][30],hQ[30][20],hR[20][20], *dP,*dQ,*dR;
    int m=(sizeof(hP)/sizeof(hP[0]));
    int n=sizeof(hP[0])/sizeof(float);
    int k=sizeof(hQ[0])/sizeof(float);
    int size=sizeof(float);
    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++) hP[i][j]=i;
    }
    for (int i=0;i<n;i++){
        for (int j=0;j<k;j++) hQ[i][j]=i;
    }
    cudaMalloc((void **) &dP,m*n*size);
    cudaMalloc((void **) &dQ,n*k*size);
    cudaMalloc((void **) &dR,m*k*size);
    dim3 grid(int(ceil(m/16.0f)),int((ceil(k/16.0f))),1);
    dim3 block(16,16,1);
    cudaMemcpy(dP,hP,m*n*size,cudaMemcpyHostToDevice);
    cudaMemcpy(dQ,hQ,n*k*size,cudaMemcpyHostToDevice);
    matmul <<<grid,block>>> (dP,dQ,dR,k,m,n);
    cudaMemcpy(hR,dR,m*k*size,cudaMemcpyDeviceToHost);
    std::cout<<hR[1][0];
    return 0;
}
