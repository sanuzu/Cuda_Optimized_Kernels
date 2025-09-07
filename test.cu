#include <iostream>
#include <cuda_runtime.h>
__global__
void vectorAdd()
{
    
    printf("Hello, World from GPU!\n");
}


int main(){
    vectorAdd<<<1,1>>>();
    return 0;
}