//
// Created by od641 on 17/11/2025.
//

#include <iostream>

__global__ void hello_world()
{
    printf("Block index: (%02d, %02d, %02d)\tThread index: (%02d, %02d, %02d)\n",
        blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
    hello_world<<<3, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
