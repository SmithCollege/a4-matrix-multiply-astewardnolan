#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

#define BLOCK_SIZE 32  

__global__ void tile_matrix_multiply(float* M, float* N, float* P, int width) {
    // Shared memory for tiles of M and N
    __shared__ float shareM[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shareN[BLOCK_SIZE][BLOCK_SIZE];
    

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float temp = 0.0f;

    for (int i = 0; i < (width + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        //put in share mem
        if (i * BLOCK_SIZE + tx < width && row < width) {
            shareM[ty][tx] = M[row * width + i * BLOCK_SIZE + tx]; 
        } else {
            shareM[ty][tx] = 0.0f;  // Padding with 0s if out of bounds
        }

        if (i * BLOCK_SIZE + ty < width && col < width) {
            shareN[ty][tx] = N[(i * BLOCK_SIZE + ty) * width + col]; 
        } else {
            shareN[ty][tx] = 0.0f;  // Padding with 0s if out of bounds
        }

        __syncthreads();  

        // Compute the product for this tile
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            temp += shareM[ty][k] * shareN[k][tx]; 
        }

        __syncthreads();  // Synchronize threads before the next iteration
    }

   
    if (row < width && col < width) {
        P[row * width + col] = temp;
    }
}

double get_clock() {
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok < 0) {
        printf("gettimeofday error");
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main(void) {
    int size = 10000;  

    float *M, *N, *P;
    cudaMallocManaged(&M, size * size * sizeof(float));
    cudaMallocManaged(&N, size * size * sizeof(float));
    cudaMallocManaged(&P, size * size * sizeof(float));

    // Initialize with 1s
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            M[i * size + j] = 1.0f;
            N[i * size + j] = 1.0f;
        }
    }

    double start = get_clock();

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    tile_matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(M, N, P, size);
    cudaDeviceSynchronize();

    double end = get_clock();
    printf("Time per call: %f ms\n", (end - start) * 1000);

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    

    printf("Result matrix P:\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%f ", P[i * size + j]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(M);
    cudaFree(N);
    cudaFree(P);

    return 0;
}
