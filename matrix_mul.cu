// include any headers
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

// define constants
#define BLOCK_SIZE 32

__global__ void matrix_mul(float* M, float* N, float* P, int Width) { //need kernel  function
    // Get the row and column index for the output matrix element that this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Width && col < Width) {
        float sum = 0.0f;

        for (int k = 0; k < Width; ++k) {
            sum += M[row * Width + k] * N[k * Width + col];
        }

        // Store the result in the output matrix P
        P[row * Width + col] = sum;
    }
   __syncthreads();
}


double get_clock() {
    struct timeval tv; int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { 
        printf("gettimeofday error"); 
        }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main(void) {
  // allocate input and output arrays
  int SIZE =10000;

  float *M, *N, *P;
  cudaMallocManaged(&M, SIZE*SIZE * sizeof(float));
  cudaMallocManaged(&N, SIZE*SIZE * sizeof(float));
  cudaMallocManaged(&P, SIZE*SIZE * sizeof(float));

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      M[i * SIZE + j] = 1.0f; // x[i][j]
      N[i * SIZE + j] = 1.0f;
    }
  }

    double start = get_clock();
//checks og
  printf("\n");

      // run the kernel
  //Initializing block size and running kernerl
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE); // Threads per block (BLOCK_SIZE x BLOCK_SIZE)
  dim3 blocksPerGrid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE); // Grid size



  matrix_mul<<<blocksPerGrid, threadsPerBlock>>>(M,N,P, SIZE);
  cudaDeviceSynchronize();

  double end = get_clock();
  printf("time per call: %f ns\n", (end-start) );
  // check results


  for (int i = 0; i < 10; i++) { // Just print the first 10x10 block
  for (int j = 0; j < 10; j++) {
    printf("%f ", P[i * SIZE + j]);
  }
  printf("\n");
}

  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  cudaFree(M);
  cudaFree(N);
  cudaFree(P);

  return 0;


}