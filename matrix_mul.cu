// include any headers
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

// define constants
#define BLOCK_SIZE 128

__global__ void matrix_mul(float* M, float* N, float* P, int Width) { //need kernel  function
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;

    if(gindex >= Width){
        return;
    }
    {
    for (int i = 0; i < Width; ++i)
    for (int j = 0; j < Width; ++j) {
    float sum = 0;
    for (int k = 0; k < Width; ++k) {
    float a = M[i * Width + k];
    float b = N[k * Width + j];
    sum += a * b;
 }
    P[i * Width + j] = sum;
 }
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
  int SIZE = 1<<20; // 1M elements
  SIZE =128;

  float *M, *N, *P;
  cudaMallocManaged(&M, SIZE*sizeof(float));
  cudaMallocManaged(&N, SIZE*sizeof(float));
  cudaMallocManaged(&P, SIZE*sizeof(float));

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      M[i * SIZE + j] = 1; // x[i][j]
      N[i * SIZE + j] = 1;
    }
  }

    double start = get_clock();
//checks og
  printf("\n");

      // run the kernel
  //Initializing block size and running kernerl
  int blocksPerGrid = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE; 
  int threadsPerBlock = BLOCK_SIZE;


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