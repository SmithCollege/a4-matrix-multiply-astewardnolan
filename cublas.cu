#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "cublas_v2.h" 

// CUDA runtime
#include <cuda_runtime.h>
#include <sys/time.h>

//resources :  that cublas github you linked micheal

#define SIZE 32

double get_clock() {
    struct timeval tv; int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { 
        printf("gettimeofday error"); 
        }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main(void) {

  printf("[Matrix Multiply CUBLAS] - Starting...\n");


  cublasHandle_t handle;

  (cublasCreate(&handle));


  const float alpha = 1.0f;
  const float beta = 0.0f;


  int size =1024;
  

  float *M, *N, *P;


  cudaMallocManaged(&M, sizeof(float)*size *size);
  cudaMallocManaged(&N, sizeof(float)*size *size);
  cudaMallocManaged(&P, sizeof(float)*size *size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      M[i * size + j] = 1; // x[i][j]
      N[i * size + j] = 1;
    }
  }

//checks og
  printf("\n");

  double start = get_clock();


    // Perform warmup operation with cublas
  cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size,
      size, &alpha, M, size, N, size, &beta, P, size);


  printf("%s\n", cudaGetErrorString(cudaGetLastError()));


  cudaDeviceSynchronize();


  double end = get_clock();
  printf("time per call: %f ns\n", (end-start) );
  // check results


  for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        printf("%f ", P[i * size + j]);
        if (P[i * size + j] != size) {
          printf("Error at P[%d][%d]: %f\n", i, j, P[i * size + j]);
        }
      }
      printf("\n");
    }


  cudaFree(M);
  cudaFree(N);
  cudaFree(P);
  cublasDestroy(handle);

  return 0;

}
