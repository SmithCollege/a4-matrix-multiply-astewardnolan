#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

void MatrixMulOnHost(float* M, float* N, float* P, int Width) {
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
}
double get_clock() {
    struct timeval tv; int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { 
        printf("gettimeofday error"); 
        }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main() {
  int size = 1024;

  float* x = malloc(sizeof(float) * size * size);
  float* y = malloc(sizeof(float) * size * size);
  float* z = malloc(sizeof(float) * size * size);
  
  double start = get_clock();

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      x[i * size + j] = 1; // x[i][j]
      y[i * size + j] = 1;
    }
  }

  MatrixMulOnHost(x, y, z, size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (z[i * size + j] != size) {
        printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
      }
    }
  }

  double end = get_clock();
  printf("time per call: %f ns\n", (end-start) );


  for (int i = 0; i < 10; i++) { // Just print the first 10x10 block
  for (int j = 0; j < 10; j++) {
    printf("%f ", z[i * size + j]);
  }
  printf("\n");
}

  return 0;
}