#ifndef utils_h
#define utils_h

#include <curand_kernel.h>

typedef struct {
  float Ti; /*Initial time*/
  float x; /*Asset price at T=T_i*/
  float j; /*Initial value of I*/
  float F; /*Option price*/
} Option_price;


float sum_array(float* array, int length);

void testCUDA(cudaError_t error, const char *file, int line);

__global__ void init_curand_state_k(curandState* state);

__global__ void init_curand_state_k_2D(curandState* state);

void write_data(char filename[], Option_price* data, int length);

#endif