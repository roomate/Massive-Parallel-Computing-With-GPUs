#ifndef utils_h
#define utils_h

#include <curand_kernel.h>

typedef struct {
  float Ti; /*Initial time*/
  float x; /*Asset price at T=T_i*/
  float j; /*Initial value of I*/
  float F; /*Option price*/
} Option_price;


float mean(float* array, int length);

float std_(float* array, int length);

// Function that catches errors
void testCUDA(cudaError_t error, const char *file, int line);

// Initiate the PRNG state for each thread
__global__ void init_curand_state_k(curandState* state, int axis);

// Initiate the PRNG state for each thread
__global__ void init_curand_state_k_2D(curandState* state);

// Initiate the PRNG state for each thread
__global__ void init_curand_state_k_init(curandState* state);

/*Save all data computed in a file. Columns are in this order: Ti, S_Ti, j, F*/
void write_data(char filename[], Option_price* data, int length);

/*Read data from filename and print it in standard output.*/
void read_data(char filename[]);

#endif