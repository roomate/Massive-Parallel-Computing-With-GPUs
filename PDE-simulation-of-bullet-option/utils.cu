#include <stdio.h>
#include "utils.cuh"
#include <curand_kernel.h>

float sum_array(float* array, int length)
{
    float sum=0;
    for (int i=0; i<length; ++i) {sum+=array[i];}
    return sum/length;
}

// Function that catches errors 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Initiate the PRNG state for each thread
__global__ void init_curand_state_k(curandState* state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
  /*seed is 0, offset is 0.*/
	curand_init(0, idx, 0, &state[idx]);
}

// Initiate the PRNG state for each thread
__global__ void init_curand_state_k_2D(curandState* state)
{
	int idx = blockIdx.x * blockDim.x * gridDim.y +
		blockDim.x * blockIdx.y + threadIdx.x;
  /*seed is 0, offset is 0.*/
	curand_init(0, idx, 0, &state[idx]);
}


/*Save all data computed in a file. Columns are in this order: Ti, S_Ti, j, F*/
void write_data(char filename[], Option_price* data, int length)
{
  FILE* file=fopen(filename, "w");
  if (file==nullptr) {printf("Error. Unable to open a file for writing.\n"); exit(EXIT_FAILURE);}
  fprintf(file, "Ti S_Ti j F(Ti, S_Ti, j)\n");
  for (int i=0; i<length; ++i)
  {
    float Ti=data[i].Ti;
    float S_Ti=data[i].x;
    int j=data[i].j;
    float F=data[i].F;

    fprintf(file, "%f %f %i %f\n", Ti, S_Ti, j, F);
  }
  fclose(file);
}