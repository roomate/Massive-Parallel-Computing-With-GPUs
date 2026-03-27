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

void write_data(char[] filename, )
{

}