#include <stdio.h>
#include "utils.cuh"
#include <curand_kernel.h>
#include <assert.h>

float mean(float* array, int length)
{
    float sum=0;
    for (int i=0; i<length; ++i) {sum+=array[i];}
    return sum/length;
}


void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error %i in file %s at line %d\n", error, file, line);
       exit(EXIT_FAILURE);
	}
}

__global__ void init_curand_state_k(curandState* state, int axis)
{
  assert (axis>=0 && axis<3);
  int idx;
  if (axis==0) {idx = blockDim.x * blockIdx.x + threadIdx.x;}

  else if (axis==1) {idx = blockDim.y * blockIdx.y + threadIdx.x;}
  
  else if (axis==2){idx = blockDim.z * blockIdx.z + threadIdx.x;}

  /*seed is 0, offset is 0.*/
	curand_init(0, idx, 0, &state[idx]);
}

__global__ void init_curand_state_k_2D(curandState* state)
{
	int idx = blockIdx.x * blockDim.x * gridDim.y +
		blockDim.x * blockIdx.y + threadIdx.x;
  /*seed is 0, offset is 0.*/
	curand_init(0, idx, 0, &state[idx]);
}

__global__ void init_curand_state_k_init(curandState* state)
{
	int idx = blockIdx.x * blockDim.x * gridDim.y +
		blockDim.x * blockIdx.y;
  /*seed is 0, offset is 0.*/
	curand_init(0, idx, 0, &state[idx]);
}

void write_data(char filename[], Option_price* data, int length)
{
  FILE* file=fopen(filename, "w");
  if (file==nullptr) {printf("Error. Unable to open a file for writing.\n"); exit(EXIT_FAILURE);}
  fprintf(file, "Ti S_Ti j F(Ti,S_Ti,j) \n");
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

void read_data(char filename[])
{
  FILE* file=fopen(filename, "r");
  if (file==nullptr) {printf("Error. Unable to open a file for reading.\n"); exit(EXIT_FAILURE);}
  char a[20], b[20], c[20], d[20];
  fscanf(file, "%s %s %s %s\n", a, b, c, d);
  float Ti, S_Ti, F;
  int j;
  while (fscanf(file, "%f %f %i %f", &Ti, &S_Ti, &j, &F) != EOF)
  {
    printf("Ti=%f, S_Ti=%f, j=%i, F(Ti, S_Ti, j)=%f\n", Ti, S_Ti, j, F);
  }
  fclose(file);
}