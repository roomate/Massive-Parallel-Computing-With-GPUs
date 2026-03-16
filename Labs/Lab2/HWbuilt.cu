/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void empty_k(void) {}

__global__ void print_k(void) {
	// printf("Hello World!\n");
  
  	// printf("threadIdx.x: %i, blockDim.x: %i, blockIdx.x: %i, gridDim.x: %i \n", 
    // threadIdx.x, blockDim.x, blockIdx.x, gridDim.x);

	// if (threadIdx.x == 0)
	// {
	// 	printf("blockIdx.x: %i \n", blockIdx.x);
	// }

	if (blockIdx.x == 0)
	{
		printf("threadIdx.x: %i, blockDim.x: %i, blockIdx.x: %i, gridDim.x: %i \n", 
		threadIdx.x, blockDim.x, blockIdx.x, gridDim.x);
	}
}

int main(void) {

	int device=0;
	int Y, y;

	cudaDeviceGetAttribute(&Y, cudaDevAttrComputeCapabilityMajor, device);
	cudaDeviceGetAttribute(&y, cudaDevAttrComputeCapabilityMinor, device);

	printf("Major and minor compute capabilities are respectively: %i.%i\n", Y , y);

	size_t FIFO_size;

	cudaDeviceGetLimit(&FIFO_size, cudaLimitPrintfFifoSize);

	printf("The printf FIFO buffer size is %lu.\n", FIFO_size);

	empty_k <<<1, 1>>> ();
	print_k <<<2, 4>>> ();
	cudaDeviceSynchronize();
	/*************************************************************

	Once requested, replace this comment by the appropriate code

	*************************************************************/


	return 0;
}