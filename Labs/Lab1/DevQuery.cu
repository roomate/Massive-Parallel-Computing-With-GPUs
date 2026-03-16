/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

int main (void){

	int count;
	cudaGetDeviceCount(&count);
	printf("The number of devices available is %i GPUS\n", count);

	cudaDeviceProp prop;
	testCUDA(cudaGetDeviceProperties(&prop, 0));
	printf("Global memory size in octet (bytes): %zd\n", prop.totalGlobalMem);
	printf("Maxmimum grid size: %i x %i x %i\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

	/*Maximum size of each dimension of a block. It represents the space in the room available, but you can not use everything simultaneously.*/
	printf("Maximum number of threads that can be launched: %i x %i x %i\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	
	/*The maximum number of threads in a block. Should be limited to 1024. This is the product of whatever your threadblock dimensions are (x*y*z), not to confuse with
	maxThreadsDim.*/
	printf("Maximum number of threads that can be launched per block: %i\n", prop.maxThreadsPerBlock);

	printf("The size of a warp in terms of threads: %i\n", prop.warpSize);
	printf("Shared memory size per block %zd\n", prop.sharedMemPerBlock);
	printf("Number of registers per block: %i\n", prop.regsPerBlock);
	printf("Number of registers per multiprocessor: %i\n", prop.regsPerMultiprocessor);
	printf("Maximum 2D texture memory: %i x %i\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
	printf("Number of multiprocessors: %i\n", prop.multiProcessorCount);
	printf("Maximum number of threads per multiprocessor: %i\n", prop.maxThreadsPerMultiProcessor);
	printf("Maximum number of resident blocks per multiprocessor: %i\n", prop.maxBlocksPerMultiProcessor);
	return 0;
}