/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include "timer.h"

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


void addVect(int *a, int *b, int *c, int length){

	int i;

	for(i=0; i<length; i++){
		c[i] = a[i] + b[i];
	}
}

__global__ void addVect_k(int* a, int* b, int* c, int L)
{
	int idx=threadIdx.x + blockIdx.x*blockDim.x;

  //Slow index to access data on memory
  // int idx=threadIdx.x*gridDim.x + blockIdx.x;

	while (idx < L)
	{
		c[idx]=a[idx]+b[idx];
		idx += blockDim.x* gridDim.x;
	}
}

int main (void){

	// Variables definition
	int *a, *b, *c;
	int *aGPU, *bGPU, *cGPU;
	int i;
	
	// Length for the size of arrays
	int length = 1e8;

	float TimerAdd;							// GPU timer instructions
	cudaEvent_t start, stop;
	testCUDA(cudaEventCreate(&start));
	testCUDA(cudaEventCreate(&stop));

	// Memory allocation of arrays 
	a = (int*)malloc(length*sizeof(int));
	b = (int*)malloc(length*sizeof(int));
	c = (int*)malloc(length*sizeof(int));

	//Cuda memory allocation
	// testCUDA(cudaMalloc(&aGPU, length*sizeof(int)));
	// testCUDA(cudaMalloc(&bGPU, length*sizeof(int)));
	// testCUDA(cudaMalloc(&cGPU, length*sizeof(int)));

	//Cuda memory allocation, without explicit call between GPUs and the host CPU. Unified memory.
	testCUDA(cudaMallocManaged(&aGPU, length*sizeof(int)));
	testCUDA(cudaMallocManaged(&bGPU, length*sizeof(int)));
	testCUDA(cudaMallocManaged(&cGPU, length*sizeof(int)));


	// Values initialization
	for(i=0; i<length; i++){
		a[i] = i;
		b[i] = 9*i;
	}

// Uncomment if you want to compute the execution time for transferring data
//---------------------------------------------------
	// testCUDA(cudaEventRecord(start,0)); //Start the timer
//---------------------------------------------------

	//Copy memory from host cpu to gpu device
	testCUDA(cudaMemcpy(aGPU, a, length*sizeof(int), cudaMemcpyHostToDevice)); //cudaMemcpyHostToDevice argument precise the direction of dataflow.
	testCUDA(cudaMemcpy(bGPU, b, length*sizeof(int), cudaMemcpyHostToDevice));

	testCUDA(cudaEventRecord(start,0)); //Start the timer

	// Executing the addition 
	addVect_k << <1000, 1024>> > (aGPU, bGPU, cGPU, length);

//Comment if you want to compute the execution time of transferring data
//-----------------------------------------------------
	testCUDA(cudaEventRecord(stop,0)); //End the timer
	testCUDA(cudaEventSynchronize(stop));
	testCUDA(cudaEventElapsedTime(&TimerAdd, start, stop));
//-----------------------------------------------------

	testCUDA(cudaMemcpy(c, cGPU, length*sizeof(int), cudaMemcpyDeviceToHost));

//Uncomment if you want to compute the execution time of transferring data.
//-----------------------------------------------------
	// testCUDA(cudaEventRecord(stop,0)); //End the timer
	// testCUDA(cudaEventSynchronize(stop));
	// testCUDA(cudaEventElapsedTime(&TimerAdd, start, stop));
//-----------------------------------------------------
	testCUDA(cudaEventDestroy(start));
	testCUDA(cudaEventDestroy(stop));

	// Displaying the results to check the correctness 
	for(i=length-50; i<length-45; i++){
		printf(" ( %i ): %i\n", a[i]+b[i], c[i]);
	}

	printf("GPU Timer for the addition on the GPU of vectors: %f ms\n", 
		   (float)TimerAdd);		// GPU timer instructions

	// Freeing the memory
	free(a);
	free(b); 
	free(c);

	testCUDA(cudaFree(aGPU));
	testCUDA(cudaFree(bGPU));
	testCUDA(cudaFree(cGPU));

	return 0;
}