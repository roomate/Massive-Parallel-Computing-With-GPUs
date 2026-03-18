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


//Run on the CPU
void addVect(int *a, int *b, int *c, int length){

	for(int i=0; i<length; i++){
		c[i] = a[i] + b[i];
	}
}

//Run on the GPU
__global__ void addVect_k(int* a, int* b, int* c, int L)
{
	int idx=threadIdx.x + blockIdx.x*blockDim.x;

	//Slow index to access data on memory
	// int idx=threadIdx.x*gridDim.x + blockIdx.x; //Using this index multiplies by 10 the execution time

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

	Timer Tim;							// CPU timer instructions

	// Memory allocation of arrays 
	a = (int*)malloc(length*sizeof(int));
	b = (int*)malloc(length*sizeof(int));
	c = (int*)malloc(length*sizeof(int));

	//Cuda memory allocation
	testCUDA(cudaMalloc(&aGPU, length*sizeof(int)));
	testCUDA(cudaMalloc(&bGPU, length*sizeof(int)));
	testCUDA(cudaMalloc(&cGPU, length*sizeof(int)));


	// Values initialization
	for(i=0; i<length; i++){
		a[i] = i;
		b[i] = 9*i;
	}

	//Copy memory from host cpu to gpu device
	testCUDA(cudaMemcpy(aGPU, a, length*sizeof(int), cudaMemcpyHostToDevice)); //cudaMemcpyHostToDevice argument precises the direction of dataflow.
	testCUDA(cudaMemcpy(bGPU, b, length*sizeof(int), cudaMemcpyHostToDevice));

	Tim.start();						// CPU timer instructions

	// Executing the addition on CPU
	// addVect(a, b, c, length);
	//Executing the addition on GPU
	addVect_k << <1000, 1024>> > (aGPU, bGPU, cGPU, length); //Add vector on GPU proc.
	cudaDeviceSynchronize(); //Used because we use a timer from CPU. It stops the CPU until the GPU ends its jobs. We do that to have an accurate measure of time.

	Tim.add();							// CPU timer instructions

	testCUDA(cudaMemcpy(c, cGPU, length*sizeof(int), cudaMemcpyDeviceToHost));

	// Displaying the results to check the correctness 
	for(i=length-50; i<length-45; i++){
		printf(" ( %i ): %i\n", a[i]+b[i], c[i]);
	}

	printf("CPU Timer for the addition on the CPU of vectors: %f s\n", 
		   (float)Tim.getsum());		// CPU timer instructions

	// Freeing the memory on CPU
	free(a);
	free(b);
	free(c);

	// Freeing the memory on GPU
	testCUDA(cudaFree(aGPU));
	testCUDA(cudaFree(bGPU));
	testCUDA(cudaFree(cGPU));

	return 0;
}