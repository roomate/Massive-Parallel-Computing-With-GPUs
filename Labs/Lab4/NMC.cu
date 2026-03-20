/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <curand_kernel.h>


// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is error %i in file %s at line %d\n", error, file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


/*One-Dimensional Normal Law. Cumulative distribution function. */
double NP(double x) {
	const double p = 0.2316419;
	const double b1 = 0.319381530;
	const double b2 = -0.356563782;
	const double b3 = 1.781477937;
	const double b4 = -1.821255978;
	const double b5 = 1.330274429;
	const double one_over_twopi = 0.39894228;
	double t;

	if (x >= 0.0) {
		t = 1.0 / (1.0 + p * x);
		return (1.0 - one_over_twopi * exp(-x * x / 2.0) * t * (t * (t *
			(t * (t * b5 + b4) + b3) + b2) + b1));
	}
	else {/* x < 0 */
		t = 1.0 / (1.0 - p * x);
		return (one_over_twopi * exp(-x * x / 2.0) * t * (t * (t * (t *
			(t * b5 + b4) + b3) + b2) + b1));
	}
}

// Set the state for each thread
__global__ void init_curand_state_k(curandState* state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}


// Monte Carlo simulation kernel from Lab4
__global__ void MC_k1(float S_0, float r, float sigma, float dt, float K,
	int N, curandState* state, float* PayGPU){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx];
	float2 G;
	float S = S_0;

	for (int i = 0; i < N; i++) {
		G = curand_normal2(&localState);
		S *= (1 + r * dt * dt + sigma * dt * G.x);
	}

	PayGPU[idx] = expf(-r * dt * dt * N) * fmaxf(0.0f, S - K);

	/* Copy state back to global memory */
	//state[idx] = localState;
}


// Monte Carlo simulation kernel
__global__ void MC_k2(float S_0, float r, float sigma, float dt, float K,
	int N, curandState* state, float* sum, int n) {
	/*N is the number of time steps, n is the number of MC simulations*/

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx]; //Retrieve the thread seed
	float2 G;
	float S = S_0;
	extern __shared__ float A[];

	float *R1s, *R2s;
	R1s = A;
	R2s = R1s + blockDim.x; /*Get the threads of the subsequent block*/

	for (int i = 0; i < N; i++) {
		G = curand_normal2(&localState);
		S *= (1 + r * dt * dt + sigma * dt * G.x);
	}
	R1s[threadIdx.x] = expf(-r * dt * dt * N) * fmaxf(0.0f, S - K) / n; /*Compute the updated option price, divide by n before the summation.*/
	R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x] * n; /*Compute the square of the updated option price, divided by n.*/

	__syncthreads(); //Sync all threads executing the kernel

	/*Threads Reduction to compute the sum*/
	int i = blockDim.x / 2; ///*It plays the role of a threshold.*/
	while (i != 0) {
		/*The idea is to combine threads per pair, one below the threshold and one above.*/
		if (threadIdx.x < i) { //There's no divergence here because no else condition
			R1s[threadIdx.x] += R1s[threadIdx.x + i];
			R2s[threadIdx.x] += R2s[threadIdx.x + i];
		}
		__syncthreads(); //Synchronize threads
		i /= 2;
	}

	/*Sum samples from first and last threads, and store it into the appropriate R1s and R2s respectively.*/
	if (threadIdx.x == 0) {
		atomicAdd(sum, R1s[0]); //At the end of reduction, only one thread left
		atomicAdd(sum + 1, R2s[0]);
	}

	/* Copy state back to global memory */
	//state[idx] = localState;
}


__global__ void init_curand_nested_state_k(curandState* state)
{
	int idx = blockIdx.x * blockDim.x * gridDim.y +
		blockDim.x * blockIdx.y + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}

__global__ void MC_nested_k(float Smin, float dS, float r, float sigma,
	float dt, float K, int N, curandState* state, float* sum, int n,
	int NBS) {
  
	int idx = blockIdx.x * blockDim.x * gridDim.y +
		blockDim.x * blockIdx.y + threadIdx.x;
	curandState localState = state[idx];
	float2 G;
	float S;
	extern __shared__ float A[];
	float* R1s, * R2s;
	R1s = A;
	R2s = R1s + blockDim.x;

	for (int j = 0; j < NBS; j++) {
		S = Smin + dS * blockIdx.x;
		for (int i = 0; i < N; i++) {
			G = curand_normal2(&localState);
			S *= (1 + r * dt * dt + sigma * dt * G.x);
		}
		R1s[threadIdx.x] = expf(-r * dt * dt * N) * fmaxf(0.0f, S - K) / n;
		R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x] * n;

		__syncthreads();
		int i = blockDim.x / 2; /*It plays the role of a threshold.*/
		/*The idea is to combine threads per pair, one below the threshold and one above.*/
		while (i != 0) {
			if (threadIdx.x < i) {
				R1s[threadIdx.x] += R1s[threadIdx.x + i]; 
				R2s[threadIdx.x] += R2s[threadIdx.x + i];
			}
			__syncthreads();
			i /= 2;
		}
		/*Sum samples from first and last threads*/
		if (threadIdx.x == 0) {
			atomicAdd(sum + blockIdx.x * 2, R1s[0]);
			atomicAdd(sum + blockIdx.x * 2 + 1, R2s[0]);
		}
	}
}

int main(void) {

	int NTPB = 512;
	int NB = 64;
	int n = NB * NTPB; /*Total number of threads.*/
	float T = 1.0f;
	float S_0 = 50.0f;
	float K = S_0;
	float sigma = 0.2f;
	float r = 0.1f;
	int N = 100;
	float dt = sqrtf(T/N);
	float* sum;
	int m = 256 * 512;
	int NBS = NB / 8;
	float Smin = 20;
	float Smax = 80;
	float dS = (Smax - Smin) / m;
	/*2D blocks, first coordinate is the number of MC simulation, and second coordinate is the number of samples per MC simulation*/
	dim3 Nblocks(m, NB / NBS);

	/*Allocate shared memory for variable sum, so that all threads can access it for reduction*/
	cudaMallocManaged(&sum, 2* m * sizeof(float));

	/*Set device sum variable to  value 0*/
	cudaMemset(sum, 0, 2 * m * sizeof(float));

	curandState* states;
	// testCuda(cudaMalloc(&states, n * sizeof(curandState)));
	//init_curand_state_k << <NB, NTPB >> > (states);
	testCUDA(cudaMalloc(&states, (NB / NBS) * NTPB * m * sizeof(curandState)));
	init_curand_nested_state_k <<<Nblocks, NTPB>>> (states);
  
	float Tim;
	cudaEvent_t start, stop;			// GPU timer instructions
	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start, 0);			// GPU timer instructions

	/*See Lab4 for MC_k1. The key difference is that the for loop lies outside the kernel, which significantly impacts the performance.*/
	// MC_k1<<<NB, NTPB>>>(S_0, r, sigma, dt, K, N, states, PayGPU);

	/*The for loop computing the actual sum of each Monte-Carlo simulation takes place in the kernel. You can see that sum is passed in arguments.*/
	// MC_k2 << <NB, NTPB, 2*NTPB*sizeof(float) >> > (S_0, r, sigma, dt, K, N, states, sum, n);

	MC_nested_k << <Nblocks, NTPB, 2 * NTPB * sizeof(float) >> > (Smin, dS,
		r, sigma, dt, K, N, states, sum, n, NBS);

	cudaEventRecord(stop, 0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
		start, stop);					// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions

	printf("The estimated price is equal to %f\n", sum[0]);
	printf("error associated to a confidence interval of 95%% = %f\n",
		1.96 * sqrt((double)(1.0f / (n - 1)) * (n*sum[m+1] - (sum[m] * sum[m]))) / sqrt((double)n));
	printf("The true price %f\n", S_0 * NP((r + 0.5 * sigma * sigma)/sigma) -
									K * expf(-r) * NP((r - 0.5 * sigma * sigma) / sigma));
	printf("Execution time %f ms\n", Tim);

	cudaFree(states);
	cudaFree(sum);

	return 0;
}