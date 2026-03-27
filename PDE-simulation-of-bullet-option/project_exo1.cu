#include <stdio.h>
#include <curand_kernel.h>
#include <assert.h>

// Function that catches errors 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__));


// Initiate the PRNG state for each thread
__global__ void init_curand_state_k(curandState* state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
  /*seed is 0, offset is 0.*/
	curand_init(0, idx, 0, &state[idx]);
}

/*M is the maximum number of time step, i is the current timestep, and j is the value of I_{T_i}. Note that one must have j<=i.*/
__global__ void MC_k1(float x, float r, float sigma, float dt, float K, float B, float P1, float P2,
	int M, int i, int j, curandState* state, float* PayGPU)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx];
	float2 G;
	float S = x;

	for (int k = i; k < M; k++) {
		G = curand_normal2(&localState); /*Sample the unit gaussian law.*/
		S *= expf((r - sigma*sigma/2) * dt * dt + sigma * dt * G.x);
    j+=(S<=B);
	}

	PayGPU[idx] = expf(-r * (M-i) * dt * dt) * fmaxf(0.0f, S - K)*(((float)j>=P1) && ((float)j<=P2));

  state[idx]=localState;
}

float sum_array(float* array, int length)
{
    float sum=0;
    for (int i=0; i<length; ++i) {sum+=array[i];}
    return sum/length;
}

/*Wrapper 1: Compute conditional expectation for a single triplet (i, x, j)*/
void wrapper_1(float x, int i, int j, float T, float r, float sigma, float K, float B, float P1, float P2)
{
    /*Simulation variables*/
    float *PayGPU, *PayCPU;
    int NB=128;
    int NTPB=512;
    int Nb_sim=NB*NTPB;

    //Parameters for numerical parameter
    unsigned int M=100; //Number of time steps
    float dt=sqrtf(T/M); //IMPORTANT: It is the square root of the simulation's step size.
    float t_i = i * dt * dt;

    printf("The interval [0, %.1f] is divided into %i sub-intervals, with steps of size %.2f \n", T, M, dt*dt);

    assert (x>=0);
    assert (j<=i);

    float TimeExec;
    float mc_value;
  	cudaEvent_t start, stop;						// GPU timer instructions
	  testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	  testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	  testCUDA(cudaEventRecord(start,0));				// GPU timer instructions


    /*Allocate memory on host*/
    PayCPU=(float*)malloc(Nb_sim*sizeof(float));
    /*To store the price computed on unified memory.*/
	  testCUDA(cudaMalloc(&PayGPU, Nb_sim * sizeof(float)));

    /*Initiate seeds for MC simulations*/
    curandState* states;
    testCUDA(cudaMalloc(&states, Nb_sim * sizeof(curandState)));
    init_curand_state_k <<<NB, NTPB>>> (states);

    MC_k1<<<NB, NTPB>>>(x, r, sigma, dt, K, B, P1, P2, M, i, j, states, PayGPU);
    
    testCUDA(cudaMemcpy(PayCPU, PayGPU, Nb_sim*sizeof(float), cudaMemcpyDeviceToHost));

	  testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	  testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	  testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
			 start, stop));							// GPU timer instructions
	  testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	  testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

    mc_value = sum_array(PayCPU, Nb_sim);

	  printf("GPU time execution for Monte Carlo: %f ms\n", TimeExec);
    printf("MC result for T_%d = %.6f, j=%d, S=%.2f: F = %f\n", i, t_i, j, x, mc_value);

    testCUDA(cudaFree(PayGPU));
    testCUDA(cudaFree(states));
    free(PayCPU);
}

/*Thread reduction with shared memory*/
__global__ void MC_k2(float x, float r, float sigma, float dt, float K, float B, float P1, float P2,
	int M, int i, int j, curandState* state, float* PayGPU)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  extern __shared__ float tmp[];

  curandState localState = state[idx];
  float2 G;
  float S = x;

  for (int k = i; k < M; k++) {
    G = curand_normal2(&localState); /*Sample the unit gaussian law.*/
    S *= expf((r - sigma*sigma/2) * dt * dt + sigma * dt * G.x);
    j+=(S<=B);
  }

  tmp[threadIdx.x] = expf(-r * (M-i) * dt * dt) * fmaxf(0.0f, S - K)*(((float)j>=P1) && ((float)j<=P2));

  __syncthreads();

  /*Threads Reduction to compute the sum*/
  int counter = blockDim.x / 2; /*blockDim.x=m, it is a decreasing counter.*/
  /*Apply dyadic thread reduction between threads in the same block.*/
  while (counter != 0) {
    if (threadIdx.x < counter) { //There's no divergence here because no else condition
      tmp[threadIdx.x] += tmp[threadIdx.x + counter];
    }
    __syncthreads(); //Synchronise all threads within the same block.
    counter /= 2;
  }

  /*Combine dyadic reduction of each blocks. There should be NBS blocks with the same initial S.*/
  if (threadIdx.x == 0) {
    atomicAdd(PayGPU, tmp[0]);
  }
  state[idx]=localState;
}

/*Thread reduction with registers*/
__global__ void MC_k3(float x, float r, float sigma, float dt, float K, float B, float P1, float P2,
	int M, int i, int j, curandState* state, float* PayGPU)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int lane=threadIdx.x & 0x1f;

  /*Static allocation on shared memory. At most 32 warps in a block.*/
  __shared__ float tmp_block[32];

  float loc_warp;

  curandState localState = state[idx];
  float2 G;
  float S = x;

  for (int k = i; k < M; k++) {
    G = curand_normal2(&localState); /*Sample the unit gaussian law.*/
    S *= expf((r - sigma*sigma/2) * dt * dt + sigma * dt * G.x);
    j+=(S<=B);
  }

  loc_warp = expf(-r * (M-i) * dt * dt) * fmaxf(0.0f, S - K)*(((float)j>=P1) && ((float)j<=P2));

  /*Threads Reduction on lane to compute the sum*/
  int counter = 16; /*blockDim.x=m, it is a decreasing counter.*/

  /*Apply dyadic thread reduction between threads in the warp.*/
  while (counter != 0) {
    loc_warp += __shfl_down_sync(0xffffffff, loc_warp, counter, 32);
    counter /= 2;
  }

  if (lane==0)
  {
    tmp_block[threadIdx.x/32]=loc_warp; 
  }
  __syncthreads();

  /*Threads Reduction on block threads to compute the sum*/
  counter = blockDim.x/(2*32); /*It is a decreasing counter.*/

  /*Apply dyadic thread reduction between threads in the block.*/
  while (counter != 0) {
    if (threadIdx.x < counter)
    {
      tmp_block[threadIdx.x]+=tmp_block[threadIdx.x + counter];
    }
    __syncthreads();
    counter/=2;
  }

  /*Combine dyadic reduction of each blocks. There should be NBS blocks with the same initial S.*/
  if (threadIdx.x==0)
  {
    atomicAdd(PayGPU, tmp_block[0]); 
  }

  state[idx]=localState;

}


/*Wrapper 2: Threads reduction within the kernel with shared memory and registers*/
void wrapper_2(float x, int i, int j, float T, float r, float sigma, float K, float B, float P1, float P2)
{
    /*Simulation variables*/
    float *PayGPU, *PayCPU;
    int NB=128;
    int NTPB=512;
    int Nb_sim=NB*NTPB;

    //Parameters for numerical parameter
    unsigned int M=100; //Number of time steps
    float dt=sqrtf(T/M); //IMPORTANT: It is the square root of the simulation's step size.
    float t_i = i * dt * dt;

    printf("The interval [0, %.1f] is divided into %i sub-intervals, with steps of size %.2f \n", T, M, dt*dt);

    assert (x>=0);
    assert (j<=i);

    float TimeExec;
    float mc_value;
  	cudaEvent_t start, stop;						// GPU timer instructions
	  testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	  testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	  testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

    /*Allocate memory on host*/
    PayCPU=(float*)malloc(sizeof(float));
    *PayCPU=0;
    /*To store the option price on GPU.*/
	  testCUDA(cudaMalloc(&PayGPU, sizeof(float)));
    /*In default stream, kernel launches are serialized.*/
    testCUDA(cudaMemset(PayGPU, 0, sizeof(float)));

    /*Initiate seeds for MC simulations*/
    curandState* states;
    testCUDA(cudaMalloc(&states, Nb_sim * sizeof(curandState)));
    init_curand_state_k <<<NB, NTPB>>> (states);
    // MC_k2<<<NB, NTPB, NTPB*sizeof(float)>>>(x, r, sigma, dt, K, B, P1, P2, M, i, j, states, PayGPU);
    MC_k3<<<NB, NTPB, NTPB*sizeof(float)>>>(x, r, sigma, dt, K, B, P1, P2, M, i, j, states, PayGPU);
    testCUDA(cudaMemcpy(PayCPU, PayGPU, sizeof(float), cudaMemcpyDeviceToHost));
    
	  testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	  testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	  testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
			 start, stop));							// GPU timer instructions
	  testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	  testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

    mc_value = *PayCPU / Nb_sim;

	  printf("GPU time execution for Monte Carlo: %f ms\n", TimeExec);
    printf("MC result for T_%d = %.6f, j=%d, S=%.2f: F = %f\n", i, t_i, j, x, mc_value);

    testCUDA(cudaFree(PayGPU));
    testCUDA(cudaFree(states));
    free(PayCPU);
}

int main(int argc, char* argv[])
{
    // financial parameters
    float sigma=0.2; //Volatility
    float r=.1; //Risk-free return
    // float S0=100; //Initial spot price
    float T=1; //Maturity
    float K=100; //Contract's strike
    float B=110; //Option's barrier
    float P1=10; //Lower bound of the interval
    float P2=40; //Upper bound of the interval
    
    float mode=atof(argv[1]);
    float x = 100.0f;
    int j = 20;
    int i = 99;

    if (argc >= 5)
    {
      x = atof(argv[2]);
      j = atoi(argv[3]);
      i = atoi(argv[4]);
    }

    if (mode==1)
    {
      wrapper_1(x, i, j, T, r, sigma, K, B, P1, P2);
    }
    else if (mode==2)
    {
      wrapper_2(x, i, j, T, r, sigma, K, B, P1, P2);
    }
}
