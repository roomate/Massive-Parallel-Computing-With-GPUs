#include "MC.cuh"
#include "utils.cuh"
#include <stdio.h>
#include <curand_kernel.h>

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

__global__ void MC_k2(float x, float r, float sigma, float dt, float K, float B, float P1, float P2,
	int M, int i, int j, curandState* state, float* PayGPU)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  /*Each block has its own copy of tmp.*/
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

  /*Block-level synchronization barrier.*/
  __syncthreads();

  /*Threads Reduction to compute the sum*/
  int counter = blockDim.x / 2; /*blockDim.x=m, it is a decreasing counter.*/
  /*Apply dyadic thread reduction between threads in the same block.*/
  while (counter != 0) {
    if (threadIdx.x < counter) { //There's no divergence here because no else condition
      tmp[threadIdx.x] += tmp[threadIdx.x + counter];
    }
    __syncthreads();
    counter /= 2;
  }

  if (threadIdx.x == 0) {
    atomicAdd(PayGPU, tmp[0]);
  }
  state[idx]=localState;
}

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

  /*Block-level synchronization barrier.*/
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

  if (threadIdx.x==0)
  {
    atomicAdd(PayGPU, tmp_block[0]); 
  }

  state[idx]=localState;
}

__global__ void MC_k4(float r, float sigma, float dt, float S0, float K, float B, float P1, float P2,
	int M, int Sample_size, curandState* state, curandState* states_MC, Option_price* PayGPU)
{
  int idx_init_state = blockIdx.y * gridDim.x + blockIdx.x;

  /*Each block has its own copy of tmp.*/
  extern __shared__ float tmp[];

  curandState localState = state[idx_init_state];

  float S_Ti=S0;
  int j_Ti=0;
  float2 G;

  /*This computation should be identical for blocks with the same (x,y) coordinates*/
  for (int i=0; i<M; ++i)
  {
    G = curand_normal2(&localState); /*Sample the unit gaussian law.*/
    S_Ti *= expf((r - sigma*sigma/2) * dt * dt + sigma * dt * G.x)*(i <= blockIdx.x) + (i > blockIdx.x);
    j_Ti+=(S_Ti<=B)*(i <= blockIdx.x);
  }

  float S=S_Ti;
  int j=j_Ti;

  /*Id of the thread on the z-axis. Block with the same z coordinate have the same seed.*/
  int idx_MC = blockIdx.z * blockDim.z + threadIdx.x;
  localState = state[idx_MC];

  for (int i = 1; i <= M; ++i) {
      G = curand_normal2(&localState); /*Sample the unit gaussian law.*/
      S *= expf((r - sigma*sigma/2) * dt * dt + sigma * dt * G.x)*(i > blockIdx.x) + (i <= blockIdx.x);
      j+=(S<=B)*(i > blockIdx.x);
  }

  tmp[threadIdx.x] = expf(-r * (M-blockIdx.x) * dt * dt) * fmaxf(0.0f, S - K); // (((float)j>=P1) && ((float)j<=P2));
  
  /*Dyadic thread reduction of blocks with the same (x,y) coordinates.*/

  /*Block-level synchronization barrier.*/
   __syncthreads();

  int counter = blockDim.x / 2; /*It is basically a decreasing counter.*/
  /*Apply dyadic thread reduction between threads in the same block.*/
  while (counter != 0) {
    if (threadIdx.x < counter) { //There's no divergence here because no else condition
      tmp[threadIdx.x] += tmp[threadIdx.x + counter];
    }
    __syncthreads();
    counter /= 2;
  }
  /*Add the result of each block to PayGPU.*/
  if (threadIdx.x == 0) {
    /*Store conditional parameters*/
    PayGPU[idx_init_state].Ti=(blockIdx.x+1)*dt*dt;
    PayGPU[idx_init_state].x=S_Ti;
    PayGPU[idx_init_state].j=j_Ti;

    atomicAdd(&(PayGPU[idx_init_state].F), tmp[0]/Sample_size);
  }
}


__global__ void MC_k_trash(float r, float sigma, float dt, float S0, float K, float B, float P1, float P2,
	int M, int Nb_sim, curandState* state, Option_price* PayGPU)
{
  int idx = blockIdx.x * blockDim.x * gridDim.y +
		blockDim.x * blockIdx.y + threadIdx.x;

  curandState localState = state[idx];

  float S_Ti=S0;
  int j_Ti=0;

  float S;
  int j;
  float F=0;

  float2 G;

  for (int i=0; i<blockIdx.x+1; ++i)
  {
    G = curand_normal2(&localState); /*Sample the unit gaussian law.*/
    S_Ti *= expf((r - sigma*sigma/2) * dt * dt + sigma * dt * G.x);
    j_Ti+=(S<=B);
  }

  /*Store conditional parameters*/
  PayGPU[idx].Ti=(blockIdx.x+1)*dt*dt;
  PayGPU[idx].x=S_Ti;
  PayGPU[idx].j=j_Ti;

  for (int k=0; k<1; ++k)
  {
    S=S_Ti;
    j=j_Ti;
    for (int p = blockIdx.x+1; p <= blockDim.x+1; ++p) {
      G = curand_normal2(&localState); /*Sample the unit gaussian law.*/
      S *= expf((r - sigma*sigma/2) * dt * dt + sigma * dt * G.x);
      j+=(S<=B);
    }
    F += expf(-r * (M-blockIdx.x) * dt * dt) * fmaxf(0.0f, S - K) * (((float)j>=P1) && ((float)j<=P2));
  }
  /*No reduction is required since threads run monte-carlo simulation independantly of each others.*/
  PayGPU[idx].F=F/Nb_sim;
}
