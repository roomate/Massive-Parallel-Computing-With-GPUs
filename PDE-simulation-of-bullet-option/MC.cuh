#ifndef MC_h
#define MC_h

#include <curand_kernel.h>
#include "utils.cuh"

/*M is the maximum number of time step, i is the current timestep, and j is the value of I_{T_i}. Note that one must have j<=i.*/
__global__ void MC_k1(float x, float r, float sigma, float dt, float K, float B, float P1, float P2,
	int M, int i, int j, curandState* state, float* PayGPU);

/*Thread reduction with shared memory*/
__global__ void MC_k2(float x, float r, float sigma, float dt, float K, float B, float P1, float P2,
	int M, int i, int j, curandState* state, float* PayGPU);

/*Thread reduction with registers*/
__global__ void MC_k3(float x, float r, float sigma, float dt, float K, float B, float P1, float P2,
	int M, int i, int j, curandState* state, float* PayGPU);

__global__ void MC_k4(float r, float sigma, float dt, float S0, float K, float B, float P1, float P2,
	int M, int Nb_sim, curandState* state, curandState* states_MC, Option_price* PayGPU);

__global__ void MC_k_trash(float r, float sigma, float dt, float S0, float K, float B, float P1, float P2,
	int M, int Nb_sim, curandState* state, Option_price* PayGPU);

#endif