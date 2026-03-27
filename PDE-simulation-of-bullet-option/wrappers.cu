#include <stdio.h>
#include <stdlib.h>
#include "MC.cuh"
#include <assert.h>
#include "utils.cuh"

#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__));

void wrapper_1(float x, int i, int j, float T, float r, float sigma, float K, float B, float P1, float P2)
{
    /*Simulation variables*/
    float *PayGPU, *PayCPU;
    int NB=128;
    int NTPB=512;
    int Nb_sim=NB*NTPB;

    //Parameters for numerical parameter
    unsigned int M=100; //Number of time steps
    float dt=sqrtf(T/(M+1)); //IMPORTANT: It is the square root of the simulation's step size.

    printf("The interval [0, %.1f] is divided into %i sub-intervals, with steps of size %.3f \n", T, M+1, dt*dt);

    assert (x>=0);
    assert (j<=i);

    float TimeExec;
  	cudaEvent_t start, stop;						// GPU timer instructions
	  testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	  testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	  testCUDA(cudaEventRecord(start,0));				// GPU timer instructions


    /*Allocate memory on host*/
    PayCPU=(float*)malloc(Nb_sim*sizeof(float));
    if (PayCPU==nullptr) {printf("Error, unable to allocate memory."); exit(EXIT_FAILURE);}
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

	  printf("GPU time execution for MC_k1 is: %f ms\n", TimeExec);

    printf("F(%i, %i, %i)=%f\n", i, (int)x, j, sum_array(PayCPU, Nb_sim));
    testCUDA(cudaFree(PayGPU));
    free(PayCPU);
}

void wrapper_2(float x, int i, int j, float T, float r, float sigma, float K, float B, float P1, float P2)
{
    /*Simulation variables*/
    float *PayGPU, *PayCPU;
    int NB=128;
    int NTPB=512;
    int Nb_sim=NB*NTPB;

    //Parameters for numerical parameter
    unsigned int M=100; //Number of time steps
    float dt=sqrtf(T/(M+1)); //IMPORTANT: It is the square root of the simulation's step size.

    printf("The interval [0, %.1f] is divided into %i sub-intervals, with steps of size %.3f \n", T, M+1, dt*dt);

    assert (x>=0);
    assert (j<=i);

    float TimeExec;
  	cudaEvent_t start, stop;						// GPU timer instructions
	  testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	  testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	  testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

    /*Allocate memory on host*/
    PayCPU=(float*)malloc(sizeof(float));
    if (PayCPU==nullptr) {printf("Error, unable to allocate memory."); exit(EXIT_FAILURE);}

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

	  printf("GPU time execution for MC_k3 is: %f ms\n", TimeExec);
    printf("F(%i, %i, %i)=%f\n", i, (int)x, j, *PayCPU/Nb_sim);

    testCUDA(cudaFree(PayGPU));
    free(PayCPU);
}


void wrapper_trash(float T, float r, float sigma, float S0, float K, float B, float P1, float P2)
{

  //Parameters for numerical parameter
  unsigned int M=100; //Number of time steps

  /*Simulation variables*/
  Option_price *PayGPU, *PayCPU;
  int NTPB=512; /*Number of threads per block*/
  int Nb_sim=10000; /*Number of samples for the Monte-Carlo estimation.*/
  dim3 Nb_blocks(M, 512);

  float dt=sqrtf(T/(M+1)); //IMPORTANT: It is the square root of the simulation's step size.

  printf("The interval [0, %.1f] is divided into %i sub-intervals, with steps of size %.3f \n", T, M+1, dt*dt);

  float TimeExec;
  cudaEvent_t start, stop;						// GPU timer instructions
  testCUDA(cudaEventCreate(&start));				// GPU timer instructions
  testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
  testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

  /*Allocate memory on host*/
  PayCPU=(Option_price*)malloc(M * 512 * NTPB * sizeof(Option_price));
  if (PayCPU==nullptr) {printf("Error, unable to allocate memory."); exit(EXIT_FAILURE);}

  /*To store the option price on GPU.*/
  testCUDA(cudaMalloc(&PayGPU, M * 512 * NTPB * sizeof(Option_price)));
  /*In default stream, kernel launches are serialized.*/
  testCUDA(cudaMemset(PayGPU, 0, M * 512 * NTPB * sizeof(Option_price)));

  /*Initiate seeds for MC simulations*/
  curandState* states;
  testCUDA(cudaMalloc(&states, 512 * M * NTPB * sizeof(curandState)));
  init_curand_state_k_2D <<<Nb_blocks, NTPB>>> (states);

  MC_k4<<<Nb_blocks, NTPB, NTPB*sizeof(float)>>>(r, sigma, dt, S0, K, B, P1, P2,
	M, Nb_sim, states, PayGPU);
  testCUDA(cudaMemcpy(PayCPU, PayGPU, sizeof(float), cudaMemcpyDeviceToHost));

  testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
  testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
  testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
      start, stop));							// GPU timer instructions
  testCUDA(cudaEventDestroy(start));				// GPU timer instructions
  testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

  printf("GPU time execution for MC_k4 is: %f ms\n", TimeExec);
  
  /*Write data in a text file.*/
  char filename[]="option_price.txt";
  write_data(filename, PayCPU, 512 * M * NTPB);

  /*Free memory*/
  testCUDA(cudaFree(PayGPU));
  free(PayCPU);
} 
