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
    init_curand_state_k <<<NB, NTPB>>> (states, 0);

    MC_k1<<<NB, NTPB>>>(x, r, sigma, dt, K, B, P1, P2, M, i, j, states, PayGPU);
    
    testCUDA(cudaMemcpy(PayCPU, PayGPU, Nb_sim*sizeof(float), cudaMemcpyDeviceToHost));

	  testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	  testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	  testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
			 start, stop));							// GPU timer instructions
	  testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	  testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

	  printf("GPU time execution for MC_k1 is: %f ms\n", TimeExec);
    printf("F(%i, %i, %i)=%f\n", i, (int)x, j, mean(PayCPU, Nb_sim));

    testCUDA(cudaFree(PayGPU));
    testCUDA(cudaFree(states));
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
    init_curand_state_k <<<NB, NTPB>>> (states, 0);


    // MC_k2<<<NB, NTPB, NTPB*sizeof(float)>>>(x, r, sigma, dt, K, B, P1, P2, M, i, j, states, PayGPU);
    MC_k3<<<NB, NTPB, NTPB*sizeof(float)>>>(x, r, sigma, dt, K, B, P1, P2, M, i, j, states, PayGPU);
    testCUDA(cudaMemcpy(PayCPU, PayGPU, sizeof(float), cudaMemcpyDeviceToHost));

    printf("F(%i, %i, %i)=%f\n", i, (int)x, j, *PayCPU/Nb_sim);

    testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	  testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	  testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
			 start, stop));							// GPU timer instructions
	  testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	  testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

	  printf("GPU time execution for MC_k3 is: %f ms\n", TimeExec);

    testCUDA(cudaFree(PayGPU));
    testCUDA(cudaFree(states));
    free(PayCPU);
}

void wrapper_3(char filename[], float T, float r, float sigma, float S0, float K, float B, float P1, float P2)
{

  //Parameters for numerical parameter
  unsigned int M=100; //Number of time steps

  /*Simulation variables*/
  Option_price *PayGPU, *PayCPU;
  unsigned int NTPB=512; /*Number of threads per block*/
  unsigned int gridDim_x=M+1; /*Each block is associated with a time instant Ti.*/
  unsigned int gridDim_y=1024; /*Number of blocks having its own triplet (T_i, S_Ti, j_Ti).*/
  unsigned int gridDim_z=128; /*Number of blocks to estimate F(Ti, S_Ti, j).*/
  /*x-axis gives the instant of time Ti, the y-axis is a couple (S_Ti, j) and z-axis parallelizes sampling to estimate F(Ti, S_Ti, j).*/
  dim3 Nb_blocks(gridDim_x, gridDim_y, gridDim_z);

  /*Total number of Monte-Carlo estimates of F*/
  int Nb_sim=gridDim_x * gridDim_y;

  /*Total number of samples per MC estimate.*/
  int Sample_size = gridDim_z * NTPB;

  float dt=sqrtf(T/(M+1)); //IMPORTANT: It is the square root of the simulation's step size.

  printf("The interval [0, %.1f] is divided into %i sub-intervals, with steps of size %.3f \n", T, M+1, dt*dt);

  float TimeExec;
  cudaEvent_t start, stop;						// GPU timer instructions
  testCUDA(cudaEventCreate(&start));				// GPU timer instructions
  testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
  testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

  /*Allocate memory on host*/
  PayCPU=(Option_price*)malloc(Nb_sim * sizeof(Option_price));
  if (PayCPU==nullptr) {printf("Error, unable to allocate memory."); exit(EXIT_FAILURE);}

  /*To store the option price on device.*/
  testCUDA(cudaMalloc(&PayGPU, Nb_sim * sizeof(Option_price)));

  /*Initiate seeds for MC simulations. All blocks on the same (x,y)-axis will receive the same seed.*/
  curandState* states;
  testCUDA(cudaMalloc(&states, Nb_sim * sizeof(curandState)));
  init_curand_state_k_init <<<Nb_sim, 1>>> (states);

  /*Seed for monte-carlo sampling. All blocks on the same z-axis will receive the same seed.*/
  curandState* states_MC;
  testCUDA(cudaMalloc(&states_MC,  Sample_size * sizeof(curandState)));
  init_curand_state_k <<<gridDim_z, NTPB>>> (states_MC, 2);

  MC_k4<<<Nb_blocks, NTPB, NTPB*sizeof(float)>>>(r, sigma, dt, S0, K, B, P1, P2,
	M, Sample_size, states, states_MC, PayGPU);

  /*Copy memory back into host*/
  testCUDA(cudaMemcpy(PayCPU, PayGPU, Nb_sim * sizeof(Option_price), cudaMemcpyDeviceToHost));

  testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
  testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
  testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
      start, stop));							// GPU timer instructions
  testCUDA(cudaEventDestroy(start));				// GPU timer instructions
  testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

  printf("GPU time execution for MC_k4 is: %f ms\n", TimeExec);
  
  /*Free memory*/
  testCUDA(cudaFree(PayGPU));
  testCUDA(cudaFree(states));
  testCUDA(cudaFree(states_MC));
  free(PayCPU);

  /*Write data in a text file.*/
  write_data(filename, PayCPU, Nb_sim);
}

void wrapper_4(char filename[], float T, float r, float sigma, float S0, float K, float B, float P1, float P2)
{

  //Parameters for numerical parameter
  unsigned int M=100; //Number of time steps

  /*Simulation variables*/
  Option_price *PayGPU, *PayCPU;
  unsigned int NTPB=512; /*Number of threads per block*/
  unsigned int gridDim_x=M+1; /*Each block is associated with a time instant Ti.*/
  unsigned int gridDim_y=50; /*Each block is associated with j \in [0, 50].*/
  unsigned int gridDim_z=128; /*Number of blocks to estimate F(Ti, S_Ti, j).*/
  /*x-axis gives the instant of time Ti, the y-axis is a couple (S_Ti, j) and z-axis parallelizes sampling to estimate F(Ti, S_Ti, j).*/
  dim3 Nb_blocks(gridDim_x, gridDim_y, gridDim_z);

  /*Total number of Monte-Carlo estimates of F*/
  int Nb_sim=gridDim_x * gridDim_y;

  /*Total number of samples per MC estimate.*/
  int Sample_size = gridDim_z * NTPB;

  float dt=sqrtf(T/(M+1)); //IMPORTANT: It is the square root of the simulation's step size.

  printf("The interval [0, %.1f] is divided into %i sub-intervals, with steps of size %.3f \n", T, M+1, dt*dt);

  float TimeExec;
  cudaEvent_t start, stop;						// GPU timer instructions
  testCUDA(cudaEventCreate(&start));				// GPU timer instructions
  testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
  testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

  /*Allocate memory on host*/
  PayCPU=(Option_price*)malloc(Nb_sim * sizeof(Option_price));
  if (PayCPU==nullptr) {printf("Error, unable to allocate memory."); exit(EXIT_FAILURE);}

  /*To store the option price on device.*/
  testCUDA(cudaMalloc(&PayGPU, Nb_sim * sizeof(Option_price)));

  /*Initiate seeds for MC simulations. All blocks on the same (x,y)-axis will receive the same seed.*/
  curandState* states;
  testCUDA(cudaMalloc(&states, Nb_sim * sizeof(curandState)));
  init_curand_state_k_init <<<Nb_sim, 1>>> (states);

  /*Seed for monte-carlo sampling. All blocks on the same z-axis will receive the same seed.*/
  curandState* states_MC;
  testCUDA(cudaMalloc(&states_MC,  Sample_size * sizeof(curandState)));
  init_curand_state_k <<<gridDim_z, NTPB>>> (states_MC, 2);

  MC_k4<<<Nb_blocks, NTPB, NTPB*sizeof(float)>>>(r, sigma, dt, S0, K, B, P1, P2,
	M, Sample_size, states, states_MC, PayGPU);

  /*Copy memory back into host*/
  testCUDA(cudaMemcpy(PayCPU, PayGPU, Nb_sim * sizeof(Option_price), cudaMemcpyDeviceToHost));

  testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
  testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
  testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
      start, stop));							// GPU timer instructions
  testCUDA(cudaEventDestroy(start));				// GPU timer instructions
  testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

  printf("GPU time execution for MC_k4 is: %f ms\n", TimeExec);
  
  /*Free memory*/
  testCUDA(cudaFree(PayGPU));
  testCUDA(cudaFree(states));
  testCUDA(cudaFree(states_MC));
  free(PayCPU);

  /*Write data in a text file.*/
  write_data(filename, PayCPU, Nb_sim);
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

  /*Initiate seeds for MC simulations*/
  curandState* states;
  testCUDA(cudaMalloc(&states, 512 * M * NTPB * sizeof(curandState)));
  init_curand_state_k_2D <<<Nb_blocks, NTPB>>> (states);

  MC_k_trash<<<Nb_blocks, NTPB>>>(r, sigma, dt, S0, K, B, P1, P2,
	M, Nb_sim, states, PayGPU);
  testCUDA(cudaMemcpy(PayCPU, PayGPU, sizeof(float), cudaMemcpyDeviceToHost));

  testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
  testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
  testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
      start, stop));							// GPU timer instructions
  testCUDA(cudaEventDestroy(start));				// GPU timer instructions
  testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

  printf("GPU time execution for MC_k4 is: %f ms\n", TimeExec);
  
  /*Free memory*/
  testCUDA(cudaFree(PayGPU));
  testCUDA(cudaFree(states));
  free(PayCPU);

  /*Write data in a text file.*/
  char filename[]="option_price.txt";
  write_data(filename, PayCPU, 512 * M * NTPB);

} 
