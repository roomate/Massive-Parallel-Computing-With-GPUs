/**************************************************************
This code is a part of a course on cuda taught by the author:
Lokman A. Abbas-Turki

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/
// Explicit
#include <stdio.h>
#include <math.h>

#define NTPB 256
#define NB 64

typedef float MyTab[NB][NTPB];

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

/*************************************************************************/
/*                   Black-Sholes Formula                                */
/*************************************************************************/
/*One-Dimensional Normal Law. Cumulative distribution function. */
double NP(double x){
  const double p= 0.2316419;
  const double b1= 0.319381530;
  const double b2= -0.356563782;
  const double b3= 1.781477937;
  const double b4= -1.821255978;
  const double b5= 1.330274429;
  const double one_over_twopi= 0.39894228;  
  double t;

  if(x >= 0.0){
	t = 1.0 / ( 1.0 + p * x );
    return (1.0 - one_over_twopi * exp( -x * x / 2.0 ) * t * ( t *( t * 
		   ( t * ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
  }else{/* x < 0 */
    t = 1.0 / ( 1.0 - p * x );
    return ( one_over_twopi * exp( -x * x / 2.0 ) * t * ( t *( t * ( t * 
		   ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
  }
}

__global__ void PDE_diff_k3(float dt, float dx, float dsig, float pmin,
	float pmax, float sigmin, float r, int N, MyTab* pt_GPU) {

	/*************************************************************

	Define pu, pm, pd and others needed values in registers

	*************************************************************/

	/*************************************************************

	N steps of the explicit scheme

	Do not forget to set limit values pmin and pmax at each step,
	without using the if statement

	*************************************************************/
  int u = (threadIdx.x + 1)*(threadIdx.x < NTPB - 1);
	int m = threadIdx.x;
	int d = (threadIdx.x - 1)*(threadIdx.x > 0);

  float sig = sigmin + dsig*blockIdx.x;
  float mu=r-0.5f * sig * sig;
  float pu=0.5f * (sig * sig * dt/(dx * dx) + mu * dt / dx);
  float pm=1.0f - sig * sig * dt / (dx * dx);
  float pd=0.5f * (sig * sig * dt / (dx * dx) - mu * dt / dx);

  for (int i=0; i< N; ++i)
  {
    pt_GPU[(i+1)%2][blockIdx.x][threadIdx.x] = 
    (threadIdx.x == 0) * pmin + (threadIdx.x == NTPB - 1) * pmax 
    + ((threadIdx.x > 0) 
    && (threadIdx.x < NTPB - 1))*(pu * pt_GPU[(i+1)%2][blockIdx.x][u] 
    + pm * pt_GPU[i%2][blockIdx.x][m] *
    + pd * pt_GPU[i%2][blockIdx.x][d]);
  }
  __syncthreads();

}

__global__ void PDE_diff_k2(float dt, float dx, float dsig, float pmin,
	float pmax, float sigmin, float r, int N, MyTab* pt_GPU) {

	/*************************************************************

	Define pu, pm, pd and others needed values in registers
  	This time, the 'for' loop occurs inside the kernel.

	*************************************************************/
	float sig = sigmin + dsig * blockIdx.x;
	float mu = r - 0.5f * sig * sig;
	float pu = 0.5f * (sig * sig * dt / (dx * dx) + mu * dt / dx);
	float pm = 1.0f - sig * sig * dt / (dx * dx);
	float pd = 0.5f * (sig * sig * dt / (dx * dx) - mu * dt / dx);
	
  	int u = threadIdx.x + 1;
	int m = threadIdx.x;
	int d = threadIdx.x - 1;

	for (int i=0; i<N; ++i)
  	{
		if (threadIdx.x == 0) {
			pt_GPU[(i+1)%2][blockIdx.x][threadIdx.x] = pmin;
		}
		else {
			if (threadIdx.x == NTPB - 1) {
				pt_GPU[(i+1)%2][blockIdx.x][threadIdx.x] = pmax;
			}
			else {
				pt_GPU[(i+1)%2][blockIdx.x][threadIdx.x] = pu * pt_GPU[i%2][blockIdx.x][u] 
        		+ pm * pt_GPU[i%2][blockIdx.x][m] 
        		+ pd * pt_GPU[i%2][blockIdx.x][d];
			}
		}
		__syncthreads(); //Synchronize threads of the kernel
	} 
}


__global__ void PDE_diff_k1(float dt, float dx, float dsig, float pmin,
	float pmax, float sigmin, float r, MyTab* pt_GPU, int i) {

	/*************************************************************

	Define pu, pm, pd and others needed values in registers

	*************************************************************/

	/*************************************************************

	One step of the explicit scheme

	Using an if statement, do not forget to set limit 
	values pmin and pmax

	*************************************************************/

	float sig=sigmin + dsig*blockIdx.x;
	float mu=r-0.5f*sig*sig;
	float pu=0.5f*(sig*sig*dt/(dx*dx) + mu*dt/dx);
	float pm=1.0f - sig*sig*dt/(dx * dx);
	float pd=0.5f*(sig*sig*dt/(dx*dx) - mu*dt/dx);

	int u=threadIdx.x + 1;
	int m=threadIdx.x;
	int d=threadIdx.x - 1;

	if (threadIdx.x == 0)
	{
		pt_GPU[(i+1)%2][blockIdx.x][threadIdx.x] = pmin;
	}
	else
	{
		if (threadIdx.x == NTPB - 1)
		{
			pt_GPU[(i+1)%2][blockIdx.x][threadIdx.x] = pmax;
		}
		else
		{
			pt_GPU[(i+1)%2][blockIdx.x][threadIdx.x] = pu * pt_GPU[i%2][blockIdx.x][u] + pm * pt_GPU[i%2][blockIdx.x][m] + pd * pt_GPU[i%2][blockIdx.x][d];
		}
	}
}


// Wrapper 
void PDE_diff (float dt, float dx, float dsig, float pmin, float pmax, 
			   float sigmin, float r, int N, MyTab* CPUTab){

	float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

	MyTab *GPUTab;
	testCUDA(cudaMalloc(&GPUTab, 2*sizeof(MyTab)));

	testCUDA(cudaMemcpy(GPUTab, CPUTab, sizeof(MyTab), cudaMemcpyHostToDevice)); //Transfer CPUTab on the GPU
	
	/*************************************************************

	Transfer the values on CPUTab to GPUTab 

	*************************************************************/

	// For loop on the host, to uncomment when needed

	// for (int i=0; i<N; i++){
	//   PDE_diff_k1<<<NB,NTPB>>>(dt, dx, dsig, pmin, pmax, sigmin, r, GPUTab, i); 
	// }

	// For loop on the device, to uncomment when needed
	PDE_diff_k2<<<NB,NTPB>>>(dt, dx, dsig, pmin, pmax, sigmin, r, N, GPUTab);
	
	// For loop on the device, to uncomment when needed
	//PDE_diff_k3<<<NB,NTPB>>>(dt, dx, dsig, pmin, pmax, sigmin, r, N, GPUTab);

	/*************************************************************

	Transfer the values on GPUTab to CPUTab

	*************************************************************/

    testCUDA(cudaMemcpy(CPUTab, GPUTab, sizeof(MyTab), cudaMemcpyDeviceToHost));
	testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&TimeExec,		// GPU timer instructions
			 start, stop));							// GPU timer instructions
	testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

	printf("GPU time execution for PDE diffusion: %f ms\n", TimeExec);

	testCUDA(cudaFree(GPUTab));	
}

///////////////////////////////////////////////////////////////////////////
// main function for a put option f(x) = max(0,K-x)
///////////////////////////////////////////////////////////////////////////

int main(void){

	float K = 100.0f;
	float T = 1.0f;
	float r = 0.1f;
	int N = 10000;
	float dt = (float)T/N;
	float xmin = log(0.5f*K);
	float xmax = log(2.0f*K);
	float dx = (xmax-xmin)/NTPB;
	float pmin = 0.5f*K;
	float pmax = 0.0f;
	float sigmin = 0.1f;
	float sigmax = 0.5f;
	float dsig = (sigmax-sigmin)/NB;
	

	MyTab *pt_CPU;
	testCUDA(cudaHostAlloc(&pt_CPU, sizeof(MyTab), cudaHostAllocDefault));
	for(int i=0; i<NB; i++){
	   for(int j=0; j<NTPB; j++){
	      pt_CPU[0][i][j] = max(0.0, K-exp(xmin + dx*j)); //Fill pt_CPU with terminal conditions
	   }	
	}

	PDE_diff(dt, dx, dsig, pmin, pmax, sigmin, r, N, pt_CPU); //Solve the PDE

    // S0 = 100 , sigma = 0.2
	printf(" %f, compare with %f\n",exp(-r*T)*pt_CPU[0][16][128],
		   K*(exp(-r*T)*NP(-(r-0.5*0.2*0.2)*sqrt(T)/0.2)-
		   NP(-(r+0.5*0.2*0.2)*sqrt(T)/0.2)));
	// S0 = 100 , sigma = 0.3
	printf(" %f, compare with %f\n",exp(-r*T)*pt_CPU[0][32][128],
		   K*(exp(-r*T)*NP(-(r-0.5*0.3*0.3)*sqrt(T)/0.3)-
		   NP(-(r+0.5*0.3*0.3)*sqrt(T)/0.3)));
	// S0 = 141.4214 , sigma = 0.3
	printf(" %f, compare with %f\n",exp(-r*T)*pt_CPU[0][32][192],
		   K*exp(-r*T)*NP(-(log(141.4214/K)+(r-0.5*0.3*0.3)*T)/(0.3*sqrt(T)))-
		   141.4214*NP(-(log(141.4214/K)+(r+0.5*0.3*0.3)*T)/(0.3*sqrt(T))));

	testCUDA(cudaFreeHost(pt_CPU));	
	return 0;
}