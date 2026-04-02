/**************************************************************
This code is a part of a course on cuda taught by the author:
Lokman A. Abbas-Turki

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/
// Optim
#include <stdio.h>
#include <math.h>

#define NTPB 256
#define NB 64

typedef float MyTab[NB][NTPB];

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

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

// Thomas algorithm for implicit part
__device__ void Thomas_d(const float pmin, const float pmax, float qu, float qm, float qd, float* rhs, float* solution, int len) {

	/*************************************************************

	Thomas' algorithm. It is assumed that the triplet (a, b, c) remains identical along the tridiagonal.

	*************************************************************/
  
  /*Inject boundary conditions into the tridiagonal system*/
  rhs[1]-=qd*pmin;
  rhs[len-2]-=qu*pmax;
  float tmp[NTPB-1];
  tmp[0]=qm;

  float w;

  /*Forward sweep*/
  for (int i=1; i<len-1; ++i)
  {
    w=qd/tmp[i-1];
    tmp[i]=qm - w*qu;
    rhs[i]-=w*rhs[i-1];
  }

  /*Backward sweep*/
  solution[len-1]=pmax;
  for (int i=len-2; i>0; --i)
  {
    solution[i]=(rhs[i]-qu*solution[i+1])/tmp[i];
  }
  solution[0]=pmin;
}

/////////////////////////////////////////////////////////////////////////////
// Optimized explicit
/////////////////////////////////////////////////////////////////////////////
__global__ void PDE_diff_k5(float dt, float dx, float dsig, float pmin,
	float pmax, float sigmin, float r, int N,
	MyTab* pt_GPU) {

	/*************************************************************

	We use here shared memory to limitate access time to data on high-latency global memory.

	*************************************************************/

	int u = (threadIdx.x + 1)*(threadIdx.x < NTPB - 1);
	int m = threadIdx.x;
	int d = (threadIdx.x - 1)*(threadIdx.x > 0);

	float sig = sigmin + dsig * blockIdx.x;
	float mu=r-0.5f * sig * sig;
	float pu=0.5f * (sig * sig * dt/(dx * dx) + mu * dt / dx);
	float pm=1.0f - sig * sig * dt / (dx * dx);
	float pd=0.5f * (sig * sig * dt / (dx * dx) - mu * dt / dx);

	extern __shared__ float A[];

	float* A1=A;
	float* A2=A+NTPB;

	A1[m] = pt_GPU[0][blockIdx.x][m];
	__syncthreads();

	for (int i=1; i<=N; ++i)
	{
		if ((i%2))
		{
			A2[m] = (m==0)*pmin + (m==NTPB - 1) * pmax + ((m>0) && (m<NTPB - 1)) * (pu*A1[u] + pm * A1[m] + pd*A1[d]);
		}
		else
		{
			A1[m] = (m==0)*pmin + (m==NTPB - 1) * pmax + ((m>0) && (m<NTPB - 1)) * (pu*A2[u] + pm * A2[m] + pd*A2[d]);
		}
	}
	if (N%2)
	{
		pt_GPU[0][blockIdx.x][m]=A2[m];
	}
	else
	{
		pt_GPU[0][blockIdx.x][m]=A1[m];
	}
}


/////////////////////////////////////////////////////////////////////////////
// Optimized implicit
/////////////////////////////////////////////////////////////////////////////
__global__ void PDE_diff_k6(float dt, float dx, float dsig, float pmin,
	float pmax, float sigmin, float r, int N,
	MyTab* pt_GPU) {

	/*************************************************************

  Only a single thread per block should execute this kernel.

	*************************************************************/

	float sig = sigmin + dsig * blockIdx.x;
	float mu=r-0.5f * sig * sig;
	float qu=-0.5f * (sig * sig * dt/(dx * dx) + mu * dt / dx);
	float qm=1.0f + sig * sig * dt / (dx * dx);
	float qd=0.5f * (-sig * sig * dt / (dx * dx) + mu * dt / dx);

	extern __shared__ float A[];

	float* A1=A;
	float* A2=A+NTPB;
	
	/*Copy data in A1*/
	for (int i=0; i<NTPB; ++i)
	{
		A1[i]=(*pt_GPU)[blockIdx.x][i];
	}
	for (int i=0; i<N; ++i)
	{
		if (i%2==0) {Thomas_d(pmin, pmax, qu, qm, qd, A1, A2, NTPB);}
		else {Thomas_d(pmin, pmax, qu, qm, qd, A2, A1, NTPB);}
	}
	if (N%2==0)
	{
		for (int i=0; i<NTPB; ++i)
		{
			(*pt_GPU)[blockIdx.x][i]=A1[i];
		}
	}
	else
	{
		for (int i=0; i<NTPB; ++i)
		{
			(*pt_GPU)[blockIdx.x][i]=A2[i];
		}
	}
}


// Wrapper 
void PDE_diff(float dt, float dx, float dsig, float pmin, float pmax,
	float sigmin, float r, int N, MyTab* CPUTab) {

	float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start, 0));			// GPU timer instructions

	MyTab* GPUTab;
	testCUDA(cudaMalloc(&GPUTab, sizeof(MyTab)));

	testCUDA(cudaMemcpy(GPUTab, CPUTab, sizeof(MyTab), cudaMemcpyHostToDevice)); //Transfer CPUTab on the GPU

	// PDE_diff_k5<<<NB, NTPB, 2*NTPB*sizeof(float)>>>(dt, dx, dsig, pmin, pmax, sigmin, r, N, GPUTab);

	PDE_diff_k6<<<NB, 1, 2*NTPB*sizeof(float)>>>(dt, dx, dsig, pmin, pmax, sigmin, r, N, GPUTab);


	/*************************************************************

	Once requested, replace this comment by the appropriate code

	*************************************************************/
	testCUDA(cudaMemcpy(CPUTab, GPUTab, sizeof(MyTab), cudaMemcpyDeviceToHost));


	testCUDA(cudaEventRecord(stop, 0));				// GPU timer instructions
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
int main(void) {

	float K = 100.0f;
	float T = 1.0f;
	float r = 0.1f;
	int N = 10000;
	float dt = (float)T / N;
	float xmin = log(0.5f * K);
	float xmax = log(2.0f * K);
	float dx = (xmax - xmin) / NTPB;
	float pmin = 0.5f * K;
	float pmax = 0.0f;
	float sigmin = 0.1f;
	float sigmax = 0.5f;
	float dsig = (sigmax - sigmin) / NB;


	MyTab* pt_CPU;
	testCUDA(cudaHostAlloc(&pt_CPU, sizeof(MyTab), cudaHostAllocDefault));
	for (int i = 0; i < NB; i++) {
		for (int j = 0; j < NTPB; j++) {
			pt_CPU[0][i][j] = max(0.0, K - exp(xmin + dx * j));
		}
	}

	PDE_diff(dt, dx, dsig, pmin, pmax, sigmin, r, N, pt_CPU);

	// S0 = 100 , sigma = 0.2
	printf(" %f, compare with %f\n", exp(-r * T) * pt_CPU[0][16][128],
		K * (exp(-r * T) * NP(-(r - 0.5 * 0.2 * 0.2) * sqrt(T) / 0.2) -
			NP(-(r + 0.5 * 0.2 * 0.2) * sqrt(T) / 0.2)));
	// S0 = 100 , sigma = 0.3
	printf(" %f, compare with %f\n", exp(-r * T) * pt_CPU[0][32][128],
		K * (exp(-r * T) * NP(-(r - 0.5 * 0.3 * 0.3) * sqrt(T) / 0.3) -
			NP(-(r + 0.5 * 0.3 * 0.3) * sqrt(T) / 0.3)));
	// S0 = 141.4214 , sigma = 0.3
	printf(" %f, compare with %f\n", exp(-r * T) * pt_CPU[0][32][192],
		K * exp(-r * T) * NP(-(log(141.4214 / K) + (r - 0.5 * 0.3 * 0.3) * T) / (0.3 * sqrt(T))) -
		141.4214 * NP(-(log(141.4214 / K) + (r + 0.5 * 0.3 * 0.3) * T) / (0.3 * sqrt(T))));

	testCUDA(cudaFreeHost(pt_CPU));
	return 0;
}