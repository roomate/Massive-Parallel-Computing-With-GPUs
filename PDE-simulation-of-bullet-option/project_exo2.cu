#include <stdio.h>
#include <math.h>

#define EPS 0.0000001f
#define NTPB 256
#define NB 41  // for j=0 to 40

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


// Parallel cyclic reduction for implicit part
__device__ void PCR_d(float* sa, float* sd, float* sc,
	float* sy, int* sl, int n) {

	int i, lL, d, tL, tR;
	float aL, dL, cL, yL;
	float aLp, dLp, cLp, yLp;

	d = (n / 2 + (n % 2)) * (threadIdx.x % 2) + (int)threadIdx.x / 2;

	tL = threadIdx.x - 1;
	if (tL < 0) tL = 0;
	tR = threadIdx.x + 1;
	if (tR >= n) tR = 0;

	for (i = 0; i < (int)(logf((float)n) / logf(2.0f)) + 1; i++) {
		lL = (int)sl[threadIdx.x];

		aL = sa[threadIdx.x];
		dL = sd[threadIdx.x];
		cL = sc[threadIdx.x];
		yL = sy[threadIdx.x];

		dLp = sd[tL];
		cLp = sc[tL];

		if (fabsf(aL) > EPS) {
			aLp = sa[tL];
			yLp = sy[tL];
			dL -= aL * cL / dLp;
			yL -= aL * yLp / dLp;
			aL = -aL * aLp / dLp;
			cL = -cLp * cL / dLp;
		}

		cLp = sc[tR];
		if (fabsf(cLp) > EPS) {
			aLp = sa[tR];
			dLp = sd[tR];
			yLp = sy[tR];
			dL -= cLp * aLp / dLp;
			yL -= cLp * yLp / dLp;
		}
		__syncthreads();

		if (i < (int)(logf((float)n) / logf(2.0f))) {
			sa[d] = aL;
			sd[d] = dL;
			sc[d] = cL;
			sy[d] = yL;
			sl[d] = (int)lL;
			__syncthreads();
		}
	}

	sy[(int)sl[threadIdx.x]] = yL / dL;
}


/////////////////////////////////////////////////////////////////////////////
// Crank-Nicolson solution
/////////////////////////////////////////////////////////////////////////////
__global__ void PDE_diff_kernel (float dt, float dx, float pmin, 
							 float r, int N, int P1, int P2, float K, MyTab *pt_GPU){


	int i;
	int u = (threadIdx.x + 1) * (threadIdx.x < NTPB - 1);
	int m = threadIdx.x;
	int d = (threadIdx.x - 1) * (threadIdx.x > 0);
	int j = blockIdx.x;
	float sig = 0.2f;
	float mu = r - 0.5f * sig * sig;
	float pmax = (j >= P1 && j <= P2) ? 1.5f * K : 0.0f;
	float pu = 0.25f * (sig * sig * dt / (dx * dx) + mu * dt / dx);
	float pm = 1.0f - 0.5 * sig * sig * dt / (dx * dx);
	float pd = 0.25f * (sig * sig * dt / (dx * dx) - mu * dt / dx);
	float qu = -0.25f * (sig * sig * dt / (dx * dx) + mu * dt / dx);
	float qm = 1.0f + 0.5 * sig * sig * dt / (dx * dx);
	float qd = -0.25f * (sig * sig * dt / (dx * dx) - mu * dt / dx);

	extern __shared__ float A[];
	float* sa = A;
	float* sd = sa + NTPB;
	float* sc = sd + NTPB;
	float* sy = sc + NTPB;
	int* sl = (int*)sy + 2 * NTPB;
	
	sy[m] = pt_GPU[0][j][m];
	__syncthreads();


	for (i = 1; i<=N; i++) {
		// explicit part
		sy[NTPB * (i % 2) + m] = (m == 0) * pmin + (m == NTPB - 1) * pmax +
			((m > 0) && (m < NTPB - 1)) * (pu * sy[NTPB * ((i + 1) % 2) + u] +
			pm * sy[NTPB * ((i + 1) % 2) + m] + pd * sy[NTPB * ((i + 1) % 2) + d]);
			
		// implicit part
		sd[m] = ((m == 0) || (m == NTPB - 1)) ? 1.0f : qm;
		sa[m] = (m > 0 && m < NTPB - 1) ? qd : 0.0f;
		sc[m] = (m > 0 && m < NTPB - 1) ? qu : 0.0f;
		sy[NTPB * (i % 2) + m] = (m == 0) * pmin + (m == NTPB - 1) * pmax +
			((m > 0) && (m < NTPB - 1)) * sy[NTPB * (i % 2) + m];
		sl[m] = m;
		__syncthreads();
		PCR_d(sa, sd, sc, sy + NTPB * (i % 2), sl, NTPB);
		__syncthreads();
		sy[NTPB * (i % 2) + m] = (m == 0) * pmin + (m == NTPB - 1) * pmax +
			((m > 0) && (m < NTPB - 1)) * sy[NTPB * (i % 2) + m];
		__syncthreads();
	}
	
	pt_GPU[0][j][m] = sy[m + NTPB*(N % 2)];
}



// Wrapper 
void PDE_diff (float dt, float dx, float pmin, float r, int N, int P1, int P2, float K, MyTab* CPUTab){

	float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

	MyTab *GPUTab;
	testCUDA(cudaMalloc(&GPUTab, sizeof(MyTab)));
	
	testCUDA(cudaMemcpy(GPUTab, CPUTab, sizeof(MyTab), cudaMemcpyHostToDevice));
	
	PDE_diff_kernel<<<NB, NTPB, 6*NTPB * sizeof(float)>>>(dt, dx, pmin, r, N, P1, P2, K, GPUTab);
	
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

int main(int argc, char* argv[]){

	float K = 100.0f;
	float T = 1.0f;
	float r = 0.1f;
	int M = 100;
	int N = 100; // Default number of Crank-Nicolson substeps on [T_{M-1}, T)
	if (argc >= 2) {
		N = atoi(argv[1]);
		if (N <= 0) {
			printf("The number of Crank-Nicolson substeps must be positive.\n");
			return EXIT_FAILURE;
		}
	}
	float interval_dt = T / M;
	float dt = interval_dt / N;
	float t_m_minus_1 = T - interval_dt;
	float xmin = log(2.0f * K / 5.0f);
	float xmax = log(5.0f * K / 2.0f);
	float dx = (xmax - xmin) / (NTPB - 1);
	float pmin = 0.0f;
	int P1 = 10;
	int P2 = 40;
	

	MyTab *pt_CPU;
	testCUDA(cudaHostAlloc(&pt_CPU, sizeof(MyTab), cudaHostAllocDefault));
	for(int i=0; i<NB; i++){
	   int j_val = i;
	   for(int j=0; j<NTPB; j++){
	      float S = exp(xmin + dx * j);
	      pt_CPU[0][i][j] = (j_val >= P1 && j_val <= P2) * fmaxf(S - K, 0.0f);
	   }	
	}

	PDE_diff(dt, dx, pmin, r, N, P1, P2, K, pt_CPU);

	// Report F(T_{M-1}, x, j) near the ATM point for comparison with Monte Carlo.
	int j_mc = 20;
	float s_target = 100.0f;
	int x_idx = (int)lroundf((logf(s_target) - xmin) / dx);
	if (x_idx < 0) x_idx = 0;
	if (x_idx >= NTPB) x_idx = NTPB - 1;
	float s_grid = expf(xmin + dx * x_idx);
	float u_value = pt_CPU[0][j_mc][x_idx];
	float f_pde = u_value * expf(-r * interval_dt);

	printf("Crank-Nicolson uses %d substeps on [%.6f, %.6f] with dt = %.6f\n",
		N, t_m_minus_1, T, dt);
	printf("PDE result for T_{M-1} = %.6f, j=%d, S~=%.2f (grid S=%.6f): F = %f\n",
		t_m_minus_1, j_mc, s_target, s_grid, f_pde);

	testCUDA(cudaFreeHost(pt_CPU));	
	return 0;
}
