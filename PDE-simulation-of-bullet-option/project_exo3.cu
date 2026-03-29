#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EPS 0.0000001f
#define NTPB 256
#define NB 64  // for j=0 to 40

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

	// Reorder unknowns so each PCR stage combines alternating neighbors.
	d = (n / 2 + (n % 2)) * (threadIdx.x % 2) + (int)threadIdx.x / 2;

	tL = threadIdx.x - 1;
	if (tL < 0) tL = 0;
	tR = threadIdx.x + 1;
	if (tR >= n) tR = 0;

	// Each iteration removes one more level of off-diagonal dependence.
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

	// Write the solved value back to its original row index.
	sy[(int)sl[threadIdx.x]] = yL / dL;
}


/////////////////////////////////////////////////////////////////////////////
// Crank-Nicolson solution
/////////////////////////////////////////////////////////////////////////////
__global__ void PDE_diff_kernel (float dt, float dx, float pmin, 
							 float r, int N, int P1, int P2, float K, MyTab *pt_GPU){


	int i;
	// Each block handles one j-layer, and each thread handles one x-grid node.
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

	// Shared memory stores the tridiagonal system for one fixed j.
	extern __shared__ float A[];
	float* sa = A;
	float* sd = sa + NTPB;
	float* sc = sd + NTPB;
	float* sy = sc + NTPB;
	int* sl = (int*)sy + 2 * NTPB;
	
	// Initial condition for this interval: the value at the later time level.
	sy[m] = pt_GPU[0][j][m];
	__syncthreads();


	for (i = 1; i<=N; i++) {
		// Explicit half-step of the Crank-Nicolson scheme.
		sy[NTPB * (i % 2) + m] = (m == 0) * pmin + (m == NTPB - 1) * pmax +
			((m > 0) && (m < NTPB - 1)) * (pu * sy[NTPB * ((i + 1) % 2) + u] +
			pm * sy[NTPB * ((i + 1) % 2) + m] + pd * sy[NTPB * ((i + 1) % 2) + d]);
			
		// Build the implicit tridiagonal system for the same time step.
		sd[m] = ((m == 0) || (m == NTPB - 1)) ? 1.0f : qm;
		sa[m] = (m > 0 && m < NTPB - 1) ? qd : 0.0f;
		sc[m] = (m > 0 && m < NTPB - 1) ? qu : 0.0f;
		sy[NTPB * (i % 2) + m] = (m == 0) * pmin + (m == NTPB - 1) * pmax +
			((m > 0) && (m < NTPB - 1)) * sy[NTPB * (i % 2) + m];
		sl[m] = m;
		__syncthreads();
		// Solve the implicit system in parallel across the x-grid.
		PCR_d(sa, sd, sc, sy + NTPB * (i % 2), sl, NTPB);
		__syncthreads();
		sy[NTPB * (i % 2) + m] = (m == 0) * pmin + (m == NTPB - 1) * pmax +
			((m > 0) && (m < NTPB - 1)) * sy[NTPB * (i % 2) + m];
		__syncthreads();
	}
	// Store the solution at the beginning of the interval.
	pt_GPU[0][j][m] = sy[m + NTPB*(N % 2)];
}

__global__ void discontinuity_kernel(float xmin, float dx, float B,
									 int Pk1, int P2, MyTab *src, MyTab *dst) {

	int j = blockIdx.x;
	int m = threadIdx.x;
	float S = expf(xmin + dx * m);
	float value = 0.0f;

	// Apply the observation-date jump in j depending on whether S crosses the barrier.
	if (j == P2) {
		value = (S >= B) * src[0][P2][m];
	}
	else if (Pk1 > 0 && j == Pk1 - 1) {
		value = (S < B) * src[0][Pk1][m];
	}
	else if (j >= Pk1 && j < P2) {
		value = (S >= B) ? src[0][j][m] : src[0][j + 1][m];
	}

	dst[0][j][m] = value;
}


// Wrapper 
void PDE_diff (float dt, float dx, float pmin, float r, int N, int P1, int P2, float K, MyTab* CPUTab){

	float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

	// Move the whole (j, x) grid to the GPU for one PDE solve.
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

void PDE_full(float interval_dt, float dx, float xmin, float B, float pmin,
			  float r, int N, int M, int P1, int P2, float K, MyTab* CPUTab) {

	float TimeExec;
	cudaEvent_t start, stop;
	testCUDA(cudaEventCreate(&start));
	testCUDA(cudaEventCreate(&stop));
	testCUDA(cudaEventRecord(start, 0));

	// Two buffers are needed because the discontinuity update reads j and j+1.
	MyTab *GPUTab;
	MyTab *TmpTab;
	testCUDA(cudaMalloc(&GPUTab, sizeof(MyTab)));
	testCUDA(cudaMalloc(&TmpTab, sizeof(MyTab)));
	testCUDA(cudaMemcpy(GPUTab, CPUTab, sizeof(MyTab), cudaMemcpyHostToDevice));

	float dt = interval_dt / N;

	// March backward from T to T_0 by alternating PDE propagation and jump updates.
	for (int k = 0; k <= M; k++) {
		int Pk1 = P1 - k;
		if (Pk1 < 0) Pk1 = 0;

		PDE_diff_kernel<<<NB, NTPB, 6 * NTPB * sizeof(float)>>>(
			dt, dx, pmin, r, N, Pk1, P2, K, GPUTab);

		discontinuity_kernel<<<NB, NTPB>>>(xmin, dx, B, Pk1, P2, GPUTab, TmpTab);

		// The updated grid becomes the input for the next time interval.
		MyTab *SwapTab = GPUTab;
		GPUTab = TmpTab;
		TmpTab = SwapTab;
	}

	testCUDA(cudaMemcpy(CPUTab, GPUTab, sizeof(MyTab), cudaMemcpyDeviceToHost));

	testCUDA(cudaEventRecord(stop, 0));
	testCUDA(cudaEventSynchronize(stop));
	testCUDA(cudaEventElapsedTime(&TimeExec, start, stop));
	testCUDA(cudaEventDestroy(start));
	testCUDA(cudaEventDestroy(stop));

	printf("GPU time execution for full PDE solve: %f ms\n", TimeExec);

	testCUDA(cudaFree(GPUTab));
	testCUDA(cudaFree(TmpTab));
}

int main(int argc, char* argv[]){

	if (argc < 3) {
		printf("Usage:\n");
		printf("  %s query <N> <j> <S>\n", argv[0]);
		printf("  %s dump <N> <output.txt>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	// Model and contract parameters from the project statement.
	float K = 100.0f;
	float T = 1.0f;
	float r = 0.1f;
	float B = 110.0f;
	int M = 100;
	int N = atoi(argv[2]);
	float interval_dt = T / (M+1);
	float dt = interval_dt / N;
	float xmin = log(2.0f * K / 5.0f);
	float xmax = log(5.0f * K / 2.0f);
	float dx = (xmax - xmin) / (NTPB - 1);
	float pmin = 0.0f;
	int P1 = 10;
	int P2 = 40;
	

	MyTab *pt_CPU;
	testCUDA(cudaHostAlloc(&pt_CPU, sizeof(MyTab), cudaHostAllocDefault));
	// Terminal payoff u(T, x, j) = (S - K)+ * 1_{j in [P1, P2]}.
	for(int i=0; i<NB; i++){
	   int j_val = i;
	   for(int j=0; j<NTPB; j++){
	      float S = exp(xmin + dx * j);
	      pt_CPU[0][i][j] = (j_val >= P1 && j_val <= P2) * fmaxf(S - K, 0.0f); /*Set terminal conditions*/
	   }	
	}

	// Solve the full exercise 3 problem by going backward through all dates.
	PDE_full(interval_dt, dx, xmin, B, pmin, r, N, M, P1, P2, K, pt_CPU);

	if (strcmp(argv[1], "query") == 0) {
		if (argc != 5) {
			printf("Usage: %s query <N> <j> <S>\n", argv[0]);
			testCUDA(cudaFreeHost(pt_CPU));
			return EXIT_FAILURE;
		}

		int j_mc = atoi(argv[3]);
		float s_target = atof(argv[4]);
		if (j_mc < 0) j_mc = 0;
		if (j_mc >= NB) j_mc = NB - 1;

		// Project the requested asset value onto the nearest log-price grid point.
		int x_idx = (int)lroundf((logf(s_target) - xmin) / dx);
		if (x_idx < 0) x_idx = 0;
		if (x_idx >= NTPB) x_idx = NTPB - 1;

		float s_grid = expf(xmin + dx * x_idx);
		float u_value = pt_CPU[0][j_mc][x_idx];
		// Convert back from u(0, x, j) to the option price F(0, S, j).
		float f_pde = u_value * expf(-r * T);

		printf("Full PDE solve uses %d substeps per interval with dt = %.6f\n", N, dt);
		printf("PDE result for T_0 = 0.000000, j=%d, S~=%.2f (grid S=%.6f): F = %f\n",
			j_mc, s_target, s_grid, f_pde);
	}
	else if (strcmp(argv[1], "dump") == 0) {
		if (argc != 4) {
			printf("Usage: %s dump <N> <output.txt>\n", argv[0]);
			testCUDA(cudaFreeHost(pt_CPU));
			exit(EXIT_FAILURE);
		}

		FILE *fp = fopen(argv[3], "w");
		if (fp == NULL) {
			printf("Could not open output file %s\n", argv[3]);
			testCUDA(cudaFreeHost(pt_CPU));
			exit(EXIT_FAILURE);
		}

		fprintf(fp, "# t=0.000000 j S F\n");
		for (int j = 0; j < NB; j++) {
			for (int x = 0; x < NTPB; x++) {
				float S = expf(xmin + dx * x);
				float F = pt_CPU[0][j][x] * expf(-r * T);
				fprintf(fp, "%d %.8f %.8f\n", j, S, F);
			}
		}

		fclose(fp);
		printf("Wrote PDE grid to %s\n", argv[3]);
	}
	else {
		printf("Unknown mode '%s'\n", argv[1]);
		printf("Usage:\n");
		printf("  %s query <N> <j> <S>\n", argv[0]);
		printf("  %s dump <N> <output.txt>\n", argv[0]);
		testCUDA(cudaFreeHost(pt_CPU));
		exit(EXIT_FAILURE);
	}

	testCUDA(cudaFreeHost(pt_CPU));	
	return 0;
}
