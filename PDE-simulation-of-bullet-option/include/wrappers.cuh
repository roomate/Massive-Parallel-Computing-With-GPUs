#ifndef wrapper_h
#define wrapper_h

/*Wrapper 1: Compute conditional expectation for a single triplet (i, x, j)*/
void wrapper_1(float x, int i, int j, float T, float r, float sigma, float K, float B, float P1, float P2);

/*Wrapper 2: Threads reduction within the kernel with shared memory (MC_k2) and registers (MC_k3)*/
void wrapper_2(float x, int i, int j, float T, float r, float sigma, float K, float B, float P1, float P2);

/*3D grids, 1D blocks. The x-axis is associated to Ti, y-axis with (S_ti, j) and z-axis with estimate of F(T_i, S_Ti, j).*/
void wrapper_3(char filename[], float T, float r, float sigma, float S0, float K, float B, float P1, float P2);

/*2D Grids, 1D blocks. Each row of block is associated with a Ti, and each thread samples its own couple (S_{T_i}, j).*/
/*In term of performance on GPU, it is pure trash. I keep it as an example.*/
void wrapper_trash(float T, float r, float sigma, float S0, float K, float B, float P1, float P2);

#endif

