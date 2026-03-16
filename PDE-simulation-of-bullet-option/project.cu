#include <stdio.h>
#include <curand_kernel.h>

__global__ void 

int main(int argc, char* argv[])
{
    // financial parameters
    float sigma=0.2; //Volatility
    float r=.1; //Risk-free return
    float S0=100; //Initial spot price
    float T=1; //Maturity
    float K=100; //Contract's strike
    
    //Parameters for numerical parameter
    float dt; //IMPORTANT: It is the square root of the simulation's step size.
    float B=110;
    unsigned int M=100;
    float P1=10;
    float P2=40;

    float* step=malloc(sizeof(float)*M);
    for (int i=0; i<M; ++i) {step[i]=i/M*T;}

    free(step);
}