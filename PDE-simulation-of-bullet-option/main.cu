#include <stdio.h>
#include <curand_kernel.h>
#include "wrappers.cuh"
#include "utils.cuh"


int main(int argc, char* argv[])
{
    // financial parameters
    float sigma=.2; //Volatility
    float r=.1; //Risk-free return
    float S0=100; //Initial spot price
    float T=1; //Maturity
    float K=100; //Contract's strike
    float B=110; //Option's barrier
    float P1=10; //Lower bound of the interval
    float P2=40; //Upper bound of the interval
    
    float mode=atof(argv[1]);

    if (mode==1)
    {
      printf("The command line must be used to pass S_Ti, j and Ti in this order.\n");
      float x=atof(argv[2]);
      int j=atoi(argv[3]);
      int i=atoi(argv[4]);
      if (i==0) {x=S0; j=0;}
      wrapper_1(x, i, j, T, r, sigma, K, B, P1, P2);
    }
    else if (mode==2)
    {
      printf("The command line must be used to pass S_Ti, j and Ti in this order.\n");
      float x=atof(argv[2]);
      int j=atoi(argv[3]);
      int i=atoi(argv[4]);
      if (i==0) {x=S0; j=0;}
      wrapper_2(x, i, j, T, r, sigma, K, B, P1, P2);
    }
    else if (mode==3)
    {
      printf("The command line must be used to pass a filename.\n");
      char* filename=argv[2];
      wrapper_3(filename, T, r, sigma, S0, K, B, P1, P2);
    }
    else if (mode==4)
    {
      printf("It is going to take a lot of time!\n");
      wrapper_trash(T, r, sigma, S0, K, B, P1, P2);
    }
}