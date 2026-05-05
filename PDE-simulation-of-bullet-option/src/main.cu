#include <stdio.h>
#include "wrappers.cuh"
#include "utils.cuh"
#include <assert.h>

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
      assert (argc==5);
      float x=atof(argv[2]);
      int j=atoi(argv[3]);
      int i=atoi(argv[4]);
      if (i==0) {x=S0; j=0;}
      bool verbose=true;
      wrapper_1(x, i, j, T, r, sigma, K, B, P1, P2, verbose);
    }
    else if (mode==2)
    {
      printf("The command line must be used to pass S_Ti, j and Ti in this order.\n");
      assert (argc==5);
      float x=atof(argv[2]);
      int j=atoi(argv[3]);
      int i=atoi(argv[4]);
      if (i==0) {x=S0; j=0;}
      bool verbose=true;
      wrapper_2(x, i, j, T, r, sigma, K, B, P1, P2, verbose);
    }
    else if (mode==3)
    {
      assert (argc==3 || argc==2);
      char* filename;
      if (argc==3) {
        printf("The command line must be used to pass a filename.\n");
        filename=argv[2];
      }
      else {filename=nullptr;}
      wrapper_3(filename, T, r, sigma, S0, K, B, P1, P2);
    }
    else if (mode==4)
    {
      printf("It is going to take a lot of time!\n");
      wrapper_trash(T, r, sigma, S0, K, B, P1, P2);
    }
    else if (mode==5)
    {
      printf("The command line must be used to pass S_Ti, j and Ti in this order.\n");
      assert (argc==6);
      float x=atof(argv[2]);
      int j=atoi(argv[3]);
      int i=atoi(argv[4]);
      int Nb_sim=atoi(argv[5]);
      if (i==0) {x=S0; j=0;}
      bool verbose=false;
      float* Exec_time=(float*)malloc(sizeof(float)*Nb_sim);
      for (int k=0; i<Nb_sim; ++i)
      {
        Exec_time[k]=wrapper_1(x, i, j, T, r, sigma, K, B, P1, P2, verbose);
      }
      printf("Execution time is %.4f \u00b1 %.4f ms.\n", mean(Exec_time, Nb_sim), std_(Exec_time, Nb_sim));
    }

    else if (mode==6)
    {
      printf("S_Ti, j and Ti must be passed by the command line, in this order.\n");
      assert (argc==6);
      float x=atof(argv[2]);
      int j=atoi(argv[3]);
      int i=atoi(argv[4]);
      int Nb_sim=atoi(argv[5]);
      if (i==0) {x=S0; j=0;}
      bool verbose=false;
      float* Exec_time=(float*)malloc(sizeof(float)*Nb_sim);
      for (int k=0; i<Nb_sim; ++i)
      {
        Exec_time[k]=wrapper_2(x, i, j, T, r, sigma, K, B, P1, P2, verbose);
      }
      printf("Execution time is %.4f \u00b1 %.4f ms.\n", mean(Exec_time, Nb_sim), std_(Exec_time, Nb_sim));
    }
    else
    {
      printf("The command line must be used to pass a mode between 1 and 4 included.");
      exit(1);
    }
    return 0;
}