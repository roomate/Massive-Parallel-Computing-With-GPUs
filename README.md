# Massive Parallel Computing with GPUs

This repository stores all the labs completed of the course [Massive Parallel Computing with GPUs](https://finance.math.upmc.fr/enseignements/2_mn_4_massive_parallel/) given at Sorbonne University between February and March 2026, as well as the project carried out for validation.

## Project *PDE Simulation of Bullet Option*

### Context

The project aims to utilise **the parallelisation capabilities of GPUs to price bullet options**, which are exotic, path-dependent contracts. Due to the highly constrained nature of option values, computing precise estimates demands high computational power, which is where GPUs come in. This is a scientifically challenging and relevant project demanding:

- The ability to write complex algorithms in the C programming language.
- Design and implement a parallelisation strategy that leverages GPU hardware.
- Leveraging low-level asynchronous interactions between the host and device with the CUDA runtime API.
- Assess the quality of a stochastic and deterministic methods.

### Visualization

To visualize the price surface plot, you first need to run:
```
#Run the compilation routine
make

# Run Monte-Carlo algorithm
./MC 3 <filename.txt>
```

You should see a text file appearing in the working directory. Eventually, run with your Python interpreter
```
python plot.py <filename.txt>
```

to plot a surface of the option price against its two conditioning parameters. You can tune the time instant with the vertical slider on the left.

### Algorithms

Two numerical solutions are proposed:

- Monte-Carlo algorithm. The device grid parallelises a large number of asset price trajectories and combines them to provide an option price Monte-Carlo estimator.

- The finite-difference scheme. The Black-Scholes partial differential equation (i.e. a backward Fokker–Planck equation) is solved using a Crank–Nicolson finite-difference scheme that leverages the Parallel Cyclic Reduction algorithm for efficient implementation.

The GPU timer, more precise and trustworthy than the CPU timer, measures the execution time. Both yield a solution within a comparable time frame of around a second.

## Labs

- Lab1: *Device Query*. This lab teaches how to query the device to get the piece of information you need on your GPU.
- Lab2: *Hello World!* This lab is your first step in the world of CUDA programming. Say your very first words through your device's processor.
- Lab3: *Add Array*. Add two arrays on you device, and clock the execution time with the CPU and GPU timer. *The Need for Speed!*
- Lab4: *Monte-Carlo*. Use your favourite Pseudo-Random Number Generator (PRNG) and compute expectations at light speed. Ideal to beat the financial markets. You can compute the arithmetic mean either:
    * on the host. It does not exploit the GPU's full power.
    * on-device. A code implementing Dyadic thread reduction using shared memory on the kernel is proposed.
- Lab5: *Partial Differential Equation*. Solve the backward Fokker–Planck equation using three different approaches with a finite-difference schemes. To a greater or lesser extent, each approach leverages the capabilities of the device in a different way.
    * Explicit Euler scheme.
    * Implicit Euler scheme with Thomas' method to solve the tridiagonal system. Despite simple, the main bottleneck is that it cannot be efficiently parallelised on GPUs.
    * Crank-Nicolson scheme, aka semi-implicit semi-explicit for better stability properties. Parallel-Cyclic Reduction algorithm is implemented to solve the resulting tridiagonal system. Although this algorithm is more complex, it lends itself much more naturally to parallelisation on GPU grids.

### Repository structure

```
.
├── Labs
│   ├── Lab1
│   │   ├── DevQuery.cu                         # Query your device
│   │   └── Device_Query_Lab.ipynb
│   ├── Lab2
│   │   ├── HW_built_Lab.ipynb                  # Your first 'Hello World!' in CUDA
│   │   └── HWbuilt.cu
│   ├── Lab3
│   │   ├── Add_timer_cpu.cu                    # Time your algorithms with the host clock
│   │   ├── Add_timer_gpu.cu                    # Time your algorithms with the device clock
│   │   ├── Array_Add_Lab.ipynb
│   │   └── timer.h                             # Header of the timer library
│   ├── Lab4
│   │   ├── MC.cu                               # Monte-Carlo algorithm
│   │   ├── MC2.cu
│   │   ├── MC_Lab.ipynb
│   │   ├── NMC.cu                              # Nested Monte-Carlo
│   │   └── NMC_Lab.ipynb
│   └── Lab5
│       ├── Crank_Lab.ipynb
│       ├── Explicit_Lab.ipynb
│       ├── Implicit_Lab.ipynb
│       ├── Optim_Lab.ipynb
│       ├── PDE_Crank.cu                    # Crank-Nicolson scheme
│       ├── PDE_explicit.cu                 # Euler explicit scheme
│       ├── PDE_implicit.cu                 # Euler implicit scheme
│       └── PDE_optim.cu                    # Uses shared memory instead of global memory
├── PDE-simulation-of-bullet-option
│   ├── HPC_GPU.pdf
│   ├── Makefile                            # Compilation routine
│   ├── include                             # Headers
│   │   ├── MC.cuh
│   │   ├── utils.cuh
│   │   └── wrappers.cuh
│   ├── plot.py                             # Routile to plot option price surface against S_Ti and I_Ti with a slider tuning Ti
│   └── src
│       ├── MC.cu                           # algorithms for sampling to estimate the option price
│       ├── PDE_full.cu                     # Solve the Black-Scholes equation over the whole interval [0, T]
│       ├── PDE_one_step.cu                 # Solve the Black-Scholes equation for a single step-size
│       ├── main.cu                         # main file
│       ├── utils.cu                        # Multiple utility functions
│       └── wrappers.cu                     # Wrapper around Monte-Carlo algorithms
└── README.md
```



## Acknowledgement

The project was carried out with Elio Moreau.
