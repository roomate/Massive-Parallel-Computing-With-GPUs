# Massive Parallel Computing with GPUs

This repository stores all the labs completed of the course [Massive Parallel Computing with GPUs](https://finance.math.upmc.fr/enseignements/2_mn_4_massive_parallel/) given at Sorbonne University between February and March 2026, as well as the project carried out for validation.

## Labs

- Lab1: *Device Query*. This lab teaches how to query the device to get the piece of information you need on your GPU.
- Lab2: *Hello World!* This lab is your first step in the world of CUDA programming. Say your very first words through your device's processor.
- Lab3: *Add Array*. Add two arrays on you device, and clock the execution time with the CPU and GPU timer. *The Need for Speed!*
- Lab4: *Monte-Carlo*. Use your favourite Pseudo-Random Number Generator (PRNG) and compute expectations at light speed. Ideal to beat the financial markets. You can compute the arithmetic mean either:
    * on the host. It does not exploit the GPU's full power.
    * on-device. A code implementing Dyadic thread reduction using shared memory on the kernel is proposed.
- Lab5: *Partial Differential Equation*. Solve the backward Fokker–Planck equation using three different approaches with a finite-difference schemes. To a greater or lesser extent, each approach leverages the capabilities of the device in a different way.
    * Explicit Euler scheme.
    * Implicit Euler scheme.
    * Crank-Nicolson scheme, aka semi-implicit semi-explicit.


## Project *PDE Simulation of Bullet Option*

The project aims to use the parallelisation capacity of GPUs for bullet option pricing. This is a highly constrained and path-dependant contract, a estimation of good quality would require around $10^6$ samples per estimate.

### Repository structure

```
.
├── Labs
│   ├── Lab1
│   │   ├── DevQuery.cu
│   │   └── Device_Query_Lab.ipynb
│   ├── Lab2
│   │   ├── HW_built_Lab.ipynb
│   │   └── HWbuilt.cu
│   ├── Lab3
│   │   ├── Add_timer_cpu.cu
│   │   ├── Add_timer_gpu.cu
│   │   ├── Array_Add_Lab.ipynb
│   │   └── timer.h
│   ├── Lab4
│   │   ├── MC.cu
│   │   ├── MC2.cu
│   │   ├── MC_Lab.ipynb
│   │   ├── NMC.cu
│   │   └── NMC_Lab.ipynb
│   └── Lab5
│       ├── Explicit_Lab.ipynb
│       └── PDE.cu
├── PDE-simulation-of-bullet-option
│   ├── MC.cu
│   ├── MC.cuh
│   ├── Makefile
│   ├── main.cu
│   ├── plot.py
│   ├── project.ipynb
│   ├── project_exo2.cu
│   ├── project_exo3.cu
│   ├── test.py
│   ├── utils.cu
│   ├── utils.cuh
│   ├── wrappers.cu
│   └── wrappers.cuh
└── README.md```



## Acknowledgement

The project was carried out with Elio Moreau.
