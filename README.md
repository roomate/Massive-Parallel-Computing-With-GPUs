# Massive Parallel Computing with GPUs

This repository stores all the labs completed of the course [Massive Parallel Computing with GPUs](https://finance.math.upmc.fr/enseignements/2_mn_4_massive_parallel/) given at Sorbonne University between February and March 2026, as well as the project carried out for validation.

## Labs

- Lab1: *Device Query*. This lab teaches how to query the device to get the piece of information you need on your GPU.
- Lab2: *Hello World!* This lab is your first step in the world of cuda programming. Say your very first words from your device's processor.
- Lab3: *Add Array*. Add two arrays on you device, and clock the execution time with the CPU and GPU timer. *The Need for Speed!*
- Lab4: *Monte-Carlo*. Use your favourite Pseudo-Random Number Generator (PRNG) and compute expectations at light speed. Ideal to beat the financial markets.
    * Compute arithmetic mean directly on the host. It does not exploit the GPU's full power.
    * Compute arithmetic mean on-device. A code implementing Dyadic thread reduction using shared memory on the kernel is proposed.
- Lab5: *Partial Differential Equation*. Solve the backward Fokker–Planck equation using three different approaches with a finite-difference schemes. To a greater or lesser extent, each approach exploits the capabilities of the device in a different way.
    * Explicit Euler scheme.
    * Implicit Euler scheme.
    * Crank-Nicolson scheme, aka semi-implicit semi-explicit.


## Project *PDE Simulation of Bullet Option*



### Repository structure

```
.
├── Labs
│   ├── Lab1
│   │   ├── DevQuery.cu
│   │   └── Device_Query_Lab.ipynb
│   ├── Lab2
│   │   ├── HW_built_Lab.ipynb
│   │   └── HWbuilt.cu
│   ├── Lab3
│   │   ├── Add_timer_cpu.cu
│   │   ├── Add_timer_gpu.cu
│   │   ├── Array_Add_Lab.ipynb
│   │   └── timer.h
│   ├── Lab4
│   │   ├── MC.cu
│   │   ├── MC2.cu
│   │   ├── MC_Lab.ipynb
│   │   ├── NMC.cu
│   │   └── NMC_Lab.ipynb
│   └── Lab5
│       ├── Explicit_Lab.ipynb
│       └── PDE.cu
├── PDE-simulation-of-bullet-option
│   ├── project.cu
│   └── project.ipynb
└── README.md
```



### Acknowledgement

The project was carried out with Elio Moreau.
