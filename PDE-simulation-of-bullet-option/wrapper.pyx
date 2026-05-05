cdef extern from "./include/wrapper.cuh":
    #3D grids, 1D blocks. The x-axis is associated to Ti, y-axis with (S_ti, j) and z-axis with estimate of F(T_i, S_Ti, j)
    void wrapper_3(char filename[], float T, float r, float sigma, float S0, float K, float B, float P1, float P2)

def wrapper(bytes filename, float T, float r, float sigma, float S0, float K, float B, float P1, float P2):
    wrapper_3(filename, T, r, sigma, S0, K, B, P1, P2)