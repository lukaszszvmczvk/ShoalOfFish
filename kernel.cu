#define GLM_FORCE_CUDA
#include "kernel.h"
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <glm/glm.hpp>
#include "fish.h"

#include <thrust/gather.h>


void Boids::init_simulation(unsigned int N)
{
    Fish* fishes = nullptr;
    constexpr unsigned int block_size = 256;
    unsigned int fish_count = 0;
        fish_count = N;
    dim3 fullBlocksPerGrid((N + block_size - 1) / block_size);

    cudaMalloc(reinterpret_cast<void**>(&fishes), N * sizeof(Fish));

    cudaDeviceSynchronize();
}