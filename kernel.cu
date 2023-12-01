#define GLM_FORCE_CUDA
#include "kernel.h"
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <glm/glm.hpp>

#include <thrust/gather.h>

unsigned int block_size = 256;
float r = 0.01f;


__global__ void update_pos(Fish* fishes, unsigned int N)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) { return; }

    fishes[index].x += fishes[index].dx;
    fishes[index].y += fishes[index].dy;

    for (int i = 0; i < N; ++i)
    {
        if (i == index)
            continue;
    }
}

void Boids::init_simulation(unsigned int N)
{
    Fish* fishes = nullptr;

    dim3 fullBlocksPerGrid((N + block_size - 1) / block_size);

    cudaMalloc(reinterpret_cast<void**>(&fishes), N * sizeof(Fish));

    cudaDeviceSynchronize();
}

void Boids::update_fishes(Fish* fishes, unsigned int N)
{
    const dim3 full_blocks_per_grid((N + block_size - 1) /
        block_size);
    const dim3 threads_per_block(block_size);

    Fish* fishes_gpu = 0;
    cudaSetDevice(0);
    cudaMalloc(reinterpret_cast<void**>(&fishes_gpu), N * sizeof(Fish));
    cudaMemcpy(fishes_gpu, fishes, N * sizeof(Fish), cudaMemcpyHostToDevice);
    update_pos << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu, N);

    cudaDeviceSynchronize();

    cudaMemcpy(fishes, fishes_gpu, N * sizeof(Fish), cudaMemcpyDeviceToHost);
}