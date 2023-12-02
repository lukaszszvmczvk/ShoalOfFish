#define GLM_FORCE_CUDA
#include "kernel.h"
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <glm/glm.hpp>

#include <thrust/gather.h>

unsigned int block_size = 256;

constexpr float max_speed = 15.0f;

constexpr float turn_factor = 1.f;

constexpr float height = 900.f;
constexpr float width = 1600.f;

constexpr float margin = 50;


__global__ void update_pos(Fish* fishes, unsigned int N, float visualRange, float minDistance, float rule1_scale, float rule2_scale, float rule3_scale)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) { return; }


    int neighbour_count = 0;
    float centerX = 0.f;
    float centerY = 0.f;
    float moveX = 0.f;
    float moveY = 0.f;
    float avgDX = 0.f;
    float avgDY = 0.f;
    
    float dx = fishes[index].dxP;
    float dy = fishes[index].dyP;

    for (int i = 0; i < N; ++i)
    {
        if (i == index)
            continue;
        float distance = glm::sqrt((fishes[index].x - fishes[i].x) * (fishes[index].x - fishes[i].x) +
            (fishes[index].y - fishes[i].y) * (fishes[index].y - fishes[i].y));
        if (distance < visualRange)
        {
            ++neighbour_count;
            centerX += fishes[i].x;
            centerY += fishes[i].y;
            avgDX += fishes[i].dx;
            avgDY += fishes[i].dy;

            if (distance < minDistance)
            {
                moveX += fishes[index].x - fishes[i].x;
                moveY += fishes[index].y - fishes[i].y;
            }
        }
    }

    if (neighbour_count > 0)
    {
        // rule1 Coherence
        centerX = centerX / neighbour_count;
        centerY = centerY / neighbour_count;

        dx += (centerX - fishes[index].x) * rule1_scale;
        dy += (centerY - fishes[index].y) * rule1_scale;

        // rule2 
        dx += moveX * rule2_scale;
        dy += moveY * rule2_scale;

        // rule3
        avgDX = avgDX / neighbour_count;
        avgDY = avgDY / neighbour_count;

        dx += (avgDX - fishes[index].dxP) * rule3_scale;
        dy += (avgDY - fishes[index].dyP) * rule3_scale;
    }


    float speed = glm::sqrt(dx * dx + dy * dy);
    if (speed > max_speed)
    {
        dx = (dx / speed) * max_speed;
        dy = (dy / speed) * max_speed;
    }

    if (fishes[index].x < margin)
        dx += turn_factor;
    if (fishes[index].y < margin)
        dy += turn_factor;
    if (fishes[index].x > width - margin)
        dx -= turn_factor;
    if (fishes[index].y > height - margin)
        dy -= turn_factor;

    fishes[index].dx = dx;
    fishes[index].dy = dy;

    fishes[index].x += fishes[index].dx;
    fishes[index].y += fishes[index].dy;
}

void Boids::init_simulation(unsigned int N)
{
    Fish* fishes = nullptr;

    dim3 fullBlocksPerGrid((N + block_size - 1) / block_size);

    cudaMalloc(reinterpret_cast<void**>(&fishes), N * sizeof(Fish));

    cudaDeviceSynchronize();
}

void Boids::update_fishes(Fish* fishes, unsigned int N, float vr, float md, float r1, float r2, float r3)
{
    const dim3 full_blocks_per_grid((N + block_size - 1) /
        block_size);
    const dim3 threads_per_block(block_size);

    Fish* fishes_gpu = 0;
    cudaSetDevice(0);
    cudaMalloc(reinterpret_cast<void**>(&fishes_gpu), N * sizeof(Fish));
    cudaMemcpy(fishes_gpu, fishes, N * sizeof(Fish), cudaMemcpyHostToDevice);
    update_pos << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu, N, vr, md, r1, r2, r3);

    cudaDeviceSynchronize();
    cudaMemcpy(fishes, fishes_gpu, N * sizeof(Fish), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
    {
        fishes[i].dxP = fishes[i].dx;
        fishes[i].dyP = fishes[i].dy;
    }
}