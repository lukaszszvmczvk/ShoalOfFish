#define GLM_FORCE_CUDA
#include "kernel.h"
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <glm/glm.hpp>

#include <thrust/gather.h>

unsigned int block_size = 256;

constexpr float max_speed = 20.0f;

constexpr float turn_factor = 1.f;

constexpr float height = 900.f;
constexpr float width = 1600.f;

constexpr float margin = 50;

glm::vec2* positions = nullptr;
glm::vec2* velocity1 = nullptr;
glm::vec2* velocity2 = nullptr;


__global__ void update_vel(glm::vec2* pos, glm::vec2* vel1, glm::vec2* vel2, unsigned int N, 
    float visualRange, float minDistance, float cohesion_scale, float separation_scale, float alignment_scale)
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
    
    float dx = vel1[index].x;
    float dy = vel1[index].y;

    for (int i = 0; i < N; ++i)
    {
        if (i == index)
            continue;
        float distance = glm::sqrt((pos[index].x - pos[i].x) * (pos[index].x - pos[i].x) +
            (pos[index].y - pos[i].y) * (pos[index].y - pos[i].y));
        if (distance < visualRange)
        {
            ++neighbour_count;
            centerX += pos[i].x;
            centerY += pos[i].y;
            avgDX += vel1[i].x;
            avgDY += vel1[i].y;

            if (distance < minDistance)
            {
                moveX += pos[index].x - pos[i].x;
                moveY += pos[index].y - pos[i].y;
            }
        }
    }

    if (neighbour_count > 0)
    {
        // rule1 Cohesion
        centerX = centerX / neighbour_count;
        centerY = centerY / neighbour_count;

        dx += (centerX - pos[index].x) * cohesion_scale;
        dy += (centerY - pos[index].y) * cohesion_scale;

        // rule2 Separation
        dx += moveX * separation_scale;
        dy += moveY * separation_scale;

        // rule3 Alignment
        avgDX = avgDX / neighbour_count;
        avgDY = avgDY / neighbour_count;

        dx += (avgDX - vel1[index].x) * alignment_scale;
        dy += (avgDY - vel1[index].y) * alignment_scale;
    }

    // keep fishes in bounds
    if (pos[index].x < margin)
        dx += turn_factor;
    if (pos[index].y < margin)
        dy += turn_factor;
    if (pos[index].x > width - margin)
        dx -= turn_factor;
    if (pos[index].y > height - margin)
        dy -= turn_factor;

    // check if speed is < max_speed
    float speed = glm::sqrt(dx * dx + dy * dy);
    if (speed > max_speed)
    {
        dx = (dx / speed) * max_speed;
        dy = (dy / speed) * max_speed;
    }

    // update velocities
    vel2[index].x = dx;
    vel2[index].y = dy;
}

__global__ void update_pos(glm::vec2* pos, glm::vec2* vel, unsigned int N)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) { return; }

    pos[index].x += vel[index].x;
    pos[index].y += vel[index].y;
}
void Boids::init_simulation(unsigned int N)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
    }

    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&positions), N * sizeof(glm::vec2));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&velocity1), N * sizeof(glm::vec2));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&velocity2), N * sizeof(glm::vec2));
    if(cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaDeviceSynchronize();
}
void Boids::end_simulation()
{
    cudaFree(positions);
    cudaFree(velocity1);
    cudaFree(velocity2);
}
void Boids::update_fishes(glm::vec2* pos, glm::vec2* vel, unsigned int N, float vr, float md, float r1, float r2, float r3)
{
    const dim3 full_blocks_per_grid((N + block_size - 1) /
        block_size);
    const dim3 threads_per_block(block_size);

    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(positions, pos, N * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaStatus = cudaMemcpy(velocity1, vel, N * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    update_vel << <full_blocks_per_grid, threads_per_block >> > (positions, velocity1, velocity2, N, vr, md, r1, r2, r3);
    cudaDeviceSynchronize();

    update_pos << <full_blocks_per_grid, threads_per_block >> > (positions, velocity2, N);
    cudaDeviceSynchronize();
    
    cudaStatus = cudaMemcpy(pos, positions, N * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaStatus = cudaMemcpy(vel, velocity2, N * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

}