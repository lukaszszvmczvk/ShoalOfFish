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
Fish* fishes_gpu = nullptr;
Fish* fishes_gpu_sorted = nullptr;
int* indices = nullptr;
int* grid_cell_indices = nullptr;
glm::vec2* pos_sorted = nullptr;
glm::vec2* vel_sorted = nullptr;


__global__ void update_vel(glm::vec2* pos, glm::vec2* vel1, glm::vec2* vel2, Fish* fishes, int* grid_cell_indices, int* grid_cell_start, int* grid_cell_end,
    unsigned int N, float visualRange, float minDistance, float cohesion_scale, float separation_scale, float alignment_scale)
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

    int cell_index = grid_cell_indices[index];

    int row_cells = width / (2 * visualRange) + 1;
    int grid_size = row_cells * (height / (2 * visualRange) + 1);

    int neighbour_cells[] = { index + 1, index - 1, index - row_cells, index + row_cells, index + row_cells - 1, index + row_cells + 1,
                index - row_cells + 1, index - row_cells - 1 };

    for (int j = 0; j < 8; ++j)
    {
        int current_cell = neighbour_cells[j];

        if (current_cell < 0 || current_cell >= grid_size)
            continue;

        for (int i = grid_cell_start[current_cell]; i < grid_cell_end[current_cell]; ++i)
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

__global__ void assign_grid_cell(glm::vec2* pos, int* grid_cells, int* indices, float cell_width, unsigned int N)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) { return; }

    float x = pos[index].x;
    float y = pos[index].y;

    int x_size = width / cell_width + 1;

    int x_cell = x / cell_width;
    int y_cell = y / cell_width;

    grid_cells[index] = y_cell * x_size + x_cell;
    indices[index] = index;
}

__global__ void compute_start_end_cell(int* grid_cell_indices, int* grid_cell_start, int* grid_cell_end, int grid_size, unsigned int N)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= grid_size) { return; }

    int start = -1;
    int i = 0;
    while (i < N && grid_cell_indices[i] <= index)
    {
        if (start == -1 && grid_cell_indices[i] == index)
        {
            start = i;
        }
        ++i;
    }
    if (start == -1)
    {
        grid_cell_start[index] = -1;
        grid_cell_end[index] = -1;
    }
    else
    {
        grid_cell_start[index] = start;
        grid_cell_end[index] = i;
    }
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
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&pos_sorted), N * sizeof(glm::vec2));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&vel_sorted), N * sizeof(glm::vec2));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&grid_cell_indices), N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&indices), N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&fishes_gpu), N * sizeof(Fish));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&fishes_gpu_sorted), N * sizeof(Fish));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaDeviceSynchronize();
}
void Boids::end_simulation()
{
    cudaFree(positions);
    cudaFree(velocity1);
    cudaFree(velocity2);
    cudaFree(indices);
    cudaFree(grid_cell_indices);
    cudaFree(fishes_gpu);
    cudaFree(fishes_gpu_sorted);
}
void Boids::update_fishes(glm::vec2* pos, glm::vec2* vel, Fish* fishes, unsigned int N, float vr, float md, float r1, float r2, float r3)
{
    cudaError_t cudaStatus;
    
    const dim3 full_blocks_per_grid((N + block_size - 1) / block_size);
    const dim3 threads_per_block(block_size);

    float cell_width = 2 * vr;
    int grid_size = (width / cell_width + 1) * (height / cell_width + 1);
    const dim3 full_blocks_per_grid2((grid_size + block_size - 1) / block_size);

    // Allocate memory for start and end indices
    int* grid_cell_start;
    int* grid_cell_end;

    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&grid_cell_start), grid_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&grid_cell_end), grid_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // Copy data to gpu
    cudaStatus = cudaMemcpy(positions, pos, N * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaStatus = cudaMemcpy(velocity1, vel, N * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaStatus = cudaMemcpy(fishes_gpu, fishes, N * sizeof(Fish), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    // Asign grid cell to every fish
    assign_grid_cell << <full_blocks_per_grid, threads_per_block >> > (positions, grid_cell_indices, indices, cell_width, N);
    cudaDeviceSynchronize();

    // Cast arrays to perform thrust operations
    auto thrust_gci = thrust::device_pointer_cast(grid_cell_indices);
    auto thrust_i = thrust::device_pointer_cast(indices);
    auto thrust_p = thrust::device_pointer_cast(positions);
    auto thrust_v = thrust::device_pointer_cast(velocity1);
    auto thrust_ps = thrust::device_pointer_cast(pos_sorted);
    auto thrust_vs = thrust::device_pointer_cast(vel_sorted);
    auto thrust_f = thrust::device_pointer_cast(fishes_gpu);
    auto thrust_fs = thrust::device_pointer_cast(fishes_gpu_sorted);

    // Sort fishes indicies by grid cell
    thrust::sort_by_key(thrust_gci, thrust_gci + N, thrust_i);

    // Compute start and end indices of grid cell
    compute_start_end_cell << <full_blocks_per_grid2, threads_per_block >> > (grid_cell_indices, grid_cell_start, grid_cell_end, grid_size, N);
    cudaDeviceSynchronize();

    // Sort fish pos and vel by indices
    thrust::gather(thrust_i, thrust_i + N, thrust_p, thrust_ps);
    thrust::gather(thrust_i, thrust_i + N, thrust_v, thrust_vs);
    thrust::gather(thrust_i, thrust_i + N, thrust_f, thrust_fs);


    update_vel << <full_blocks_per_grid, threads_per_block >> > (pos_sorted, vel_sorted, velocity2, fishes_gpu_sorted, grid_cell_indices, grid_cell_start, grid_cell_end,
        N, vr, md, r1, r2, r3);
    cudaDeviceSynchronize();

    update_pos << <full_blocks_per_grid, threads_per_block >> > (pos_sorted, velocity2, N);
    cudaDeviceSynchronize();
    

    // Przerzucenie pamieci na cpu
    cudaStatus = cudaMemcpy(pos, pos_sorted, N * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaStatus = cudaMemcpy(vel, velocity2, N * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaStatus = cudaMemcpy(fishes, fishes_gpu_sorted, N * sizeof(Fish), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    // Zwolnienie pamiêci
    cudaFree(grid_cell_start);
    cudaFree(grid_cell_end);
}