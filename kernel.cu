#define GLM_FORCE_CUDA
#include "kernel.h"
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <glm/glm.hpp>
#include <thrust/gather.h>

// Const variables
constexpr unsigned int block_size = 256;
constexpr float max_speed = 15.0f;
constexpr float turn_factor = 5.f;
constexpr float height = 900.f;
constexpr float width = 1600.f;
constexpr float margin = 50;


// Assign helpful variables
glm::vec2* velocity_buffer = nullptr;
Fish* fishes_gpu = nullptr;
Fish* fishes_gpu_sorted = nullptr;
unsigned int* indices = nullptr;
unsigned int* grid_cell_indices = nullptr;
float* vertices_array_gpu = nullptr;

// Compute velocity kernel function
__global__ void compute_vel(Fish* fishes, glm::vec2* vel2, unsigned int* grid_cell_indices, int* grid_cell_start, int* grid_cell_end,
    unsigned int N, float visualRange, float minDistance, float cohesion_scale, float separation_scale, float alignment_scale, unsigned int grid_size)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) { return; }

    // Assign helpful variables
    int neighbour_count = 0;
    float centerX = 0.f;
    float centerY = 0.f;
    float moveX = 0.f;
    float moveY = 0.f;
    float avgDX = 0.f;
    float avgDY = 0.f;
    
    float dx = fishes[index].dx;
    float dy = fishes[index].dy;

    // Find neighbours cells
    int cell_index = grid_cell_indices[index];
    int row_cells = width / (2 * visualRange) + 1;
    int neighbour_cells[] = {cell_index, cell_index + 1, cell_index - 1, cell_index + row_cells, cell_index - row_cells, cell_index - row_cells - 1,
                    cell_index - row_cells + 1, cell_index + row_cells - 1, cell_index + row_cells + 1 };
    // Go through neighbour grids
    for (int j = 0; j < 9; ++j)
    {
        int current_cell = neighbour_cells[j];

        if (current_cell < 0 || current_cell >= grid_size)
            continue;

        // Iterate through fishes from neighbour cell
        for (int i = grid_cell_start[current_cell]; i < grid_cell_end[current_cell]; ++i)
        {
            if (i == index)
                continue;

            // Check if fish is in distance
            float distance = glm::sqrt((fishes[index].x - fishes[i].x) * (fishes[index].x - fishes[i].x) +
                (fishes[index].y - fishes[i].y) * (fishes[index].y - fishes[i].y));
            if (distance < visualRange)
            {
                // Compute algorithm values
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
    }

    if (neighbour_count > 0)
    {
        // rule1 Cohesion
        centerX = centerX / neighbour_count;
        centerY = centerY / neighbour_count;

        dx += (centerX - fishes[index].x) * cohesion_scale;
        dy += (centerY - fishes[index].y) * cohesion_scale;

        // rule2 Separation
        dx += moveX * separation_scale;
        dy += moveY * separation_scale;

        // rule3 Alignment
        avgDX = avgDX / neighbour_count;
        avgDY = avgDY / neighbour_count;

        dx += (avgDX - fishes[index].dx) * alignment_scale;
        dy += (avgDY - fishes[index].dy) * alignment_scale;
    }

    // keep fishes in bounds
    if (fishes[index].x < margin)
        dx += turn_factor;
    if (fishes[index].y < margin)
        dy += turn_factor;
    if (fishes[index].x > width - margin)
        dx -= turn_factor;
    if (fishes[index].y > height - margin)
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

// Update pos and velocity kernel function
__global__ void update_pos_vel(Fish* fishes, glm::vec2* vel, unsigned int N, float dt)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) { return; }

    fishes[index].x += vel[index].x * dt;
    fishes[index].y += vel[index].y * dt;
    fishes[index].dx = vel[index].x;
    fishes[index].dy = vel[index].y;
}

// Assign grid cell to fish
__global__ void assign_grid_cell(Fish* fishes, unsigned int* grid_cells, unsigned int* indices, float cell_width, unsigned int N)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) { return; }

    float x = fishes[index].x;
    float y = fishes[index].y;

    int x_size = width / cell_width + 1;

    int x_cell = x / cell_width;
    int y_cell = y / cell_width;

    grid_cells[index] = y_cell * x_size + x_cell;
    indices[index] = index;
}

// Compute start and end indices of cell
__global__ void compute_start_end_cell(unsigned int* grid_cell_indices, int* grid_cell_start, int* grid_cell_end, unsigned int N)
{
    const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) { return; }

    unsigned int grid_cell_id = grid_cell_indices[index];

    if (index == 0)
    {
        grid_cell_start[grid_cell_id] = 0;
        return;
    }
    unsigned int prev_grid_cell_id = grid_cell_indices[index - 1];
    if (grid_cell_id != prev_grid_cell_id)
    {
        grid_cell_end[prev_grid_cell_id] = index;
        grid_cell_start[grid_cell_id] = index;
        if (index == N - 1) 
        { 
            grid_cell_end[grid_cell_id] = index;
        }
    }
}

__global__ void copy_fishes_kernel(Fish* fishes, float* vertices, unsigned int N)
{
    const auto i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= N) { return; }

    Fish fish = fishes[i];
    float centerX = fish.x;
    float centerY = fish.y;

    float sideLength = fish.species.size;

    glm::vec2 A = glm::vec2(centerX, centerY + sideLength / 1.73);
    glm::vec2 B = glm::vec2(centerX - sideLength / 2, centerY - sideLength / (2 * 1.73));
    glm::vec2 C = glm::vec2(centerX + sideLength / 2, centerY - sideLength / (2 * 1.73));

    vertices[i * 3 * 5] = A.x;
    vertices[i * 3 * 5 + 1] = A.y;
    vertices[i * 3 * 5 + 2] = fish.species.color.r; vertices[i * 3 * 5 + 3] = fish.species.color.g; vertices[i * 3 * 5 + 4] = fish.species.color.b;

    vertices[i * 3 * 5 + 5] = B.x;
    vertices[i * 3 * 5 + 6] = B.y;
    vertices[i * 3 * 5 + 7] = fish.species.color.r; vertices[i * 3 * 5 + 8] = fish.species.color.g; vertices[i * 3 * 5 + 9] = fish.species.color.b;

    vertices[i * 3 * 5 + 10] = C.x;
    vertices[i * 3 * 5 + 11] = C.y;
    vertices[i * 3 * 5 + 12] = fish.species.color.r; vertices[i * 3 * 5 + 13] = fish.species.color.g; vertices[i * 3 * 5 + 14] = fish.species.color.b;

}

// Allocate needed memory
void Boids::init_simulation(unsigned int N)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
    }

    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&velocity_buffer), N * sizeof(glm::vec2));
    if(cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&grid_cell_indices), N * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&indices), N * sizeof(unsigned int));
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
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&vertices_array_gpu), 15 * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaDeviceSynchronize();
}
// Dellocate memory
void Boids::end_simulation()
{
    cudaFree(velocity_buffer);
    cudaFree(indices);
    cudaFree(grid_cell_indices);
    cudaFree(fishes_gpu);
    cudaFree(fishes_gpu_sorted);
    cudaFree(vertices_array_gpu);
}
// Function to upddate fish pos and vel
void Boids::update_fishes(Fish* fishes, unsigned int N, float vr, float md, float r1, float r2, float r3, float dt)
{
    cudaError_t cudaStatus;
    
    const dim3 full_blocks_per_grid((N + block_size - 1) / block_size);
    const dim3 threads_per_block(block_size);

    float cell_width = 2 * vr;
    unsigned int grid_size = (width / cell_width + 1) * (height / cell_width + 1);
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
    cudaStatus = cudaMemcpy(fishes_gpu, fishes, N * sizeof(Fish), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    // Asign grid cell to every fish
    assign_grid_cell << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu, grid_cell_indices, indices, cell_width, N);
    cudaDeviceSynchronize();


    // Cast arrays to perform thrust operations
    auto thrust_gci = thrust::device_pointer_cast(grid_cell_indices);
    auto thrust_i = thrust::device_pointer_cast(indices);
    auto thrust_f = thrust::device_pointer_cast(fishes_gpu);
    auto thrust_fs = thrust::device_pointer_cast(fishes_gpu_sorted);

    // Sort fishes indicies by grid cell
    thrust::sort_by_key(thrust_gci, thrust_gci + N, thrust_i);

    // Compute start and end indices of grid cell
    compute_start_end_cell << <full_blocks_per_grid, threads_per_block >> > (grid_cell_indices, grid_cell_start, grid_cell_end, N);
    cudaDeviceSynchronize();

    // Sort fish pos and vel by indices
    thrust::gather(thrust_i, thrust_i + N, thrust_f, thrust_fs);


    // Update velocity
    compute_vel << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu_sorted, velocity_buffer, grid_cell_indices, grid_cell_start, grid_cell_end,
        N, vr, md, r1, r2, r3, grid_size);
    cudaDeviceSynchronize();


    // Update position
    update_pos_vel << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu_sorted, velocity_buffer, N, dt);
    cudaDeviceSynchronize();
    

    // Przerzucenie pamieci na cpu
    cudaStatus = cudaMemcpy(fishes, fishes_gpu_sorted, N * sizeof(Fish), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    // Zwolnienie pami�ci
    cudaFree(grid_cell_start);
    cudaFree(grid_cell_end);
}

void Boids::copy_fishes(Fish* fishes, float* vertices_array, unsigned int N)
{
    cudaError_t cudaStatus;

    const dim3 full_blocks_per_grid((N + block_size - 1) / block_size);
    const dim3 threads_per_block(block_size);

    // Copy data to gpu
    cudaStatus = cudaMemcpy(fishes_gpu, fishes, N * sizeof(Fish), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaStatus = cudaMemcpy(vertices_array_gpu, vertices_array, 15 * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    copy_fishes_kernel << <full_blocks_per_grid, threads_per_block >> > (fishes_gpu, vertices_array_gpu, N);
    cudaDeviceSynchronize();

    // Przerzucenie pamieci na cpu
    cudaStatus = cudaMemcpy(vertices_array, vertices_array_gpu, 15 * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

}