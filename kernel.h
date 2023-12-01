#pragma once

#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <vector>

namespace Boids
{
	void init_simulation(unsigned int N);
}