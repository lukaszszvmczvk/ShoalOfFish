#pragma once

#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <vector>
#include "fish.h"

namespace Boids
{
	void init_simulation(unsigned int N);
	void end_simulation();
	void update_fishes(Fish* fishes, unsigned int N, float vr, float md, float r1, float r2, float r3);
}