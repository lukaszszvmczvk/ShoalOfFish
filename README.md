# ShoalOfFish

**ShoalOfFish** is a GPU-accelerated simulation of a fish school using the **Boid Algorithm**. The project demonstrates how **parallel computing with CUDA** can efficiently simulate the complex flocking behavior of fish by applying the three core rules of the Boid Algorithm: **separation, alignment, and cohesion**. It's also extended by a feature where the fish avoid the cursor.

---

## How It Works
The simulation models individual fish (boids) as agents that interact based on simple local rules:
1. **Separation**: Avoid crowding nearby fish.
2. **Alignment**: Steer towards the average direction of neighboring fish.
3. **Cohesion**: Move towards the average position of nearby fish.

By updating these rules in parallel for all fish using **CUDA kernels**, the simulation achieves high performance even for large fish schools.

---

## Tech Stack
- **Language**: C++
- **GPU Framework**: CUDA

---

## Presentation

![](example.gif)
