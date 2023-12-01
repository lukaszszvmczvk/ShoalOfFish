#include<iostream>
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include "kernel.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "shaderClass.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "kernel.h"
#include "fish.h"

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900
#define N 100

GLFWwindow* window = nullptr;
GLuint VAO, VBO;
Shader shaderProgram;
Fish fishes[N];

void init_fishes()
{
	for (int i = 0; i < N; ++i)
	{
		float X = (static_cast<float>(rand() % static_cast<int>(WINDOW_WIDTH)) / WINDOW_WIDTH) * 2.0f - 1.0f;
		float Y = (static_cast<float>(rand() % static_cast<int>(WINDOW_HEIGHT)) / WINDOW_HEIGHT) * 2.0f - 1.0f;
		Species species;
		fishes[i] = Fish(X, Y, species);

		if (i % 3 == 0)
		{
			fishes[i].species.color = glm::vec3(0, 1, 0);
			fishes[i].species.size = 0.04;
			fishes[i].dx = 0.001;
		}
		else if (i % 3 == 1)
		{
			fishes[i].species.color = glm::vec3(0, 0, 1);
			fishes[i].species.size = 0.02;
			fishes[i].dy = 0.002;
		}
		else
		{
			fishes[i].dy = 0.0005;
			fishes[i].dx = 0.0003;
		}
	}
}
bool initialize()
{
	cudaDeviceProp deviceProp;
	int gpuDevice = 0;
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	if (gpuDevice > device_count) {
		std::cout << "Error: GPU device number is greater than the number of devices!";
		return false;
	}
	cudaGetDeviceProperties(&deviceProp, gpuDevice);

	// Initialize GLFW
	if (!glfwInit())
	{
		std::cout << "GLFW init failed";
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	// Create a GLFWwindow object
	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Shoal of fish", NULL, NULL);

	// Error check if the window fails to create
	if (window == NULL)
	{
		std::cout << "Failed to create window";
		glfwTerminate();
		return false;
	}

	// Introduce the window into the current context
	glfwMakeContextCurrent(window);

	gladLoadGL();

	shaderProgram = Shader("default.vert", "default.frag");
	init_fishes();
	Boids::init_simulation(N);

	return true;
}

void setup_triangles() 
{
	float vertices[3 * 5 * N];

	for (int i = 0; i < N; ++i)
	{
		Fish fish = fishes[i];
		float centerX = fish.x;
		float centerY = fish.y;

		float sideLength = fish.species.size;

		vertices[i * 3 * 5] = centerX;
		vertices[i * 3 * 5 + 1] = centerY + sideLength / sqrt(3);
		vertices[i * 3 * 5 + 2] = fish.species.color.r; vertices[i * 3 * 5 + 3] = fish.species.color.g; vertices[i * 3 * 5 + 4] = fish.species.color.b;

		vertices[i * 3 * 5 + 5] = centerX - sideLength / 2;
		vertices[i * 3 * 5 + 6] = centerY - sideLength / (2 * sqrt(3));
		vertices[i * 3 * 5 + 7] = fish.species.color.r; vertices[i * 3 * 5 + 8] = fish.species.color.g; vertices[i * 3 * 5 + 9] = fish.species.color.b;

		vertices[i * 3 * 5 + 10] = centerX + sideLength / 2;
		vertices[i * 3 * 5 + 11] = centerY - sideLength / (2 * sqrt(3));
		vertices[i * 3 * 5 + 12] = fish.species.color.r; vertices[i * 3 * 5 + 13] = fish.species.color.g; vertices[i * 3 * 5 + 14] = fish.species.color.b;
	}

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(sizeof(float) * 2));

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}
void update_triangles()
{
	float vertices[3 * 5 * N];

	for (int i = 0; i < N; ++i)
	{
		Fish fish = fishes[i];
		float centerX = fish.x;
		float centerY = fish.y;

		float sideLength = fish.species.size;

		vertices[i * 3 * 5] = centerX;
		vertices[i * 3 * 5 + 1] = centerY + sideLength / sqrt(3);
		vertices[i * 3 * 5 + 2] = fish.species.color.r; vertices[i * 3 * 5 + 3] = fish.species.color.g; vertices[i * 3 * 5 + 4] = fish.species.color.b;

		vertices[i * 3 * 5 + 5] = centerX - sideLength / 2;
		vertices[i * 3 * 5 + 6] = centerY - sideLength / (2 * sqrt(3));
		vertices[i * 3 * 5 + 7] = fish.species.color.r; vertices[i * 3 * 5 + 8] = fish.species.color.g; vertices[i * 3 * 5 + 9] = fish.species.color.b;

		vertices[i * 3 * 5 + 10] = centerX + sideLength / 2;
		vertices[i * 3 * 5 + 11] = centerY - sideLength / (2 * sqrt(3));
		vertices[i * 3 * 5 + 12] = fish.species.color.r; vertices[i * 3 * 5 + 13] = fish.species.color.g; vertices[i * 3 * 5 + 14] = fish.species.color.b;
	}

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}
void update_fish()
{
	Boids::update_fishes(fishes, N);
	update_triangles();
}
void draw_triangles()
{
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, 3 * N);
	glBindVertexArray(0);
}
void program_loop()
{
	setup_triangles();
	 
	// Main while loop
	while (!glfwWindowShouldClose(window))
	{
		// Take care of all GLFW events
		glfwPollEvents();

		// Specify the color of the background
		glClearColor(0.67f, 0.84f, 0.9f, 1.0f);
		// Clean the back buffer and assign the new color to it
		glClear(GL_COLOR_BUFFER_BIT);

		// Tell OpenGL which Shader Program we want to use
		shaderProgram.Activate();

		update_fish();
		draw_triangles();


		// Swap the back buffer with the front buffer
		glfwSwapBuffers(window);
	}

	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	shaderProgram.Delete();
	// Delete window before ending the program
	glfwDestroyWindow(window);
	// Terminate GLFW before ending the program
	glfwTerminate();
}
int main()
{
	if (initialize())
	{
		program_loop();
		return 0;
	}
	else
	{
		return -1;
	}
}