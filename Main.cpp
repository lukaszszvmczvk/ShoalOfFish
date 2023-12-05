#include<iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
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
#include <ctime>
#include <imgui_impl_opengl3.h>
#include <chrono>

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900
#define N 200


// OpenGL variables
GLFWwindow* window = nullptr;
GLuint VAO, VBO;
Shader shaderProgram;

// Fishes info
Fish fishes[N];

// Initialzie fishes in random positions
void init_fishes()
{
	for (int i = 0; i < N; ++i)
	{
		for (int i = 0; i < N; ++i)
		{
			float X = (static_cast<float>(rand() % static_cast<int>(WINDOW_WIDTH)));
			float Y = (static_cast<float>(rand() % static_cast<int>(WINDOW_HEIGHT)));

			int x_vel = rand() % 5 + 1;
			int y_vel = rand() % 5 + 1;
			int x_rand = rand() % 2;
			int y_rand = rand() % 2;
			
			Species species;
			fishes[i] = Fish(X, Y, species);

			if (i % 3 == 0)
			{
				fishes[i].species.color = glm::vec3(0, 0.7f, 0);
				fishes[i].species.size = 10.f;
			}
			else if (i % 3 == 1)
			{
				fishes[i].species.color = glm::vec3(0, 0, 1);
				fishes[i].species.size = 20.f;
			}

			if (x_rand == 0)
				fishes[i].dx = x_vel;
			else
				fishes[i].dx = -x_vel;

			if (y_rand == 0)
				fishes[i].dy = y_vel;
			else
				fishes[i].dy = -y_vel;

		}
	}
}
// Initialize application
bool initialize()
{
	srand(static_cast<unsigned int>(time(0)));

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
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);

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

	// Initialize shaders
	shaderProgram = Shader("default");
	shaderProgram.bind();
	glm::mat4 proj = glm::ortho(0.f, static_cast<float>(WINDOW_WIDTH), 0.f, static_cast<float>(WINDOW_HEIGHT), -1.f, 1.f);
	shaderProgram.setUniformMat4fv("projection", proj);

	// Initialize ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	init_fishes();
	Boids::init_simulation(N);

	return true;
}
// Setup fishes pos on screen and pass to VBO 
void setup_fishes() 
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
// Update fishes pos on screen and pass to VBO
void update_fishes()
{
	float vertices[3 * 5 * N];

	Boids::copy_fishes(fishes, vertices, N);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}
// Run functions to update fishes
void update_fish(float vr, float md, float r1, float r2, float r3, float dt)
{
	Boids::update_fishes(fishes, N, vr, md, r1, r2, r3, dt);
	update_fishes();
}
// Draw fishes on screen
void draw_fishes()
{
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, 3 * N);
	glBindVertexArray(0);
}
// Main program loop
void program_loop()
{
	// Initalize basic values
	float visualRange = 35.f;
	float minDistance = 20.f;
	float cohesion_scale = 0.001f;
	float separation_scale = 0.05f;
	float alignment_scale = 0.05f;
	float dt = 0.7f;

	setup_fishes();

	// Main while loop
	while (!glfwWindowShouldClose(window))
	{
		// Start fps timer
		auto startFrameTime = std::chrono::high_resolution_clock::now();

		// Take care of all GLFW events
		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Specify the color of the background
		glClearColor(0.67f, 0.84f, 0.9f, 1.0f);
		// Clean the back buffer and assign the new color to it
		glClear(GL_COLOR_BUFFER_BIT);


		update_fish(visualRange, minDistance, cohesion_scale, separation_scale, alignment_scale, dt);
		draw_fishes();


		ImGui::Begin("Set properties");
		ImGui::SliderFloat("Visual range of fish", &visualRange, 5.0f, 100.0f);
		ImGui::SliderFloat("Min. separation distance", &minDistance, 0.0f, 50.0f);
		ImGui::SliderFloat("Cohesion rule scale", &cohesion_scale, 0.0f, 0.1f);
		ImGui::SliderFloat("Separation rule scale", &separation_scale, 0.0f, 0.1f);
		ImGui::SliderFloat("Alignment rule scale", &alignment_scale, 0.0f, 0.1f);
		ImGui::SliderFloat("Speed scale", &dt, 0.1f, 1.f);

		ImGui::End();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		// Swap the back buffer with the front buffer
		//glfwSwapBuffers(window);
		glFlush();

		// Count and display fps
		auto endFrameTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endFrameTime - startFrameTime).count();
		float fps = 1000000.0f / static_cast<float>(duration);
		std::cout << "FPS: " << fps << std::endl;
	}


	// Clear memory and end simulation
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	shaderProgram.unbind();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	Boids::end_simulation();

	// Delete window before ending the program
	glfwDestroyWindow(window);
	// Terminate GLFW before ending the program
	glfwTerminate();
}
// Main function
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