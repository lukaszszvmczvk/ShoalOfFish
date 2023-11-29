#include<iostream>
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include "kernel.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "shaderClass.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900
#define N 500

GLFWwindow* window = nullptr;
GLuint VAO, VBO;
Shader shaderProgram;

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

	GLfloat vertices[] =
	{
		0.5f, 0.3f,
		-0.4f, 0.2f,
		0.8f, -0.6f
	};

	shaderProgram = Shader("default.vert", "default.frag");

	return true;
}
void setupRectangle() {
	float vertices[] = {
		-0.8f, -0.8f,  // Lewy dolny róg
		 0.8f, -0.8f,  // Prawy dolny róg
		 0.8f,  0.8f,  // Prawy górny róg
		-0.8f,  0.8f   // Lewy górny róg
	};

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}
void draw_rectangle()
{
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINE_LOOP, 0, 4);
	glBindVertexArray(0);
}
//void initVAO()
//{
//	std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (N)] };
//	std::unique_ptr<GLuint[]> bindices{ new GLuint[N] };
//
//	glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
//	glm::vec4 lr(1.0, 1.0, 0.0, 0.0);
//
//	for (int i = 0; i < N; i++) {
//		bodies[4 * i + 0] = 0.0f;
//		bodies[4 * i + 1] = 0.0f;
//		bodies[4 * i + 2] = 0.0f;
//		bodies[4 * i + 3] = 1.0f;
//		bindices[i] = i;
//	}
//
//	glGenVertexArrays(
//		1, &boidVAO); // Attach everything needed to draw a particle to this
//	glGenBuffers(1, &boidVBO_positions);
//	glGenBuffers(1, &boidVBO_velocities);
//	glGenBuffers(1, &boidIBO);
//
//	glBindVertexArray(boidVAO);
//
//	// Bind the positions array to the boidVAO by way of the boidVBO_positions
//	glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
//	glBufferData(GL_ARRAY_BUFFER, 4 * (N) * sizeof(GLfloat), bodies.get(),
//		GL_DYNAMIC_DRAW); // transfer data
//
//	glEnableVertexAttribArray(positionLocation);
//	glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);
//
//	// Bind the velocities array to the boidVAO by way of the boidVBO_velocities
//	glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
//	glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(),
//		GL_DYNAMIC_DRAW);
//	glEnableVertexAttribArray(velocitiesLocation);
//	glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0,
//		0);
//
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS) * sizeof(GLuint),
//		bindices.get(), GL_STATIC_DRAW);
//
//	glBindVertexArray(0);
//}
void program_loop()
{
	//setupRectangle();
	 
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

		//draw_rectangle();



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