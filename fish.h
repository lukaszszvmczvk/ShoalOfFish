#pragma once

#include<GLFW/glfw3.h>
#include <glm/glm.hpp>

struct Fish
{
	int speciesID;
	float size;
	glm::vec3 color;
	Fish(int id, float size, glm::vec3 color)
	{
		speciesID = id;
		this->size = size;
		this->color = color;
	}
	Fish()
	{
		speciesID = 0;
		size = 15;
		color = glm::vec3(1.0f, 0, 0);
	}
};