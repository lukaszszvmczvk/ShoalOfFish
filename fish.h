#pragma once

#include<GLFW/glfw3.h>
#include <glm/glm.hpp>

struct Species
{
public:
	int id;
	float size;
	glm::vec3 color;
	Species(float size, glm::vec3 color, int id)
	{
		this->size = size;
		this->color = color;
		this->id = id;
	}
	Species()
	{
		size = 15;
		color = glm::vec3(1.0f, 0, 0);
		id = 0;
	}
};

struct Fish
{
	float x;
	float y;
	float dx;
	float dy;
	Species species;
	Fish(float x, float y, Species species)
	{
		this->x = x;
		this->y = y;
		this->species = species;
		dx = dy = 0;
	}
	Fish()
	{
		x = y = dx = dy = 0;
		species = Species();
	}
};