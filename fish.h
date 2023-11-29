#include<GLFW/glfw3.h>

struct Species
{
	GLuint size;
	//color
};

struct Fish
{
	GLint x;
	GLint y;
	GLfloat dx;
	GLfloat dy;
	Species species;
	Fish(GLint x, GLint y, Species species)
	{
		this->x = x;
		this->y = y;
		this->species = species;
		dx = dy = 0;
	}
};