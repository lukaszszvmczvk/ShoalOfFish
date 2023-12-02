#ifndef SHADER_CLASS_H
#define SHADER_CLASS_H

#include<glad/glad.h>
#include<string>
#include<fstream>
#include<sstream>
#include<iostream>
#include<cerrno>
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>

void glClearError();
bool glCheckError();


#define Call(x) x; \
				glCheckError() 


std::string read_from_file(const std::string& filename);

class Shader
{
	std::unordered_map<std::string, int> locationCache;
	unsigned int m_RendererID;
public:
	Shader() { m_RendererID = 0; };
	Shader(const std::string& name);
	void createFromFile(const std::string& name);
	void bind() const;
	void unbind() const;
	void setUniform1i(const std::string& name, int v0);
	void setUniform1f(const std::string& name, float v0);
	void setUniform3f(const std::string& name, glm::vec3 v);
	void setUniform4f(const std::string& name, glm::vec4 v);
	void setUniformMat4fv(const std::string& name, glm::mat4 matrix);
private:
	int GetUniformLocation(const std::string& name);
	unsigned int CreateShader(GLenum type, const std::string& strShaderFile);
	unsigned int CompileProgram(std::vector<unsigned int> shaderList);
};
#endif