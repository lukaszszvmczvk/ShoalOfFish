#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 color;
out vec3 out_color;
void main()
{
   gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
   out_color = color;
};