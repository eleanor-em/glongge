#version 460

layout(push_constant) uniform VertPC {
    float window_width;
    float window_height;
};
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color; // 0-255 sRGB
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec2 f_uv;

void main() {
    gl_Position = vec4(2.0 * pos.x / window_width - 1.0,
                       2.0 * pos.y / window_height - 1.0,
                       0.0,
                       1.0);
    f_color = color;
    f_uv = uv;
}
