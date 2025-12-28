#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 f_uv;
layout(location = 1) flat in uint f_texture_id;
layout(location = 2) in vec4 f_blend_col;
layout(location = 3) in vec2 f_clip_min;
layout(location = 4) in vec2 f_clip_max;

layout(location = 0) out vec4 f_col;

layout(set = 0, binding = 0) uniform sampler s;
layout(set = 0, binding = 1) uniform texture2D tex[1023];

void main() {
    if (gl_FragCoord.x < f_clip_min.x || gl_FragCoord.y < f_clip_min.y
            || gl_FragCoord.x > f_clip_max.x || gl_FragCoord.y > f_clip_max.y) {
    } else {
        const vec4 tex_col = texture(sampler2D(nonuniformEXT(tex[f_texture_id]), s), f_uv);
        f_col = tex_col * f_blend_col;
    }
}
