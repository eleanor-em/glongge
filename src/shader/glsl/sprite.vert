#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 translation;
layout(location = 2) in float rotation;
layout(location = 3) in vec2 scale;
layout(location = 4) in uint material_id;
layout(location = 5) in vec4 blend_col;
layout(location = 6) in vec2 clip_min;
layout(location = 7) in vec2 clip_max;

layout(location = 0) out vec2 f_uv;
layout(location = 1) out uint f_texture_id;
layout(location = 2) out vec4 f_blend_col;
layout(location = 3) out vec2 f_clip_min;
layout(location = 4) out vec2 f_clip_max;

layout(push_constant) uniform WindowData {
    float window_width;
    float window_height;
};

struct Material {
    vec2 uv_top_left;
    vec2 uv_bottom_right;
    uint texture_id;
    uint dummy1;
    uint dummy2;
    uint dummy3;
};

layout(std140, set = 0, binding = 0) readonly buffer MaterialData {
    Material data[];
} materials;

void main() {
    // map ([0, window_width], [0, window_height]) to ([-1, 1], [-1, 1])
    const mat4 projection = mat4(
        vec4( 2 / window_width,  0, 0, 0),
        vec4( 0,  2 / window_height, 0, 0),
        vec4( 0,  0, 1, 0),
        vec4(-1, -1, 0, 1));

    // TODO: scale and rotation may require rounding too, it interacts poorly with
    //       additional scaling factor (i.e. not just due to HiDPI) to round the entire
    //       transformed vector.
    const mat4 scale_mat = mat4(
        vec4(scale.x, 0, 0, 0),
        vec4(0, scale.y, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, 0, 0, 1));
    const mat4 rotation_mat = mat4(
        vec4(cos(rotation), sin(rotation), 0, 0),
        vec4(-sin(rotation), cos(rotation), 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, 0, 0, 1));
    const mat4 translation_mat = mat4(
        vec4(1, 0, 0, 0),
        vec4(0, 1, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(round(translation), 0, 1));
    gl_Position = projection * translation_mat * rotation_mat * scale_mat * vec4(position, 0, 1);

    Material material = materials.data[nonuniformEXT(material_id)];
    vec2 uvs[] = {
        material.uv_top_left,
        vec2(material.uv_bottom_right.x, material.uv_top_left.y),
        vec2(material.uv_top_left.x, material.uv_bottom_right.y),
        vec2(material.uv_bottom_right.x, material.uv_top_left.y),
        material.uv_bottom_right,
        vec2(material.uv_top_left.x, material.uv_bottom_right.y),
    };
    f_uv = uvs[gl_VertexIndex % 6];
    f_texture_id = material.texture_id;
    f_blend_col = blend_col;
    f_clip_min = clip_min;
    f_clip_max = clip_max;
}
