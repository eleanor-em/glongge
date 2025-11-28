#version 460

layout(set = 0, binding = 0) uniform sampler s;
layout(set = 0, binding = 1) uniform texture2D font_texture;

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;
layout(location = 0) out vec4 f_col;

// 0-1 sRGB gamma from 0-1 linear
vec3 srgb_gamma_from_linear(vec3 rgb) {
    bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
    vec3 lower = rgb * vec3(12.92);
    vec3 higher = vec3(1.055) * pow(rgb, vec3(1.0 / 2.4)) - vec3(0.055);
    return mix(higher, lower, vec3(cutoff));
}

// 0-1 sRGBA gamma from 0-1 linear
vec4 srgba_gamma_from_linear(vec4 rgba) {
    return vec4(srgb_gamma_from_linear(rgba.rgb), rgba.a);
}

void main() {
//    vec4 texture_in_gamma = srgba_gamma_from_linear(texture(sampler2D(font_texture, s), v_uv));
    vec4 texture_in_gamma = texture(sampler2D(font_texture, s), v_uv);
    vec4 frag_color_gamma = v_color * texture_in_gamma;
    f_col = frag_color_gamma;
}
