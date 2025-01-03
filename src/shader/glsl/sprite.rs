use vulkano::buffer::BufferContents;
use crate::shader::VkVertex;

#[derive(BufferContents, VkVertex, Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct Vertex {
    // (2+2+1+2+2+1+4) * 4 = 56 bytes = 1 (x86) or 2 (Apple) vertices per cache line
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub translation: [f32; 2],
    #[format(R32_SFLOAT)]
    pub rotation: f32,
    #[format(R32G32_SFLOAT)]
    pub scale: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    #[format(R16_UINT)]
    pub texture_id: u16,
    #[format(R32G32B32A32_SFLOAT)]
    pub blend_col: [f32; 4],
}

pub mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;
            layout(location = 1) in vec2 translation;
            layout(location = 2) in float rotation;
            layout(location = 3) in vec2 scale;
            layout(location = 4) in vec2 uv;
            layout(location = 5) in uint texture_id;
            layout(location = 6) in vec4 blend_col;

            layout(location = 0) out vec2 f_uv;
            layout(location = 1) out uint f_texture_id;
            layout(location = 2) out vec4 f_blend_col;

            layout(push_constant) uniform WindowData {
                float window_width;
                float window_height;
                float scale_factor;
            };

            void main() {
                // map ([0, window_width], [0, window_height]) to ([0, 1], [0, 1])
                const mat4 window_pixel_scale = mat4(
                    vec4(scale_factor / window_width, 0, 0, 0),
                    vec4(0, scale_factor / window_height, 0, 0),
                    vec4(0, 0, 1, 0),
                    vec4(0, 0, 0, 1));
                // map ([0, 1], [0, 1]) to ([-1, 1], [-1, 1])
                const mat4 window_translation = mat4(
                    vec4( 2,  0, 0, 0),
                    vec4( 0,  2, 0, 0),
                    vec4( 0,  0, 1, 0),
                    vec4(-1, -1, 0, 1));
                const mat4 projection = window_translation * window_pixel_scale;

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
                f_uv = uv;
                f_texture_id = texture_id;
                f_blend_col = blend_col;
            }
        ",
    }
}
pub mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460
            #extension GL_EXT_fragment_shader_barycentric  : require

            layout(location = 0) in vec2 f_uv;
            layout(location = 1) flat in uint f_texture_id;
            layout(location = 2) in vec4 f_blend_col;

            layout(location = 0) out vec4 f_col;

            layout(set = 0, binding = 0) uniform sampler2D tex[1023];

            void main() {
                const vec4 tex_col = texture(tex[f_texture_id], f_uv);
                f_col = tex_col * f_blend_col;
            }
        ",
    }
}
