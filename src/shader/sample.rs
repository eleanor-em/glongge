pub mod basic_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;
            layout(location = 1) in vec2 translation;
            layout(location = 2) in float rotation;
            layout(location = 3) in vec4 blend_colour;
            layout(location = 0) out vec4 colour;

            layout(set = 0, binding = 0) uniform Data {
                float window_width;
                float window_height;
                float scale_factor;
            };

            void main() {
                // map ([0, window_width], [0, window_height]) to ([0, 1], [0, 1])
                mat4 window_pixel_scale = mat4(
                    vec4(scale_factor / window_width, 0, 0, 0),
                    vec4(0, scale_factor / window_height, 0, 0),
                    vec4(0, 0, 1, 0),
                    vec4(0, 0, 0, 1));
                // map ([0, 1], [0, 1]) to ([-1, 1], [-1, 1])
                mat4 window_translation = mat4(
                    vec4( 2,  0, 0, 0),
                    vec4( 0,  2, 0, 0),
                    vec4( 0,  0, 1, 0),
                    vec4(-1, -1, 0, 1));
                mat4 projection =  window_translation * window_pixel_scale;

                mat4 rotation_mat = mat4(
                    vec4(cos(rotation), -sin(rotation), 0, 0),
                    vec4(sin(rotation), cos(rotation), 0, 0),
                    vec4(0, 0, 1, 0),
                    vec4(0, 0, 0, 1));
                mat4 translation_mat = mat4(
                    vec4(1, 0, 0, 0),
                    vec4(0, 1, 0, 0),
                    vec4(0, 0, 1, 0),
                    vec4(translation, 0, 1));
                gl_Position = projection * translation_mat * rotation_mat * vec4(position, 0, 1);
                colour = blend_colour;
            }
        ",
    }
}
pub mod basic_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) in vec4 colour;
            layout(location = 0) out vec4 f_colour;

            void main() {
                f_colour = colour;
            }
        ",
    }
}
