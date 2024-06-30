pub mod basic_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;
            layout(set = 0, binding = 0) uniform Data {
                mat4 transform;
            };

            void main() {
                // Transforms (width=1, height=1) -> (width=1024, height=768)
                mat4 window_pixel_scale = mat4(
                    vec4(1/1024.0, 0, 0, 0),
                    vec4(0, 1/768.0, 0, 0),
                    vec4(0, 0, 1, 0),
                    vec4(0, 0, 0, 1));
                // Transforms (-1024/2, -768/2) top-left -> (0, 0) top-left
                mat4 window_translation = mat4(
                    vec4(2, 0, 0, 0),
                    vec4(0, 2, 0, 0),
                    vec4(0, 0, 1, 0),
                    vec4(-1, -1, 0, 1));
                mat4 projection = window_translation * window_pixel_scale;
                gl_Position = projection * transform * vec4(position, 0.0, 1.0);
            }
        ",
    }
}
pub mod basic_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}
