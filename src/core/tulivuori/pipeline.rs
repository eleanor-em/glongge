use crate::check_eq;
use crate::core::tulivuori::GgViewport;
use crate::core::tulivuori::TvWindowContext;
use crate::core::tulivuori::shader::VertFragShader;
use crate::core::tulivuori::swapchain::Swapchain;
use anyhow::Result;
use ash::vk;
use std::sync::Arc;

pub(crate) struct Pipeline {
    ctx: Arc<TvWindowContext>,
    _shader: Arc<VertFragShader>,
    #[allow(clippy::struct_field_names)]
    pipeline_layout: vk::PipelineLayout,
    graphics_pipelines: Vec<vk::Pipeline>,
}

impl Pipeline {
    pub fn new(
        ctx: Arc<TvWindowContext>,
        swapchain: &Arc<Swapchain>,
        shader: &Arc<VertFragShader>,
        pipeline_layout: vk::PipelineLayout,
        viewport: &GgViewport,
    ) -> Result<Arc<Self>> {
        match unsafe {
            check_eq!(
                swapchain.surface_resolution().width as f32,
                viewport.physical_width()
            );
            check_eq!(
                swapchain.surface_resolution().height as f32,
                viewport.physical_height()
            );
            ctx.device().create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[shader
                    .graphics_pipeline_create_info()
                    .color_blend_state(
                        &vk::PipelineColorBlendStateCreateInfo::default()
                            .logic_op(vk::LogicOp::CLEAR)
                            .attachments(&[vk::PipelineColorBlendAttachmentState {
                                blend_enable: 1,
                                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                                color_blend_op: vk::BlendOp::ADD,
                                src_alpha_blend_factor: vk::BlendFactor::SRC_ALPHA,
                                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                                alpha_blend_op: vk::BlendOp::ADD,
                                color_write_mask: vk::ColorComponentFlags::RGBA,
                            }]),
                    )
                    .viewport_state(
                        &vk::PipelineViewportStateCreateInfo::default()
                            .scissors(&[swapchain.surface_resolution().into()])
                            .viewports(&[viewport.inner]),
                    )
                    .rasterization_state(&vk::PipelineRasterizationStateCreateInfo {
                        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                        line_width: 1.0,
                        polygon_mode: vk::PolygonMode::FILL,
                        ..Default::default()
                    })
                    .multisample_state(&vk::PipelineMultisampleStateCreateInfo {
                        rasterization_samples: vk::SampleCountFlags::TYPE_1,
                        ..Default::default()
                    })
                    .dynamic_state(
                        &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                            vk::DynamicState::VIEWPORT,
                            vk::DynamicState::SCISSOR,
                        ]),
                    )
                    .layout(pipeline_layout)
                    .render_pass(vk::RenderPass::null())
                    .push_next(
                        &mut vk::PipelineRenderingCreateInfo::default()
                            .color_attachment_formats(&[swapchain.surface_format().format]),
                    )],
                None,
            )
        } {
            Ok(graphics_pipelines) => Ok(Arc::new(Self {
                ctx,
                _shader: shader.clone(),
                pipeline_layout,
                graphics_pipelines,
            })),
            Err((_graphics_pipelines, err_code)) => Err(err_code.into()),
        }
    }

    pub fn bind(&self, command_buffer: vk::CommandBuffer, viewport: &GgViewport) {
        unsafe {
            let scissor = vk::Rect2D::default().extent(vk::Extent2D {
                width: viewport.physical_width().floor() as u32,
                height: viewport.physical_height().floor() as u32,
            });
            let mut viewport_for_cmd = viewport.inner;
            viewport_for_cmd.x = 0.0;
            viewport_for_cmd.y = 0.0;
            viewport_for_cmd.width = viewport.physical_width();
            viewport_for_cmd.height = viewport.physical_height();
            self.ctx
                .device()
                .cmd_set_viewport(command_buffer, 0, &[viewport_for_cmd]);
            self.ctx
                .device()
                .cmd_set_scissor(command_buffer, 0, &[scissor]);
            self.ctx.device().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipelines[0],
            );
            // layout(push_constant) uniform WindowData {
            //     float window_width;
            let mut bytes = (viewport.physical_width() / viewport.combined_scale_factor())
                .to_le_bytes()
                .to_vec();
            //     float window_height;
            bytes.extend(
                (viewport.physical_height() / viewport.combined_scale_factor())
                    .to_le_bytes()
                    .to_vec(),
            );
            // };
            self.ctx.device().cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &bytes,
            );
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device().device_wait_idle().unwrap();
            for pipeline in &self.graphics_pipelines {
                self.ctx.device().destroy_pipeline(*pipeline, None);
            }
        }
    }
}
