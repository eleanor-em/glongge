use crate::{
    check_eq, check_false, core::tulivuori::GgViewport, core::tulivuori::TvWindowContext,
    core::tulivuori::shader::VertFragShader, core::tulivuori::swapchain::Swapchain,
};
use anyhow::Result;
use ash::vk;
use std::{
    sync::Arc,
    sync::atomic::{AtomicBool, Ordering},
};
use tracing::error;

pub(crate) struct Pipeline {
    ctx: Arc<TvWindowContext>,
    _shader: Arc<VertFragShader>,
    #[allow(clippy::struct_field_names)]
    pipeline_layout: vk::PipelineLayout,
    graphics_pipelines: Vec<vk::Pipeline>,
    did_vk_free: AtomicBool,
}

impl Pipeline {
    pub fn new(
        ctx: Arc<TvWindowContext>,
        swapchain: &Swapchain,
        shader: &Arc<VertFragShader>,
        pipeline_layout: vk::PipelineLayout,
        viewport: &GgViewport,
    ) -> Result<Self> {
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
                    .rasterization_state(
                        &vk::PipelineRasterizationStateCreateInfo::default()
                            .cull_mode(vk::CullModeFlags::NONE)
                            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                            .line_width(1.0)
                            .polygon_mode(vk::PolygonMode::FILL),
                    )
                    .multisample_state(
                        &vk::PipelineMultisampleStateCreateInfo::default()
                            .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                    )
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
            Ok(graphics_pipelines) => Ok(Self {
                ctx,
                _shader: shader.clone(),
                pipeline_layout,
                graphics_pipelines,
                did_vk_free: AtomicBool::new(false),
            }),
            Err((_graphics_pipelines, err_code)) => Err(err_code.into()),
        }
    }

    pub fn bind(&self, command_buffer: vk::CommandBuffer, viewport: &GgViewport) {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
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

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            self.ctx.device().device_wait_idle().unwrap();
            for pipeline in &self.graphics_pipelines {
                self.ctx.device().destroy_pipeline(*pipeline, None);
            }
        }
        self.did_vk_free.store(true, Ordering::Relaxed);
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if !self.did_vk_free.load(Ordering::Relaxed) {
            error!("leaked resource: Pipeline");
        }
    }
}
