use crate::core::config::FONT_SAMPLE_RATIO;
use crate::core::tulivuori::swapchain::Swapchain;
use crate::core::tulivuori::texture::TvInternalTexture;
use crate::{check_false, core::tulivuori::TvWindowContext};
use anyhow::{Context, Result};
use ash::{util::read_spv, vk};
use egui::epaint;
use std::io::Cursor;
use std::mem::offset_of;
use std::{
    sync::Arc,
    sync::atomic::{AtomicBool, Ordering},
};
use tracing::error;

pub trait ShaderInfo {
    fn graphics_pipeline_create_info(&'_ self) -> vk::GraphicsPipelineCreateInfo<'_>;
}

pub struct VertFragShader {
    ctx: Arc<TvWindowContext>,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    shader_stage_create_infos: [vk::PipelineShaderStageCreateInfo<'static>; 2],
    vertex_input_assembly_state_info: vk::PipelineInputAssemblyStateCreateInfo<'static>,
    vertex_input_attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
    vertex_input_binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    vertex_input_state: vk::PipelineVertexInputStateCreateInfo<'static>,
    did_vk_free: AtomicBool,
}

impl VertFragShader {
    pub fn new<R: std::io::Read + std::io::Seek>(
        ctx: Arc<TvWindowContext>,
        vertex_spv_file: &mut R,
        frag_spv_file: &mut R,
        vertex_input_binding_descriptions: Vec<vk::VertexInputBindingDescription>,
        vertex_input_attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
    ) -> Result<Self> {
        let vertex_code =
            read_spv(vertex_spv_file).context("Failed to read vertex shader spv file")?;
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&vertex_code);

        let frag_code =
            read_spv(frag_spv_file).context("Failed to read fragment shader spv file")?;
        let frag_shader_info = vk::ShaderModuleCreateInfo::default().code(&frag_code);

        let vert = unsafe { ctx.device().create_shader_module(&vertex_shader_info, None) }
            .context("Vertex shader module error")?;

        let frag = unsafe { ctx.device().create_shader_module(&frag_shader_info, None) }
            .context("Fragment shader module error")?;

        let shader_entry_name = c"main";
        let shader_stage_create_infos = [
            vk::PipelineShaderStageCreateInfo::default()
                .module(vert)
                .name(shader_entry_name)
                .stage(vk::ShaderStageFlags::VERTEX),
            vk::PipelineShaderStageCreateInfo::default()
                .module(frag)
                .name(shader_entry_name)
                .stage(vk::ShaderStageFlags::FRAGMENT),
        ];

        let mut rv = Self {
            ctx,
            vert,
            frag,
            shader_stage_create_infos,
            vertex_input_attribute_descriptions,
            vertex_input_binding_descriptions,
            vertex_input_assembly_state_info: vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
            vertex_input_state: vk::PipelineVertexInputStateCreateInfo::default(),
            did_vk_free: AtomicBool::new(false),
        };

        // Avoid annoying lifetime issues.
        rv.vertex_input_state.p_vertex_attribute_descriptions =
            rv.vertex_input_attribute_descriptions.as_ptr();
        rv.vertex_input_state.vertex_attribute_description_count =
            rv.vertex_input_attribute_descriptions.len() as u32;
        rv.vertex_input_state.p_vertex_binding_descriptions =
            rv.vertex_input_binding_descriptions.as_ptr();
        rv.vertex_input_state.vertex_binding_description_count =
            rv.vertex_input_binding_descriptions.len() as u32;

        Ok(rv)
    }

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.swap(true, Ordering::Relaxed));
        unsafe {
            self.ctx.device().device_wait_idle().unwrap();
            self.ctx.device().destroy_shader_module(self.vert, None);
            self.ctx.device().destroy_shader_module(self.frag, None);
        }
    }
}

impl Drop for VertFragShader {
    fn drop(&mut self) {
        if !self.did_vk_free.load(Ordering::Relaxed) {
            error!("leaked resource: VertFragShader");
        }
    }
}

impl ShaderInfo for VertFragShader {
    fn graphics_pipeline_create_info(&'_ self) -> vk::GraphicsPipelineCreateInfo<'_> {
        vk::GraphicsPipelineCreateInfo::default()
            .stages(&self.shader_stage_create_infos)
            .vertex_input_state(&self.vertex_input_state)
            .input_assembly_state(&self.vertex_input_assembly_state_info)
    }
}

pub struct GuiVertFragShader {
    inner: VertFragShader,
    sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    desc_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    did_vk_free: AtomicBool,
}

impl GuiVertFragShader {
    #[allow(clippy::too_many_lines)]
    pub fn new(ctx: Arc<TvWindowContext>) -> Result<Self> {
        unsafe {
            let sampler = ctx.device().create_sampler(
                &vk::SamplerCreateInfo {
                    mag_filter: vk::Filter::LINEAR,
                    min_filter: vk::Filter::LINEAR,
                    min_lod: 0.0,
                    max_lod: FONT_SAMPLE_RATIO.log2() + 1.0,
                    mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                    address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
                    address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
                    address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
                    max_anisotropy: 1.0,
                    border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
                    compare_op: vk::CompareOp::NEVER,
                    ..Default::default()
                },
                None,
            )?;

            let descriptor_pool = ctx.device().create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::SAMPLER,
                            descriptor_count: 1,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::SAMPLED_IMAGE,
                            descriptor_count: 2,
                        },
                    ])
                    .max_sets(1),
                None,
            )?;
            let desc_set_layout = ctx.device().create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .bindings(&[
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            p_immutable_samplers: &raw const sampler,
                            ..Default::default()
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                            descriptor_count: 2,
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            ..Default::default()
                        },
                    ])
                    .push_next(
                        &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                            .binding_flags(&[
                                vk::DescriptorBindingFlags::empty(),
                                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                            ]),
                    ),
                None,
            )?;
            let descriptor_set = ctx.device().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[desc_set_layout]),
            )?[0];

            let pipeline_layout = ctx.device().create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[desc_set_layout])
                    .push_constant_ranges(&[
                        vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::VERTEX)
                            .offset(0)
                            .size(8),
                        vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                            .offset(8)
                            .size(4),
                    ]),
                None,
            )?;

            let inner = VertFragShader::new(
                ctx,
                &mut Cursor::new(&include_bytes!("../../shader/glsl/gui-vert.spv")[..]),
                &mut Cursor::new(&include_bytes!("../../shader/glsl/gui-frag.spv")[..]),
                vec![vk::VertexInputBindingDescription {
                    binding: 0,
                    stride: size_of::<epaint::Vertex>() as u32,
                    input_rate: vk::VertexInputRate::VERTEX,
                }],
                vec![
                    vk::VertexInputAttributeDescription {
                        location: 0,
                        binding: 0,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: offset_of!(epaint::Vertex, pos) as u32,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 1,
                        binding: 0,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: offset_of!(epaint::Vertex, uv) as u32,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 2,
                        binding: 0,
                        format: vk::Format::R8G8B8A8_UNORM,
                        offset: offset_of!(epaint::Vertex, color) as u32,
                    },
                ],
            )?;

            Ok(Self {
                inner,
                sampler,
                descriptor_pool,
                desc_set_layout,
                descriptor_set,
                pipeline_layout,
                did_vk_free: AtomicBool::new(false),
            })
        }
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    fn update_font_texture_inner(&self, font_texture: &Arc<TvInternalTexture>, index: usize) {
        unsafe {
            self.inner.ctx.device().update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(index as u32)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&[vk::DescriptorImageInfo {
                        sampler: self.sampler,
                        image_view: font_texture.tex_image_view(),
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    }])],
                &[],
            );
        }
    }
    pub fn init_font_texture(&self, font_texture: &Arc<TvInternalTexture>) {
        self.update_font_texture_inner(font_texture, 0);
        self.update_font_texture_inner(font_texture, 1);
    }
    pub fn update_font_texture(
        &self,
        font_texture: &Arc<TvInternalTexture>,
        swapchain: &Swapchain,
    ) {
        self.update_font_texture_inner(font_texture, swapchain.current_frame_index());
    }

    pub fn bind(&self, command_buffer: vk::CommandBuffer) {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            self.inner.ctx.device().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
        }
    }

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.swap(true, Ordering::Relaxed));
        unsafe {
            self.inner.ctx.device().device_wait_idle().unwrap();
            self.inner.vk_free();
            self.inner
                .ctx
                .device()
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.inner
                .ctx
                .device()
                .destroy_descriptor_set_layout(self.desc_set_layout, None);
            self.inner
                .ctx
                .device()
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.inner.ctx.device().destroy_sampler(self.sampler, None);
        }
    }
}

impl ShaderInfo for GuiVertFragShader {
    fn graphics_pipeline_create_info(&'_ self) -> vk::GraphicsPipelineCreateInfo<'_> {
        self.inner.graphics_pipeline_create_info()
    }
}
