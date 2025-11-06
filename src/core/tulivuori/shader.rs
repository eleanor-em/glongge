use crate::{check_false, core::tulivuori::TvWindowContext};
use anyhow::{Context, Result};
use ash::{util::read_spv, vk};
use std::{
    sync::Arc,
    sync::atomic::{AtomicBool, Ordering},
};
use tracing::error;

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
    ) -> Result<Arc<Self>> {
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

        Ok(Arc::new(rv))
    }

    pub fn graphics_pipeline_create_info(&'_ self) -> vk::GraphicsPipelineCreateInfo<'_> {
        vk::GraphicsPipelineCreateInfo::default()
            .stages(&self.shader_stage_create_infos)
            .vertex_input_state(&self.vertex_input_state)
            .input_assembly_state(&self.vertex_input_assembly_state_info)
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
