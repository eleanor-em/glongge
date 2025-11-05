use crate::{
    core::prelude::*,
    core::tulivuori::swapchain::Swapchain,
    core::tulivuori::{TvWindowContext, tv},
};
use anyhow::{Context, Result};
use ash::{util::Align, vk};
use std::{
    marker::PhantomData,
    sync::Arc,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

pub(crate) struct GenericBuffer<T: Copy> {
    ctx: Arc<TvWindowContext>,
    copy_count: usize,
    buffer_memory: Vec<vk::DeviceMemory>,
    buffer: Vec<vk::Buffer>,
    buffer_memory_req: vk::MemoryRequirements,
    length: usize,
    did_vk_free: AtomicBool,
    phantom: PhantomData<T>,
}

impl<T: Copy> GenericBuffer<T> {
    pub fn new(
        ctx: Arc<TvWindowContext>,
        copy_count: usize,
        length: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        unsafe {
            let mut buffer_memory_vec = Vec::new();
            let mut buffer_vec = Vec::new();
            for _ in 0..copy_count {
                let buffer = ctx.device().create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size((length * size_of::<T>()) as u64)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )?;
                let buffer_memory_req = ctx.device().get_buffer_memory_requirements(buffer);
                let buffer_memory_index = tv::find_memorytype_index(
                    &buffer_memory_req,
                    &ctx.get_physical_device_memory_properties(),
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .context("Unable to find suitable memorytype for the vertex buffer.")?;
                let buffer_memory = ctx.device().allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(buffer_memory_req.size)
                        .memory_type_index(buffer_memory_index),
                    None,
                )?;
                ctx.device().bind_buffer_memory(buffer, buffer_memory, 0)?;
                buffer_vec.push(buffer);
                buffer_memory_vec.push(buffer_memory);
            }

            let buffer_memory_req = ctx.device().get_buffer_memory_requirements(buffer_vec[0]);
            Ok(Self {
                ctx,
                copy_count,
                buffer_memory: buffer_memory_vec,
                buffer: buffer_vec,
                buffer_memory_req,
                length,
                did_vk_free: AtomicBool::new(false),
                phantom: PhantomData,
            })
        }
    }

    pub fn write(&self, data: &[T], copy_index: usize) -> Result<()> {
        check_lt!(copy_index, self.copy_count);
        unsafe {
            let ptr = self.ctx.device().map_memory(
                self.buffer_memory[copy_index],
                0,
                self.buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )?;

            Align::new(ptr, align_of::<T>() as u64, self.buffer_memory_req.size)
                .copy_from_slice(data);
            self.ctx
                .device()
                .unmap_memory(self.buffer_memory[copy_index]);
        }
        Ok(())
    }

    pub fn buffer(&self, copy_index: usize) -> vk::Buffer {
        self.buffer[copy_index]
    }

    pub fn len(&self) -> usize {
        self.length
    }
    pub fn size(&self) -> vk::DeviceSize {
        self.buffer_memory_req.size
    }

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            self.ctx.device().device_wait_idle().unwrap();
            for &buffer_memory in &self.buffer_memory {
                self.ctx.device().free_memory(buffer_memory, None);
            }
            for &buffer in &self.buffer {
                self.ctx.device().destroy_buffer(buffer, None);
            }
        }
        self.did_vk_free.store(true, Ordering::Relaxed);
    }
}

impl<T: Copy> Drop for GenericBuffer<T> {
    fn drop(&mut self) {
        if !self.did_vk_free.load(Ordering::Relaxed) {
            error!("leaked resource: GenericBuffer");
        }
    }
}

struct SwapchainGenericBuffer<T: Copy> {
    inner: GenericBuffer<T>,
    current_frame_index: Arc<AtomicUsize>,
    frames_in_flight: usize,
}

impl<T: Copy> SwapchainGenericBuffer<T> {
    pub fn new(
        ctx: Arc<TvWindowContext>,
        swapchain: &Swapchain,
        length: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        Ok(Self {
            inner: GenericBuffer::new(ctx, swapchain.frames_in_flight(), length, usage)?,
            current_frame_index: swapchain.current_frame_index.clone(),
            frames_in_flight: swapchain.frames_in_flight(),
        })
    }

    pub fn write(&self, data: &[T]) -> Result<()> {
        let current_frame_index = self.current_frame_index.load(Ordering::Relaxed);
        let safe_frame = (current_frame_index + self.frames_in_flight - 1) % self.frames_in_flight;
        check_ne!(safe_frame, current_frame_index);
        self.inner.write(data, safe_frame)
    }

    pub fn current_buffer(&self) -> vk::Buffer {
        self.inner
            .buffer(self.current_frame_index.load(Ordering::Relaxed))
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn vk_free(&self) {
        self.inner.vk_free();
    }
}

pub struct VertexBuffer<T: Copy> {
    inner: SwapchainGenericBuffer<T>,
}

impl<T: Copy> VertexBuffer<T> {
    pub fn new(ctx: Arc<TvWindowContext>, swapchain: &Swapchain, length: usize) -> Result<Self> {
        Ok(Self {
            inner: SwapchainGenericBuffer::new(
                ctx,
                swapchain,
                length,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?,
        })
    }

    pub fn write(&self, data: &[T]) -> Result<()> {
        self.inner.write(data)
    }

    pub fn bind(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.inner.inner.ctx.device().cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.inner.current_buffer()],
                &[0],
            );
        }
    }

    pub fn vk_free(&self) {
        self.inner.vk_free();
    }
}

pub struct IndexBuffer32 {
    inner: SwapchainGenericBuffer<u32>,
}

impl IndexBuffer32 {
    pub fn new(
        ctx: Arc<TvWindowContext>,
        swapchain: &Swapchain,
        length: usize,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            inner: SwapchainGenericBuffer::new(
                ctx,
                swapchain,
                length,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?,
        }))
    }

    pub fn write(&self, data: &[u32]) -> Result<()> {
        self.inner.write(data)
    }

    pub fn bind(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.inner.inner.ctx.device().cmd_bind_index_buffer(
                command_buffer,
                self.inner.current_buffer(),
                0,
                vk::IndexType::UINT32,
            );
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    pub fn vk_free(&self) {
        self.inner.vk_free();
    }
}
