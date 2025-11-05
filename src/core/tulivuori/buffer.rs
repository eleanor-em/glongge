use crate::core::prelude::*;
use crate::core::tulivuori;
use crate::core::tulivuori::TvWindowContext;
use crate::core::tulivuori::swapchain::Swapchain;
use anyhow::Context;
use anyhow::Result;
use ash::util::Align;
use ash::vk;
use std::marker::PhantomData;
use std::sync::Arc;

pub(crate) struct GenericBuffer<T: Copy> {
    ctx: Arc<TvWindowContext>,
    copy_count: usize,
    buffer_memory: Vec<vk::DeviceMemory>,
    buffer: Vec<vk::Buffer>,
    buffer_memory_req: vk::MemoryRequirements,
    length: usize,
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
                    &vk::BufferCreateInfo {
                        size: (length * size_of::<T>()) as u64,
                        usage,
                        sharing_mode: vk::SharingMode::EXCLUSIVE,
                        ..Default::default()
                    },
                    None,
                )?;

                let buffer_memory_req = ctx.device().get_buffer_memory_requirements(buffer);

                let buffer_memory_index = tulivuori::find_memorytype_index(
                    &buffer_memory_req,
                    &ctx.get_physical_device_memory_properties(),
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .context("Unable to find suitable memorytype for the vertex buffer.")?;

                let buffer_memory = ctx.device().allocate_memory(
                    &vk::MemoryAllocateInfo {
                        allocation_size: buffer_memory_req.size,
                        memory_type_index: buffer_memory_index,
                        ..Default::default()
                    },
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

            let mut align = Align::new(ptr, align_of::<T>() as u64, self.buffer_memory_req.size);
            align.copy_from_slice(data);
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
}

impl<T: Copy> Drop for GenericBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device().device_wait_idle().unwrap();
            for buffer_memory in self.buffer_memory.drain(..) {
                self.ctx.device().free_memory(buffer_memory, None);
            }
            for buffer in self.buffer.drain(..) {
                self.ctx.device().destroy_buffer(buffer, None);
            }
        }
    }
}

struct SwapchainGenericBuffer<T: Copy> {
    inner: GenericBuffer<T>,
    swapchain: Arc<Swapchain>,
}

impl<T: Copy> SwapchainGenericBuffer<T> {
    pub fn new(
        ctx: Arc<TvWindowContext>,
        swapchain: Arc<Swapchain>,
        length: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        Ok(Self {
            inner: GenericBuffer::new(ctx, swapchain.frames_in_flight(), length, usage)?,
            swapchain,
        })
    }

    pub fn write(&self, data: &[T]) -> Result<()> {
        let safe_frame = (self.swapchain.current_frame_index() + self.swapchain.frames_in_flight()
            - 1)
            % self.swapchain.frames_in_flight();
        check_ne!(safe_frame, self.swapchain.current_frame_index());
        self.inner.write(data, safe_frame)
    }

    pub fn current_buffer(&self) -> vk::Buffer {
        self.inner.buffer(self.swapchain.current_frame_index())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

pub struct VertexBuffer<T: Copy> {
    inner: SwapchainGenericBuffer<T>,
}

impl<T: Copy> VertexBuffer<T> {
    pub fn new(
        ctx: Arc<TvWindowContext>,
        swapchain: Arc<Swapchain>,
        length: usize,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            inner: SwapchainGenericBuffer::new(
                ctx,
                swapchain,
                length,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?,
        }))
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
}

pub struct IndexBuffer32 {
    inner: SwapchainGenericBuffer<u32>,
}

impl IndexBuffer32 {
    pub fn new(
        ctx: Arc<TvWindowContext>,
        swapchain: Arc<Swapchain>,
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
}
