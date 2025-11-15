use crate::core::tulivuori::tv_mem::TvAllocation;
use crate::{
    core::prelude::*, core::tulivuori::TvWindowContext, core::tulivuori::swapchain::Swapchain,
};
use anyhow::Result;
use ash::{util::Align, vk};
use std::{
    marker::PhantomData,
    sync::Arc,
    sync::atomic::{AtomicBool, Ordering},
};
use vk_mem::Alloc;
pub(crate) struct GenericDeviceBuffer<T: Copy> {
    ctx: Arc<TvWindowContext>,
    copy_count: usize,
    buffer_vec: Vec<vk::Buffer>,
    buffer_alloc_vec: Vec<TvAllocation>,
    did_vk_free: AtomicBool,
    phantom: PhantomData<T>,
}

impl<T: Copy> GenericDeviceBuffer<T> {
    pub fn new(
        ctx: Arc<TvWindowContext>,
        copy_count: usize,
        length: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        unsafe {
            let mut buffer_alloc_vec = Vec::new();
            let mut buffer_vec = Vec::new();
            for _ in 0..copy_count {
                let (buffer, alloc) = ctx.allocator("GenericDeviceBuffer::new").create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size((length * size_of::<T>()) as u64)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::AutoPreferDevice,
                        ..Default::default()
                    },
                )?;
                buffer_vec.push(buffer);
                buffer_alloc_vec.push(TvAllocation::new(&ctx, alloc));
            }

            Ok(Self {
                ctx,
                copy_count,
                buffer_vec,
                buffer_alloc_vec,
                did_vk_free: AtomicBool::new(false),
                phantom: PhantomData,
            })
        }
    }

    pub fn buffer(&self, copy_index: usize) -> vk::Buffer {
        check_lt!(copy_index, self.copy_count);
        self.buffer_vec[copy_index]
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.buffer_alloc_vec[0].size()
    }

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.swap(true, Ordering::Relaxed));
        unsafe {
            for (&buffer, alloc) in self.buffer_vec.iter().zip(&self.buffer_alloc_vec) {
                self.ctx
                    .allocator("GenericDeviceBuffer::vk_free")
                    .destroy_buffer(buffer, alloc.as_mut());
            }
        }
    }
}

pub(crate) struct GenericBuffer<T: Copy> {
    ctx: Arc<TvWindowContext>,
    copy_count: usize,
    buffer_vec: Vec<vk::Buffer>,
    buffer_alloc_vec: Vec<TvAllocation>,
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
            let mut buffer_alloc_vec = Vec::new();
            let mut buffer_vec = Vec::new();
            for _ in 0..copy_count {
                let (buffer, alloc) = ctx.allocator("GenericBuffer::new").create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size((length * size_of::<T>()) as u64)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    &vk_mem::AllocationCreateInfo {
                        flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        usage: vk_mem::MemoryUsage::AutoPreferHost,
                        ..Default::default()
                    },
                )?;
                buffer_vec.push(buffer);
                buffer_alloc_vec.push(TvAllocation::new(&ctx, alloc));
            }

            Ok(Self {
                ctx,
                copy_count,
                buffer_vec,
                buffer_alloc_vec,
                length,
                did_vk_free: AtomicBool::new(false),
                phantom: PhantomData,
            })
        }
    }

    pub fn write(&self, data: &[T], copy_index: usize) -> Result<()> {
        if copy_index >= self.copy_count {
            bail!(
                "GenericBuffer::write(): copy_index out of bounds: {copy_index} >= {}",
                self.copy_count
            );
        }
        check_lt!(copy_index, self.copy_count);
        unsafe {
            let alloc = &self.buffer_alloc_vec[copy_index];
            let alloc_count = alloc.size() as usize / size_of::<T>();
            if data.len() > alloc_count {
                bail!(
                    "GenericBuffer::write(): out of space: tried to write {} elements, but have space for {alloc_count}",
                    data.len()
                );
            }
            let allocator = self.ctx.allocator("GenericBuffer::write");
            let ptr = allocator.map_memory(alloc.as_mut())?;
            Align::new(ptr.cast(), align_of::<T>() as u64, alloc.size()).copy_from_slice(data);
            allocator.unmap_memory(alloc.as_mut());
        }
        Ok(())
    }

    pub fn buffer(&self, copy_index: usize) -> vk::Buffer {
        check_lt!(copy_index, self.copy_count);
        self.buffer_vec[copy_index]
    }

    pub fn len(&self) -> usize {
        self.length
    }
    pub fn size(&self) -> vk::DeviceSize {
        self.buffer_alloc_vec[0].size()
    }

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.swap(true, Ordering::Relaxed));
        unsafe {
            for (&buffer, alloc) in self.buffer_vec.iter().zip(&self.buffer_alloc_vec) {
                self.ctx
                    .allocator("GenericBuffer::vk_free")
                    .destroy_buffer(buffer, alloc.as_mut());
            }
        }
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
        })
    }

    pub fn write(&self, swapchain: &Swapchain, data: &[T]) -> Result<()> {
        self.inner
            .write(data, swapchain.current_frame_index())
            .context("caused by: SwapchainGenericBuffer::write()")
    }

    pub fn current_buffer(&self, swapchain: &Swapchain) -> vk::Buffer {
        self.inner.buffer(swapchain.current_frame_index())
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

    pub fn write(&self, swapchain: &Swapchain, data: &[T]) -> Result<()> {
        self.inner
            .write(swapchain, data)
            .context("caused by: VertexBuffer::write()")
    }

    pub fn bind(&self, swapchain: &Swapchain, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.inner.inner.ctx.device().cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.inner.current_buffer(swapchain)],
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
    pub fn new(ctx: Arc<TvWindowContext>, swapchain: &Swapchain, length: usize) -> Result<Self> {
        Ok(Self {
            inner: SwapchainGenericBuffer::new(
                ctx,
                swapchain,
                length,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?,
        })
    }

    pub fn write(&self, swapchain: &Swapchain, data: &[u32]) -> Result<()> {
        self.inner
            .write(swapchain, data)
            .context("caused by: IndexBuffer32::write()")
    }

    pub fn bind(&self, swapchain: &Swapchain, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.inner.inner.ctx.device().cmd_bind_index_buffer(
                command_buffer,
                self.inner.current_buffer(swapchain),
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
