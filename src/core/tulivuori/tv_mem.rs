use crate::core::tulivuori::TvWindowContext;
use anyhow::Result;
use ash::vk;
use std::cell::UnsafeCell;
use std::sync::Arc;

pub struct TvAllocation {
    alloc: UnsafeCell<vk_mem::Allocation>,
    memory_type: u32,
    device_memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
    user_data: usize,
}

impl TvAllocation {
    pub fn new(ctx: &Arc<TvWindowContext>, alloc: vk_mem::Allocation) -> Result<Self> {
        let info = ctx
            .allocator("TvAllocation::new")?
            .get_allocation_info(&alloc);
        Ok(Self {
            alloc: UnsafeCell::new(alloc),
            memory_type: info.memory_type,
            device_memory: info.device_memory,
            offset: info.offset,
            size: info.size,
            user_data: info.user_data,
        })
    }

    pub fn memory_type(&self) -> u32 {
        self.memory_type
    }
    pub fn device_memory(&self) -> vk::DeviceMemory {
        self.device_memory
    }
    pub fn offset(&self) -> vk::DeviceSize {
        self.offset
    }
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }
    pub fn user_data(&self) -> usize {
        self.user_data
    }

    /// The `vk_mem` API is a bit crappy and requires mutable references for no particular reason.
    #[allow(clippy::mut_from_ref)]
    pub fn as_mut(&self) -> &mut vk_mem::Allocation {
        unsafe { &mut *self.alloc.get() }
    }
}
