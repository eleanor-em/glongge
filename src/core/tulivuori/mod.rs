use crate::util::gg_sync;
use crate::{
    core::prelude::*,
    core::render::RenderHandler,
    core::scene::SceneHandler,
    gui::GuiContext,
    info_every_seconds,
    resource::ResourceHandler,
    util::gg_time::TimeIt,
    util::{SceneHandlerBuilder, UniqueShared, gg_float},
    warn_every_seconds,
};
use anyhow::{Context, Result};
use ash::{Device, Entry, Instance, ext::debug_utils, khr::surface, vk};
use egui::ViewportId;
use egui_winit::{
    winit::application::ApplicationHandler,
    winit::dpi::PhysicalSize,
    winit::event_loop::ActiveEventLoop,
    winit::keyboard::PhysicalKey,
    winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    winit::window::{Window, WindowAttributes, WindowId},
    winit::{dpi::LogicalSize, event::WindowEvent, event_loop::EventLoop},
};
use std::sync::MutexGuard;
use std::time::Duration;
use std::{
    borrow::Cow,
    ffi,
    ffi::CString,
    mem::offset_of,
    str::FromStr,
    sync::atomic::{AtomicBool, Ordering},
    sync::mpsc,
    sync::mpsc::{Receiver, Sender},
    sync::{Arc, Mutex},
    time::Instant,
    time::SystemTime,
};
use tracing::{error, info_span, warn};

pub mod tv {
    use crate::core::config::USE_VSYNC;
    use ash::vk;

    pub fn default_command_buffer_begin_info() -> vk::CommandBufferBeginInfo<'static> {
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
    }
    pub fn default_component_mapping() -> vk::ComponentMapping {
        vk::ComponentMapping::default()
            .r(vk::ComponentSwizzle::R)
            .g(vk::ComponentSwizzle::G)
            .b(vk::ComponentSwizzle::B)
            .a(vk::ComponentSwizzle::A)
    }
    pub fn default_image_subresource_range() -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .level_count(1)
    }

    pub fn find_memorytype_index(
        memory_req: &vk::MemoryRequirements,
        memory_prop: &vk::PhysicalDeviceMemoryProperties,
        flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        memory_prop.memory_types[..memory_prop.memory_type_count as _]
            .iter()
            .enumerate()
            .find(|(index, memory_type)| {
                (1 << index) & memory_req.memory_type_bits != 0
                    && memory_type.property_flags & flags == flags
            })
            .map(|(index, _memory_type)| index as _)
    }

    pub fn present_mode_key(present_mode: &vk::PresentModeKHR) -> usize {
        if !USE_VSYNC && *present_mode == vk::PresentModeKHR::IMMEDIATE {
            return 0;
        }
        match *present_mode {
            vk::PresentModeKHR::MAILBOX => 1,
            vk::PresentModeKHR::FIFO => 2,
            vk::PresentModeKHR::IMMEDIATE => 3,
            _ => 4,
        }
    }

    pub(crate) unsafe fn any_as_u32_slice<T: Sized>(p: &T) -> &[u32] {
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(p).cast::<u32>(),
                size_of::<T>() / size_of::<u32>(),
            )
        }
    }
}

pub struct TvWindowContext {
    _entry: Entry,
    instance: Instance,
    debug_handler: Option<DebugHandler>,

    // Just seems prudent to also have this.
    #[allow(unused)]
    window: Arc<Window>,

    surface: vk::SurfaceKHR,
    surface_loader: surface::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    device: Device,
    present_queue: Arc<Mutex<vk::Queue>>,

    allocator: UniqueShared<Option<Arc<vk_mem::Allocator>>>,

    did_vk_free: AtomicBool,
}

impl TvWindowContext {
    pub fn surface(&self) -> vk::SurfaceKHR {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        self.surface
    }
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }
    pub fn device(&self) -> &Device {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        &self.device
    }
    pub fn present_queue(&self) -> Result<MutexGuard<'_, vk::Queue>> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        gg_sync::spin_lock_for(&self.present_queue, Duration::from_millis(100))
    }
    pub fn allocator(&self) -> Arc<vk_mem::Allocator> {
        self.allocator.lock().as_ref().cloned().unwrap()
    }

    pub(crate) fn create_swapchain_device(&self) -> ash::khr::swapchain::Device {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        ash::khr::swapchain::Device::new(&self.instance, &self.device)
    }

    pub fn get_physical_device_memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        }
    }
    pub fn get_physical_device_surface_capabilities(&self) -> Result<vk::SurfaceCapabilitiesKHR> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            Ok(self
                .surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)?)
        }
    }
    pub fn get_physical_device_surface_formats(&self) -> Result<Vec<vk::SurfaceFormatKHR>> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            Ok(self
                .surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)?)
        }
    }
    pub fn get_physical_device_surface_present_modes(&self) -> Result<Vec<vk::PresentModeKHR>> {
        check_false!(self.did_vk_free.load(Ordering::Relaxed));
        unsafe {
            Ok(self
                .surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)?)
        }
    }

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.swap(true, Ordering::Relaxed));
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.allocator.lock().take().unwrap();
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some(debug_handler) = self.debug_handler.as_ref() {
                debug_handler.vk_free();
            }
            self.instance.destroy_instance(None);
        }
    }
    pub fn did_vk_free(&self) -> bool {
        self.did_vk_free.load(Ordering::Relaxed)
    }
}

impl Drop for TvWindowContext {
    fn drop(&mut self) {
        if !self.did_vk_free.load(Ordering::Relaxed) {
            error!("leaked resource: TvWindowContext");
            // vk_mem uses RAII for freeing resources, which could crash in this case.
            std::mem::forget(self.allocator.lock().take().unwrap());
        }
    }
}

pub struct TvWindowContextBuilder {
    app_name: CString,
    instance_extension_names: Vec<*const ffi::c_char>,
    logical_device_extension_names: Vec<*const ffi::c_char>,

    flag_add_validation_layers: bool,
    flag_use_debug_tools: bool,
    flag_verbose_logging: bool,
}

impl Default for TvWindowContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TvWindowContextBuilder {
    pub fn new() -> TvWindowContextBuilder {
        TvWindowContextBuilder {
            app_name: CString::from_str("tulivuori").unwrap(),
            instance_extension_names: Vec::new(),
            logical_device_extension_names: Vec::new(),
            flag_add_validation_layers: false,
            flag_use_debug_tools: false,
            flag_verbose_logging: false,
        }
    }

    #[must_use]
    pub fn with_app_name(mut self, app_name: CString) -> Self {
        self.app_name = app_name;
        self
    }
    #[must_use]
    pub fn with_instance_extension(mut self, extension: &ffi::CStr) -> Self {
        self.instance_extension_names.push(extension.as_ptr());
        self
    }
    #[must_use]
    pub fn with_logical_device_extension(mut self, logical_device_extension: &ffi::CStr) -> Self {
        self.logical_device_extension_names
            .push(logical_device_extension.as_ptr());
        self
    }
    #[must_use]
    pub fn with_flag_debug_tools(mut self) -> Self {
        self.flag_use_debug_tools = true;
        self
    }
    #[must_use]
    pub fn with_flag_validation_layers(mut self) -> Self {
        self.flag_add_validation_layers = true;
        self
    }
    #[must_use]
    pub fn with_flag_verbose_logging(mut self) -> Self {
        self.flag_verbose_logging = true;
        self
    }

    #[allow(clippy::too_many_lines)]
    pub fn build(self, window: &Arc<Window>) -> Result<Arc<TvWindowContext>> {
        let span = info_span!("TvWindowContext");
        let _enter = span.enter();

        let perf_stats = PerfStats::new("vulkan_init");
        perf_stats.start();

        let entry = Entry::linked();

        let mut layers_names_raw = Vec::new();
        if self.flag_add_validation_layers {
            layers_names_raw.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
        }

        let mut instance_extension_names = Self::create_min_instance_extension_names(window)?;
        instance_extension_names.extend(self.instance_extension_names);
        if self.flag_verbose_logging {
            info!("create instance");
        }
        let instance = unsafe {
            Self::create_instance(
                &self.app_name,
                &entry,
                &layers_names_raw,
                &instance_extension_names,
            )?
        };
        perf_stats.lap("create instance");

        let debug_handler = if self.flag_use_debug_tools {
            #[cfg(not(debug_assertions))]
            {
                error!("should not enable debug tools in release mode!");
            }
            Some(DebugHandler::new(&entry, &instance)?)
        } else {
            None
        };

        if self.flag_verbose_logging {
            info!("create surface");
        }
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?
        };
        perf_stats.lap("create surface");

        // TODO: find a way to make these into struct members.
        let features = vk::PhysicalDeviceFeatures::default();
        let features_v12 = vk::PhysicalDeviceVulkan12Features::default()
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true);
        let features_v13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        if self.flag_verbose_logging {
            info!("create physical device");
        }
        let (surface_loader, physical_device, queue_family_index) = Self::create_physical_device(
            &entry,
            &instance,
            surface,
            &features,
            &features_v12,
            &features_v13,
        )?;
        perf_stats.lap("create physical device");

        if self.flag_verbose_logging {
            info!("create logical device (and present queue)");
        }
        let mut logical_device_extension_names = vec![
            ash::khr::swapchain::NAME.as_ptr(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ash::khr::portability_subset::NAME.as_ptr(),
        ];
        logical_device_extension_names.extend(self.logical_device_extension_names);
        let device = Self::create_device(
            &instance,
            physical_device,
            queue_family_index,
            &logical_device_extension_names,
            features,
            features_v12,
            features_v13,
        )?;
        let present_queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        perf_stats.lap("create logical device");

        let allocator = unsafe {
            vk_mem::Allocator::new(vk_mem::AllocatorCreateInfo::new(
                &instance,
                &device,
                physical_device,
            ))?
        };
        perf_stats.lap("create allocator");
        perf_stats.report(0);

        Ok(Arc::new(TvWindowContext {
            _entry: entry,
            instance,
            debug_handler,
            window: window.clone(),
            surface,
            surface_loader,
            physical_device,
            queue_family_index,
            device,
            present_queue: Arc::new(Mutex::new(present_queue)),
            allocator: UniqueShared::new(Some(Arc::new(allocator))),
            did_vk_free: AtomicBool::new(false),
        }))
    }

    fn create_min_instance_extension_names(
        window: &Arc<Window>,
    ) -> Result<Vec<*const ffi::c_char>> {
        let mut min_extension_names =
            ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?.to_vec();
        min_extension_names.push(debug_utils::NAME.as_ptr());

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            min_extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
        }

        Ok(min_extension_names)
    }

    unsafe fn create_instance(
        app_name: &ffi::CStr,
        entry: &Entry,
        layers_names_raw: &[*const ffi::c_char],
        min_extension_names: &[*const ffi::c_char],
    ) -> Result<Instance> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let var = match std::env::var("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS") {
                Ok(var) => var,
                Err(e) => {
                    panic!(
                        "on macOS, environment variable `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` must be set; \
                        do you have .cargo/config.toml set up correctly? got: {e:?}"
                    );
                }
            };
            check_eq!(var, "1");
        }
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let create_flags = {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                vk::InstanceCreateFlags::empty()
            }
        };
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(layers_names_raw)
            .enabled_extension_names(min_extension_names)
            .flags(create_flags);
        let instance = unsafe { entry.create_instance(&create_info, None)? };
        Ok(instance)
    }

    fn create_physical_device(
        entry: &Entry,
        instance: &Instance,
        surface: vk::SurfaceKHR,
        required_features: &vk::PhysicalDeviceFeatures,
        required_features_v12: &vk::PhysicalDeviceVulkan12Features,
        required_features_v13: &vk::PhysicalDeviceVulkan13Features,
    ) -> Result<(surface::Instance, vk::PhysicalDevice, u32)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let surface_loader = surface::Instance::new(entry, instance);
        let (physical_device, queue_family_index) = physical_devices
            .iter()
            // API support:
            .filter(|&&candidate| unsafe {
                let props = instance.get_physical_device_properties(candidate);
                let rv = vk::api_version_major(props.api_version) >= 1
                    && vk::api_version_minor(props.api_version) >= 3;
                check!(rv);
                rv
            })
            // Features support:
            .filter(|&&candidate| unsafe {
                Self::does_support_features(
                    instance,
                    required_features,
                    required_features_v12,
                    required_features_v13,
                    candidate,
                )
            })
            // Surface and graphics queue support:
            .filter_map(|&candidate| unsafe {
                let (queue_family_index, _queue_family_properties) = instance
                    .get_physical_device_queue_family_properties(candidate)
                    .iter()
                    .enumerate()
                    .filter(|(_, info)| info.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                    .find(|(index, _)| {
                        surface_loader
                            .get_physical_device_surface_support(candidate, *index as u32, surface)
                            .unwrap_or_else(|e| {
                                error!(
                                    "get_physical_device_surface_support(): index={index}: {e:?}"
                                );
                                false
                            })
                    })?;
                Some((candidate, queue_family_index))
            })
            // Preferred device type:
            .sorted_by_key(|&(candidate, _)| unsafe {
                let props = instance.get_physical_device_properties(candidate);
                match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    vk::PhysicalDeviceType::CPU => 3,
                    _ => 4,
                }
            })
            // Preferred present mode:
            .min_by_key(|(candidate, _)| {
                let present_modes = match unsafe {
                    surface_loader.get_physical_device_surface_present_modes(*candidate, surface)
                } {
                    Ok(present_modes) => {
                        if present_modes.is_empty() {
                            error!("get_physical_device_surface_present_modes() returned empty?");
                        }
                        present_modes
                    }
                    Err(e) => {
                        error!("get_physical_device_surface_present_modes(): {e:?}");
                        return usize::MAX;
                    }
                };
                present_modes
                    .iter()
                    .map(tv::present_mode_key)
                    .min()
                    .unwrap_or(usize::MAX)
            })
            .context("Couldn't find suitable device.")?;
        let queue_family_index = queue_family_index as u32;
        Ok((surface_loader, physical_device, queue_family_index))
    }

    unsafe fn does_support_features(
        instance: &Instance,
        required_features: &vk::PhysicalDeviceFeatures,
        required_features_v12: &vk::PhysicalDeviceVulkan12Features,
        required_features_v13: &vk::PhysicalDeviceVulkan13Features,
        candidate: vk::PhysicalDevice,
    ) -> bool {
        unsafe {
            let mut supported_features_v12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut supported_features_v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut supported_features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut supported_features_v12)
                .push_next(&mut supported_features_v13);
            let mut rv = true;
            instance.get_physical_device_features2(candidate, &mut supported_features);
            {
                let required_features_bytes = tv::any_as_u32_slice(required_features);
                let supported_features_bytes = tv::any_as_u32_slice(&supported_features.features);
                check_gt!(supported_features_bytes.iter().sum::<u32>(), 0u32);
                for (i, (&required, &supported)) in required_features_bytes
                    .iter()
                    .zip(supported_features_bytes)
                    .enumerate()
                {
                    check!(required == 0 || required == 1);
                    check!(supported == 0 || supported == 1);
                    if required != 0 && supported == 0 {
                        error!("missing required feature: {i}");
                        rv = false;
                    }
                }
            }
            {
                let offset = offset_of!(
                    vk::PhysicalDeviceVulkan12Features,
                    sampler_mirror_clamp_to_edge
                ) / size_of::<u32>();
                let required_features_bytes_v12 =
                    &tv::any_as_u32_slice(required_features_v12)[offset..];
                let supported_features_bytes_v12 =
                    &tv::any_as_u32_slice(&supported_features_v12)[offset..];
                // last element is a marker
                let required_features_bytes_v12 =
                    &required_features_bytes_v12[..required_features_bytes_v12.len() - 1];
                let supported_features_bytes_v12 =
                    &supported_features_bytes_v12[..supported_features_bytes_v12.len() - 1];
                check_gt!(supported_features_bytes_v12.iter().sum::<u32>(), 0);
                for (i, (&required, &supported)) in required_features_bytes_v12
                    .iter()
                    .zip(supported_features_bytes_v12)
                    .enumerate()
                {
                    check!(required == 0 || required == 1, i);
                    check!(supported == 0 || supported == 1, i);
                    if required != 0 && supported == 0 {
                        error!("missing required Vulkan 1.2 feature: {i}");
                        rv = false;
                    }
                }
            }
            {
                let offset = offset_of!(vk::PhysicalDeviceVulkan13Features, robust_image_access)
                    / size_of::<u32>();
                let required_features_bytes_v13 =
                    &tv::any_as_u32_slice(required_features_v13)[offset..];
                let supported_features_bytes_v13 =
                    &tv::any_as_u32_slice(&supported_features_v13)[offset..];
                // last element is a marker
                let required_features_bytes_v13 =
                    &required_features_bytes_v13[..required_features_bytes_v13.len() - 1];
                let supported_features_bytes_v13 =
                    &supported_features_bytes_v13[..supported_features_bytes_v13.len() - 1];
                check_gt!(supported_features_bytes_v13.iter().sum::<u32>(), 0);
                for (i, (&required, &supported)) in required_features_bytes_v13
                    .iter()
                    .zip(supported_features_bytes_v13)
                    .enumerate()
                {
                    check!(required == 0 || required == 1, i);
                    check!(supported == 0 || supported == 1, i);
                    if required != 0 && supported == 0 {
                        error!("missing required Vulkan 1.3 feature: {i}");
                        rv = false;
                    }
                }
            }
            rv
        }
    }

    fn create_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        device_extension_names_raw: &[*const ffi::c_char],
        features: vk::PhysicalDeviceFeatures,
        mut features_v12: vk::PhysicalDeviceVulkan12Features,
        mut features_v13: vk::PhysicalDeviceVulkan13Features,
    ) -> Result<Device> {
        let priorities = [1.0];

        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(device_extension_names_raw)
            .enabled_features(&features)
            .push_next(&mut features_v12)
            .push_next(&mut features_v13);
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        Ok(device)
    }
}

pub(crate) struct DebugHandler {
    debug_utils_loader: debug_utils::Instance,
    debug_callback: vk::DebugUtilsMessengerEXT,
    did_vk_free: AtomicBool,
}

impl DebugHandler {
    pub fn new(entry: &Entry, instance: &Instance) -> Result<Self> {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(Self::vulkan_debug_callback));
        let debug_utils_loader = debug_utils::Instance::new(entry, instance);
        let debug_callback =
            unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_info, None)? };
        Ok(Self {
            debug_utils_loader,
            debug_callback,
            did_vk_free: AtomicBool::new(false),
        })
    }

    pub fn vk_free(&self) {
        check_false!(self.did_vk_free.swap(true, Ordering::Relaxed));
        unsafe {
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
        }
    }

    /// # Safety: lol
    unsafe extern "system" fn vulkan_debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        let vk_span = info_span!("vulkan");
        let _enter = vk_span.enter();
        let callback_data = unsafe { *p_callback_data };
        let message_id_number = callback_data.message_id_number;

        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            unsafe { ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
        };

        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            unsafe { ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() }
        };

        match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
                info!("{message_type:?} [{message_id_name} ({message_id_number})] : {message}",);
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
                warn!("{message_type:?} [{message_id_name} ({message_id_number})] : {message}",);
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
                error!("{message_type:?} [{message_id_name} ({message_id_number})] : {message}",);
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
                // ignored
            }
            value => {
                panic!("unexpected DebugUtilsMessageSeverityFlagsEXT value: {value:?}");
            }
        }

        vk::FALSE
    }
}

impl Drop for DebugHandler {
    fn drop(&mut self) {
        if !self.did_vk_free.load(Ordering::Relaxed) {
            error!("leaked resource: DebugHandler");
        }
    }
}

#[derive(Clone)]
pub(crate) struct GgWindow {
    pub(crate) inner: Arc<Window>,
    refresh_time: f32,
}

impl GgWindow {
    pub(crate) fn new(event_loop: &ActiveEventLoop, size: impl Into<Vec2i>) -> Result<Self> {
        let size = size.into();
        let mut window_attrs = WindowAttributes::default();
        // TODO: allow setting window title.
        window_attrs.title = "glongge".to_string();
        window_attrs.resizable = true;
        window_attrs.inner_size = Some(egui_winit::winit::dpi::Size::Logical(LogicalSize::new(
            f64::from(size.x),
            f64::from(size.y),
        )));
        let window = Arc::new(event_loop.create_window(window_attrs)?);
        let refresh_time = window
            .current_monitor()
            .and_then(|m| m.refresh_rate_millihertz())
            .map_or_else(
                || {
                    warn!("failed to determine refresh rate, assuming 60 Hz");
                    1_000.0 / 60.0
                },
                |r| {
                    let refresh_time = 1_000_000.0 / r as f32;
                    info!("refresh every: {refresh_time:.2} ms");
                    refresh_time
                },
            );
        check_gt!(refresh_time, 1.0);
        check_lt!(refresh_time, 1_000.0 / 30.0);
        Ok(Self {
            inner: window,
            refresh_time,
        })
    }

    pub(crate) fn inner_size(&self) -> PhysicalSize<u32> {
        self.inner.inner_size()
    }
    pub(crate) fn winit_scale_factor(&self) -> f32 {
        self.inner.scale_factor() as f32
    }
}

#[derive(Clone)]
pub(crate) struct GgViewport {
    window: GgWindow,
    inner: vk::Viewport,
    winit_scale_factor: f32,
    extra_scale_factor: f32,
}

impl GgViewport {
    pub(crate) fn new(window: &GgWindow) -> Self {
        Self {
            window: window.clone(),
            inner: vk::Viewport::default()
                .width(window.inner_size().width as f32)
                .height(window.inner_size().height as f32)
                .min_depth(0.0)
                .max_depth(1.0),
            winit_scale_factor: window.winit_scale_factor(),
            extra_scale_factor: 1.0,
        }
    }

    pub(crate) fn physical_width(&self) -> f32 {
        self.inner.width
    }
    pub(crate) fn physical_height(&self) -> f32 {
        self.inner.height
    }
    pub(crate) fn physical_left(&self) -> f32 {
        self.inner.x
    }
    pub(crate) fn physical_top(&self) -> f32 {
        self.inner.y
    }
    pub(crate) fn physical_top_left(&self) -> Vec2 {
        Vec2 {
            x: self.physical_left(),
            y: self.physical_top(),
        }
    }
    pub(crate) fn logical_left(&self) -> f32 {
        self.inner.x / self.winit_scale_factor
    }
    pub(crate) fn logical_top(&self) -> f32 {
        self.inner.y / self.winit_scale_factor
    }
    pub(crate) fn world_left(&self) -> f32 {
        self.inner.x / self.combined_scale_factor()
    }
    pub(crate) fn world_top(&self) -> f32 {
        self.inner.y / self.combined_scale_factor()
    }
    pub(crate) fn world_top_left(&self) -> Vec2 {
        Vec2 {
            x: self.world_left(),
            y: self.world_top(),
        }
    }

    pub(crate) fn combined_scale_factor(&self) -> f32 {
        self.winit_scale_factor * self.extra_scale_factor
    }
    pub(crate) fn set_extra_scale_factor(&mut self, extra_scale_factor: f32) {
        self.extra_scale_factor = extra_scale_factor;
    }
    pub(crate) fn extra_scale_factor(&self) -> f32 {
        self.extra_scale_factor
    }
    pub(crate) fn refresh_time(&self) -> f32 {
        self.window.refresh_time
    }

    pub(crate) fn set_physical_top_left(&mut self, top_left: Vec2) {
        self.inner.x = top_left.x;
        self.inner.y = top_left.y;
    }
}

#[allow(unused)]
struct WindowEventHandlerInner {
    window: GgWindow,
    platform: egui_winit::State,
    winit_scale_factor_for_logging: f32,
    input_handler: Arc<Mutex<InputHandler>>,
    render_stats: RenderPerfStats,
    last_render_stats: Option<RenderPerfStats>,
    is_first_window_event: bool,

    window_event_rx: Receiver<WindowEvent>,
    scale_factor_rx: Receiver<f32>,
    recreate_swapchain_rx: Receiver<Instant>,
    render_handler: Option<RenderHandler>,
    render_count: usize,
}
impl WindowEventHandlerInner {
    fn handle_window_events(&mut self) {
        while let Ok(event) = self.window_event_rx.try_recv() {
            let _response = self.platform.on_window_event(&self.window.inner, &event);
            if event == WindowEvent::CloseRequested {
                self.render_handler.take().unwrap().vk_free();
                std::process::exit(0);
            }
        }
        if let Some(new_scale_factor) = self.scale_factor_rx.try_iter().last() {
            // Since scale_factor is given by winit, we expect an exact comparison to work.
            #[allow(clippy::float_cmp)]
            if self.winit_scale_factor_for_logging != new_scale_factor {
                // TODO: verbose_every_seconds!
                info_every_seconds!(
                    1,
                    "WindowEvent::ScaleFactorChanged: {} -> {}: recreating swapchain",
                    self.winit_scale_factor_for_logging,
                    new_scale_factor
                );
                self.winit_scale_factor_for_logging = new_scale_factor;
                // self.recreate_swapchain().unwrap();
            }
        }
        if let Some(request_time) = self.recreate_swapchain_rx.try_iter().last() {
            info!(
                "recreating swapchain: {:.2} ms old",
                request_time.elapsed().as_micros() as f32 / 1000.0
            );
            // self.recreate_swapchain().unwrap();
        }
    }

    fn render_update(&mut self) {
        let n = self.render_count;
        let span = info_span!("render_update", n);
        let _enter = span.enter();
        self.render_handler.as_ref().unwrap().wait_update_done();
        self.handle_window_events();

        self.render_handler
            .as_mut()
            .unwrap()
            .render_update()
            .unwrap();
        self.render_count += 1;
    }
}

struct WindowEventHandlerCreateInfo<F>
where
    F: FnOnce(SceneHandlerBuilder) -> SceneHandler + Send + 'static,
{
    window_size: Vec2i,
    scene_handler_builder_callback: Option<F>,
    extra_scale_factor: f32,
    clear_col: Colour,
}

pub(crate) struct WindowEventHandler<F>
where
    F: FnOnce(SceneHandlerBuilder) -> SceneHandler + Send + 'static,
{
    create_info: WindowEventHandlerCreateInfo<F>,
    input_handler: Arc<Mutex<InputHandler>>,

    gui_ctx: Option<GuiContext>,

    window_event_tx: Sender<WindowEvent>,
    window_event_rx: Option<Receiver<WindowEvent>>,
    scale_factor_tx: Sender<f32>,
    scale_factor_rx: Option<Receiver<f32>>,
    recreate_swapchain_tx: Sender<Instant>,
    recreate_swapchain_rx: Option<Receiver<Instant>>,
}

impl<SceneHandlerBuilderCallback> WindowEventHandler<SceneHandlerBuilderCallback>
where
    SceneHandlerBuilderCallback: FnOnce(SceneHandlerBuilder) -> SceneHandler + Send + 'static,
{
    pub(crate) fn create_and_run(
        window_size: Vec2i,
        extra_scale_factor: f32,
        clear_col: Colour,
        gui_ctx: GuiContext,
        scene_handler_builder_callback: SceneHandlerBuilderCallback,
    ) -> Result<()> {
        let (window_event_tx, window_event_rx) = mpsc::channel();
        let (scale_factor_tx, scale_factor_rx) = mpsc::channel();
        let (recreate_swapchain_tx, recreate_swapchain_rx) = mpsc::channel();
        let mut this = Self {
            create_info: WindowEventHandlerCreateInfo {
                window_size,
                extra_scale_factor,
                clear_col,
                scene_handler_builder_callback: Some(scene_handler_builder_callback),
            },
            input_handler: InputHandler::new(),
            gui_ctx: Some(gui_ctx),
            window_event_tx,
            window_event_rx: Some(window_event_rx),
            scale_factor_tx,
            scale_factor_rx: Some(scale_factor_rx),
            recreate_swapchain_tx,
            recreate_swapchain_rx: Some(recreate_swapchain_rx),
        };

        let event_loop = EventLoop::new()?;
        Ok(event_loop.run_app(&mut this)?)
    }

    fn create_inner(
        &mut self,
        event_loop: &ActiveEventLoop,
        scene_handler_builder_callback: SceneHandlerBuilderCallback,
    ) -> Result<()> {
        let window = GgWindow::new(event_loop, self.create_info.window_size)?;
        let ctx = TvWindowContextBuilder::new()
            .with_app_name(CString::from_str("ash-noodling").unwrap())
            .with_flag_debug_tools()
            .with_flag_validation_layers()
            .with_flag_verbose_logging()
            .build(&window.inner)?;
        let platform = egui_winit::State::new(
            self.gui_ctx.clone().unwrap().inner.clone(),
            ViewportId::ROOT,
            &event_loop,
            Some(window.winit_scale_factor()),
            None,
            None,
        );

        let input_handler = self.input_handler.clone();
        let window_event_rx = self.window_event_rx.take().unwrap();
        let scale_factor_rx = self.scale_factor_rx.take().unwrap();
        let recreate_swapchain_rx = self.recreate_swapchain_rx.take().unwrap();
        let resource_handler = ResourceHandler::new(&ctx)?;
        let viewport = UniqueShared::new(GgViewport::new(&window));
        let render_handler = RenderHandler::new(
            ctx.clone(),
            self.gui_ctx.clone().unwrap(),
            window.clone(),
            viewport,
            resource_handler,
        )?
        .with_extra_scale_factor(self.create_info.extra_scale_factor)
        .with_clear_col(self.create_info.clear_col);
        let scene_handler_builder =
            SceneHandlerBuilder::new(input_handler.clone(), render_handler.as_lite());
        std::thread::spawn(move || {
            let mut scene_handler = scene_handler_builder_callback(scene_handler_builder);
            loop {
                scene_handler.run_update();
            }
        });
        let winit_scale_factor_for_logging = window.winit_scale_factor();
        std::thread::spawn(move || {
            let render_stats = RenderPerfStats::new(&window);
            let mut inner = WindowEventHandlerInner {
                window,
                platform,
                winit_scale_factor_for_logging,
                input_handler,
                render_stats,
                last_render_stats: None,
                is_first_window_event: true,
                window_event_rx,
                scale_factor_rx,
                recreate_swapchain_rx,
                render_handler: Some(render_handler),
                render_count: 0,
            };
            loop {
                inner.render_update();
            }
        });
        Ok(())
    }
}

impl<F> ApplicationHandler for WindowEventHandler<F>
where
    F: FnOnce(SceneHandlerBuilder) -> SceneHandler + Send + 'static,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(scene_handler_builder_callback) =
            self.create_info.scene_handler_builder_callback.take()
        {
            // First event. Note winit documentation:
            // "This is a common indicator that you can create a window."
            self.create_inner(event_loop, scene_handler_builder_callback)
                .expect("error initialising");
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        self.window_event_tx.send(event.clone()).unwrap();
        match event {
            WindowEvent::KeyboardInput { event, .. } => match event.physical_key {
                PhysicalKey::Code(keycode) => {
                    self.input_handler
                        .lock()
                        .unwrap()
                        .queue_key_event(keycode, event.state);
                }
                PhysicalKey::Unidentified(keycode) => {
                    info!("PhysicalKey::Unidentified({keycode:?}), ignoring");
                }
            },
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor_tx.send(scale_factor as f32).unwrap();
            }
            WindowEvent::Resized(physical_size) => {
                // TODO: verbose_every_seconds!
                info_every_seconds!(
                    1,
                    "WindowEvent::Resized: {:?}: recreating swapchain",
                    physical_size
                );
                self.recreate_swapchain_tx.send(Instant::now()).unwrap();
            }
            WindowEvent::RedrawRequested => {
                // self.expect_inner().run_update();
                // self.expect_inner().window.inner.request_redraw();
            }
            _other_event => {}
        }
    }
}

#[allow(unused)]
#[derive(Clone)]
pub(crate) struct RenderPerfStats {
    update_gui: TimeIt,
    execute: TimeIt,
    between_renders: TimeIt,
    extra_debug: TimeIt,

    total: TimeIt,
    on_time: u64,
    count: u64,
    penultimate_step: Instant,
    last_step: Instant,
    last_report: Instant,
    totals_ms: Vec<f32>,

    refresh_time: f32,

    last_perf_stats: Option<Box<RenderPerfStats>>,
}

impl RenderPerfStats {
    fn new(window: &GgWindow) -> Self {
        Self {
            update_gui: TimeIt::new("update_gui"),
            execute: TimeIt::new("execute"),
            between_renders: TimeIt::new("between renders"),
            extra_debug: TimeIt::new("extra_debug"),
            total: TimeIt::new("total"),
            on_time: 0,
            count: 0,
            penultimate_step: Instant::now(),
            last_step: Instant::now(),
            last_report: Instant::now(),
            totals_ms: Vec::with_capacity(10),
            refresh_time: window.refresh_time,
            last_perf_stats: None,
        }
    }

    #[allow(unused)]
    fn start(&mut self) {
        self.between_renders.stop();
    }

    #[allow(unused)]
    fn end(&mut self) -> Option<Self> {
        // Allow a bit of slack.
        let deadline_ms = self.refresh_time * 1.2;

        self.between_renders.start();

        if self.totals_ms.len() == self.totals_ms.capacity() {
            self.totals_ms.remove(0);
        }
        let render_time = gg_float::micros(self.last_step.elapsed()) * 1000.0;
        self.totals_ms.push(render_time);

        let late_in_row = self
            .totals_ms
            .iter()
            .rev()
            .take_while(|&&t| t > deadline_ms)
            .collect_vec();
        if late_in_row.len() > 1 {
            let mut msg = format!("{} frames late in a row: ", late_in_row.len());
            for time in &late_in_row[..late_in_row.len() - 1] {
                msg += format!("{time:.1}, ").as_str();
            }
            msg += format!("{:.1}", late_in_row.last().unwrap()).as_str();
            warn_every_seconds!(1, "{msg}");
        }
        if render_time <= deadline_ms {
            self.on_time += 1;
        }
        self.count += 1;

        self.total.stop();
        self.total.start();
        self.penultimate_step = self.last_step;
        self.last_step = Instant::now();

        if self.last_report.elapsed().as_secs() >= 2 {
            #[allow(clippy::cast_precision_loss)]
            let on_time_rate = self.on_time as f32 / self.count as f32 * 100.0;
            if on_time_rate.round() < 100.0 {
                // TODO: this calculation is dubious, improve.
                warn!("frames on time: {on_time_rate:.1}%");
            }
            self.last_perf_stats = Some(Box::new(Self {
                update_gui: self.update_gui.report_take(),
                execute: self.execute.report_take(),
                between_renders: self.between_renders.report_take(),
                extra_debug: self.extra_debug.report_take(),
                total: self.total.report_take(),
                on_time: 0,
                count: 0,
                last_perf_stats: None,
                last_report: Instant::now(),
                penultimate_step: self.penultimate_step,
                last_step: self.last_step,
                totals_ms: vec![],
                refresh_time: self.refresh_time,
            }));
            self.last_report = Instant::now();
            self.last_report = Instant::now();
            self.on_time = 0;
            self.count = 0;
        }

        self.last_perf_stats.clone().map(|s| *s)
    }

    pub(crate) fn as_tuples_ms(&self) -> Vec<(String, f32, f32, f32)> {
        let mut default = vec![
            self.total.as_tuple_ms(),
            self.update_gui.as_tuple_ms(),
            self.execute.as_tuple_ms(),
            self.between_renders.as_tuple_ms(),
        ];
        if self.extra_debug.last_ms() != 0.0 {
            default.push(self.extra_debug.as_tuple_ms());
        }
        default
    }
}

pub mod buffer;
pub mod pipeline;
pub mod shader;
pub mod swapchain;
pub mod texture;
pub mod tv_mem;

#[derive(Clone)]
pub struct PerfStats {
    stats: Arc<Mutex<Vec<(String, SystemTime)>>>,
    name: String,
}

impl PerfStats {
    pub fn new(name: &'static str) -> Self {
        Self {
            stats: Arc::new(Mutex::new(Vec::new())),
            name: "@".to_string() + name,
        }
    }

    pub fn start(&self) {
        self.lap("");
    }

    pub fn lap(&self, name: impl AsRef<str>) {
        if let Some((_, last)) = self.stats.try_lock().unwrap().last().cloned() {
            check_ge!(SystemTime::now(), last);
        }
        self.stats
            .try_lock()
            .unwrap()
            .push((name.as_ref().to_string(), SystemTime::now()));
    }

    pub fn report(&self, threshold_us: u128) {
        let span = info_span!("PerfStats", %self.name);
        let _enter = span.enter();
        let stats_ms = self
            .stats
            .try_lock()
            .unwrap()
            .drain(..)
            .tuple_windows()
            .map(|((_, i1), (name, i2))| {
                (
                    name,
                    i2.duration_since(i1).unwrap().as_micros() as f32 / 1000.0,
                )
            })
            .collect_vec();
        let total_ms = stats_ms
            .iter()
            .map(|(_, elapsed_ms)| elapsed_ms)
            .sum::<f32>();
        if total_ms >= threshold_us as f32 / 1000.0 {
            for (name, elapsed_ms) in stats_ms {
                info!("{name}: {:.2} ms", elapsed_ms);
            }
            info!("total: {:.2} ms", total_ms);
        } else {
            info_every_millis!(1000, "{total_ms:.2} ms");
        }
    }
}
