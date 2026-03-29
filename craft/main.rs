#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod vulkan_bindings;
use vulkan_bindings::*;

use bytemuck::{Pod, Zeroable};
use image::GenericImageView;
use rayon::prelude::*;
use rootcause::{Report, report};
use objc2::rc::Retained;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::Path;
use std::ptr;
use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
    color: [f32; 3],
    ao: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ShadowUniform {
    view_proj: [[f32; 4]; 4],
}

struct Camera {
    eye: glam::Vec3,
    yaw: f32,
    pitch: f32,
    up: glam::Vec3,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> glam::Mat4 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();

        let forward = glam::vec3(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();

        let view = glam::Mat4::look_at_rh(self.eye, self.eye + forward, self.up);
        let proj = glam::Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        proj * view
    }
}

struct CameraController {
    speed: f32,
    sensitivity: f32,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    velocity: glam::Vec3,
    on_ground: bool,
}

impl CameraController {
    fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            is_left_pressed: false,
            is_right_pressed: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            velocity: glam::Vec3::ZERO,
            on_ground: false,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        logical_key: char_key @ winit::keyboard::Key::Character(_),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                if let winit::keyboard::Key::Character(c) = char_key {
                    match c.as_str() {
                        "w" | "W" => {
                            self.is_forward_pressed = is_pressed;
                            true
                        }
                        "a" | "A" => {
                            self.is_left_pressed = is_pressed;
                            true
                        }
                        "s" | "S" => {
                            self.is_backward_pressed = is_pressed;
                            true
                        }
                        "d" | "D" => {
                            self.is_right_pressed = is_pressed;
                            true
                        }
                        " " => {
                            self.is_up_pressed = is_pressed;
                            true
                        }
                        _ => false,
                    }
                } else {
                    false
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Shift),
                        ..
                    },
                ..
            } => {
                self.is_down_pressed = *state == ElementState::Pressed;
                true
            }
            _ => false,
        }
    }

    fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal += mouse_dx as f32;
        self.rotate_vertical += mouse_dy as f32;
    }

    fn update_camera(&mut self, camera: &mut Camera, world: &World, dt: f32) {
        let (sin_yaw, cos_yaw) = camera.yaw.sin_cos();
        let forward = glam::vec3(cos_yaw, 0.0, sin_yaw).normalize();
        let right = glam::vec3(-sin_yaw, 0.0, cos_yaw).normalize();

        // Horizontal movement
        let mut move_dir = glam::Vec3::ZERO;
        if self.is_forward_pressed {
            move_dir += forward;
        }
        if self.is_backward_pressed {
            move_dir -= forward;
        }
        if self.is_right_pressed {
            move_dir += right;
        }
        if self.is_left_pressed {
            move_dir -= right;
        }

        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize();
        }

        // Acceleration / Friction
        let target_vel = move_dir * self.speed * 20.0;
        let accel = if self.on_ground { 15.0 } else { 2.0 };
        let friction = if self.on_ground { 10.0 } else { 0.5 };

        let horizontal_vel = glam::vec3(self.velocity.x, 0.0, self.velocity.z);
        let dv = target_vel - horizontal_vel;
        self.velocity.x += dv.x * accel * dt;
        self.velocity.z += dv.z * accel * dt;

        // Friction
        let friction_factor = (1.0 - friction * dt).max(0.0);
        self.velocity.x *= friction_factor;
        self.velocity.z *= friction_factor;

        // Gravity
        self.velocity.y -= 30.0 * dt;

        // Jump
        if self.is_up_pressed && self.on_ground {
            self.velocity.y = 10.0;
            self.on_ground = false;
        }

        // Apply movement with collision handling
        let player_height: f32 = 1.6; // eyes are at 1.6 height
        let p_r = 0.3; // player radius

        let mut new_eye = camera.eye;

        // Helper to check 4 corners
        let is_colliding = |pos: glam::Vec3, height_offset: f32| {
            world.is_blocked(pos.x - p_r, pos.y + height_offset, pos.z - p_r)
                || world.is_blocked(pos.x + p_r, pos.y + height_offset, pos.z - p_r)
                || world.is_blocked(pos.x - p_r, pos.y + height_offset, pos.z + p_r)
                || world.is_blocked(pos.x + p_r, pos.y + height_offset, pos.z + p_r)
        };

        // Y Movement (Vertical)
        new_eye.y += self.velocity.y * dt;
        if self.velocity.y < 0.0 {
            // Check at feet (-player_height from eye)
            if is_colliding(new_eye, -player_height) {
                new_eye.y = (new_eye.y - player_height).ceil() + player_height;
                self.velocity.y = 0.0;
                self.on_ground = true;
            } else {
                self.on_ground = false;
            }
        } else if self.velocity.y > 0.0 {
            // Check at head (0.2 above eye)
            if is_colliding(new_eye, 0.2) {
                new_eye.y = (new_eye.y + 0.2).floor() - 0.2;
                self.velocity.y = 0.0;
            }
        }

        // X Movement
        let old_x = new_eye.x;
        new_eye.x += self.velocity.x * dt;
        // Check at feet and head
        if is_colliding(new_eye, -player_height + 0.1) || is_colliding(new_eye, 0.0) {
            new_eye.x = old_x;
            self.velocity.x = 0.0;
        }

        // Z Movement
        let old_z = new_eye.z;
        new_eye.z += self.velocity.z * dt;
        if is_colliding(new_eye, -player_height + 0.1) || is_colliding(new_eye, 0.0) {
            new_eye.z = old_z;
            self.velocity.z = 0.0;
        }

        camera.eye = new_eye;

        camera.yaw += self.rotate_horizontal * self.sensitivity;
        camera.pitch -= self.rotate_vertical * self.sensitivity;

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        if camera.pitch < -std::f32::consts::FRAC_PI_2 + 0.001 {
            camera.pitch = -std::f32::consts::FRAC_PI_2 + 0.001;
        } else if camera.pitch > std::f32::consts::FRAC_PI_2 - 0.001 {
            camera.pitch = std::f32::consts::FRAC_PI_2 - 0.001;
        }
    }
}

const CHUNK_SIZE: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
struct BlockType(u8);

impl BlockType {
    const AIR: Self = Self(0);
    const DIRT: Self = Self(1);
    const GRASS: Self = Self(2);
    const STONE: Self = Self(3);
}

#[derive(Clone)]
struct Chunk {
    data: [BlockType; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
}

struct CompressedChunk {
    data: Vec<u8>,
}

impl CompressedChunk {
    fn from_chunk(chunk: &Chunk) -> Self {
        let bytes: &[u8] = bytemuck::cast_slice(&chunk.data);
        let compressed = lz4_flex::compress(bytes);
        Self { data: compressed }
    }

    fn to_chunk(&self) -> Chunk {
        let decompressed =
            lz4_flex::decompress(&self.data, CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE).unwrap();
        let data: [BlockType; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE] =
            bytemuck::cast_slice(&decompressed).try_into().unwrap();
        Chunk { data }
    }
}

impl Chunk {
    fn new(pos: (i32, i32, i32)) -> Self {
        let mut data = [BlockType::AIR; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = pos.0 * CHUNK_SIZE as i32 + x as i32;
                let world_z = pos.2 * CHUNK_SIZE as i32 + z as i32;

                // Better terrain gen with multiple octaves
                let h1 = (world_x as f32 * 0.05).sin() * (world_z as f32 * 0.05).cos() * 8.0;
                let h2 = (world_x as f32 * 0.01).sin() * (world_z as f32 * 0.01).cos() * 15.0;
                let height = (20.0 + h1 + h2) as i32;

                let chunk_y_start = pos.1 * CHUNK_SIZE as i32;
                for y in 0..CHUNK_SIZE {
                    let world_y = chunk_y_start + y as i32;
                    if world_y < height {
                        let idx = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
                        if world_y == height - 1 {
                            data[idx] = BlockType::GRASS;
                        } else if world_y > height - 4 {
                            data[idx] = BlockType::DIRT;
                        } else {
                            data[idx] = BlockType::STONE;
                        }
                    }
                }
            }
        }
        Self { data }
    }

    fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if x < 0
            || x >= CHUNK_SIZE as i32
            || y < 0
            || y >= CHUNK_SIZE as i32
            || z < 0
            || z >= CHUNK_SIZE as i32
        {
            return BlockType::AIR;
        }
        self.data[x as usize + (y as usize) * CHUNK_SIZE + (z as usize) * CHUNK_SIZE * CHUNK_SIZE]
    }
}

struct World {
    chunks: HashMap<(i32, i32, i32), CompressedChunk>,
    cache: std::sync::RwLock<HashMap<(i32, i32, i32), (Arc<Chunk>, Instant)>>,
}

#[allow(dead_code)]
impl World {
    fn new() -> Self {
        if !Path::new("saves").exists() {
            fs::create_dir("saves").unwrap();
        }
        Self {
            chunks: HashMap::new(),
            cache: std::sync::RwLock::new(HashMap::new()),
        }
    }

    fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        let cx = x.div_euclid(CHUNK_SIZE as i32);
        let cy = y.div_euclid(CHUNK_SIZE as i32);
        let cz = z.div_euclid(CHUNK_SIZE as i32);

        let bx = x.rem_euclid(CHUNK_SIZE as i32);
        let by = y.rem_euclid(CHUNK_SIZE as i32);
        let bz = z.rem_euclid(CHUNK_SIZE as i32);

        // Decompress and cache
        self.get_chunk(cx, cy, cz)
            .map(|chunk| chunk.get_block(bx, by, bz))
            .unwrap_or(BlockType::AIR)
    }

    fn get_chunk(&self, x: i32, y: i32, z: i32) -> Option<Arc<Chunk>> {
        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some((chunk, _)) = cache.get(&(x, y, z)) {
                return Some(chunk.clone());
            }
        }

        // Decompress and cache
        if let Some(compressed) = self.chunks.get(&(x, y, z)) {
            let chunk = Arc::new(compressed.to_chunk());
            let mut cache = self.cache.write().unwrap();

            // Cache eviction (updated to 2048 for better performance)
            if cache.len() > 2048 {
                let mut keys: Vec<_> = cache.iter().map(|(&k, &(_, v))| (k, v)).collect();
                keys.sort_by_key(|&(_, v)| v);
                for (k, _) in keys.iter().take(512) {
                    cache.remove(k);
                }
            }

            cache.insert((x, y, z), (chunk.clone(), Instant::now()));
            return Some(chunk);
        }
        None
    }

    fn is_blocked(&self, x: f32, y: f32, z: f32) -> bool {
        self.get_block(x.floor() as i32, y.floor() as i32, z.floor() as i32) != BlockType::AIR
    }

    fn generate_around(&mut self, player_pos: glam::Vec3, radius: i32) -> bool {
        let px = (player_pos.x / CHUNK_SIZE as f32).floor() as i32;
        let py = (player_pos.y / CHUNK_SIZE as f32).floor() as i32;
        let pz = (player_pos.z / CHUNK_SIZE as f32).floor() as i32;

        let mut chunks_to_gen = Vec::new();
        for x in (px - radius)..(px + radius) {
            for z in (pz - radius)..(pz + radius) {
                for y in (py - 2)..(py + 2) {
                    if !self.chunks.contains_key(&(x, y, z)) {
                        chunks_to_gen.push((x, y, z));
                    }
                }
            }
        }

        if chunks_to_gen.is_empty() {
            return false;
        }

        let new_chunks: Vec<((i32, i32, i32), CompressedChunk)> = chunks_to_gen
            .into_par_iter()
            .map(|(x, y, z)| {
                let chunk = self
                    .load_chunk(x, y, z)
                    .unwrap_or_else(|| CompressedChunk::from_chunk(&Chunk::new((x, y, z))));
                ((x, y, z), chunk)
            })
            .collect();

        for (pos, chunk) in new_chunks {
            self.chunks.insert(pos, chunk);
        }
        true
    }

    fn save_chunk(&self, x: i32, y: i32, z: i32) -> std::io::Result<()> {
        if let Some(chunk) = self.chunks.get(&(x, y, z)) {
            let path = format!("saves/chunk_{}_{}_{}.lz4", x, y, z);
            fs::write(path, &chunk.data)?;
        }
        Ok(())
    }

    fn load_chunk(&self, x: i32, y: i32, z: i32) -> Option<CompressedChunk> {
        let path = format!("saves/chunk_{}_{}_{}.lz4", x, y, z);
        fs::read(path).ok().map(|data| CompressedChunk { data })
    }

    fn save_all(&self) {
        for (&(x, y, z), _) in &self.chunks {
            let _ = self.save_chunk(x, y, z);
        }
    }
}

const VK_KHR_SURFACE_EXTENSION_NAME: &[u8] = b"VK_KHR_surface\0";
const VK_EXT_METAL_SURFACE_EXTENSION_NAME: &[u8] = b"VK_EXT_metal_surface\0";
const VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME: &[u8] = b"VK_KHR_portability_enumeration\0";
const VK_KHR_SWAPCHAIN_EXTENSION_NAME: &[u8] = b"VK_KHR_swapchain\0";

const fn vk_make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}
const VK_API_VERSION_1_0: u32 = vk_make_api_version(0, 1, 0, 0);

struct VulkanState {
    instance: VkInstance,
    physical_device: VkPhysicalDevice,
    device: VkDevice,
    queue: VkQueue,
    surface: VkSurfaceKHR,
    swapchain: VkSwapchainKHR,
    swapchain_images: Vec<VkImage>,
    swapchain_image_views: Vec<VkImageView>,
    swapchain_format: VkFormat,
    swapchain_extent: VkExtent2D,
}

#[allow(non_camel_case_types)]
type PFN_vkCreateMetalSurfaceEXT = unsafe extern "system" fn(
    instance: VkInstance,
    pCreateInfo: *const VkMetalSurfaceCreateInfoEXT,
    pAllocator: *const VkAllocationCallbacks,
    pSurface: *mut VkSurfaceKHR,
) -> VkResult;

#[allow(non_camel_case_types)]
type PFN_vkGetPhysicalDeviceSurfaceSupportKHR = unsafe extern "system" fn(
    physicalDevice: VkPhysicalDevice,
    queueFamilyIndex: u32,
    surface: VkSurfaceKHR,
    pSupported: *mut VkBool32,
) -> VkResult;

#[allow(non_camel_case_types)]
type PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR = unsafe extern "system" fn(
    physicalDevice: VkPhysicalDevice,
    surface: VkSurfaceKHR,
    pSurfaceCapabilities: *mut VkSurfaceCapabilitiesKHR,
) -> VkResult;

#[allow(non_camel_case_types)]
type PFN_vkGetPhysicalDeviceSurfaceFormatsKHR = unsafe extern "system" fn(
    physicalDevice: VkPhysicalDevice,
    surface: VkSurfaceKHR,
    pSurfaceFormatCount: *mut u32,
    pSurfaceFormats: *mut VkSurfaceFormatKHR,
) -> VkResult;

#[allow(non_camel_case_types)]
type PFN_vkGetPhysicalDeviceSurfacePresentModesKHR = unsafe extern "system" fn(
    physicalDevice: VkPhysicalDevice,
    surface: VkSurfaceKHR,
    pPresentModeCount: *mut u32,
    pPresentModes: *mut VkPresentModeKHR,
) -> VkResult;

#[allow(non_camel_case_types)]
type PFN_vkCreateSwapchainKHR = unsafe extern "system" fn(
    device: VkDevice,
    pCreateInfo: *const VkSwapchainCreateInfoKHR,
    pAllocator: *const VkAllocationCallbacks,
    pSwapchain: *mut VkSwapchainKHR,
) -> VkResult;

#[allow(non_camel_case_types)]
type PFN_vkGetSwapchainImagesKHR = unsafe extern "system" fn(
    device: VkDevice,
    swapchain: VkSwapchainKHR,
    pSwapchainImageCount: *mut u32,
    pSwapchainImages: *mut VkImage,
) -> VkResult;

macro_rules! load_instance_func {
    ($instance:expr, $name:ident, $type:ty) => {
        unsafe {
            let name = CString::new(stringify!($name)).unwrap();
            let addr = vkGetInstanceProcAddr($instance, name.as_ptr());
            let func: Option<$type> = std::mem::transmute(addr);
            func.ok_or_else(|| report!(format!("Failed to load instance function: {}", stringify!($name))))?
        }
    };
}

macro_rules! load_device_func {
    ($device:expr, $name:ident, $type:ty) => {
        unsafe {
            let name = CString::new(stringify!($name)).unwrap();
            let addr = vkGetDeviceProcAddr($device, name.as_ptr());
            let func: Option<$type> = std::mem::transmute(addr);
            func.ok_or_else(|| report!(format!("Failed to load device function: {}", stringify!($name))))?
        }
    };
}

impl VulkanState {
    unsafe fn new(window: &dyn Window) -> Result<Self, Report> {
        // Instance creation
        let app_info = VkApplicationInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext: ptr::null(),
            pApplicationName: b"Craft\0".as_ptr() as *const i8,
            applicationVersion: 1,
            pEngineName: b"No Engine\0".as_ptr() as *const i8,
            engineVersion: 1,
            apiVersion: VK_API_VERSION_1_0,
        };

        let mut extension_prop_count = 0;
        vkEnumerateInstanceExtensionProperties(
            ptr::null(),
            &mut extension_prop_count,
            ptr::null_mut(),
        );
        let mut extension_props: Vec<VkExtensionProperties> =
            Vec::with_capacity(extension_prop_count as usize);
        vkEnumerateInstanceExtensionProperties(
            ptr::null(),
            &mut extension_prop_count,
            extension_props.as_mut_ptr(),
        );
        extension_props.set_len(extension_prop_count as usize);

        println!("Available extensions:");
        for prop in &extension_props {
            let name = CStr::from_ptr(prop.extensionName.as_ptr()).to_string_lossy();
            println!("  {}", name);
        }
        std::io::stdout().flush().unwrap();

        // For macOS, we need VK_KHR_portability_enumeration and VK_KHR_surface + VK_EXT_metal_surface
        let extensions = vec![
            VK_KHR_SURFACE_EXTENSION_NAME.as_ptr() as *const i8,
            VK_EXT_METAL_SURFACE_EXTENSION_NAME.as_ptr() as *const i8,
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME.as_ptr() as *const i8,
        ];

        let create_info = VkInstanceCreateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext: ptr::null(),
            flags: VkInstanceCreateFlagBits::VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
                as u32,
            pApplicationInfo: &app_info,
            enabledLayerCount: 0,
            ppEnabledLayerNames: ptr::null(),
            enabledExtensionCount: extensions.len() as u32,
            ppEnabledExtensionNames: extensions.as_ptr(),
        };

        let mut instance = std::mem::MaybeUninit::uninit();
        let result = vkCreateInstance(&create_info, ptr::null(), instance.as_mut_ptr());
        if result != VkResult::VK_SUCCESS {
            return Err(report!(format!("Failed to create Vulkan instance: {:?}", result)).into());
        }
        let instance = instance.assume_init();

        // Load instance functions
        let vkCreateMetalSurfaceEXT = load_instance_func!(instance, vkCreateMetalSurfaceEXT, PFN_vkCreateMetalSurfaceEXT);
        let vkGetPhysicalDeviceSurfaceSupportKHR = load_instance_func!(instance, vkGetPhysicalDeviceSurfaceSupportKHR, PFN_vkGetPhysicalDeviceSurfaceSupportKHR);
        let vkGetPhysicalDeviceSurfaceCapabilitiesKHR = load_instance_func!(instance, vkGetPhysicalDeviceSurfaceCapabilitiesKHR, PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
        let vkGetPhysicalDeviceSurfaceFormatsKHR = load_instance_func!(instance, vkGetPhysicalDeviceSurfaceFormatsKHR, PFN_vkGetPhysicalDeviceSurfaceFormatsKHR);
        let vkGetPhysicalDeviceSurfacePresentModesKHR = load_instance_func!(instance, vkGetPhysicalDeviceSurfacePresentModesKHR, PFN_vkGetPhysicalDeviceSurfacePresentModesKHR);

        // Surface creation
        let mut surface = std::mem::MaybeUninit::uninit();
        #[cfg(target_os = "macos")]
        {
            use objc2_app_kit::NSView;
            use objc2_quartz_core::CAMetalLayer;
            use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

            if let Ok(handle) = window.window_handle() {
                if let RawWindowHandle::AppKit(handle) = handle.as_raw() {
                    println!("Creating Metal surface on macOS...");
                    std::io::stdout().flush().unwrap();

                    unsafe {
                        let view = handle.ns_view.as_ptr() as *mut NSView;
                        (*view).setWantsLayer(true);
                        let layer = CAMetalLayer::new();
                        (*view).setLayer(Some(std::mem::transmute(layer.clone())));

                        let surface_create_info = VkMetalSurfaceCreateInfoEXT {
                            sType: VkStructureType::VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
                            pNext: ptr::null(),
                            flags: 0,
                            pLayer: Retained::as_ptr(&layer) as *const _,
                        };
                        let result = vkCreateMetalSurfaceEXT(
                            instance,
                            &surface_create_info,
                            ptr::null(),
                            surface.as_mut_ptr(),
                        );
                        if result != VkResult::VK_SUCCESS {
                            return Err(report!(format!(
                                "Failed to create Metal surface: {:?}",
                                result
                            ))
                            .into());
                        }
                    }
                    println!("Metal surface created successfully");
                    std::io::stdout().flush().unwrap();
                } else {
                    return Err(report!("Not an AppKit window handle").into());
                }
            } else {
                return Err(report!("Failed to get window handle").into());
            }
        }

        let surface = surface.assume_init();
        if surface == ptr::null_mut() {
            return Err(report!("Surface handle is null after creation").into());
        }

        // Physical device selection
        let mut device_count = 0;
        vkEnumeratePhysicalDevices(instance, &mut device_count, ptr::null_mut());
        println!("Found {} physical devices", device_count);
        std::io::stdout().flush().unwrap();
        if device_count == 0 {
            return Err(report!("No Vulkan physical devices found").into());
        }
        let mut devices: Vec<VkPhysicalDevice> = Vec::with_capacity(device_count as usize);
        vkEnumeratePhysicalDevices(instance, &mut device_count, devices.as_mut_ptr());
        devices.set_len(device_count as usize);
        let physical_device = devices[0]; // Just take the first one for now

        // Find queue family with graphics and presentation support
        let mut queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &mut queue_family_count, ptr::null_mut());
        let mut queue_families: Vec<VkQueueFamilyProperties> = Vec::with_capacity(queue_family_count as usize);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &mut queue_family_count, queue_families.as_mut_ptr());
        queue_families.set_len(queue_family_count as usize);

        let mut graphics_family = None;
        for (i, family) in queue_families.iter().enumerate() {
            if (family.queueFlags & VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT as u32) != 0 {
                let mut present_support = 0;
                unsafe {
                    vkGetPhysicalDeviceSurfaceSupportKHR(
                        physical_device,
                        i as u32,
                        surface,
                        &mut present_support,
                    );
                }
                if present_support != 0 {
                    graphics_family = Some(i as u32);
                    break;
                }
            }
        }

        let queue_family_index = graphics_family
            .ok_or_else(|| report!("Failed to find a suitable queue family"))?;
        println!("Selected queue family index: {}", queue_family_index);
        std::io::stdout().flush().unwrap();

        let mut dev_extension_prop_count = 0;
        vkEnumerateDeviceExtensionProperties(
            physical_device,
            ptr::null(),
            &mut dev_extension_prop_count,
            ptr::null_mut(),
        );
        let mut dev_extension_props: Vec<VkExtensionProperties> =
            Vec::with_capacity(dev_extension_prop_count as usize);
        vkEnumerateDeviceExtensionProperties(
            physical_device,
            ptr::null(),
            &mut dev_extension_prop_count,
            dev_extension_props.as_mut_ptr(),
        );
        dev_extension_props.set_len(dev_extension_prop_count as usize);

        println!(
            "Available device extensions ({}):",
            dev_extension_prop_count
        );
        let mut has_portability_subset = false;
        for prop in &dev_extension_props {
            let name = CStr::from_ptr(prop.extensionName.as_ptr()).to_string_lossy();
            println!("  {}", name);
            if name == "VK_KHR_portability_subset" {
                has_portability_subset = true;
            }
        }
        std::io::stdout().flush().unwrap();

        // Device creation
        let queue_priority = 1.0f32;
        let queue_create_info = VkDeviceQueueCreateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queueFamilyIndex: queue_family_index,
            queueCount: 1,
            pQueuePriorities: &queue_priority,
        };

        let mut device_extensions = vec![VK_KHR_SWAPCHAIN_EXTENSION_NAME.as_ptr() as *const i8];
        if has_portability_subset {
            device_extensions.push(b"VK_KHR_portability_subset\0".as_ptr() as *const i8);
        }

        let device_create_info = VkDeviceCreateInfo {
            sType: VkStructureType::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queueCreateInfoCount: 1,
            pQueueCreateInfos: &queue_create_info,
            enabledLayerCount: 0,
            ppEnabledLayerNames: ptr::null(),
            enabledExtensionCount: device_extensions.len() as u32,
            ppEnabledExtensionNames: device_extensions.as_ptr(),
            pEnabledFeatures: ptr::null(),
        };

        let mut device = std::mem::MaybeUninit::uninit();
        println!("Creating logical device...");
        std::io::stdout().flush().unwrap();
        let result = vkCreateDevice(
            physical_device,
            &device_create_info,
            ptr::null(),
            device.as_mut_ptr(),
        );
        if result != VkResult::VK_SUCCESS {
            return Err(report!(format!("Failed to create logical device: {:?}", result)).into());
        }
        let device = device.assume_init();
        println!("Logical device created successfully");
        std::io::stdout().flush().unwrap();

        // Load device functions
        let vkCreateSwapchainKHR =
            load_device_func!(device, vkCreateSwapchainKHR, PFN_vkCreateSwapchainKHR);
        let vkGetSwapchainImagesKHR =
            load_device_func!(device, vkGetSwapchainImagesKHR, PFN_vkGetSwapchainImagesKHR);

        let mut queue = std::mem::MaybeUninit::uninit();
        vkGetDeviceQueue(device, queue_family_index, 0, queue.as_mut_ptr());
        let queue = queue.assume_init();

        // Swapchain creation
        let mut surface_capabilities = std::mem::MaybeUninit::uninit();
        unsafe {
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                physical_device,
                surface,
                surface_capabilities.as_mut_ptr(),
            );
        }
        let surface_capabilities = surface_capabilities.assume_init();

        let mut format_count = 0;
        unsafe {
            vkGetPhysicalDeviceSurfaceFormatsKHR(
                physical_device,
                surface,
                &mut format_count,
                ptr::null_mut(),
            );
        }
        let mut surface_formats: Vec<VkSurfaceFormatKHR> =
            Vec::with_capacity(format_count as usize);
        unsafe {
            vkGetPhysicalDeviceSurfaceFormatsKHR(
                physical_device,
                surface,
                &mut format_count,
                surface_formats.as_mut_ptr(),
            );
        }
        surface_formats.set_len(format_count as usize);

        let surface_format = surface_formats
            .iter()
            .find(|f| {
                f.format == VkFormat::VK_FORMAT_B8G8R8A8_UNORM
                    && f.colorSpace == VkColorSpaceKHR::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
            })
            .cloned()
            .unwrap_or(surface_formats[0]);

        let extent = if surface_capabilities.currentExtent.width != u32::MAX {
            surface_capabilities.currentExtent
        } else {
            VkExtent2D {
                width: window
                    .outer_size()
                    .width
                    .clamp(
                        surface_capabilities.minImageExtent.width,
                        surface_capabilities.maxImageExtent.width,
                    ),
                height: window
                    .outer_size()
                    .height
                    .clamp(
                        surface_capabilities.minImageExtent.height,
                        surface_capabilities.maxImageExtent.height,
                    ),
            }
        };

        let mut image_count = surface_capabilities.minImageCount + 1;
        if surface_capabilities.maxImageCount > 0
            && image_count > surface_capabilities.maxImageCount
        {
            image_count = surface_capabilities.maxImageCount;
        }

        println!(
            "Swapchain info: format={:?}, extent={:?}x{:?}, images={}",
            surface_format.format, extent.width, extent.height, image_count
        );
        std::io::stdout().flush().unwrap();

        let swapchain_create_info = VkSwapchainCreateInfoKHR {
            sType: VkStructureType::VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            pNext: ptr::null(),
            flags: 0,
            surface,
            minImageCount: image_count,
            imageFormat: surface_format.format,
            imageColorSpace: surface_format.colorSpace,
            imageExtent: extent,
            imageArrayLayers: 1,
            imageUsage: VkImageUsageFlagBits::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT as u32,
            imageSharingMode: VkSharingMode::VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
            preTransform: surface_capabilities.currentTransform,
            compositeAlpha: VkCompositeAlphaFlagBitsKHR::VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode: VkPresentModeKHR::VK_PRESENT_MODE_FIFO_KHR,
            clipped: VK_TRUE as u32,
            oldSwapchain: ptr::null_mut(),
        };

        let mut swapchain = std::mem::MaybeUninit::uninit();
        println!("Creating swapchain...");
        std::io::stdout().flush().unwrap();
        let result = unsafe {
            vkCreateSwapchainKHR(
                device,
                &swapchain_create_info,
                ptr::null(),
                swapchain.as_mut_ptr(),
            )
        };
        if result != VkResult::VK_SUCCESS {
            return Err(report!(format!("Failed to create swapchain: {:?}", result)).into());
        }
        let swapchain = swapchain.assume_init();
        println!("Swapchain created successfully");
        std::io::stdout().flush().unwrap();

        let mut image_count = 0;
        unsafe {
            vkGetSwapchainImagesKHR(device, swapchain, &mut image_count, ptr::null_mut());
        }
        let mut swapchain_images: Vec<VkImage> = Vec::with_capacity(image_count as usize);
        unsafe {
            vkGetSwapchainImagesKHR(
                device,
                swapchain,
                &mut image_count,
                swapchain_images.as_mut_ptr(),
            );
        }
        swapchain_images.set_len(image_count as usize);

        let mut swapchain_image_views = Vec::new();
        for image in &swapchain_images {
            let view_create_info = VkImageViewCreateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                image: *image,
                viewType: VkImageViewType::VK_IMAGE_VIEW_TYPE_2D,
                format: surface_format.format,
                components: VkComponentMapping {
                    r: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
                    g: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
                    b: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
                    a: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
                },
                subresourceRange: VkImageSubresourceRange {
                    aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as u32,
                    baseMipLevel: 0,
                    levelCount: 1,
                    baseArrayLayer: 0,
                    layerCount: 1,
                },
            };
            let mut view = std::mem::MaybeUninit::uninit();
            vkCreateImageView(device, &view_create_info, ptr::null(), view.as_mut_ptr());
            swapchain_image_views.push(view.assume_init());
        }

        Ok(Self {
            instance,
            physical_device,
            device,
            queue,
            surface,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format: surface_format.format,
            swapchain_extent: extent,
        })
    }
}

struct State {
    vulkan: VulkanState,
    window: Arc<dyn Window>,
    camera: Camera,
    camera_controller: CameraController,
    last_frame: Instant,
    world: World,
    mouse_locked: bool,
}

impl State {
    fn new(window: Arc<dyn Window>) -> Result<Self, Report> {
        let vulkan = unsafe { VulkanState::new(window.as_ref())? };

        let camera = Camera {
            eye: (0.0, 64.0, 0.0).into(),
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            up: glam::Vec3::Y,
            aspect: window.outer_size().width as f32 / window.outer_size().height as f32,
            fovy: 45.0f32.to_radians(),
            znear: 0.1,
            zfar: 1000.0,
        };

        let camera_controller = CameraController::new(10.0, 0.005);
        let mut world = World::new();
        let _ = world.generate_around(camera.eye, 6);

        Ok(Self {
            vulkan,
            window,
            camera,
            camera_controller,
            last_frame: Instant::now(),
            world,
            mouse_locked: true,
        })
    }

    pub fn window(&self) -> &dyn Window {
        self.window.as_ref()
    }

    fn resize(&mut self, _new_size: winit::dpi::PhysicalSize<u32>) {
        // Handle swapchain recreation here
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        self.camera_controller
            .update_camera(&mut self.camera, &self.world, dt);

        self.world.generate_around(self.camera.eye, 16);
    }

    fn render(&mut self) {
        // Implement Vulkan rendering here
    }
}

struct App {
    window: Option<Arc<dyn Window>>,
    state: Option<State>,
    minimized: bool,
    focused: bool,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            state: None,
            minimized: false,
            focused: true,
        }
    }

    fn init(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        println!("Creating window...");
        std::io::stdout().flush().unwrap();

        let window_attributes = WindowAttributes::default()
            .with_title("Craft")
            .with_visible(true);

        let window = match event_loop.create_window(window_attributes) {
            Ok(w) => Arc::from(w),
            Err(e) => {
                eprintln!("Failed to create window: {:?}", e);
                event_loop.exit();
                return;
            }
        };

        self.window = Some(window);
    }

    fn ensure_state(&mut self) -> bool {
        if self.state.is_some() {
            return true;
        }
        if let Some(window) = &self.window {
            println!("Initializing Vulkan state...");
            std::io::stdout().flush().unwrap();
            match State::new(window.clone()) {
                Ok(state) => {
                    println!("State initialized successfully");
                    std::io::stdout().flush().unwrap();
                    state.window().set_cursor_visible(!state.mouse_locked);
                    let grab_mode = if state.mouse_locked {
                        winit::window::CursorGrabMode::Locked
                    } else {
                        winit::window::CursorGrabMode::None
                    };
                    let _ = state.window().set_cursor_grab(grab_mode);
                    self.state = Some(state);
                    return true;
                }
                Err(e) => {
                    eprintln!("Failed to initialize state: {}", e);
                    std::io::stdout().flush().unwrap();
                    return false;
                }
            }
        }
        false
    }
}

impl ApplicationHandler for App {
    fn new_events(&mut self, _event_loop: &dyn ActiveEventLoop, cause: StartCause) {
        if matches!(cause, StartCause::Init) {
            println!("new_events called: {:?}", cause);
            std::io::stdout().flush().unwrap();
        }
    }

    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        println!("can_create_surfaces called");
        std::io::stdout().flush().unwrap();
        self.init(event_loop);
    }

    fn resumed(&mut self, event_loop: &dyn ActiveEventLoop) {
        println!("resumed called");
        std::io::stdout().flush().unwrap();
        self.init(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if !self.minimized && self.ensure_state() {
                    if let Some(state) = &mut self.state {
                        state.update();
                        state.render();
                    }
                }
            }
            _ => {
                if let Some(state) = &mut self.state {
                    if !state.input(&event) {
                        match event {
                            WindowEvent::Focused(focused) => {
                                self.focused = focused;
                                if focused && state.mouse_locked {
                                    let _ = state
                                        .window()
                                        .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                                        .or_else(|_| {
                                            state
                                                .window()
                                                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                                        });
                                    state.window().set_cursor_visible(false);
                                }
                            }
                            WindowEvent::Occluded(occluded) => {
                                self.minimized = occluded;
                            }
                            WindowEvent::KeyboardInput {
                                event:
                                    KeyEvent {
                                        state: ElementState::Pressed,
                                        logical_key:
                                            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
                                        ..
                                    },
                                ..
                            } => {
                                state.mouse_locked = !state.mouse_locked;
                                state.window().set_cursor_visible(!state.mouse_locked);
                                let grab_mode = if state.mouse_locked {
                                    winit::window::CursorGrabMode::Locked
                                } else {
                                    winit::window::CursorGrabMode::None
                                };
                                let _ = state.window().set_cursor_grab(grab_mode).or_else(|_| {
                                    state
                                        .window()
                                        .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                                });
                            }
                            WindowEvent::PointerButton {
                                state: ElementState::Pressed,
                                button: ButtonSource::Mouse(MouseButton::Left),
                                ..
                            } => {
                                if !state.mouse_locked {
                                    state.mouse_locked = true;
                                    state.window().set_cursor_visible(false);
                                    let _ = state
                                        .window()
                                        .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                                        .or_else(|_| {
                                            state
                                                .window()
                                                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                                        });
                                }
                            }
                            WindowEvent::SurfaceResized(physical_size) => {
                                if physical_size.width == 0 || physical_size.height == 0 {
                                    self.minimized = true;
                                } else {
                                    self.minimized = false;
                                    state.resize(physical_size);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &dyn ActiveEventLoop,
        _device_id: Option<DeviceId>,
        event: DeviceEvent,
    ) {
        if let Some(state) = &mut self.state {
            if state.mouse_locked {
                if let winit::event::DeviceEvent::PointerMotion { delta } = event {
                    state.camera_controller.process_mouse(delta.0, delta.1);
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &dyn ActiveEventLoop) {
        if let Some(window) = &self.window {
            if !self.minimized {
                window.request_redraw();
            }
        }
    }
}

fn main() {
    println!("Main starting...");
    std::io::stdout().flush().unwrap();
    tracing_subscriber::fmt::init();
    println!("EventLoop creating...");
    std::io::stdout().flush().unwrap();
    let event_loop = EventLoop::new().unwrap();
    let app = App::new();
    println!("Running app...");
    std::io::stdout().flush().unwrap();
    event_loop.run_app(app).unwrap();
}
