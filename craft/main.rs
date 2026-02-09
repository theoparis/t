use bytemuck::{Pod, Zeroable};
use image::GenericImageView;
use rayon::prelude::*;
use rootcause::{Report, report};
use std::collections::HashMap;
use std::fs;
use std::io::{Cursor, Read};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
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
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>())
                        as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>()
                        + std::mem::size_of::<[f32; 2]>()
                        + std::mem::size_of::<[f32; 3]>())
                        as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CameraUniform {
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
        let compressed = zstd::encode_all(bytes, 3).unwrap();
        Self { data: compressed }
    }

    fn to_chunk(&self) -> Chunk {
        let decompressed = zstd::decode_all(Cursor::new(&self.data)).unwrap();
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
    cache: std::sync::RwLock<HashMap<(i32, i32, i32), (Chunk, Instant)>>,
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

        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some((chunk, _)) = cache.get(&(cx, cy, cz)) {
                return chunk.get_block(bx, by, bz);
            }
        }

        // Decompress and cache
        if let Some(compressed) = self.chunks.get(&(cx, cy, cz)) {
            let chunk = compressed.to_chunk();
            let mut cache = self.cache.write().unwrap();

            // Basic cache eviction (stay under 64 chunks)
            if cache.len() > 64 {
                let mut keys: Vec<_> = cache.iter().map(|(&k, &(_, v))| (k, v)).collect();
                keys.sort_by_key(|&(_, v)| v);
                for (k, _) in keys.iter().take(16) {
                    cache.remove(k);
                }
            }

            cache.insert((cx, cy, cz), (chunk.clone(), Instant::now()));
            return chunk.get_block(bx, by, bz);
        }

        BlockType::AIR
    }

    fn is_blocked(&self, x: f32, y: f32, z: f32) -> bool {
        self.get_block(x.floor() as i32, y.floor() as i32, z.floor() as i32) != BlockType::AIR
    }

    fn generate_around(&mut self, player_pos: glam::Vec3, radius: i32) -> bool {
        let px = (player_pos.x / CHUNK_SIZE as f32).floor() as i32;
        let py = (player_pos.y / CHUNK_SIZE as f32).floor() as i32;
        let pz = (player_pos.z / CHUNK_SIZE as f32).floor() as i32;

        let mut changed = false;
        for x in (px - radius)..(px + radius) {
            for z in (pz - radius)..(pz + radius) {
                for y in (py - 2)..(py + 2) {
                    if !self.chunks.contains_key(&(x, y, z)) {
                        let chunk = self
                            .load_chunk(x, y, z)
                            .unwrap_or_else(|| CompressedChunk::from_chunk(&Chunk::new((x, y, z))));
                        self.chunks.insert((x, y, z), chunk);
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    fn save_chunk(&self, x: i32, y: i32, z: i32) -> std::io::Result<()> {
        if let Some(chunk) = self.chunks.get(&(x, y, z)) {
            let path = format!("saves/chunk_{}_{}_{}.zst", x, y, z);
            fs::write(path, &chunk.data)?;
        }
        Ok(())
    }

    fn load_chunk(&self, x: i32, y: i32, z: i32) -> Option<CompressedChunk> {
        let path = format!("saves/chunk_{}_{}_{}.zst", x, y, z);
        fs::read(path).ok().map(|data| CompressedChunk { data })
    }

    fn save_all(&self) {
        for (&(x, y, z), _) in &self.chunks {
            let _ = self.save_chunk(x, y, z);
        }
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Arc<dyn Window>,
    render_pipeline: wgpu::RenderPipeline,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_texture: Texture,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    diffuse_bind_group: wgpu::BindGroup,
    last_frame: Instant,
    world: World,
    mouse_locked: bool,
}

impl State {
    async fn new(window: Box<dyn Window>) -> Result<Self, Report> {
        rustls_graviola::default_provider()
            .install_default()
            .unwrap();

        let window: Arc<dyn Window> = Arc::from(window);
        let size = window.surface_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .map_err(Report::new)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| report!(e))?;

        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e: wgpu::RequestDeviceError| report!(e))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Assets
        let diffuse_bytes = download_assets().await?;
        let diffuse_texture = Texture::from_bytes(&device, &queue, &diffuse_bytes, "atlas.png")?;

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let camera = Camera {
            eye: (0.0, 64.0, 0.0).into(),
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            up: glam::Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0f32.to_radians(),
            znear: 0.1,
            zfar: 1000.0,
        };

        let camera_controller = CameraController::new(1.0, 0.005);

        let camera_uniform = CameraUniform {
            view_proj: camera.build_view_projection_matrix().to_cols_array_2d(),
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                ..Default::default()
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
            multiview_mask: None,
        });

        let depth_texture = Texture::create_depth_texture(&device, &config, "depth_texture");

        // Generate World
        let mut world = World::new();
        world.generate_around(camera.eye, 4);

        let vertices = generate_world_mesh(&world);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            depth_texture,
            vertex_buffer,
            num_vertices: vertices.len() as u32,
            diffuse_bind_group,
            last_frame: Instant::now(),
            world,
            mouse_locked: true,
        })
    }

    pub fn window(&self) -> &dyn Window {
        self.window.as_ref()
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.camera.aspect = self.config.width as f32 / self.config.height as f32;
        }
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

        // Infinite world: generate chunks and update mesh if needed
        if self.world.generate_around(self.camera.eye, 6) {
            let vertices = generate_world_mesh(&self.world);
            self.vertex_buffer.destroy(); // Simple approach: recreate buffer
            self.vertex_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Vertex Buffer"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
            self.num_vertices = vertices.len() as u32;
        }

        self.camera_uniform.view_proj = self
            .camera
            .build_view_projection_matrix()
            .to_cols_array_2d();
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn generate_world_mesh(world: &World) -> Vec<Vertex> {
    // 1. Pre-decompress all chunks to avoid redundant decompression during meshing
    let decompressed_chunks: HashMap<(i32, i32, i32), Chunk> = world
        .chunks
        .iter()
        .map(|(&pos, compressed)| (pos, compressed.to_chunk()))
        .collect();

    // 2. Parallel meshing using the decompressed data
    decompressed_chunks
        .par_iter()
        .map(|(&(cx, cy, cz), chunk)| {
            let mut vertices = Vec::new();
            let chunk_pos = [cx as f32 * 16.0, cy as f32 * 16.0, cz as f32 * 16.0];

            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        let block = chunk.get_block(x as i32, y as i32, z as i32);
                        if block == BlockType::AIR {
                            continue;
                        }

                        let pos = [
                            x as f32 + chunk_pos[0],
                            y as f32 + chunk_pos[1],
                            z as f32 + chunk_pos[2],
                        ];

                        let directions = [
                            ([0, 0, 1], [0.0, 0.0, 1.0]),   // Front
                            ([0, 0, -1], [0.0, 0.0, -1.0]), // Back
                            ([1, 0, 0], [1.0, 0.0, 0.0]),   // Right
                            ([-1, 0, 0], [-1.0, 0.0, 0.0]), // Left
                            ([0, 1, 0], [0.0, 1.0, 0.0]),   // Top
                            ([0, -1, 0], [0.0, -1.0, 0.0]), // Bottom
                        ];

                        for (dir, normal) in directions {
                            let world_x = cx * CHUNK_SIZE as i32 + x as i32 + dir[0];
                            let world_y = cy * CHUNK_SIZE as i32 + y as i32 + dir[1];
                            let world_z = cz * CHUNK_SIZE as i32 + z as i32 + dir[2];

                            let n_cx = world_x.div_euclid(CHUNK_SIZE as i32);
                            let n_cy = world_y.div_euclid(CHUNK_SIZE as i32);
                            let n_cz = world_z.div_euclid(CHUNK_SIZE as i32);

                            let n_bx = world_x.rem_euclid(CHUNK_SIZE as i32);
                            let n_by = world_y.rem_euclid(CHUNK_SIZE as i32);
                            let n_bz = world_z.rem_euclid(CHUNK_SIZE as i32);

                            let neighbor = decompressed_chunks
                                .get(&(n_cx, n_cy, n_cz))
                                .map(|c| c.get_block(n_bx, n_by, n_bz))
                                .unwrap_or(BlockType::AIR);

                            if neighbor == BlockType::AIR {
                                append_face(&mut vertices, pos, normal, block);
                            }
                        }
                    }
                }
            }
            vertices
        })
        .flatten()
        .collect()
}

fn append_face(vertices: &mut Vec<Vertex>, pos: [f32; 3], normal: [f32; 3], block: BlockType) {
    let (u_start, v_start, color) = match block {
        BlockType::GRASS => {
            if normal[1] > 0.5 {
                (0.0, 0.0, [0.47, 0.73, 0.33])
            }
            // Top (Green tint)
            else if normal[1] < -0.5 {
                (0.5, 0.0, [1.0, 1.0, 1.0])
            }
            // Bottom (Dirt)
            else {
                (0.5, 0.5, [1.0, 1.0, 1.0])
            } // Side
        }
        BlockType::DIRT => (0.5, 0.0, [1.0, 1.0, 1.0]),
        BlockType::STONE => (0.0, 0.5, [1.0, 1.0, 1.0]),
        _ => (0.0, 0.0, [1.0, 1.0, 1.0]),
    };

    let u_end = u_start + 0.5;
    let v_end = v_start + 0.5;

    let (v0, v1, v2, v3) = if normal[2] > 0.5 {
        // Front
        (
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        )
    } else if normal[2] < -0.5 {
        // Back
        (
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        )
    } else if normal[0] > 0.5 {
        // Right
        (
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        )
    } else if normal[0] < -0.5 {
        // Left
        (
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        )
    } else if normal[1] > 0.5 {
        // Top
        (
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        )
    } else {
        // Bottom
        (
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        )
    };

    let tv0 = [u_start, v_end];
    let tv1 = [u_end, v_end];
    let tv2 = [u_end, v_start];
    let tv3 = [u_start, v_start];

    vertices.push(Vertex {
        position: [pos[0] + v0[0], pos[1] + v0[1], pos[2] + v0[2]],
        tex_coords: tv0,
        normal,
        color,
    });
    vertices.push(Vertex {
        position: [pos[0] + v1[0], pos[1] + v1[1], pos[2] + v1[2]],
        tex_coords: tv1,
        normal,
        color,
    });
    vertices.push(Vertex {
        position: [pos[0] + v2[0], pos[1] + v2[1], pos[2] + v2[2]],
        tex_coords: tv2,
        normal,
        color,
    });

    vertices.push(Vertex {
        position: [pos[0] + v0[0], pos[1] + v0[1], pos[2] + v0[2]],
        tex_coords: tv0,
        normal,
        color,
    });
    vertices.push(Vertex {
        position: [pos[0] + v2[0], pos[1] + v2[1], pos[2] + v2[2]],
        tex_coords: tv2,
        normal,
        color,
    });
    vertices.push(Vertex {
        position: [pos[0] + v3[0], pos[1] + v3[1], pos[2] + v3[2]],
        tex_coords: tv3,
        normal,
        color,
    });
}

async fn download_assets() -> Result<Vec<u8>, Report> {
    let client = reqwest::Client::new();
    let url = "https://piston-data.mojang.com/v1/objects/c9028621e9dccd9b162d33b5188db8518299b792/client.jar";
    let bytes = client
        .get(url)
        .send()
        .await
        .map_err(Report::new)?
        .bytes()
        .await
        .map_err(Report::new)?;

    let cursor = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| report!(e))?;

    let texture_paths = [
        "assets/minecraft/textures/block/grass_block_top.png",
        "assets/minecraft/textures/block/dirt.png",
        "assets/minecraft/textures/block/stone.png",
        "assets/minecraft/textures/block/grass_block_side.png",
    ];

    let mut images = Vec::new();
    for path in texture_paths {
        let mut file = archive.by_name(path).map_err(|e| report!(e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e: std::io::Error| report!(e))?;
        images.push(image::load_from_memory(&buffer).map_err(|e| report!(e))?);
    }

    // Create 2x2 atlas (32x32 pixels since each is 16x16)
    let mut atlas = image::RgbaImage::new(32, 32);
    for (i, img) in images.iter().enumerate() {
        let x = (i % 2) as u32 * 16;
        let y = (i / 2) as u32 * 16;
        image::imageops::replace(&mut atlas, &img.to_rgba8(), x as i64, y as i64);
    }

    let mut cursor = std::io::Cursor::new(Vec::new());
    atlas
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(Report::new)?;
    Ok(cursor.into_inner())
}

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str,
    ) -> Result<Self, Report> {
        let img = image::load_from_memory(bytes).map_err(Report::new)?;
        Self::from_image(device, queue, &img, Some(label))
    }

    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &image::DynamicImage,
        label: Option<&str>,
    ) -> Result<Self, Report> {
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self {
            texture,
            view,
            sampler,
        })
    }

    pub fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
}

struct App {
    state: Option<State>,
    minimized: bool,
    focused: bool,
}

impl App {
    fn new() -> Self {
        Self {
            state: None,
            minimized: false,
            focused: true,
        }
    }
}

impl ApplicationHandler for App {
    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        let window = event_loop
            .create_window(WindowAttributes::default())
            .unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .enable_io()
            .build()
            .unwrap();
        match rt.block_on(State::new(window)) {
            Ok(state) => {
                state.window().set_cursor_visible(!state.mouse_locked);
                let grab_mode = if state.mouse_locked {
                    winit::window::CursorGrabMode::Locked
                } else {
                    winit::window::CursorGrabMode::None
                };
                let _ = state.window().set_cursor_grab(grab_mode);
                self.state = Some(state);
            }
            Err(e) => {
                eprintln!("{}", e);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(state) = &mut self.state {
            if !state.input(&event) {
                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
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
                    WindowEvent::RedrawRequested => {
                        if !self.minimized {
                            state.update();
                            match state.render() {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                                Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                                Err(e) => eprintln!("{:?}", e),
                            }
                        }
                    }
                    _ => {}
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
        if let Some(state) = &self.state {
            if !self.minimized {
                state.window().request_redraw();
                if state.mouse_locked {
                    let _ = state.window().set_cursor_position(
                        winit::dpi::PhysicalPosition::new(
                            state.size.width / 2,
                            state.size.height / 2,
                        )
                        .into(),
                    );
                }
            }
        }
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    let event_loop = EventLoop::new().unwrap();
    let app = Box::leak(Box::new(App::new()));
    event_loop.run_app(app).unwrap();
}
