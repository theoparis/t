use bytemuck::{Pod, Zeroable};
use image::GenericImageView;
use rayon::prelude::*;
use rootcause::Report;
use std::collections::HashMap;
use std::sync::Arc;
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
    target: glam::Vec3,
    up: glam::Vec3,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> glam::Mat4 {
        let view = glam::Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = glam::Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        proj * view
    }
}

struct CameraController {
    speed: f32,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_left_pressed: false,
            is_right_pressed: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        logical_key: winit::keyboard::Key::Character(c),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
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
                    "Shift" => {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.length();

        if self.is_forward_pressed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);
        if self.is_right_pressed {
            camera.eye += right * self.speed;
        }
        if self.is_left_pressed {
            camera.eye -= right * self.speed;
        }

        if self.is_up_pressed {
            camera.eye += camera.up * self.speed;
        }
        if self.is_down_pressed {
            camera.eye -= camera.up * self.speed;
        }

        camera.target = camera.eye + forward_norm * forward_mag;
    }
}

const CHUNK_SIZE: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockType {
    Air = 0,
    Dirt = 1,
    Grass = 2,
    Stone = 3,
}

struct Chunk {
    data: [BlockType; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
}

impl Chunk {
    fn new(pos: (i32, i32, i32)) -> Self {
        let mut data = [BlockType::Air; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = pos.0 * CHUNK_SIZE as i32 + x as i32;
                let world_z = pos.2 * CHUNK_SIZE as i32 + z as i32;
                let height = (10.0
                    + (world_x as f32 * 0.1).sin() * 5.0
                    + (world_z as f32 * 0.1).cos() * 5.0) as i32;
                let chunk_height = height - pos.1 * CHUNK_SIZE as i32;
                for y in 0..CHUNK_SIZE {
                    if (y as i32) < chunk_height {
                        let idx = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
                        if (y as i32) == chunk_height - 1 {
                            data[idx] = BlockType::Grass;
                        } else if (y as i32) > chunk_height - 4 {
                            data[idx] = BlockType::Dirt;
                        } else {
                            data[idx] = BlockType::Stone;
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
            return BlockType::Air;
        }
        self.data[x as usize + (y as usize) * CHUNK_SIZE + (z as usize) * CHUNK_SIZE * CHUNK_SIZE]
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
}

impl State {
    async fn new(window: Box<dyn Window>) -> Self {
        rustls_graviola::default_provider()
            .install_default()
            .unwrap();

        let window: Arc<dyn Window> = Arc::from(window);

        let size = window.surface_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

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
        let diffuse_bytes = download_assets(&queue).await;
        let diffuse_texture =
            Texture::from_bytes(&device, &queue, &diffuse_bytes, "atlas.png").unwrap();

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
            eye: (0.0, 15.0, 30.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: glam::Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0f32.to_radians(),
            znear: 0.1,
            zfar: 1000.0,
        };

        let camera_controller = CameraController::new(0.5);

        let mut camera_uniform = CameraUniform {
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
                immediate_size: 0,
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
        let mut chunks = Vec::new();
        for x in -2..2 {
            for z in -2..2 {
                for y in 0..1 {
                    chunks.push(Chunk::new((x, y, z)));
                }
            }
        }

        let vertices = generate_world_mesh(&chunks);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
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
        }
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
        self.camera_controller.update_camera(&mut self.camera);
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

fn generate_world_mesh(chunks: &[Chunk]) -> Vec<Vertex> {
    chunks
        .par_iter()
        .enumerate()
        .map(|(id, chunk)| {
            let mut vertices = Vec::new();
            let chunk_x = id as i32 % 4 - 2;
            let chunk_z = id as i32 / 4 - 2;
            let chunk_pos = [chunk_x as f32 * 16.0, 0.0, chunk_z as f32 * 16.0];

            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        let block = chunk.get_block(x as i32, y as i32, z as i32);
                        if block == BlockType::Air {
                            continue;
                        }

                        let pos = [
                            x as f32 + chunk_pos[0],
                            y as f32 + chunk_pos[1],
                            z as f32 + chunk_pos[2],
                        ];

                        // Simple face culling
                        let directions = [
                            ([0, 0, 1], [0.0, 0.0, 1.0]),   // Front
                            ([0, 0, -1], [0.0, 0.0, -1.0]), // Back
                            ([1, 0, 0], [1.0, 0.0, 0.0]),   // Right
                            ([-1, 0, 0], [-1.0, 0.0, 0.0]), // Left
                            ([0, 1, 0], [0.0, 1.0, 0.0]),   // Top
                            ([0, -1, 0], [0.0, -1.0, 0.0]), // Bottom
                        ];

                        for (dir, normal) in directions {
                            let neighbor = chunk.get_block(
                                x as i32 + dir[0],
                                y as i32 + dir[1],
                                z as i32 + dir[2],
                            );
                            if neighbor == BlockType::Air {
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
    let (u_start, v_start) = match block {
        BlockType::Grass => {
            if normal[1] > 0.5 {
                (0.0, 0.0)
            }
            // Top
            else if normal[1] < -0.5 {
                (0.5, 0.0)
            }
            // Bottom (Dirt)
            else {
                (0.5, 0.5)
            } // Side
        }
        BlockType::Dirt => (0.5, 0.0),
        BlockType::Stone => (0.0, 0.5),
        _ => (0.0, 0.0),
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
    });
    vertices.push(Vertex {
        position: [pos[0] + v1[0], pos[1] + v1[1], pos[2] + v1[2]],
        tex_coords: tv1,
        normal,
    });
    vertices.push(Vertex {
        position: [pos[0] + v2[0], pos[1] + v2[1], pos[2] + v2[2]],
        tex_coords: tv2,
        normal,
    });

    vertices.push(Vertex {
        position: [pos[0] + v0[0], pos[1] + v0[1], pos[2] + v0[2]],
        tex_coords: tv0,
        normal,
    });
    vertices.push(Vertex {
        position: [pos[0] + v2[0], pos[1] + v2[1], pos[2] + v2[2]],
        tex_coords: tv2,
        normal,
    });
    vertices.push(Vertex {
        position: [pos[0] + v3[0], pos[1] + v3[1], pos[2] + v3[2]],
        tex_coords: tv3,
        normal,
    });
}

async fn download_assets(_queue: &wgpu::Queue) -> Vec<u8> {
    // Mojang official assets for 1.20.1
    // Dirt: 827ef613470719875e52da3fbe9cc8665675cedf -> 82/827ef...
    // Grass Top: 3ba8f44f6f43685f0967a147e44955eb49f99f24
    // Stone: 08647acae4d03e2c3359d9f58330768e98bc017f
    // Grass Side: 2c1e2f7b82f061c7de35017c6999a38f376a911a

    let client = reqwest::Client::new();
    let textures = [
        "3ba8f44f6f43685f0967a147e44955eb49f99f24", // Grass Top
        "827ef613470719875e52da3fbe9cc8665675cedf", // Dirt
        "08647acae4d03e2c3359d9f58330768e98bc017f", // Stone
        "2c1e2f7b82f061c7de35017c6999a38f376a911a", // Grass Side
    ];

    let mut images = Vec::new();
    for hash in textures {
        let url = format!(
            "https://resources.download.minecraft.net/{}/{}",
            &hash[..2],
            hash
        );
        let bytes = client.get(url).send().await.unwrap().bytes().await.unwrap();
        images.push(image::load_from_memory(&bytes).unwrap());
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
        .unwrap();
    cursor.into_inner()
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
        let img = image::load_from_memory(bytes)?;
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
}

impl App {
    fn new() -> Self {
        Self { state: None }
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
        let state = rt.block_on(State::new(window));
        self.state = Some(state);
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
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                state: ElementState::Pressed,
                                logical_key:
                                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
                                ..
                            },
                        ..
                    } => event_loop.exit(),
                    WindowEvent::SurfaceResized(physical_size) => {
                        state.resize(physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        state.update();
                        match state.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                            Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                            Err(e) => eprintln!("{:?}", e),
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &dyn ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window().request_redraw();
        }
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(App::new()).unwrap();
}
