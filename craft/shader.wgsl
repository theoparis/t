struct ViewProj {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> view_proj: ViewProj;

struct ShadowSupplement {
    view_proj: mat4x4<f32>,
};

@group(2) @binding(0)
var<uniform> shadow_supp: ShadowSupplement;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) color: vec3<f32>,
    @location(4) ao: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) light: f32,
    @location(2) color: vec3<f32>,
    @location(3) ao: f32,
    @location(4) shadow_pos: vec4<f32>,
    @location(5) world_pos: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.color = model.color;
    out.ao = model.ao;
    
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    out.light = max(dot(model.normal, light_dir), 0.0);
    
    let world_pos = vec4<f32>(model.position, 1.0);
    out.world_pos = model.position;
    out.clip_position = view_proj.view_proj * world_pos;
    
    let pos_from_indices = shadow_supp.view_proj * world_pos;
    let shadow_uv = pos_from_indices.xyz / pos_from_indices.w;
    out.shadow_pos = vec4<f32>(
        shadow_uv.x * 0.5 + 0.5,
        1.0 - (shadow_uv.y * 0.5 + 0.5),
        shadow_uv.z,
        1.0
    );
    
    return out;
}

@vertex
fn vs_shadow(model: VertexInput) -> @builtin(position) vec4<f32> {
    return view_proj.view_proj * vec4<f32>(model.position, 1.0);
}

@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

@group(2) @binding(1)
var t_shadow: texture_depth_2d;
@group(2) @binding(2)
var s_shadow: sampler_comparison;

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + vec2<f32>(0.0, 0.0)), hash(i + vec2<f32>(1.0, 0.0)), u.x),
               mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    for (var i = 0; i < 5; i++) {
        v += a * noise(p + f32(i) * 10.0);
        a *= 0.5;
    }
    return v;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    if (object_color.a < 0.1) {
        discard;
    }
    
    // Improved PCF with slightly larger bias and jitter-free coordinates
    var shadow_amount = 0.0;
    let texel_size = 1.0 / 2048.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow_amount += textureSampleCompare(
                t_shadow,
                s_shadow,
                in.shadow_pos.xy + offset,
                in.shadow_pos.z - 0.0012 // Increased bias
            );
        }
    }
    shadow_amount /= 9.0;
    
    let ambient = 0.4; // Slightly brighter ambient
    let light_factor = in.light * shadow_amount;
    let final_light = (light_factor * 0.6 + ambient) * (in.ao * 0.8 + 0.2);
    var final_color = object_color.rgb * in.color * final_light;
    
    // Cloud shadows on ground
    let cloud_shadow = smoothstep(0.4, 0.7, fbm(in.world_pos.xz * 0.02));
    final_color = mix(final_color, final_color * 0.8, cloud_shadow * 0.5);
    
    // Atmospheric fog
    let fog_color = vec3<f32>(0.5, 0.7, 0.9);
    let fog_start = 80.0;
    let fog_end = 220.0;
    let dist = in.clip_position.w;
    let fog_factor = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    
    return vec4<f32>(mix(final_color, fog_color, fog_factor), object_color.a);
}

struct SkyVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_dir: vec3<f32>,
};

@vertex
fn vs_sky(@location(0) position: vec3<f32>) -> SkyVertexOutput {
    var out: SkyVertexOutput;
    var view_no_translate = view_proj.view_proj;
    view_no_translate[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.clip_position = (view_no_translate * vec4<f32>(position, 1.0)).xyww;
    out.view_dir = position;
    return out;
}

@fragment
fn fs_sky(in: SkyVertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.view_dir);
    
    let sky_top = vec3<f32>(0.05, 0.3, 0.7);
    let sky_horizon = vec3<f32>(0.5, 0.7, 0.9);
    var color = mix(sky_horizon, sky_top, pow(max(dir.y, 0.0), 0.6));
    
    let sun_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let sun_dot = dot(dir, sun_dir);
    
    let sun = smoothstep(0.998, 0.999, sun_dot);
    color += sun * vec3<f32>(1.0, 1.0, 0.8) * 2.0;
    color += pow(max(sun_dot, 0.0), 8.0) * vec3<f32>(0.5, 0.4, 0.2);
    
    if (dir.y > 0.0) {
        let uv = dir.xz / (dir.y + 0.05);
        let n = fbm(uv * 0.05);
        let cloud = smoothstep(0.4, 0.6, n);
        color = mix(color, vec3<f32>(1.0, 1.0, 1.0), cloud * 0.8 * pow(dir.y, 0.5));
    }
    
    return vec4<f32>(color, 1.0);
}
