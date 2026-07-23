// 2D Rotary Position Embedding for bonsai diffuser
// Layout: (head_dim, n_heads, seq) — different from standard RoPE
// Rotates pairs of elements within head_dim using precomputed cos/sin tables
//
// cos/sin shape: (head_dim/2, seq) = (half, seq)
// src shape:     (head_dim, n_heads, seq)
// dst shape:     same as src

struct Params {
    offset_src: u32,
    offset_cos: u32,
    offset_sin: u32,
    offset_dst: u32,
    head_dim: u32,
    n_heads: u32,
    seq: u32,
    half: u32,
}

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> cos_table: array<f32>;

@group(0) @binding(2)
var<storage, read_write> sin_table: array<f32>;

@group(0) @binding(3)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_pairs = params.half * params.n_heads * params.seq;
    if (gid.x >= total_pairs) {
        return;
    }

    let pair_idx = gid.x;
    let d_pair = pair_idx % params.half;           // 0..half-1
    let rest = pair_idx / params.half;
    let h = rest % params.n_heads;                 // 0..n_heads-1
    let s = rest / params.n_heads;                 // 0..seq-1

    let even_idx = d_pair * 2u + h * params.head_dim + s * params.head_dim * params.n_heads;
    let odd_idx  = even_idx + 1u;

    let angle_idx = d_pair + s * params.half;

    let a = src[params.offset_src + even_idx];
    let b = src[params.offset_src + odd_idx];
    let c = cos_table[params.offset_cos + angle_idx];
    let s_val = sin_table[params.offset_sin + angle_idx];

    dst[params.offset_dst + even_idx] = a * c - b * s_val;
    dst[params.offset_dst + odd_idx]  = a * s_val + b * c;
}
