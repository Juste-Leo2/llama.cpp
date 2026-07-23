// B1_0 binary-quantized matrix multiply for bonsai diffuser
//
// Weight format: packed as (out_dim, in_dim/32*6) uint8 bytes
//   Block of 32 weights → 6 bytes:
//     bytes 0-1:  scale (float16, little-endian)
//     bytes 2-5:  sign bits (packed, bit 0 = weight 0)
//   row_stride = (in_dim/32) * 6 bytes
//
// act:  (batch, in_dim) f32
// dst:  (batch, out_dim) f32
// weight: u8 bytes (row_stride bytes per output row)

struct Params {
    offset_act: u32,
    offset_weight: u32,    // byte offset in the weight buffer
    offset_dst: u32,
    batch: u32,
    in_dim: u32,
    out_dim: u32,
    num_blocks: u32,       // in_dim / 32
    row_stride: u32,       // num_blocks * 6
}

@group(0) @binding(0)
var<storage, read_write> act: array<f32>;

@group(0) @binding(1)
var<storage, read_write> weight: array<u32>;  // read as u32, extract bytes manually

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

// Read a single byte at byte_offset from a u32-backed buffer
fn read_u8(byte_off: u32) -> u32 {
    let word = weight[byte_off / 4u];
    let shift = (byte_off & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

// Convert float16 bits (lower 16 bits of u32) to f32
fn fp16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp  = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;

    var v: f32;
    if (exp == 0u) {
        v = f32(mant) * 5.960464477539063e-08;  // 2^(-24)
    } else {
        v = ldexp(f32(mant | 0x400u), i32(exp) - 25);
    }
    return select(v, -v, sign != 0u);
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.batch * params.out_dim;
    if (gid.x >= total) {
        return;
    }

    let r = gid.x % params.out_dim;     // output row
    let b = gid.x / params.out_dim;     // batch index

    let act_base   = params.offset_act + b * params.in_dim;
    let wgt_base   = params.offset_weight + r * params.row_stride;

    var sum: f32 = 0.0;

    for (var blk = 0u; blk < params.num_blocks; blk++) {
        let byte_off = wgt_base + blk * 6u;

        // Read scale (fp16, 2 bytes little-endian)
        let lo = read_u8(byte_off);
        let hi = read_u8(byte_off + 1u);
        let scale_bits = lo | (hi << 8u);
        let scale = fp16_to_f32(scale_bits);

        // Read sign bits (4 bytes little-endian)
        let b0 = read_u8(byte_off + 2u);
        let b1 = read_u8(byte_off + 3u);
        let b2 = read_u8(byte_off + 4u);
        let b3 = read_u8(byte_off + 5u);
        let bits = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);

        // Accumulate: sum += scale * sum_i(±act[i])
        var block_sum: f32 = 0.0;
        for (var i = 0u; i < 32u; i++) {
            let a = act[act_base + blk * 32u + i];
            if ((bits >> i) & 1u) != 0u {
                block_sum += a;
            } else {
                block_sum -= a;
            }
        }
        sum += scale * block_sum;
    }

    dst[params.offset_dst + r + b * params.out_dim] = sum;
}
