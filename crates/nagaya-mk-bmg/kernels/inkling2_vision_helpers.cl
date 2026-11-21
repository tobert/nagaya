// The helpers of the vision ops of Inkling 2 (MK_INKLING2), at file scope. megakernel.cl includes this file before
// the kernel, after inkling2_attn_helpers.cl. The host form is crates/nagaya-mk-inkling2-bmg/src/host/vision.rs.

// The query rows of one head for each subgroup of OP_INKLING2_VISION_ATTN; at most 16. It is VISION_ATTN_TILE of
// the LFM3 binary, which this binary does not define.
#define INKLING2_VISION_ATTN_TILE 4
// The side of the table of learned positions (host `GRID`).
#define INKLING2_VISION_GRID      32
// The frequencies of the rotary embedding: the pairs of one axis of a head of 64 (host `HEAD_DIM / 4`).
#define INKLING2_VISION_FREQS     16
// The positions of the rotary table: the class row and the 206 patch lines of a side of 4,096 px (host
// `ROPE_POSITIONS`).
#define INKLING2_VISION_POSITIONS 207

// The line and the column of patch n of an image of `columns` image tokens across, in patches. The 4 patches of a
// token follow each other, line by line (host `patch_position`).
inline uint2 inkling2_patch_position(uint n, uint columns) {
    uint token = n / 4u, inner = n % 4u;
    return (uint2)(2u * (token / columns) + inner / 2u, 2u * (token % columns) + inner % 2u);
}

// The tap of `offset` (0 or 1) into the table, and its bilinear weight, for `index` on an axis of `size` patches
// (host `axis_taps`): F.interpolate with align_corners=False and a clamp at the border. The division of the card
// can differ from the host by its rounding, so the weight can differ by a few ulp.
inline float inkling2_axis_tap(uint index, uint size, uint offset, uint* tap) {
#pragma OPENCL FP_CONTRACT OFF
    float src = ((float)index + 0.5f) * (float)INKLING2_VISION_GRID / (float)size - 0.5f;
    float fl = floor(src);
    int raw = (int)fl + (int)offset;
    *tap = (uint)clamp(raw, 0, INKLING2_VISION_GRID - 1);
    float distance = fabs(src - fl - (float)offset);
    return fmax(1.0f - distance, 0.0f);
}
