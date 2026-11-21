//! The rules of the image processor of Inkling 2 (engine-abi "The images of Inkling 2"): the grid, the convert of
//! an alpha channel, the pixel values and the patch rows, from transformers `inkling2` at `046bdb8cc2`
//! (`image_processing_inkling2.py`, `processing_inkling2.py`, `modeling_inkling2.py:1230`).
//!
//! This is the one copy of each rule. The family calls it for the rows of an image, and the host form of the engine
//! (`nagaya_mk_inkling2_bmg::host::vision`) calls it too, so the oracle checks the rules that the family uses.

/// The side of an image token in px, and the side of a patch. A token holds `MERGE x MERGE` patches.
pub const TOKEN_PX: usize = 40;
pub const PATCH_PX: usize = 20;
pub const MERGE: usize = 2;
/// The values of one patch row: 20 lines of 20 pixels of 3 channels.
pub const PATCH_VALUES: usize = PATCH_PX * PATCH_PX * 3;
/// The placeholder of an image token, `<|unused_200054|>` (`image_token_id` of the config).
pub const IMAGE_TOKEN: u32 = 200_054;
/// The sides of an image in px that the family accepts: the range of the model card (engine-abi "Sizes"). The
/// largest image has `MAX_TOKENS` image tokens, 103 by 103.
pub const MIN_SIDE: usize = 40;
pub const MAX_SIDE: usize = 4096;
pub const MAX_TOKENS: usize = MAX_SIDE.div_ceil(TOKEN_PX) * MAX_SIDE.div_ceil(TOKEN_PX);
/// The most patch lines or columns of one image.
pub const MAX_PATCH_SIDE: usize = MERGE * MAX_SIDE.div_ceil(TOKEN_PX);
/// The value of a pad pixel on the scale of 0 to 255 (`fill_value`, `image_processing_inkling2.py:90`).
pub const PAD_VALUE: f32 = -1.0;

/// The grid of an image, in image tokens of 40 px.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub rows: usize,
    pub columns: usize,
}

impl Grid {
    /// The count of image tokens, which is the count of placeholders and of projector rows.
    pub fn tokens(self) -> usize {
        self.rows * self.columns
    }

    /// The count of patches: 4 for each token.
    pub fn patches(self) -> usize {
        self.tokens() * MERGE * MERGE
    }

    /// The `grid_t`, `grid_h` and `grid_w` of the meta of a span, in patches.
    pub fn meta_grid(self) -> [u32; 3] {
        [1, (MERGE * self.rows) as u32, (MERGE * self.columns) as u32]
    }
}

/// The grid of an image of `width x height` px (`image_processing_inkling2.py:87`). A side outside 40 to 4,096 px
/// is refused, as the model card advises. The council never resizes an image.
pub fn grid(width: usize, height: usize) -> Result<Grid, String> {
    if !(MIN_SIDE..=MAX_SIDE).contains(&width) || !(MIN_SIDE..=MAX_SIDE).contains(&height) {
        return Err(format!(
            "the image is {width} x {height} px. Each side must be {MIN_SIDE} to {MAX_SIDE} px. Send an image of \
             that size."
        ));
    }
    Ok(Grid {
        rows: height.div_ceil(TOKEN_PX),
        columns: width.div_ceil(TOKEN_PX),
    })
}

/// The RGB bytes of an RGBA image over the checkerboard of the processor: squares of 8 px of 204 and 153
/// (`Inkling2ImageProcessor.convert_to_rgb`, `image_processing_inkling2.py:63`).
pub fn rgba_to_rgb(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert_eq!(
        rgba.len(),
        width * height * 4,
        "rgba_to_rgb: {} bytes for {width} x {height}",
        rgba.len()
    );
    let mut out = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let p = &rgba[4 * (y * width + x)..4 * (y * width + x + 1)];
            let background: u32 = if (y / 8 + x / 8).is_multiple_of(2) { 204 } else { 153 };
            let alpha = u32::from(p[3]);
            for c in &p[..3] {
                out.push(((u32::from(*c) * alpha + background * (255 - alpha) + 127) / 255) as u8);
            }
        }
    }
    out
}

/// The output of the processor for one image of RGB bytes, line by line: `T x 40 x 40 x 3` values, one square of
/// 40 px for each image token, in the order of the tokens (`image_processing_inkling2.py:85` to 94).
/// The pad goes on the right and at the bottom. Each value is `(v - 127.5) / 127.5`, as the fused rescale and
/// normalize of transformers computes it.
pub fn pixel_values(rgb: &[u8], width: usize, height: usize) -> Result<(Grid, Vec<f32>), String> {
    let g = grid(width, height)?;
    assert_eq!(
        rgb.len(),
        width * height * 3,
        "pixel_values: {} bytes for {width} x {height}",
        rgb.len()
    );
    let normalize = |v: f32| (v - 127.5) / 127.5;
    let pad = normalize(PAD_VALUE);
    let mut out = Vec::with_capacity(g.tokens() * TOKEN_PX * TOKEN_PX * 3);
    for tr in 0..g.rows {
        for tc in 0..g.columns {
            for dy in 0..TOKEN_PX {
                for dx in 0..TOKEN_PX {
                    let (y, x) = (tr * TOKEN_PX + dy, tc * TOKEN_PX + dx);
                    for c in 0..3 {
                        out.push(if y < height && x < width {
                            normalize(f32::from(rgb[3 * (y * width + x) + c]))
                        } else {
                            pad
                        });
                    }
                }
            }
        }
    }
    Ok((g, out))
}

/// The patch rows of the output of the processor: `4T x 1200` values, the rows that the family gives
/// (`Inkling2VisionEmbeddings.forward`, `modeling_inkling2.py:1230`). The 4 patches of a token follow each other,
/// line by line, and a row holds 20 lines of 20 pixels of 3 channels.
pub fn patch_rows(pixels: &[f32], g: Grid) -> Vec<f32> {
    assert_eq!(
        pixels.len(),
        g.tokens() * TOKEN_PX * TOKEN_PX * 3,
        "patch_rows: the pixels do not fit the grid"
    );
    let mut out = Vec::with_capacity(pixels.len());
    for token in pixels.as_chunks::<{ TOKEN_PX * TOKEN_PX * 3 }>().0 {
        for my in 0..MERGE {
            for mx in 0..MERGE {
                for py in 0..PATCH_PX {
                    let start = 3 * ((my * PATCH_PX + py) * TOKEN_PX + mx * PATCH_PX);
                    out.extend_from_slice(&token[start..start + 3 * PATCH_PX]);
                }
            }
        }
    }
    out
}
