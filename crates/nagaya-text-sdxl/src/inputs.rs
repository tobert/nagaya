//! The inputs of an image job (spec 11.9b): the input PNG, the mask, its blur, and the paste after the decode.
//!
//! Each function gives the bytes of diffusers 0.42.1 and Pillow 13.0.0, the versions of the oracle. The route decodes
//! each input before the job starts, so a bad input is a 400 and not a failed job.

use std::io::Cursor;

/// Why the bytes of an input are not an image that the nagaya reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputError {
    /// The bytes are not a PNG, or the PNG is damaged. The text says what the decoder found.
    NotPng(String),
    /// The PNG has this count of bits in a sample, not 8.
    Depth(u8),
}

impl std::fmt::Display for InputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputError::NotPng(why) => write!(formatter, "is not a PNG that the server can read ({why}); send a PNG"),
            InputError::Depth(bits) => write!(
                formatter,
                "is a PNG with {bits} bits in a sample; send a PNG with 8 bits in each channel"
            ),
        }
    }
}

/// A PNG whose header is read. Its pixels are not decoded yet, so the route can check the size first.
pub struct Png<'a> {
    reader: png::Reader<Cursor<&'a [u8]>>,
}

fn damaged(error: png::DecodingError) -> InputError {
    InputError::NotPng(error.to_string())
}

impl<'a> Png<'a> {
    /// Read the header of a PNG with 8 bits in a sample: gray, gray with alpha, RGB, RGBA, or a palette.
    pub fn read(bytes: &'a [u8]) -> Result<Png<'a>, InputError> {
        let mut decoder = png::Decoder::new(Cursor::new(bytes));
        // The decoder gives the samples as the file holds them. This code maps a palette, and it ignores `tRNS`, as
        // Pillow's `convert` does.
        decoder.set_transformations(png::Transformations::IDENTITY);
        let reader = decoder.read_info().map_err(damaged)?;
        let depth = reader.info().bit_depth as u8;
        if depth != 8 {
            return Err(InputError::Depth(depth));
        }
        Ok(Png { reader })
    }

    pub fn width(&self) -> u32 {
        self.reader.info().width
    }

    pub fn height(&self) -> u32 {
        self.reader.info().height
    }

    /// The samples of the image, and the RGB of each palette entry for a palette.
    fn samples(mut self) -> Result<(png::ColorType, Vec<u8>, Vec<u8>), InputError> {
        let size = self
            .reader
            .output_buffer_size()
            .ok_or_else(|| InputError::NotPng("the image is too large to decode".to_owned()))?;
        let mut samples = vec![0; size];
        let frame = self.reader.next_frame(&mut samples).map_err(damaged)?;
        samples.truncate(frame.buffer_size());
        let info = self.reader.info();
        let palette = info.palette.as_deref().unwrap_or_default().to_vec();
        let color = info.color_type;
        if color == png::ColorType::Indexed {
            let entries = palette.len() / 3;
            if let Some(index) = samples.iter().find(|&&index| usize::from(index) >= entries) {
                return Err(InputError::NotPng(format!(
                    "a pixel has the palette index {index}, and the palette has {entries} entries"
                )));
            }
        }
        Ok((color, samples, palette))
    }

    /// The RGB bytes of the image, 3 for each pixel, as Pillow's `convert("RGB")` gives them. An alpha channel has
    /// no effect (spec 11.9b).
    pub fn rgb(self) -> Result<Vec<u8>, InputError> {
        let (color, samples, palette) = self.samples()?;
        Ok(match color {
            png::ColorType::Grayscale => samples.iter().flat_map(|&v| [v, v, v]).collect(),
            png::ColorType::GrayscaleAlpha => pixels::<2>(&samples).iter().flat_map(|p| [p[0], p[0], p[0]]).collect(),
            png::ColorType::Rgb => samples,
            png::ColorType::Rgba => pixels::<4>(&samples).iter().flat_map(|p| [p[0], p[1], p[2]]).collect(),
            png::ColorType::Indexed => samples
                .iter()
                .flat_map(|&i| {
                    let at = 3 * usize::from(i);
                    [palette[at], palette[at + 1], palette[at + 2]]
                })
                .collect(),
        })
    }

    /// The gray bytes of the image, 1 for each pixel, as Pillow's `convert("L")` gives them. An alpha channel has no
    /// effect (spec 11.9b, "The mask").
    pub fn gray(self) -> Result<Vec<u8>, InputError> {
        let (color, samples, palette) = self.samples()?;
        Ok(match color {
            png::ColorType::Grayscale => samples,
            png::ColorType::GrayscaleAlpha => pixels::<2>(&samples).iter().map(|p| p[0]).collect(),
            png::ColorType::Rgb => pixels::<3>(&samples).iter().map(|p| luma(p)).collect(),
            png::ColorType::Rgba => pixels::<4>(&samples).iter().map(|p| luma(p)).collect(),
            png::ColorType::Indexed => samples.iter().map(|&i| luma(&palette[3 * usize::from(i)..])).collect(),
        })
    }
}

/// The pixels of `N` samples each. A rest would be a decoder fault, so it stops the process (spec 8.5).
fn pixels<const N: usize>(samples: &[u8]) -> &[[u8; N]] {
    let (pixels, rest) = samples.as_chunks::<N>();
    assert!(
        rest.is_empty(),
        "the decoder gave {} samples, not a multiple of {N}",
        samples.len()
    );
    pixels
}

/// The gray value of an RGB pixel: ITU-R 601-2 in 16-bit fixed point, as Pillow's `L24` (`Convert.c`).
fn luma(rgb: &[u8]) -> u8 {
    let [r, g, b] = [rgb[0], rgb[1], rgb[2]].map(u32::from);
    ((19595 * r + 38470 * g + 7471 * b + 0x8000) >> 16) as u8
}

/// The passes of Pillow's box blur for a Gaussian blur (`_imaging.c`, `_gaussian_blur`).
const PASSES: u32 = 3;

/// The radius of the extended box of one pass, for a Gaussian of standard deviation `radius`. This is Pillow's
/// `_gaussian_blur_radius` (`BoxBlur.c`), with its mix of f64 and f32 steps kept.
fn box_radius(radius: f32) -> f32 {
    let sigma2 = radius * radius / PASSES as f32;
    let length = (12.0 * f64::from(sigma2) + 1.0).sqrt() as f32;
    let whole = ((f64::from(length) - 1.0) / 2.0).floor() as f32;
    let mut part = (2.0 * whole + 1.0) * (whole * (whole + 1.0) - 3.0 * sigma2);
    part /= 6.0 * (sigma2 - (whole + 1.0) * (whole + 1.0));
    whole + part
}

/// One line of Pillow's extended box blur (`ImagingLineBoxBlur8`). The sums wrap in u32, as the C sums do.
fn line_blur(out: &mut [u8], line: &[u8], radius: usize, ww: u32, fw: u32) {
    let last = line.len() - 1;
    let edge_a = (radius + 1).min(line.len());
    let edge_b = line.len().saturating_sub(radius + 1);
    let at = |x: usize| u32::from(line[x]);
    let mut acc = at(0).wrapping_mul(radius as u32 + 1);
    for x in 0..edge_a.saturating_sub(1) {
        acc = acc.wrapping_add(at(x));
    }
    acc = acc.wrapping_add(at(last).wrapping_mul((radius + 1 - edge_a) as u32));
    let mut save = |x: usize, acc: u32, left: usize, right: usize| {
        let bulk = acc
            .wrapping_mul(ww)
            .wrapping_add((at(left) + at(right)).wrapping_mul(fw));
        out[x] = (bulk.wrapping_add(1 << 23) >> 24) as u8;
    };
    let shift = |acc: u32, subtract: usize, add: usize| acc.wrapping_add(at(add)).wrapping_sub(at(subtract));
    if edge_a <= edge_b {
        for x in 0..edge_a {
            acc = shift(acc, 0, x + radius);
            save(x, acc, 0, x + radius + 1);
        }
        for x in edge_a..edge_b {
            acc = shift(acc, x - radius - 1, x + radius);
            save(x, acc, x - radius - 1, x + radius + 1);
        }
        for x in edge_b..=last {
            acc = shift(acc, x - radius - 1, last);
            save(x, acc, x - radius - 1, last);
        }
    } else {
        for x in 0..edge_b {
            acc = shift(acc, 0, x + radius);
            save(x, acc, 0, x + radius + 1);
        }
        for x in edge_b..edge_a {
            acc = shift(acc, 0, last);
            save(x, acc, 0, last);
        }
        for x in edge_a..=last {
            acc = shift(acc, x - radius - 1, last);
            save(x, acc, x - radius - 1, last);
        }
    }
}

/// Blur each row of a gray image in place, as Pillow's `ImagingHorizontalBoxBlur` does.
fn rows_blur(pixels: &mut [u8], width: usize, radius: f32) {
    let whole = radius as usize;
    let ww = (16_777_216.0f32 / (radius * 2.0 + 1.0)) as u32;
    let fw = ((1u32 << 24) - (whole as u32 * 2 + 1) * ww) / 2;
    let mut out = vec![0; width];
    for row in pixels.chunks_exact_mut(width) {
        line_blur(&mut out, row, whole, ww, fw);
        row.copy_from_slice(&out);
    }
}

fn transpose(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0; pixels.len()];
    for y in 0..height {
        for x in 0..width {
            out[x * height + y] = pixels[y * width + x];
        }
    }
    out
}

/// The gray mask after Pillow's `ImageFilter.GaussianBlur(radius)`, as diffusers' `blur` makes it (spec 11.9b, "The
/// mask"). A radius of 0 gives the mask with no change.
pub fn blur(gray: &[u8], width: u32, height: u32, radius: u32) -> Vec<u8> {
    let (width, height) = (width as usize, height as usize);
    assert_eq!(gray.len(), width * height, "a gray mask has 1 byte for each pixel");
    if radius == 0 || gray.is_empty() {
        return gray.to_vec();
    }
    let extended = box_radius(radius as f32);
    let mut pixels = gray.to_vec();
    for _ in 0..PASSES {
        rows_blur(&mut pixels, width, extended);
    }
    let mut columns = transpose(&pixels, width, height);
    for _ in 0..PASSES {
        rows_blur(&mut columns, height, extended);
    }
    transpose(&columns, height, width)
}

/// Each channel `v` as `2 * v / 255 - 1` in f32, as diffusers' `pil_to_numpy` and `normalize` make it (spec 11.9b,
/// "The input PNG").
pub fn unit(pixels: &[u8]) -> Vec<f32> {
    pixels.iter().map(|&v| 2.0 * (f32::from(v) / 255.0) - 1.0).collect()
}

/// The binary mask of a blurred mask: 1 repaints a pixel whose value is 128 or more, and 0 keeps it (spec 11.9b).
pub fn binary(blurred: &[u8]) -> Vec<u8> {
    blurred.iter().map(|&v| u8::from(v >= 128)).collect()
}

/// The latent mask: the binary mask at pixel `(8x, 8y)` for latent `(x, y)`, as torch's `interpolate` in the mode
/// `nearest` takes it (spec 11.9b).
pub fn latent_mask(binary: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (width, height) = (width as usize, height as usize);
    assert_eq!(binary.len(), width * height, "a binary mask has 1 byte for each pixel");
    (0..height / 8)
        .flat_map(|y| (0..width / 8).map(move |x| binary[8 * y * width + 8 * x]))
        .collect()
}

/// The paste after the decode: each channel becomes `(a * new + (255 - a) * old + 127) / 255` in integers, with `a`
/// the blurred mask of its pixel (spec 11.9b). `new` and `old` are RGB, 3 bytes for each pixel.
pub fn paste(new: &[u8], old: &[u8], blurred: &[u8]) -> Vec<u8> {
    assert_eq!(new.len(), old.len(), "the decode and the input image have one size");
    assert_eq!(new.len(), 3 * blurred.len(), "the mask has 1 byte for each RGB pixel");
    pixels::<3>(new)
        .iter()
        .zip(pixels::<3>(old))
        .zip(blurred)
        .flat_map(|((new, old), &a)| {
            let a = u32::from(a);
            [0, 1, 2].map(|c| ((a * u32::from(new[c]) + (255 - a) * u32::from(old[c]) + 127) / 255) as u8)
        })
        .collect()
}
