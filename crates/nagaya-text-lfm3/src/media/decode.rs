//! The decode of a video part with ffmpeg's libraries (spec 11.11, "The video of a request"). The bytes go to ffmpeg
//! through a memfd, so the decode writes no file. The steps after the decode are in `video`.

use std::ffi::CString;
use std::io::Write;
use std::os::fd::{FromRawFd, OwnedFd};

use ffmpeg_next as ff;

use super::MediaError;
use super::image::Rgb;
use super::video;

/// What a video gives: the sampled frames in RGB, their indices, the stream's facts and its sound.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    /// The frames of the stream.
    pub total: usize,
    /// The frames each second, from the container, rounded to 6 decimals as Omni's processor does.
    pub fps: f64,
    /// The sampled indices and their frames, in the order of the stream.
    pub indices: Vec<usize>,
    pub frames: Vec<Rgb>,
    /// The sound track as 16 kHz mono f32 samples, or `None` for a video with no sound.
    pub sound: Option<Vec<f32>>,
}

/// The versions of the four libraries, which the recipe item of a video part names (spec 11.11).
pub fn versions() -> String {
    let v = |n: u32| format!("{}.{}.{}", n >> 16, (n >> 8) & 0xff, n & 0xff);
    format!(
        "avformat={};avcodec={};swscale={};swresample={}",
        v(ff::format::version()),
        v(ff::codec::version()),
        v(ff::software::scaling::version()),
        v(ff::software::resampling::version())
    )
}

/// The samples of each frame that the flush of the resampler fills.
const FLUSH_SAMPLES: usize = 4_096;

/// True when the decoder gave a frame, false when it needs a packet or has no more. Any other error stops the decode,
/// so a corrupt frame never makes a shorter video in silence (spec 8.5).
fn more(result: Result<(), ff::Error>) -> Result<bool, MediaError> {
    match result {
        Ok(()) => Ok(true),
        Err(ff::Error::Eof) => Ok(false),
        Err(ff::Error::Other { errno }) if errno == libc::EAGAIN => Ok(false),
        Err(e) => Err(error(format!("ffmpeg cannot decode a frame: {e}"))),
    }
}

fn error(why: impl std::fmt::Display) -> MediaError {
    MediaError::Video(why.to_string())
}

/// A memfd that holds `bytes`, and its path for ffmpeg.
fn memfd(bytes: &[u8]) -> Result<(OwnedFd, String), MediaError> {
    let name = CString::new("nagaya-video").expect("the name has no NUL");
    // SAFETY: memfd_create takes a NUL-terminated name and flags; the result is a new fd or -1.
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(error(format!(
            "the server cannot make a memfd: {}",
            std::io::Error::last_os_error()
        )));
    }
    // SAFETY: the fd is new and owned by nobody else.
    let owned = unsafe { OwnedFd::from_raw_fd(fd) };
    let mut file = std::fs::File::from(owned.try_clone().map_err(error)?);
    file.write_all(bytes).map_err(error)?;
    Ok((owned, format!("/proc/self/fd/{fd}")))
}

/// Decode the frames of the best video stream, and call `keep` with the index and the RGB pixels of each frame that
/// `wanted` gives. It returns the count of frames.
fn frames(path: &str, wanted: &dyn Fn(usize) -> bool, keep: &mut dyn FnMut(usize, Rgb)) -> Result<usize, MediaError> {
    let mut input = ff::format::input(&path).map_err(|e| error(format!("ffmpeg cannot open it: {e}")))?;
    let stream = input
        .streams()
        .best(ff::media::Type::Video)
        .ok_or_else(|| error("it has no video stream"))?;
    let index = stream.index();
    let context = ff::codec::context::Context::from_parameters(stream.parameters()).map_err(error)?;
    let mut decoder = context.decoder().video().map_err(error)?;
    let mut scaler: Option<ff::software::scaling::Context> = None;
    let mut count = 0usize;
    let mut take = |decoder: &mut ff::decoder::Video, count: &mut usize| -> Result<(), MediaError> {
        let mut frame = ff::frame::Video::empty();
        while more(decoder.receive_frame(&mut frame))? {
            if wanted(*count) {
                let scaler = match &mut scaler {
                    Some(s) => s,
                    None => scaler.insert(
                        ff::software::scaling::Context::get(
                            frame.format(),
                            frame.width(),
                            frame.height(),
                            ff::format::Pixel::RGB24,
                            frame.width(),
                            frame.height(),
                            ff::software::scaling::Flags::BILINEAR,
                        )
                        .map_err(error)?,
                    ),
                };
                let mut rgb = ff::frame::Video::empty();
                scaler
                    .run(&frame, &mut rgb)
                    .map_err(|e| error(format!("the color conversion failed: {e}")))?;
                let (height, width) = (rgb.height() as usize, rgb.width() as usize);
                let stride = rgb.stride(0);
                let data = rgb.data(0);
                let mut bytes = Vec::with_capacity(height * width * 3);
                for y in 0..height {
                    bytes.extend_from_slice(&data[y * stride..y * stride + width * 3]);
                }
                keep(*count, Rgb { height, width, bytes });
            }
            *count += 1;
        }
        Ok(())
    };
    for (stream, packet) in input.packets() {
        if stream.index() == index {
            decoder.send_packet(&packet).map_err(error)?;
            take(&mut decoder, &mut count)?;
        }
    }
    decoder.send_eof().map_err(error)?;
    take(&mut decoder, &mut count)?;
    Ok(count)
}

/// The rate of the best video stream from the container, rounded to 6 decimals.
fn rate(path: &str) -> Result<f64, MediaError> {
    let input = ff::format::input(&path).map_err(|e| error(format!("ffmpeg cannot open it: {e}")))?;
    let stream = input
        .streams()
        .best(ff::media::Type::Video)
        .ok_or_else(|| error("it has no video stream"))?;
    let r = stream.avg_frame_rate();
    if r.denominator() == 0 || r.numerator() <= 0 {
        return Err(error(format!(
            "its frame rate is {}/{}",
            r.numerator(),
            r.denominator()
        )));
    }
    let fps = f64::from(r.numerator()) / f64::from(r.denominator());
    Ok((fps * 1e6).round_ties_even() / 1e6)
}

/// The sound track as 16 kHz mono f32, or `None` when the video has no sound.
fn sound(path: &str) -> Result<Option<Vec<f32>>, MediaError> {
    let mut input = ff::format::input(&path).map_err(|e| error(format!("ffmpeg cannot open it: {e}")))?;
    let Some(stream) = input.streams().best(ff::media::Type::Audio) else {
        return Ok(None);
    };
    let index = stream.index();
    let context = ff::codec::context::Context::from_parameters(stream.parameters()).map_err(error)?;
    let mut decoder = context.decoder().audio().map_err(error)?;
    let mut resampler: Option<ff::software::resampling::Context> = None;
    let mut out: Vec<f32> = Vec::new();
    let target = ff::format::Sample::F32(ff::format::sample::Type::Packed);
    let mut push = |resampler: &mut ff::software::resampling::Context, frame: Option<&ff::frame::Audio>| {
        let mut resampled = ff::frame::Audio::empty();
        let delay = match frame {
            Some(f) => resampler.run(f, &mut resampled),
            None => {
                // ffmpeg-next's flush passes the frame as it is, and swresample needs its format and layout set.
                // SAFETY: alloc gives an empty frame its buffers for the samples of that format and layout.
                unsafe { resampled.alloc(target, FLUSH_SAMPLES, ff::ChannelLayout::MONO) };
                resampler.flush(&mut resampled)
            }
        }
        .map_err(|e| error(format!("the resampler failed: {e}")))?;
        let _ = delay;
        let n = resampled.samples();
        if n > 0 {
            let data = resampled.data(0);
            out.extend(data[..n * 4].as_chunks::<4>().0.iter().map(|b| f32::from_ne_bytes(*b)));
        }
        Ok::<_, MediaError>(n)
    };
    let mut drain = |decoder: &mut ff::decoder::Audio, resampler: &mut Option<ff::software::resampling::Context>| {
        let mut frame = ff::frame::Audio::empty();
        while more(decoder.receive_frame(&mut frame))? {
            // An AAC frame often names no channel layout. The resampler needs one, so it takes the default of the count.
            if frame.channel_layout().is_empty() {
                frame.set_channel_layout(ff::ChannelLayout::default(i32::from(frame.channels())));
            }
            let r = match resampler {
                Some(r) => r,
                None => resampler.insert(
                    ff::software::resampling::Context::get(
                        frame.format(),
                        frame.channel_layout(),
                        frame.rate(),
                        target,
                        ff::ChannelLayout::MONO,
                        video::SOUND_RATE,
                    )
                    .map_err(error)?,
                ),
            };
            push(r, Some(&frame))?;
        }
        Ok::<_, MediaError>(())
    };
    for (stream, packet) in input.packets() {
        if stream.index() == index {
            decoder.send_packet(&packet).map_err(error)?;
            drain(&mut decoder, &mut resampler)?;
        }
    }
    decoder.send_eof().map_err(error)?;
    drain(&mut decoder, &mut resampler)?;
    // The flush gives the samples that the resampler still holds, then none.
    if let Some(r) = &mut resampler {
        while push(r, None)? > 0 {}
    }
    Ok(Some(out))
}

/// Decode a video: count its frames, sample them, and keep the sampled frames and the sound track.
pub fn decode(bytes: &[u8]) -> Result<Decoded, MediaError> {
    ff::init().map_err(error)?;
    // The errors of a decode come back as values, so ffmpeg's own log keeps its errors only.
    ff::util::log::set_level(ff::util::log::Level::Error);
    let (_fd, path) = memfd(bytes)?;
    let fps = rate(&path)?;
    let total = frames(&path, &|_| false, &mut |_, _| {})?;
    let indices = video::sample(total, fps)?;
    let mut kept: Vec<(usize, Rgb)> = Vec::with_capacity(indices.len());
    let wanted = |i: usize| indices.binary_search(&i).is_ok();
    frames(&path, &wanted, &mut |i, rgb| kept.push((i, rgb)))?;
    // The sampled indices rise strictly, so the kept frames are the sampled frames in order. They move, with no copy.
    if kept.iter().map(|(i, _)| *i).ne(indices.iter().copied()) {
        return Err(error(format!(
            "ffmpeg gave {} of the {} sampled frames on the second pass",
            kept.len(),
            indices.len()
        )));
    }
    let frames = kept.into_iter().map(|(_, rgb)| rgb).collect();
    Ok(Decoded {
        total,
        fps,
        indices,
        frames,
        sound: sound(&path)?,
    })
}
