//! The WAV reader takes 22,050 Hz mono in 16-bit integers or 32-bit floats, and refuses the rest by name.

use std::io::Cursor;

use nagaya_family_beatthis::Error;
use nagaya_family_beatthis::audio::read_wav;

fn wav(rate: u32, channels: u16, bits: u16, float: bool, frames: usize) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels,
        sample_rate: rate,
        bits_per_sample: bits,
        sample_format: if float {
            hound::SampleFormat::Float
        } else {
            hound::SampleFormat::Int
        },
    };
    let mut out = Cursor::new(Vec::new());
    let mut w = hound::WavWriter::new(&mut out, spec).unwrap();
    for i in 0..frames * usize::from(channels) {
        match (float, bits) {
            (true, _) => w.write_sample(i as f32 / 1000.0 - 0.5).unwrap(),
            (false, 16) => w.write_sample((i as i32 * 37 % 65_536 - 32_768) as i16).unwrap(),
            (false, _) => w.write_sample(i as i32).unwrap(),
        }
    }
    w.finalize().unwrap();
    out.into_inner()
}

#[test]
fn reads_16_bit_mono_as_torchaudio_does() {
    let got = read_wav(&wav(22_050, 1, 16, false, 1000)).unwrap();
    assert_eq!(got.len(), 1000);
    for (i, v) in got.iter().enumerate() {
        let want = f32::from((i as i32 * 37 % 65_536 - 32_768) as i16) / 32_768.0;
        assert_eq!(v.to_bits(), want.to_bits(), "sample {i}");
    }
}

#[test]
fn reads_32_bit_float_mono_unchanged() {
    let got = read_wav(&wav(22_050, 1, 32, true, 1000)).unwrap();
    for (i, v) in got.iter().enumerate() {
        assert_eq!(v.to_bits(), (i as f32 / 1000.0 - 0.5).to_bits(), "sample {i}");
    }
}

#[test]
fn refuses_another_rate_and_names_the_rate_and_channels() {
    let e = read_wav(&wav(44_100, 1, 16, false, 100)).unwrap_err();
    assert_eq!(
        e,
        Error::Layout {
            rate: 44_100,
            channels: 1
        }
    );
    let text = e.to_string();
    assert!(
        text.contains("44100 Hz") && text.contains("1 channels") && text.contains("22050 Hz mono"),
        "{text}"
    );
}

#[test]
fn refuses_stereo() {
    assert_eq!(
        read_wav(&wav(22_050, 2, 16, false, 100)).unwrap_err(),
        Error::Layout {
            rate: 22_050,
            channels: 2
        }
    );
}

#[test]
fn refuses_stereo_at_another_rate_in_one_error() {
    assert_eq!(
        read_wav(&wav(48_000, 2, 16, false, 100)).unwrap_err(),
        Error::Layout {
            rate: 48_000,
            channels: 2
        }
    );
}

#[test]
fn refuses_24_bit_integers() {
    assert_eq!(
        read_wav(&wav(22_050, 1, 24, false, 100)).unwrap_err(),
        Error::Format { bits: 24, float: false }
    );
}

#[test]
fn refuses_bytes_that_are_not_wav() {
    assert!(matches!(
        read_wav(b"not a riff file at all").unwrap_err(),
        Error::Wav(_)
    ));
}

#[test]
fn refuses_a_truncated_file() {
    let mut bytes = wav(22_050, 1, 16, false, 1000);
    bytes.truncate(bytes.len() - 7);
    assert!(matches!(read_wav(&bytes).unwrap_err(), Error::Wav(_)));
}
