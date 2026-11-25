//! The media spans of a pass: the layout of `meta`, and the rules that tie a span to the tokens and the rows of a
//! segment (engine-abi "LFM3", "Inputs" and "The rows of a span"). None of this needs a card.
//!
//! `meta` holds one record of [`RECORD_BYTES`] bytes for each media span, in the order of the positions. A record is
//! little-endian: the kind (`u32`), the count of positions (`u32`), the first position (`u64`), then four `u32`. For
//! an image they are `grid_t`, `grid_h`, `grid_w` and 0. For a clip they are the count of mel frames and three 0.
//! For a latent span they are the width of a row and three 0.
//! The rows of the segment are the rows of each span whole, in f32, in the order of the records.
//!
//! A span can be longer than a feed block, so a pass can hold only some of its positions. The pass still gets the
//! rows of the whole span, because the encoder reads all of them. The engine runs the encoder and keeps the output
//! rows of the positions of its block.

use std::ops::Range;

/// The bytes of one record of `meta`.
pub const RECORD_BYTES: usize = 32;

/// The kind of an image span in a record.
pub const KIND_IMAGE: u32 = 1;

/// The kind of a clip span in a record.
pub const KIND_AUDIO: u32 = 2;

/// The kind of a run of a clip in a record: the sound of one pair of a video (spec 11.11, "The video of a request").
pub const KIND_CLIP_PART: u32 = 3;

/// The kind of the speech of an `assistant` audio part in a record: the codes of its frames, the end frame last
/// (spec 11.11, "Assistant audio").
pub const KIND_CODES: u32 = 4;

/// The kind of a latent span in a record: rows of the width of the residual that take the place of the embedding rows
/// of their positions (engine-abi "Latent rows of LFM3").
pub const KIND_LATENT: u32 = 5;

/// The codes of one frame of speech.
pub const FRAME_CODES: usize = 12;

/// The codes of a codebook, the end code too (`depth::CODES`).
pub const CODEBOOK_CODES: u32 = 2017;

/// The values of one patch row of an image: 3 channels, 2 copies in time, 16 by 16 pixels
/// (`image_processing_lfm3_vl.py:186`, `nagaya-text-lfm3` `media::image::ROW`).
pub const PATCH_ROW: usize = 1536;

/// The values of one mel row of a clip (`nagaya-text-lfm3` `media::audio::BINS`).
pub const MEL_ROW: usize = 128;

/// The positions of the largest image: 2,097,152 pixels over patches of 16 and a merge of 2 by 2 (spec 11.11).
pub const MAX_IMAGE_POSITIONS: usize = 2048;

/// The patches of the largest image.
pub const MAX_IMAGE_PATCHES: usize = 4 * MAX_IMAGE_POSITIONS;

/// What a span holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanKind {
    /// An image of `grid_t * grid_h * grid_w` patches.
    Image { grid: [usize; 3] },
    /// A clip of `frames` mel frames.
    Audio { frames: usize },
    /// The encoder rows from `start` of a clip of `frames` mel frames. The rows of the segment are the whole clip.
    ClipPart { frames: usize, start: usize },
    /// Frames of speech: one position for each frame, whose row is the sum of its 12 codebook rows. The rows of the
    /// segment are the codes, 12 `u32` for each frame.
    Codes,
    /// Latent rows: one row of `width` f32 values for each position. The pass copies each row over the embedding row
    /// of its position, with no encoder.
    Latent { width: usize },
}

/// One media span of `meta`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub kind: SpanKind,
    /// The first position of the span.
    pub first: u64,
    /// The count of positions.
    pub count: usize,
}

impl Span {
    /// The positions of the span.
    pub fn positions(&self) -> Range<u64> {
        self.first..self.first + self.count as u64
    }

    /// The count of rows that the span takes from the rows of the segment.
    pub fn rows(&self) -> usize {
        match self.kind {
            SpanKind::Image { grid } => grid.iter().product(),
            SpanKind::Audio { frames } | SpanKind::ClipPart { frames, .. } => frames,
            SpanKind::Codes | SpanKind::Latent { .. } => self.count,
        }
    }

    /// The values of one of its rows.
    pub fn width(&self) -> usize {
        match self.kind {
            SpanKind::Image { .. } => PATCH_ROW,
            SpanKind::Audio { .. } | SpanKind::ClipPart { .. } => MEL_ROW,
            SpanKind::Codes => FRAME_CODES,
            SpanKind::Latent { width } => width,
        }
    }

    /// The bytes of its rows.
    pub fn bytes(&self) -> usize {
        self.rows() * self.width() * 4
    }
}

/// The bytes of `meta` for `spans`.
pub fn encode(spans: &[Span]) -> Vec<u8> {
    let mut out = Vec::with_capacity(spans.len() * RECORD_BYTES);
    for span in spans {
        let (kind, rest) = match span.kind {
            SpanKind::Image { grid } => (KIND_IMAGE, [grid[0] as u32, grid[1] as u32, grid[2] as u32, 0]),
            SpanKind::Audio { frames } => (KIND_AUDIO, [frames as u32, 0, 0, 0]),
            SpanKind::ClipPart { frames, start } => (KIND_CLIP_PART, [frames as u32, start as u32, 0, 0]),
            SpanKind::Codes => (KIND_CODES, [FRAME_CODES as u32, 0, 0, 0]),
            SpanKind::Latent { width } => (KIND_LATENT, [width as u32, 0, 0, 0]),
        };
        out.extend(kind.to_le_bytes());
        out.extend((span.count as u32).to_le_bytes());
        out.extend(span.first.to_le_bytes());
        for word in rest {
            out.extend(word.to_le_bytes());
        }
    }
    out
}

/// The spans of `meta`. A record that is not whole, or of an unknown kind, or that breaks the shape of its kind, is an
/// error that names the record.
pub fn decode(meta: &[u8]) -> Result<Vec<Span>, String> {
    if !meta.len().is_multiple_of(RECORD_BYTES) {
        return Err(format!(
            "the meta has {} bytes, not whole records of {RECORD_BYTES} bytes",
            meta.len()
        ));
    }
    let mut spans = Vec::with_capacity(meta.len() / RECORD_BYTES);
    for (i, record) in meta.as_chunks::<RECORD_BYTES>().0.iter().enumerate() {
        let word = |at: usize| u32::from_le_bytes(record[at..at + 4].try_into().expect("4 bytes")) as usize;
        let first = u64::from_le_bytes(record[8..16].try_into().expect("8 bytes"));
        let count = word(4);
        let rest = [word(16), word(20), word(24), word(28)];
        let kind = match word(0) as u32 {
            KIND_IMAGE => {
                let grid = [rest[0], rest[1], rest[2]];
                if rest[3] != 0 {
                    return Err(format!(
                        "media record {i} is an image with a fourth word of {}",
                        rest[3]
                    ));
                }
                if grid[0] != 1 {
                    return Err(format!(
                        "media record {i} is an image with grid_t {}; an image has one frame",
                        grid[0]
                    ));
                }
                if grid[1] < 2 || grid[2] < 2 || !grid[1].is_multiple_of(2) || !grid[2].is_multiple_of(2) {
                    return Err(format!(
                        "media record {i} is an image of {} by {} patches; each side is even, because the merge \
                         takes 2 by 2 patches",
                        grid[1], grid[2]
                    ));
                }
                let positions = grid[1] * grid[2] / 4;
                if count != positions {
                    return Err(format!(
                        "media record {i} is an image of {} by {} patches with {count} positions; it has {positions}",
                        grid[1], grid[2]
                    ));
                }
                if count > MAX_IMAGE_POSITIONS {
                    return Err(format!(
                        "media record {i} is an image of {count} positions; an image has at most \
                         {MAX_IMAGE_POSITIONS}"
                    ));
                }
                SpanKind::Image { grid }
            }
            KIND_AUDIO => {
                let frames = rest[0];
                if rest[1..] != [0, 0, 0] {
                    return Err(format!(
                        "media record {i} is a clip with words {rest:?} after its frames"
                    ));
                }
                if frames == 0 || count != frames.div_ceil(8) {
                    return Err(format!(
                        "media record {i} is a clip of {frames} frames with {count} positions; it has {}",
                        frames.div_ceil(8)
                    ));
                }
                SpanKind::Audio { frames }
            }
            KIND_CLIP_PART => {
                let (frames, start) = (rest[0], rest[1]);
                if rest[2..] != [0, 0] {
                    return Err(format!("media record {i} is a run of a clip with words {rest:?}"));
                }
                if count == 0 {
                    return Err(format!("media record {i} is a run of a clip with no positions"));
                }
                let rows = frames.div_ceil(8);
                if frames == 0 || start + count > rows {
                    return Err(format!(
                        "media record {i} is a run of {count} rows from row {start} of a clip of {frames} frames, \
                         past the {rows} rows of the clip"
                    ));
                }
                SpanKind::ClipPart { frames, start }
            }
            KIND_CODES => {
                if rest != [FRAME_CODES, 0, 0, 0] {
                    return Err(format!(
                        "media record {i} is speech with words {rest:?}; it names {FRAME_CODES} codes for each frame"
                    ));
                }
                if count == 0 {
                    return Err(format!("media record {i} is speech with no frames"));
                }
                SpanKind::Codes
            }
            KIND_LATENT => {
                let width = rest[0];
                if width == 0 || rest[1..] != [0, 0, 0] {
                    return Err(format!(
                        "media record {i} is a latent span with words {rest:?}; it names the width of its rows, then \
                         three 0"
                    ));
                }
                if count == 0 {
                    return Err(format!("media record {i} is a latent span with no positions"));
                }
                SpanKind::Latent { width }
            }
            other => {
                return Err(format!(
                    "media record {i} has the kind {other}, not an image, a clip, a run, speech or latent rows"
                ));
            }
        };
        spans.push(Span { kind, first, count });
    }
    Ok(spans)
}

/// The placeholder tokens of the media of a model, and the kinds that the engine can encode. A model with no media
/// has none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MediaRules {
    /// The placeholder of an image (`image_token_id`). A model with it has a vision tower.
    pub image_token: Option<u32>,
    /// The placeholder of a clip (`audio_token_id`).
    pub audio_token: Option<u32>,
    /// The placeholder of a video (`video_token_id`). The pair of a video is an image span at its positions.
    pub video_token: Option<u32>,
    /// The placeholder of a frame of speech (`audio_out_token_id`), when the engine has the codebook table.
    pub audio_out_token: Option<u32>,
    /// True when the engine runs the audio encoder: the model has one.
    pub audio_encoder: bool,
    /// The placeholder and the width of latent rows, when the engine takes them.
    pub latent: Option<LatentRule>,
}

/// The rule of latent rows (engine-abi "Latent rows of LFM3"): the placeholder token of a latent position, and the
/// width of a row, which is the width of the residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentRule {
    pub token: u32,
    pub width: usize,
}

impl LatentRule {
    /// The rule of a model from its `config.json`. The placeholder is `pad_token_id`: at the root for `LFM3-1.7B`, and
    /// in `text_config` for Omni. The width is the width of the residual.
    pub fn of(config_text: &str, width: usize) -> Result<LatentRule, String> {
        let root: serde_json::Value =
            serde_json::from_str(config_text).map_err(|e| format!("config.json is not JSON: {e}"))?;
        let pad = match &root["text_config"] {
            serde_json::Value::Object(text) => &text["pad_token_id"],
            _ => &root["pad_token_id"],
        };
        let token = pad
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("config.json has pad_token_id {pad}, not a token id"))?;
        Ok(LatentRule { token, width })
    }
}

impl MediaRules {
    /// The placeholder of a span's kind, or an error when the engine has no encoder for it.
    fn token(&self, span: &Span, i: usize) -> Result<u32, String> {
        match span.kind {
            SpanKind::Image { .. } => self.image_token.ok_or_else(|| {
                format!("media record {i} is an image, and this model has no vision tower; send text only")
            }),
            SpanKind::Audio { frames } | SpanKind::ClipPart { frames, .. }
                if frames > crate::audio::MAX_CLIP_FRAMES =>
            {
                Err(format!(
                    "media record {i} is a clip of {frames} frames, and the audio encoder takes at most {} frames \
                 (about 41 s); send a shorter clip",
                    crate::audio::MAX_CLIP_FRAMES
                ))
            }
            SpanKind::Codes => self.audio_out_token.ok_or_else(|| {
                format!("media record {i} is speech, and this model has no codebook table; send its text only")
            }),
            SpanKind::Latent { width } => match self.latent {
                Some(rule) if rule.width == width => Ok(rule.token),
                Some(rule) => Err(format!(
                    "media record {i} is a latent span of rows of {width} values, and a row of this model has {} \
                     values; send rows of the width of the residual",
                    rule.width
                )),
                None => Err(format!(
                    "media record {i} is a latent span, and this engine takes no latent rows"
                )),
            },
            SpanKind::Audio { .. } | SpanKind::ClipPart { .. } => match self.audio_token {
                Some(token) if self.audio_encoder => Ok(token),
                Some(_) => Err(format!(
                    "media record {i} is a clip, and this engine has no audio encoder yet; \
                     send no audio"
                )),
                None => Err(format!(
                    "media record {i} is a clip, and this model has no audio encoder; send text only"
                )),
            },
        }
    }
}

/// One span that a block reads: the span, where its rows start in the rows of the segment, and the positions of the
/// block that it fills.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanUse {
    pub span: Span,
    /// The first byte of its rows in the rows of the segment.
    pub rows_at: usize,
    /// The positions of the span that the block holds, relative to the first position of the span.
    pub take: Range<usize>,
    /// The row of the block of the first position that it fills.
    pub block_row: usize,
}

/// Check the spans of a block of `tokens` from position `pos0` against its rows, and give what the block reads.
///
/// - Each span reads at least one position of the block, and the spans do not overlap.
/// - Each position of a span in the block holds the placeholder of its kind. Each placeholder of the block is in a
///   span, so no placeholder enters the backbone as its embedding row.
/// - The rows are the rows of every span whole, and nothing more.
pub fn check_spans(
    rules: &MediaRules,
    spans: &[Span],
    tokens: &[u32],
    pos0: u64,
    rows: usize,
) -> Result<Vec<SpanUse>, String> {
    let end = pos0 + tokens.len() as u64;
    let mut uses = Vec::with_capacity(spans.len());
    let mut covered = vec![false; tokens.len()];
    let mut rows_at = 0usize;
    let mut last_end = 0u64;
    for (i, span) in spans.iter().enumerate() {
        let mut token = rules.token(span, i)?;
        let p = span.positions();
        // An image span holds the image placeholder, or the video placeholder for a pair of a video. Its first
        // position in the block names which, and each of its positions holds that one.
        if let (SpanKind::Image { .. }, Some(video)) = (span.kind, rules.video_token)
            && p.start.max(pos0) < end
            && tokens.get((p.start.max(pos0).saturating_sub(pos0)) as usize) == Some(&video)
        {
            token = video;
        }
        if i > 0 && p.start < last_end {
            return Err(format!(
                "media record {i} starts at {} inside the span before it, which ends at {last_end}",
                p.start
            ));
        }
        last_end = p.end;
        if p.end <= pos0 || p.start >= end {
            return Err(format!(
                "media record {i} holds the positions {} to {}, and the block holds {pos0} to {end}; a block names \
                 only the spans that it reads",
                p.start, p.end
            ));
        }
        let (lo, hi) = (p.start.max(pos0), p.end.min(end));
        for at in lo..hi {
            let t = (at - pos0) as usize;
            if tokens[t] != token {
                return Err(format!(
                    "position {at} is in media record {i}, and it holds the token {}, not the placeholder {token}",
                    tokens[t]
                ));
            }
            covered[t] = true;
        }
        uses.push(SpanUse {
            span: *span,
            rows_at,
            take: (lo - p.start) as usize..(hi - p.start) as usize,
            block_row: (lo - pos0) as usize,
        });
        rows_at += span.bytes();
    }
    if rows != rows_at {
        return Err(format!(
            "the segment has {rows} bytes of rows, and its media records need {rows_at}"
        ));
    }
    // The speech placeholder is not here: a frame of the speech loop feeds it alone, and the frame call writes its
    // row (`depth::frame_embedding`). A span of speech still checks its own positions.
    let placeholders: Vec<u32> = [rules.image_token, rules.audio_token, rules.video_token]
        .into_iter()
        .flatten()
        .collect();
    if let Some(t) = (0..tokens.len()).find(|t| !covered[*t] && placeholders.contains(&tokens[*t])) {
        return Err(format!(
            "position {} holds the media placeholder {}, and no media record names it",
            pos0 + t as u64,
            tokens[t]
        ));
    }
    Ok(uses)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE: u32 = 125_030;
    const AUDIO: u32 = 125_031;
    const VIDEO: u32 = 125_029;
    const AUDIO_OUT: u32 = 125_032;

    fn rules() -> MediaRules {
        MediaRules {
            image_token: Some(IMAGE),
            audio_token: Some(AUDIO),
            video_token: Some(VIDEO),
            audio_out_token: Some(AUDIO_OUT),
            audio_encoder: false,
            latent: Some(LatentRule {
                token: PAD,
                width: 2048,
            }),
        }
    }

    const PAD: u32 = 124_893;

    fn latent(first: u64, count: usize, width: usize) -> Span {
        Span {
            kind: SpanKind::Latent { width },
            first,
            count,
        }
    }

    /// A latent span names the width of its rows, and it takes one row of that width for each position (engine-abi
    /// "Latent rows of LFM3").
    #[test]
    fn a_latent_span_goes_through_its_bytes_and_takes_one_row_for_each_position() {
        let span = latent(30, 2, 2048);
        let meta = encode(&[span]);
        assert_eq!(&meta[..4], &KIND_LATENT.to_le_bytes());
        assert_eq!(&meta[4..8], &2u32.to_le_bytes());
        assert_eq!(&meta[8..16], &30u64.to_le_bytes());
        assert_eq!(&meta[16..32], &[0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(decode(&meta).unwrap(), [span]);
        assert_eq!(span.bytes(), 2 * 2048 * 4);
        let uses = check_spans(&rules(), &[span], &[PAD, PAD], 30, span.bytes()).unwrap();
        assert_eq!((uses[0].take.clone(), uses[0].block_row), (0..2, 0));
        // A block of text before the latent position.
        let one = latent(32, 1, 2048);
        let uses = check_spans(&rules(), &[one], &[7, 7, PAD], 30, one.bytes()).unwrap();
        assert_eq!((uses[0].take.clone(), uses[0].block_row), (0..1, 2));
    }

    #[test]
    fn a_latent_span_that_breaks_its_rule_is_refused() {
        let bad = |edit: &dyn Fn(&mut Vec<u8>), want: &str| {
            let mut meta = encode(&[latent(0, 1, 2048)]);
            edit(&mut meta);
            let e = decode(&meta).unwrap_err();
            assert!(e.contains(want), "{e}");
        };
        bad(
            &|m| m[16..20].copy_from_slice(&[0; 4]),
            "latent span with words [0, 0, 0, 0]",
        );
        bad(&|m| m[24] = 1, "latent span with words [2048, 0, 1, 0]");
        bad(&|m| m[4] = 0, "latent span with no positions");
        // A row of another width, a position that does not hold the placeholder, and rows of another size.
        let span = latent(0, 1, 1024);
        let e = check_spans(&rules(), &[span], &[PAD], 0, span.bytes()).unwrap_err();
        assert!(
            e.contains("rows of 1024 values") && e.contains("has 2048 values"),
            "{e}"
        );
        let span = latent(0, 1, 2048);
        let e = check_spans(&rules(), &[span], &[7], 0, span.bytes()).unwrap_err();
        assert!(e.contains("not the placeholder 124893"), "{e}");
        let e = check_spans(&rules(), &[span], &[PAD], 0, 4).unwrap_err();
        assert!(e.contains("need 8192"), "{e}");
        let none = MediaRules {
            latent: None,
            ..rules()
        };
        let e = check_spans(&none, &[span], &[PAD], 0, span.bytes()).unwrap_err();
        assert!(e.contains("takes no latent rows"), "{e}");
    }

    /// The placeholder of a latent position is `pad_token_id`: at the root of `LFM3-1.7B`, in `text_config` of Omni.
    #[test]
    fn the_latent_rule_takes_the_pad_token_of_each_config() {
        let text = include_str!("../tests/fixtures/lfm3.config.json");
        let omni = include_str!("../tests/fixtures/lfm3-omni.config.json");
        assert_eq!(
            LatentRule::of(text, 2048).unwrap(),
            LatentRule {
                token: PAD,
                width: 2048
            }
        );
        assert_eq!(
            LatentRule::of(omni, 2048).unwrap(),
            LatentRule {
                token: PAD,
                width: 2048
            }
        );
        let e = LatentRule::of(r#"{"pad_token_id": "x"}"#, 2048).unwrap_err();
        assert!(e.contains("pad_token_id \"x\""), "{e}");
    }

    fn image(first: u64, grid_h: usize, grid_w: usize) -> Span {
        Span {
            kind: SpanKind::Image {
                grid: [1, grid_h, grid_w],
            },
            first,
            count: grid_h * grid_w / 4,
        }
    }

    /// Tokens of a block from `pos0`: text, with the placeholder at the positions of `spans`.
    fn tokens(pos0: u64, n: usize, spans: &[Span]) -> Vec<u32> {
        (pos0..pos0 + n as u64)
            .map(|p| {
                if spans.iter().any(|s| s.positions().contains(&p)) {
                    IMAGE
                } else {
                    7
                }
            })
            .collect()
    }

    #[test]
    fn a_record_goes_through_its_bytes_and_back() {
        let spans = [
            image(5, 14, 22),
            Span {
                kind: SpanKind::Audio { frames: 100 },
                first: 90,
                count: 13,
            },
        ];
        let meta = encode(&spans);
        assert_eq!(meta.len(), 2 * RECORD_BYTES);
        assert_eq!(&meta[..4], &KIND_IMAGE.to_le_bytes());
        assert_eq!(&meta[4..8], &77u32.to_le_bytes());
        assert_eq!(&meta[8..16], &5u64.to_le_bytes());
        assert_eq!(&meta[16..28], &[1, 0, 0, 0, 14, 0, 0, 0, 22, 0, 0, 0]);
        assert_eq!(decode(&meta).unwrap(), spans);
        assert_eq!(spans[0].bytes(), 308 * 1536 * 4);
        assert_eq!(spans[1].bytes(), 100 * 128 * 4);
    }

    #[test]
    fn a_record_that_breaks_its_shape_is_refused() {
        let bad = |edit: &dyn Fn(&mut Vec<u8>), want: &str| {
            let mut meta = encode(&[image(0, 14, 22)]);
            edit(&mut meta);
            let e = decode(&meta).unwrap_err();
            assert!(e.contains(want), "{e}");
        };
        bad(&|m| m.truncate(31), "not whole records");
        bad(&|m| m[0] = 9, "the kind 9");
        bad(&|m| m[4] = 78, "with 78 positions; it has 77");
        bad(&|m| m[16] = 2, "grid_t 2");
        bad(&|m| m[20] = 15, "15 by 22 patches; each side is even");
        bad(&|m| m[28] = 1, "fourth word");
        let big = encode(&[image(0, 96, 96)]);
        assert!(decode(&big).unwrap_err().contains("at most 2048"));
        let clip = Span {
            kind: SpanKind::Audio { frames: 100 },
            first: 0,
            count: 12,
        };
        assert!(decode(&encode(&[clip])).unwrap_err().contains("it has 13"));
    }

    #[test]
    fn a_span_longer_than_the_block_gives_the_positions_of_the_block() {
        // An image of 117 positions from 5, read by a block of 64 from 64: positions 64 to 121 of the span's 5 to 121.
        let span = image(5, 18, 26);
        let toks = tokens(64, 64, &[span]);
        let uses = check_spans(&rules(), &[span], &toks, 64, span.bytes()).unwrap();
        assert_eq!(
            uses,
            vec![SpanUse {
                span,
                rows_at: 0,
                take: 59..117,
                block_row: 0
            }]
        );
        // The block of the start of the span.
        let toks = tokens(0, 64, &[span]);
        let uses = check_spans(&rules(), &[span], &toks, 0, span.bytes()).unwrap();
        assert_eq!((uses[0].take.clone(), uses[0].block_row), (0..59, 5));
    }

    #[test]
    fn two_spans_take_their_rows_in_order() {
        let (a, b) = (image(2, 8, 8), image(30, 16, 16));
        let toks = tokens(0, 200, &[a, b]);
        let uses = check_spans(&rules(), &[a, b], &toks, 0, a.bytes() + b.bytes()).unwrap();
        assert_eq!(
            (uses[1].rows_at, uses[1].take.clone(), uses[1].block_row),
            (a.bytes(), 0..64, 30)
        );
        let e = check_spans(&rules(), &[a, b], &toks, 0, a.bytes()).unwrap_err();
        assert!(e.contains("need"), "{e}");
        let e = check_spans(&rules(), &[b, a], &toks, 0, a.bytes() + b.bytes()).unwrap_err();
        assert!(e.contains("inside the span before it"), "{e}");
    }

    #[test]
    fn a_span_must_match_the_placeholders_of_the_block() {
        let span = image(4, 8, 8);
        let mut toks = tokens(0, 40, &[span]);
        toks[10] = 7;
        let e = check_spans(&rules(), &[span], &toks, 0, span.bytes()).unwrap_err();
        assert!(e.contains("position 10 is in media record 0"), "{e}");
        // A placeholder outside each span.
        let mut toks = tokens(0, 40, &[span]);
        toks[30] = IMAGE;
        let e = check_spans(&rules(), &[span], &toks, 0, span.bytes()).unwrap_err();
        assert!(e.contains("position 30 holds the media placeholder"), "{e}");
        // No span at all.
        let e = check_spans(&rules(), &[], &tokens(0, 40, &[span]), 0, 0).unwrap_err();
        assert!(e.contains("position 4 holds the media placeholder"), "{e}");
        // A span that the block does not read.
        let e = check_spans(&rules(), &[image(100, 8, 8)], &tokens(0, 40, &[]), 0, span.bytes()).unwrap_err();
        assert!(e.contains("names only the spans that it reads"), "{e}");
    }

    /// With the audio encoder, a clip of up to 4,096 frames fills its positions, and a longer clip is refused before
    /// the encoder sees it (engine-abi "The audio encoder of LFM3").
    #[test]
    fn a_clip_is_taken_with_the_encoder_up_to_the_longest_clip() {
        let rules = MediaRules {
            audio_encoder: true,
            ..rules()
        };
        let clip = |frames: usize| Span {
            kind: SpanKind::Audio { frames },
            first: 0,
            count: frames.div_ceil(8),
        };
        let uses = check_spans(&rules, &[clip(8)], &[AUDIO], 0, clip(8).bytes()).expect("a clip of 8 frames");
        assert_eq!((uses[0].take.clone(), uses[0].block_row), (0..1, 0));
        let longest = clip(crate::audio::MAX_CLIP_FRAMES);
        let tokens = vec![AUDIO; longest.count];
        check_spans(&rules, &[longest], &tokens, 0, longest.bytes()).expect("the longest clip");
        let long = clip(crate::audio::MAX_CLIP_FRAMES + 1);
        let tokens = vec![AUDIO; long.count];
        let e = check_spans(&rules, &[long], &tokens, 0, long.bytes()).unwrap_err();
        assert!(e.contains("4097 frames") && e.contains("at most 4096"), "{e}");
    }

    #[test]
    fn a_kind_with_no_encoder_is_refused() {
        let clip = Span {
            kind: SpanKind::Audio { frames: 8 },
            first: 0,
            count: 1,
        };
        let e = check_spans(&rules(), &[clip], &[AUDIO], 0, clip.bytes()).unwrap_err();
        assert!(e.contains("no audio encoder yet"), "{e}");
        // A clip placeholder with no span is refused too, so it never enters the backbone as its embedding row.
        let e = check_spans(&rules(), &[], &[7, AUDIO], 0, 0).unwrap_err();
        assert!(e.contains("position 1 holds the media placeholder 125031"), "{e}");
        let span = image(0, 2, 2);
        let e = check_spans(&MediaRules::default(), &[span], &[IMAGE], 0, span.bytes()).unwrap_err();
        assert!(e.contains("no vision tower"), "{e}");
    }

    fn run(first: u64, frames: usize, start: usize, count: usize) -> Span {
        Span {
            kind: SpanKind::ClipPart { frames, start },
            first,
            count,
        }
    }

    /// A run of a clip takes the whole clip's mel rows and names the first encoder row of its positions (spec 11.11,
    /// "The video of a request").
    #[test]
    fn a_run_of_a_clip_goes_through_its_bytes_and_takes_the_rows_of_its_positions() {
        let r = run(40, 249, 15, 15);
        let meta = encode(&[r]);
        assert_eq!(&meta[..4], &KIND_CLIP_PART.to_le_bytes());
        assert_eq!(&meta[16..24], &[249, 0, 0, 0, 15, 0, 0, 0]);
        assert_eq!(decode(&meta).unwrap(), [r]);
        assert_eq!(r.bytes(), 249 * 128 * 4, "the rows of the whole clip");
        // The clip has ceil(249 / 8) = 32 encoder rows, so a run from 30 holds at most 2.
        assert!(decode(&encode(&[run(0, 249, 30, 2)])).is_ok());
        let e = decode(&encode(&[run(0, 249, 30, 3)])).unwrap_err();
        assert!(e.contains("past the 32 rows"), "{e}");
        let e = decode(&encode(&[run(0, 249, 0, 0)])).unwrap_err();
        assert!(e.contains("no positions"), "{e}");
        let rules = MediaRules {
            audio_encoder: true,
            ..rules()
        };
        let uses = check_spans(&rules, &[r], &[AUDIO; 15], 40, r.bytes()).unwrap();
        assert_eq!((uses[0].take.clone(), uses[0].block_row), (0..15, 0));
    }

    /// The pair of a video is an image of one frame in time at the positions of the video placeholder.
    #[test]
    fn an_image_span_takes_the_video_placeholder_too() {
        let span = image(0, 8, 10);
        check_spans(&rules(), &[span], &[VIDEO; 20], 0, span.bytes()).expect("a pair of a video");
        let e = check_spans(&rules(), &[], &[7, VIDEO], 0, 0).unwrap_err();
        assert!(e.contains("position 1 holds the media placeholder 125029"), "{e}");
        let mixed: Vec<u32> = (0..20).map(|i| if i < 10 { VIDEO } else { IMAGE }).collect();
        let e = check_spans(&rules(), &[span], &mixed, 0, span.bytes()).unwrap_err();
        assert!(e.contains("position 10 is in media record 0"), "{e}");
    }

    /// The speech of an assistant audio part: one position for each frame, and 12 codes of rows for each.
    #[test]
    fn speech_goes_through_its_bytes_and_takes_12_codes_for_each_frame() {
        let span = Span {
            kind: SpanKind::Codes,
            first: 30,
            count: 47,
        };
        let meta = encode(&[span]);
        assert_eq!(&meta[..4], &KIND_CODES.to_le_bytes());
        assert_eq!(&meta[16..20], &12u32.to_le_bytes());
        assert_eq!(decode(&meta).unwrap(), [span]);
        assert_eq!(span.bytes(), 47 * 12 * 4);
        let uses = check_spans(&rules(), &[span], &[AUDIO_OUT; 47], 30, span.bytes()).unwrap();
        assert_eq!((uses[0].take.clone(), uses[0].block_row), (0..47, 0));
        let mut bad = meta.clone();
        bad[16] = 11;
        assert!(decode(&bad).unwrap_err().contains("12 codes"));
        // A frame of the speech loop feeds the placeholder alone, with no span: the frame call writes its row.
        assert!(check_spans(&rules(), &[], &[7, AUDIO_OUT], 0, 0).unwrap().is_empty());
        let none = MediaRules {
            audio_out_token: None,
            ..rules()
        };
        let e = check_spans(&none, &[span], &[AUDIO_OUT; 47], 30, span.bytes()).unwrap_err();
        assert!(e.contains("no codebook table"), "{e}");
    }
}
