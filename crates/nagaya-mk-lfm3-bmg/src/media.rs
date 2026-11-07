//! The media spans of a pass: the layout of `meta`, and the rules that tie a span to the tokens and the rows of a
//! segment (engine-abi "LFM3", "Inputs" and "The rows of a span"). None of this needs a card.
//!
//! `meta` holds one record of [`RECORD_BYTES`] bytes for each media span, in the order of the positions. A record is
//! little-endian: the kind (`u32`), the count of positions (`u32`), the first position (`u64`), then four `u32`. For
//! an image they are `grid_t`, `grid_h`, `grid_w` and 0. For a clip they are the count of mel frames and three 0.
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
            SpanKind::Audio { frames } => frames,
        }
    }

    /// The values of one of its rows.
    pub fn width(&self) -> usize {
        match self.kind {
            SpanKind::Image { .. } => PATCH_ROW,
            SpanKind::Audio { .. } => MEL_ROW,
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
            other => return Err(format!("media record {i} has the kind {other}, not an image or a clip")),
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
    /// True when the engine runs the audio encoder: the model has one.
    pub audio_encoder: bool,
}

impl MediaRules {
    /// The placeholder of a span's kind, or an error when the engine has no encoder for it.
    fn token(&self, span: &Span, i: usize) -> Result<u32, String> {
        match span.kind {
            SpanKind::Image { .. } => self.image_token.ok_or_else(|| {
                format!("media record {i} is an image, and this model has no vision tower; send text only")
            }),
            SpanKind::Audio { frames } if frames > crate::audio::MAX_CLIP_FRAMES => Err(format!(
                "media record {i} is a clip of {frames} frames, and the audio encoder takes at most {} frames \
                 (about 41 s); send a shorter clip",
                crate::audio::MAX_CLIP_FRAMES
            )),
            SpanKind::Audio { .. } => match self.audio_token {
                Some(token) if self.audio_encoder => Ok(token),
                Some(_) => Err(format!(
                    "media record {i} is a clip, and this engine has no audio encoder yet (milestone 24 step 7); \
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
        let token = rules.token(span, i)?;
        let p = span.positions();
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
    let placeholders: Vec<u32> = [rules.image_token, rules.audio_token].into_iter().flatten().collect();
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

    fn rules() -> MediaRules {
        MediaRules {
            image_token: Some(IMAGE),
            audio_token: Some(AUDIO),
            audio_encoder: false,
        }
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
}
