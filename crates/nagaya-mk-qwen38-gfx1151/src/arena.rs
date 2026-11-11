//! The weight arena: the plan of every uploaded tensor, and the sealed memfd that holds them (milestone 15 step 6).
//!
//! mk's `engine.hip` planned and filled the arena in C++. This module does it in Rust, byte for byte: the header, the
//! manifest and the tensor data are mk's `arena.h` layout. `examples/arena_digest` gives the arena's SHA-256, and the
//! digest of the copy must equal mk's. The kernels read the arena where it is mapped. The engine is never closed, so
//! the arena stays mapped and registered for the life of the process.

use std::ffi::c_void;
use std::path::Path;

use crate::binding::{Binding, RtHostUnregister};
use crate::dequant::GgmlType;
use crate::gguf::{Model, TensorInfo};

/// The arena's magic, `arena.h` `kMagic`.
const MAGIC: [u8; 8] = *b"MKARENA\0";
/// The layout version, `arena.h` `kFormat`.
const FORMAT: u32 = 1;
/// The alignment of each tensor, `arena.h` `kAlign`.
const ALIGN: u64 = 4096;
/// The bytes kept for the header, `arena.h` `kHeaderBytes`.
const HEADER_BYTES: u64 = 4096;
/// The alignment of the first tensor, for transparent huge pages, `arena.h` `kDataAlign`.
const DATA_ALIGN: u64 = 2 << 20;
/// The most shards that the header records, `arena.h` `kMaxShards`.
const MAX_SHARDS: usize = 4;
/// The bytes of one manifest entry, `sizeof(arena::Entry)`.
const ENTRY_BYTES: usize = 128;
/// The bytes of the name in an entry.
const NAME_BYTES: usize = 64;
/// The bytes of one shard identity, `sizeof(arena::ShardId)`.
const SHARD_ID_BYTES: usize = 256;
/// The bytes of the path in a shard identity.
const SHARD_PATH_BYTES: usize = 240;
/// The offset of the shard identities in the header, `offsetof(Header, shards)`.
const SHARDS_OFFSET: usize = 48;
/// The seals of a filled arena: no new seal, no resize, and no new writable map.
const SEALS: libc::c_int = libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_FUTURE_WRITE;
/// The PLE table stays in the GGUF map. The host gathers its rows for each pass (`ple.rs`).
pub(crate) const PLE_TABLE: &str = "per_layer_token_embd.weight";

/// The type ids of the read-once tiles, `quant.h`. They are not ggml types.
const IQ3_S_TILES: u32 = 121;
const Q8_0_TILES: u32 = 108;
const Q6_K_TILES: u32 = 114;
const IQ4_NL_TILES: u32 = 120;
/// The bytes of one tile of 16 rows, `tiles.h`.
const IQ3_S_TILE: u64 = 1792;
const Q8_0_TILE: u64 = 1088;
const Q6_K_TILE: u64 = 3360;
const IQ4_NL_TILE: u64 = 1152;

/// A tensor's read-once tile layout (`tiles.h`), or none. The value is the id that `mk_relayout` takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum Tile {
    None = 0,
    Iq3s = 1,
    Q8 = 2,
    Q6 = 3,
    Iq4 = 4,
}

/// One uploaded tensor: its index in the model, its final layout, and its place in the arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Planned {
    pub gguf: usize,
    pub name: String,
    /// The final type id: the ggml type, or a tile type.
    pub kind: u32,
    pub ne: [u64; 3],
    pub row_bytes: u64,
    pub tile: Tile,
    pub rows: u32,
    pub k: u32,
    pub src_bytes: u64,
    pub bytes: u64,
    pub offset: u64,
}

/// Values per block of a final type id (`quant.h` `block_values`).
fn block_values(kind: u32) -> u64 {
    match kind {
        Q8_0_TILES => 64,
        IQ4_NL_TILES => 128,
        IQ3_S_TILES | Q6_K_TILES => 256,
        other => GgmlType::from_id(other)
            .unwrap_or_else(|| panic!("type id {other} has no block"))
            .block_values(),
    }
}

/// Bytes per block of a final type id (`quant.h` `block_bytes`).
fn block_bytes(kind: u32) -> u64 {
    match kind {
        Q8_0_TILES => 68,
        IQ4_NL_TILES => 72,
        IQ3_S_TILES => 112,
        Q6_K_TILES => 210,
        other => GgmlType::from_id(other)
            .unwrap_or_else(|| panic!("type id {other} has no block"))
            .block_bytes(),
    }
}

/// The bytes of one row of `k` values (`quant.h` `row_bytes`).
pub(crate) fn row_bytes(kind: u32, k: u32) -> u64 {
    u64::from(k) / block_values(kind) * block_bytes(kind)
}

fn round_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

/// The tile of one tensor, from its name and type (`engine.hip` `plan`, mk 555a2a3). The IQ3_S expert gate and up
/// matrices, the IQ4_NL expert down matrices, every Q8_0 matrix but the token embedding table and the expert matrices,
/// and the Q6_K output head take tiles.
fn tile_of(name: &str, kind: GgmlType) -> Result<Tile, String> {
    let long = name.len() > 20;
    let expert_gu = long && (name.contains(".ffn_gate_exps.weight") || name.contains(".ffn_up_exps.weight"));
    let expert_dn = long && name.contains(".ffn_down_exps.weight");
    if expert_dn && kind != GgmlType::Iq4Nl && kind != GgmlType::Q8_0 {
        return Err(format!(
            "{name}: expert down matrices are IQ4_NL or Q8_0, this one is type {}",
            kind.id()
        ));
    }
    Ok(if expert_gu && kind == GgmlType::Iq3S {
        Tile::Iq3s
    } else if kind == GgmlType::Q8_0 && name != "token_embd.weight" && !name.contains("_exps.") {
        Tile::Q8
    } else if kind == GgmlType::Q6K && name == "output.weight" {
        Tile::Q6
    } else if expert_dn && kind == GgmlType::Iq4Nl {
        Tile::Iq4
    } else {
        Tile::None
    })
}

/// Every uploaded tensor's place in the arena and its final layout, from the GGUF records and the tile setting
/// alone. Returns the plan and the arena's size in bytes.
pub(crate) fn plan(tensors: &[TensorInfo], tiles: bool) -> Result<(Vec<Planned>, u64), String> {
    let mut planned = Vec::new();
    for (index, info) in tensors.iter().enumerate() {
        if info.name == PLE_TABLE {
            continue;
        }
        let name = &info.name;
        if name.len() >= NAME_BYTES {
            return Err(format!("{name}: name longer than the arena's {} bytes", NAME_BYTES - 1));
        }
        let values = info.kind.block_values();
        if !info.ne[0].is_multiple_of(values) || (values > 1 && !info.ne[0].is_multiple_of(32)) {
            return Err(format!(
                "{name}: ne0 {} is not a multiple of its block ({values})",
                info.ne[0]
            ));
        }
        let k = u32::try_from(info.ne[0]).map_err(|_| format!("{name}: ne0 {} is past 32 bits", info.ne[0]))?;
        let rows = u32::try_from(info.ne[1] * info.ne[2])
            .map_err(|_| format!("{name}: {} rows are past 32 bits", info.ne[1] * info.ne[2]))?;
        let mut p = Planned {
            gguf: index,
            name: name.clone(),
            kind: info.kind.id(),
            ne: [info.ne[0], info.ne[1], info.ne[2]],
            row_bytes: row_bytes(info.kind.id(), k),
            tile: Tile::None,
            rows,
            k,
            src_bytes: info.n_bytes,
            bytes: info.n_bytes,
            offset: 0,
        };
        if tiles {
            p.tile = tile_of(name, info.kind)?;
            if p.tile != Tile::None {
                let (n, kk) = (u64::from(rows), u64::from(k));
                let (bytes, kind) = match p.tile {
                    Tile::Q6 => (n / 16 * (kk / 256) * Q6_K_TILE, Q6_K_TILES),
                    Tile::Q8 => (n / 16 * (kk / 64) * Q8_0_TILE, Q8_0_TILES),
                    Tile::Iq4 => (n / 16 * (kk / 128) * IQ4_NL_TILE, IQ4_NL_TILES),
                    Tile::Iq3s => (n / 16 * (kk / 256) * IQ3_S_TILE, IQ3_S_TILES),
                    Tile::None => unreachable!("an untiled tensor has no tile bytes"),
                };
                if p.tile != Tile::Iq3s && bytes != p.row_bytes * n {
                    return Err(format!(
                        "{name}: type {} tiles take {bytes} bytes, the rows {}",
                        p.kind,
                        p.row_bytes * n
                    ));
                }
                p.kind = kind;
                p.row_bytes = row_bytes(kind, k);
                p.bytes = bytes;
            }
        }
        planned.push(p);
    }
    let mut offset = round_up(HEADER_BYTES + (planned.len() * ENTRY_BYTES) as u64, DATA_ALIGN);
    for p in &mut planned {
        p.offset = offset;
        offset = round_up(offset + p.bytes, ALIGN);
    }
    Ok((planned, offset))
}

/// A shard as the creator saw it (`arena::ShardId`): its real path, its size and its mtime in nanoseconds.
pub(crate) fn shard_id(path: &Path) -> Result<[u8; SHARD_ID_BYTES], String> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;
    let real = std::fs::canonicalize(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let bytes = real.as_os_str().as_bytes();
    if bytes.len() >= SHARD_PATH_BYTES {
        return Err(format!(
            "{}: path longer than {} bytes",
            real.display(),
            SHARD_PATH_BYTES - 1
        ));
    }
    let meta = std::fs::metadata(&real).map_err(|error| format!("stat {}: {error}", real.display()))?;
    let mut id = [0u8; SHARD_ID_BYTES];
    id[0..8].copy_from_slice(&meta.size().to_le_bytes());
    let mtime_ns = meta.mtime() * 1_000_000_000 + meta.mtime_nsec();
    id[8..16].copy_from_slice(&mtime_ns.to_le_bytes());
    id[16..16 + bytes.len()].copy_from_slice(bytes);
    Ok(id)
}

/// The header (`arena::Header`) of an arena of `planned` tensors and `bytes` bytes. The rest of its 4096 bytes are 0.
pub(crate) fn header(planned: &[Planned], bytes: u64, tiles: bool, shards: &[[u8; SHARD_ID_BYTES]]) -> Vec<u8> {
    assert!(
        shards.len() <= MAX_SHARDS,
        "{} shards, the arena records {MAX_SHARDS}",
        shards.len()
    );
    let mut h = vec![0u8; SHARDS_OFFSET + MAX_SHARDS * SHARD_ID_BYTES];
    h[0..8].copy_from_slice(&MAGIC);
    h[8..12].copy_from_slice(&FORMAT.to_le_bytes());
    h[12..16].copy_from_slice(&u32::from(tiles).to_le_bytes());
    h[16..20].copy_from_slice(&(planned.len() as u32).to_le_bytes());
    h[20..24].copy_from_slice(&(shards.len() as u32).to_le_bytes());
    h[24..32].copy_from_slice(&bytes.to_le_bytes());
    h[32..40].copy_from_slice(&HEADER_BYTES.to_le_bytes());
    let data = planned.first().map_or(bytes, |p| p.offset);
    h[40..48].copy_from_slice(&data.to_le_bytes());
    for (index, id) in shards.iter().enumerate() {
        let at = SHARDS_OFFSET + index * SHARD_ID_BYTES;
        h[at..at + SHARD_ID_BYTES].copy_from_slice(id);
    }
    h
}

/// The manifest entry (`arena::Entry`) of one tensor.
pub(crate) fn entry(p: &Planned) -> [u8; ENTRY_BYTES] {
    let mut e = [0u8; ENTRY_BYTES];
    e[..p.name.len()].copy_from_slice(p.name.as_bytes());
    e[64..68].copy_from_slice(&p.kind.to_le_bytes());
    for (index, value) in p.ne.iter().enumerate() {
        e[72 + 8 * index..80 + 8 * index].copy_from_slice(&value.to_le_bytes());
    }
    e[96..104].copy_from_slice(&p.row_bytes.to_le_bytes());
    e[104..112].copy_from_slice(&p.offset.to_le_bytes());
    e[112..120].copy_from_slice(&p.bytes.to_le_bytes());
    e
}

/// A filled, sealed arena, mapped and registered with HIP. It is never unmapped: the engine is never closed.
pub(crate) struct Arena {
    pub ptr: *mut u8,
    pub bytes: u64,
    /// The memfd. It stays open for the life of the arena.
    _fd: libc::c_int,
    /// The call that removes the device's registration of the map, before the drop unmaps it. A test arena has none.
    unregister: Option<RtHostUnregister>,
}

/// The close of the weights (engine-abi "close"): the drop removes the registration, unmaps the arena, and closes the
/// memfd. The kernels must be done with the arena; the worker drops it only when no context is open.
impl Drop for Arena {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if let Some(unregister) = self.unregister {
                // SAFETY: `create` registered this map, and no kernel reads it now.
                unsafe { unregister(self.ptr.cast()) };
            }
            // SAFETY: `create` mapped `bytes` bytes at `ptr`, and nothing holds a slice of them after the drop.
            let code = unsafe { libc::munmap(self.ptr.cast(), self.bytes as usize) };
            assert_eq!(
                code,
                0,
                "munmap of the arena failed: {}",
                std::io::Error::last_os_error()
            );
        }
        if self._fd >= 0 {
            // SAFETY: the arena owns the fd. systemd's store holds its own copy (spec 3.3a).
            unsafe { libc::close(self._fd) };
        }
    }
}

impl Arena {
    /// The first byte of a planned tensor in the arena. A tensor past the arena stops the program.
    pub(crate) fn tensor(&self, p: &Planned) -> *const u8 {
        assert!(
            p.offset + p.bytes <= self.bytes,
            "{} ends past the arena of {} bytes",
            p.name,
            self.bytes
        );
        // SAFETY: the offset is inside the arena (the assert above).
        unsafe { self.ptr.add(p.offset as usize) }
    }
}

/// Copy `src` to `dst` with 16 threads. One thread's copy out of the GGUF map is far below what the page cache and
/// the arena take. The threads make no HIP call.
fn par_memcpy(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "a copy needs two ranges of one size");
    let per = dst.len().div_ceil(16).max(1);
    std::thread::scope(|scope| {
        for (to, from) in dst.chunks_mut(per).zip(src.chunks(per)) {
            scope.spawn(move || to.copy_from_slice(from));
        }
    });
}

fn os_error(what: &str) -> String {
    format!("{what}: {}", std::io::Error::last_os_error())
}

/// Fill a new arena (`engine.hip` `create_arena`, mk 555a2a3). The CPU copies each tensor out of the GGUF map
/// straight into place, or through a pinned scratch buffer and the relayout kernel for a tiled one. The file pages of
/// each copied chunk leave the page cache. Then the header and the manifest go in, and the seals.
///
/// # Safety
///
/// `binding` is the linked engine's, and this runs on the worker thread, which makes every HIP call.
pub(crate) unsafe fn create(
    binding: &Binding,
    model: &Model,
    planned: &[Planned],
    bytes: u64,
    tiles: bool,
) -> Result<Arena, String> {
    let start = std::time::Instant::now();
    let shards = model.shard_paths().map(shard_id).collect::<Result<Vec<_>, _>>()?;
    if shards.len() > MAX_SHARDS {
        return Err(format!("{} shards, the arena records {MAX_SHARDS}", shards.len()));
    }
    let len = usize::try_from(bytes).map_err(|_| format!("an arena of {bytes} bytes is past the address space"))?;
    let fd = unsafe { libc::memfd_create(c"mk-weights".as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if fd < 0 {
        return Err(os_error("memfd_create"));
    }
    if unsafe { libc::ftruncate(fd, bytes as libc::off_t) } != 0 {
        return Err(os_error(&format!("ftruncate the arena to {bytes}")));
    }
    let map = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    if map == libc::MAP_FAILED {
        return Err(os_error("mmap the arena"));
    }
    if unsafe { libc::madvise(map, len, libc::MADV_HUGEPAGE) } != 0 {
        return Err(os_error("madvise(MADV_HUGEPAGE) on the arena"));
    }
    // Lock the arena before the GPU maps it (milestone 30 step 5). Swap could move its pages, and a move remaps them
    // under a running kernel that the driver cannot preempt (`rt.h`): on gfx1151host that gave spin-timeouts and a GPU fault.
    if unsafe { libc::mlock2(map, len, libc::MLOCK_ONFAULT) } != 0 {
        return Err(os_error(&format!(
            "mlock the arena of {bytes} bytes (the unit needs LimitMEMLOCK=infinity)"
        )));
    }
    unsafe { (binding.rt_host_register)(map, bytes) };
    // SAFETY: the map is `len` writable bytes, and nothing else holds it yet.
    let arena = unsafe { std::slice::from_raw_parts_mut(map as *mut u8, len) };
    let scratch_bytes = planned
        .iter()
        .filter(|p| p.tile != Tile::None)
        .map(|p| p.src_bytes)
        .max()
        .unwrap_or(0);
    let scratch = if scratch_bytes > 0 {
        unsafe { (binding.rt_host_malloc)(scratch_bytes) as *mut u8 }
    } else {
        std::ptr::null_mut()
    };
    const CHUNK: u64 = 512 << 20;
    let mut tiled = 0u64;
    for p in planned {
        let src = model.tensor_bytes(p.gguf);
        assert_eq!(
            src.len() as u64,
            p.src_bytes,
            "{}: the plan and the model differ",
            p.name
        );
        // SAFETY: the scratch holds `scratch_bytes` >= `src_bytes` bytes, and the arena range is the tensor's place.
        let to: &mut [u8] = if p.tile == Tile::None {
            &mut arena[p.offset as usize..(p.offset + p.src_bytes) as usize]
        } else {
            unsafe { std::slice::from_raw_parts_mut(scratch, p.src_bytes as usize) }
        };
        let mut at = 0u64;
        while at < p.src_bytes {
            let n = CHUNK.min(p.src_bytes - at);
            let range = at as usize..(at + n) as usize;
            par_memcpy(&mut to[range.clone()], &src[range.clone()]);
            model.drop_cached(src[range].as_ptr(), n)?;
            at += n;
        }
        if p.tile != Tile::None {
            // The relayout synchronizes, so the scratch is free for the next tiled tensor.
            unsafe {
                (binding.relayout)(
                    p.tile as u32,
                    scratch,
                    arena.as_mut_ptr().add(p.offset as usize),
                    p.rows,
                    p.k,
                )
            };
            tiled += p.bytes;
        }
    }
    if !scratch.is_null() {
        unsafe { (binding.rt_host_free)(scratch as *mut c_void) };
    }
    let head = header(planned, bytes, tiles, &shards);
    arena[..head.len()].copy_from_slice(&head);
    for (index, p) in planned.iter().enumerate() {
        let at = HEADER_BYTES as usize + index * ENTRY_BYTES;
        arena[at..at + ENTRY_BYTES].copy_from_slice(&entry(p));
    }
    if unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, SEALS) } != 0 {
        return Err(os_error("sealing the arena"));
    }
    let seals = unsafe { libc::fcntl(fd, libc::F_GET_SEALS) };
    if seals != SEALS {
        return Err(format!("the arena's seals are {seals:#x}, want {SEALS:#x}"));
    }
    eprintln!(
        "engine: loaded {:.2} GiB of weights ({:.2} GiB in read-once tiles) into the arena in {:.1} s",
        bytes as f64 / f64::from(1u32 << 30),
        tiled as f64 / f64::from(1u32 << 30),
        start.elapsed().as_secs_f64()
    );
    Ok(Arena {
        ptr: map as *mut u8,
        bytes,
        _fd: fd,
        unregister: Some(binding.rt_host_unregister),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(name: &str, kind: GgmlType, ne: [u64; 4]) -> TensorInfo {
        let n_elements = ne.iter().product::<u64>();
        TensorInfo {
            name: name.to_owned(),
            kind,
            n_dims: 3,
            ne,
            n_elements,
            n_bytes: n_elements / kind.block_values() * kind.block_bytes(),
            shard: 0,
            offset: 0,
        }
    }

    fn model_like() -> Vec<TensorInfo> {
        vec![
            info("token_embd.weight", GgmlType::Q8_0, [2560, 64, 1, 1]),
            info(PLE_TABLE, GgmlType::Q8_0, [160, 64, 1, 1]),
            info("output.weight", GgmlType::Q6K, [2560, 32, 1, 1]),
            info("blk.0.attn_qkv.weight", GgmlType::Q8_0, [2560, 32, 1, 1]),
            info("blk.0.ffn_gate_exps.weight", GgmlType::Iq3S, [2560, 640, 4, 1]),
            info("blk.0.ffn_down_exps.weight", GgmlType::Iq4Nl, [640, 2560, 4, 1]),
            info("blk.1.ffn_down_exps.weight", GgmlType::Q8_0, [640, 2560, 4, 1]),
            info("blk.0.ssm_a", GgmlType::F32, [48, 1, 1, 1]),
        ]
    }

    #[test]
    fn the_plan_tiles_the_bulky_matrices_and_leaves_the_rest() {
        let (planned, bytes) = plan(&model_like(), true).unwrap();
        let kinds: Vec<(&str, u32, Tile)> = planned.iter().map(|p| (p.name.as_str(), p.kind, p.tile)).collect();
        assert_eq!(
            kinds,
            [
                ("token_embd.weight", 8, Tile::None),
                ("output.weight", Q6_K_TILES, Tile::Q6),
                ("blk.0.attn_qkv.weight", Q8_0_TILES, Tile::Q8),
                ("blk.0.ffn_gate_exps.weight", IQ3_S_TILES, Tile::Iq3s),
                ("blk.0.ffn_down_exps.weight", IQ4_NL_TILES, Tile::Iq4),
                ("blk.1.ffn_down_exps.weight", 8, Tile::None),
                ("blk.0.ssm_a", 0, Tile::None),
            ]
        );
        // IQ3_S tiles take 112 bytes for each 256 values of a row, not 110.
        let gate = &planned[3];
        assert_eq!(gate.bytes, 2560 / 16 * 10 * 1792);
        assert_eq!(gate.row_bytes, 10 * 112);
        assert_eq!(gate.src_bytes, 2560 * 10 * 110);
        // The data starts at 2 MiB, and each tensor at a multiple of 4096 after the one before it.
        assert_eq!(planned[0].offset, 2 << 20);
        for pair in planned.windows(2) {
            assert_eq!(pair[1].offset, round_up(pair[0].offset + pair[0].bytes, 4096));
        }
        let last = planned.last().unwrap();
        assert_eq!(bytes, round_up(last.offset + last.bytes, 4096));
    }

    #[test]
    fn the_plan_without_tiles_keeps_the_gguf_layout() {
        let (planned, _) = plan(&model_like(), false).unwrap();
        assert!(planned.iter().all(|p| p.tile == Tile::None && p.bytes == p.src_bytes));
        assert_eq!(planned[1].kind, 14);
    }

    #[test]
    fn an_expert_down_matrix_of_another_type_and_a_long_name_are_refused() {
        let mut tensors = model_like();
        tensors[5] = info("blk.0.ffn_down_exps.weight", GgmlType::Q6K, [768, 2560, 4, 1]);
        let error = plan(&tensors, true).unwrap_err();
        assert!(error.contains("IQ4_NL or Q8_0"), "{error}");
        let long = info(&"x".repeat(64), GgmlType::F32, [32, 1, 1, 1]);
        let error = plan(&[long], true).unwrap_err();
        assert!(error.contains("name longer"), "{error}");
    }

    #[test]
    fn the_header_and_an_entry_have_mk_layout() {
        let (planned, bytes) = plan(&model_like(), true).unwrap();
        let mut shard = [0u8; SHARD_ID_BYTES];
        shard[0] = 7;
        shard[16] = b'/';
        let h = header(&planned, bytes, true, &[shard]);
        assert_eq!(h.len(), 1072);
        assert_eq!(&h[0..8], b"MKARENA\0");
        assert_eq!(u32::from_le_bytes(h[8..12].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(h[12..16].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(h[16..20].try_into().unwrap()), 7);
        assert_eq!(u32::from_le_bytes(h[20..24].try_into().unwrap()), 1);
        assert_eq!(u64::from_le_bytes(h[24..32].try_into().unwrap()), bytes);
        assert_eq!(u64::from_le_bytes(h[32..40].try_into().unwrap()), 4096);
        assert_eq!(u64::from_le_bytes(h[40..48].try_into().unwrap()), 2 << 20);
        assert_eq!(h[48], 7);
        assert_eq!(h[64], b'/');
        let e = entry(&planned[1]);
        assert_eq!(&e[..13], b"output.weight");
        assert_eq!(e[13], 0);
        assert_eq!(u32::from_le_bytes(e[64..68].try_into().unwrap()), Q6_K_TILES);
        assert_eq!(u64::from_le_bytes(e[72..80].try_into().unwrap()), 2560);
        assert_eq!(u64::from_le_bytes(e[80..88].try_into().unwrap()), 32);
        assert_eq!(u64::from_le_bytes(e[96..104].try_into().unwrap()), planned[1].row_bytes);
        assert_eq!(u64::from_le_bytes(e[104..112].try_into().unwrap()), planned[1].offset);
        assert_eq!(u64::from_le_bytes(e[112..120].try_into().unwrap()), planned[1].bytes);
        assert_eq!(&e[120..128], &[0; 8]);
    }

    #[test]
    #[should_panic(expected = "ends past the arena")]
    fn a_tensor_past_the_arena_stops_the_program() {
        let arena = Arena {
            ptr: std::ptr::null_mut(),
            bytes: 4096,
            _fd: -1,
            unregister: None,
        };
        let (planned, _) = plan(&model_like(), false).unwrap();
        arena.tensor(&planned[0]);
    }

    #[test]
    fn a_parallel_copy_gives_the_same_bytes() {
        let src: Vec<u8> = (0..100_003u32).map(|value| (value % 251) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        par_memcpy(&mut dst, &src);
        assert_eq!(dst, src);
    }
}
