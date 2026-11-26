//! The instruction set of the kernels, chosen at run time: SSE (each x86_64 CPU), AVX2, or AVX-512 (AVX-512F). Each
//! level gives the same bits for each output (spec 8.5): the kernels of each level do the same IEEE operations on each
//! value, in the same order, with no fused multiply-add. Only the count of values in one register changes.
//!
//! A kernel reads the level on the thread that calls it: the best level of the CPU, capped by `with_level`. Tests and
//! the bench use `with_level` to run each level and compare the bits.

use std::cell::Cell;
use std::sync::OnceLock;

/// An instruction set of the kernels, from the least to the best.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// SSE2 on x86_64 (each x86_64 CPU has it); plain arrays on other machines.
    Sse,
    /// AVX2: 8 f32 in one register.
    Avx2,
    /// AVX-512F: 16 f32 in one register.
    Avx512,
}

impl Level {
    pub const ALL: [Level; 3] = [Level::Sse, Level::Avx2, Level::Avx512];

    pub fn name(self) -> &'static str {
        match self {
            Level::Sse => "sse",
            Level::Avx2 => "avx2",
            Level::Avx512 => "avx512",
        }
    }
}

/// The best level of this CPU.
pub fn detected() -> Level {
    static D: OnceLock<Level> = OnceLock::new();
    *D.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx2") {
                return Level::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Level::Avx2;
            }
        }
        Level::Sse
    })
}

thread_local! {
    static CAP: Cell<Level> = const { Cell::new(Level::Avx512) };
}

/// The level of the kernels that this thread calls.
pub fn level() -> Level {
    detected().min(CAP.with(Cell::get))
}

/// Runs `f` with the kernels of this thread capped at `cap`. Panics when the CPU does not have `cap`: a test that
/// asks for a level must know that it runs it.
pub fn with_level<R>(cap: Level, f: impl FnOnce() -> R) -> R {
    assert!(
        cap <= detected(),
        "the CPU does not have {}; its best level is {}",
        cap.name(),
        detected().name()
    );
    struct Restore(Level);
    impl Drop for Restore {
        fn drop(&mut self) {
            CAP.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(CAP.with(|c| c.replace(cap)));
    f()
}

/// `leveled! { fn name(args) -> R = body; }` makes `fn name(level: Level, args) -> R`, which calls `body::<Sse>`,
/// `body::<Avx2>` or `body::<Avx512>` for `level`. The last two run in functions compiled for their instructions
/// (`#[target_feature]`). `body` must be `#[inline(always)]`, and so must each function that it calls with vectors,
/// so that all its code is compiled for the instructions of the level.
macro_rules! leveled {
    ($(#[$m:meta])* fn $name:ident($($a:ident: $t:ty),* $(,)?) $(-> $r:ty)? = $body:ident;) => {
        $(#[$m])*
        fn $name(level: $crate::isa::Level, $($a: $t),*) $(-> $r)? {
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2")]
            unsafe fn avx2($($a: $t),*) $(-> $r)? {
                $body::<$crate::lanes::Avx2>($($a),*)
            }
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx512f")]
            unsafe fn avx512($($a: $t),*) $(-> $r)? {
                $body::<$crate::lanes::Avx512>($($a),*)
            }
            match level {
                $crate::isa::Level::Sse => $body::<$crate::lanes::Sse>($($a),*),
                // SAFETY: `isa::level()` is at most `isa::detected()`, so the CPU has the instructions.
                #[cfg(target_arch = "x86_64")]
                $crate::isa::Level::Avx2 => unsafe { avx2($($a),*) },
                // SAFETY: as above.
                #[cfg(target_arch = "x86_64")]
                $crate::isa::Level::Avx512 => unsafe { avx512($($a),*) },
                #[cfg(not(target_arch = "x86_64"))]
                _ => unreachable!("only SSE is detected on this machine"),
            }
        }
    };
}
pub(crate) use leveled;
