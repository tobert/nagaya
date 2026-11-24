//! The bytes that an open may take from the memory of a GPU. The rule needs no GPU.
//!
//! On a GPU with unified memory (the GB10), `cuMemGetInfo` gives a free count that leaves out the page cache. The kernel
//! reclaims the page cache for a GPU allocation, so that count is too low. The bytes that an open may take are
//! `MemAvailable` of the host, capped at the total of the driver. On a discrete GPU they are the free bytes of the driver.

/// Which count gave the available bytes. A message names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Count {
    /// `MemAvailable` of `/proc/meminfo`, capped at the total of the driver.
    MemAvailable,
    /// The free bytes that `cuMemGetInfo` gives.
    GpuFree,
}

impl Count {
    /// The words of a message for this count.
    pub fn label(self) -> &'static str {
        match self {
            Count::MemAvailable => "MemAvailable of the host",
            Count::GpuFree => "free bytes of the GPU",
        }
    }
}

/// The bytes that an open may take, and the count that gave them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Available {
    pub bytes: u64,
    pub count: Count,
}

/// The `MemAvailable` line of a `/proc/meminfo` text, in bytes. A missing line or a bad number is an error.
pub fn parse_mem_available(text: &str) -> Result<u64, String> {
    let line = text
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .ok_or_else(|| {
            "the text of /proc/meminfo has no `MemAvailable` line. The kernel is too old for this engine: use a kernel \
             of version 3.14 or later"
                .to_string()
        })?;
    let mut words = line.split_whitespace();
    let number = words.next().unwrap_or("");
    let kib: u64 = number.parse().map_err(|_| {
        format!("the `MemAvailable` line of /proc/meminfo has the bad number `{number}`. Report this line: `{line}`")
    })?;
    match (words.next(), words.next()) {
        (Some("kB"), None) => kib.checked_mul(1024).ok_or_else(|| {
            format!("the `MemAvailable` line of /proc/meminfo is too large: `{line}`. Report this line")
        }),
        _ => Err(format!(
            "the `MemAvailable` line of /proc/meminfo has no unit `kB` after the number: `{line}`. Report this line"
        )),
    }
}

/// The bytes that an open may take. `integrated` is the device attribute `CU_DEVICE_ATTRIBUTE_INTEGRATED`. `free` and
/// `total` come from `cuMemGetInfo`. `meminfo` is the text of `/proc/meminfo`, and a discrete GPU does not read it.
pub fn available_bytes(integrated: bool, free: u64, total: u64, meminfo: &str) -> Result<Available, String> {
    if !integrated {
        return Ok(Available {
            bytes: free,
            count: Count::GpuFree,
        });
    }
    Ok(Available {
        bytes: parse_mem_available(meminfo)?.min(total),
        count: Count::MemAvailable,
    })
}
