//! The memory of this process on one `xe` card, from the DRM keys of `/proc/self/fdinfo` (spec 9.6). An open reads it
//! before and after its allocations, so it can see bytes that the driver put in system memory. It needs no card, so
//! the host tests cover the parse.

use std::collections::BTreeSet;

/// The bytes of this process on one card, summed once per DRM client.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CardMemory {
    /// `drm-total-vram0`: the bytes of the buffers that the driver placed in the memory of the card.
    pub vram: u64,
    /// `drm-resident-vram0`: the part of `vram` that is in the memory of the card now.
    pub resident: u64,
}

/// The bytes of a DRM value such as `31224552 KiB`.
fn bytes_of(key: &str, value: &str) -> Result<u64, String> {
    let mut parts = value.split_whitespace();
    let number = parts.next().and_then(|n| n.parse::<u64>().ok());
    let scale = match parts.next() {
        None | Some("B") => Some(1),
        Some("KiB") => Some(1 << 10),
        Some("MiB") => Some(1 << 20),
        Some("GiB") => Some(1 << 30),
        Some(_) => None,
    };
    match (number, scale) {
        (Some(n), Some(s)) => Ok(n * s),
        _ => Err(format!("the fdinfo key {key} holds `{value}`, not a count of bytes")),
    }
}

/// The memory of the card at the PCI address `pci` in the fdinfo texts `files`. A file of another driver or another
/// card adds nothing. An `xe` file with no `drm-pdev` is an error, because the sum must not miss a card.
pub fn parse<'a>(files: impl IntoIterator<Item = &'a str>, pci: &str) -> Result<CardMemory, String> {
    let mut clients = BTreeSet::new();
    let mut out = CardMemory::default();
    for text in files {
        let value = |key: &str| {
            text.lines().find_map(|line| {
                line.split_once(':')
                    .filter(|(k, _)| k.trim() == key)
                    .map(|(_, v)| v.trim())
            })
        };
        if value("drm-driver") != Some("xe") {
            continue;
        }
        let pdev = value("drm-pdev").ok_or_else(|| {
            "an xe fd of this process names no drm-pdev, so its card is not known. The kernel is too old for the \
             memory check of an open"
                .to_owned()
        })?;
        if pdev != pci {
            continue;
        }
        if let Some(id) = value("drm-client-id")
            && !clients.insert(id.to_owned())
        {
            continue;
        }
        for (key, field) in [
            ("drm-total-vram0", &mut out.vram),
            ("drm-resident-vram0", &mut out.resident),
        ] {
            if let Some(v) = value(key) {
                *field += bytes_of(key, v)?;
            }
        }
    }
    Ok(out)
}

/// The memory of this process on the card at `pci`, from `/proc/self/fdinfo`. An fd that closes during the walk adds
/// nothing.
pub fn card_memory(pci: &str) -> Result<CardMemory, String> {
    let dir = "/proc/self/fdinfo";
    let entries = std::fs::read_dir(dir).map_err(|e| format!("cannot read {dir}: {e}"))?;
    let mut texts = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot read {dir}: {e}"))?;
        match std::fs::read_to_string(entry.path()) {
            Ok(text) => texts.push(text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("cannot read {}: {e}", entry.path().display())),
        }
    }
    parse(texts.iter().map(String::as_str), pci)
}

/// Check that an open of `planned` device bytes on the card at `pci` stayed in the memory of the card. `before` and
/// `after` are the memory of the process before and after the open. The driver can put a buffer that does not fit in
/// system memory, and then each pass runs 5 to 6 times slower (engine-abi "Inkling 2", "The cost of a slot").
pub fn check_open(pci: &str, planned: u64, before: CardMemory, after: CardMemory) -> Result<(), String> {
    let placed = after.vram.saturating_sub(before.vram);
    if placed < planned {
        return Err(format!(
            "the card at {pci} did not hold the open: the engine asked for {planned} device bytes, and the driver put \
             only {placed} of them in the memory of the card. The rest went to system memory, where each pass runs 5 \
             to 6 times slower. Give fewer or shorter slots, or unload another engine of the card"
        ));
    }
    if after.resident < after.vram {
        return Err(format!(
            "the card at {pci} did not hold the open: {} bytes of the buffers of this process on it are in system \
             memory now. Give fewer or shorter slots, or unload another engine of the card",
            after.vram - after.resident
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CARD0: &str = "drm-driver:\txe\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t7\ndrm-total-system:\t197040 KiB\n\
                         drm-total-vram0:\t30855348 KiB\ndrm-resident-vram0:\t30855348 KiB\n";
    const CARD1: &str = "drm-driver:\txe\ndrm-pdev:\t0000:08:00.0\ndrm-client-id:\t8\n\
                         drm-total-vram0:\t31224552 KiB\ndrm-resident-vram0:\t31224552 KiB\n";

    #[test]
    fn the_parse_sums_the_clients_of_one_card_once() {
        let other = "drm-driver:\tamdgpu\ndrm-pdev:\t0000:03:00.0\ndrm-total-vram:\t5 KiB\n";
        let same_client = CARD0;
        let second = "drm-driver:\txe\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t9\n\
                      drm-total-vram0:\t4 MiB\ndrm-resident-vram0:\t3 MiB\n";
        let memory = parse([CARD0, CARD1, other, same_client, second], "0000:03:00.0").unwrap();
        assert_eq!(
            memory,
            CardMemory {
                vram: 30_855_348 * 1024 + (4 << 20),
                resident: 30_855_348 * 1024 + (3 << 20),
            }
        );
        assert_eq!(parse([CARD1], "0000:03:00.0").unwrap(), CardMemory::default());
        let e = parse(["drm-driver:\txe\n"], "0000:03:00.0").unwrap_err();
        assert!(e.contains("no drm-pdev"), "{e}");
        let e = parse(
            ["drm-driver:\txe\ndrm-pdev:\t0000:03:00.0\ndrm-total-vram0:\tmany\n"],
            "0000:03:00.0",
        )
        .unwrap_err();
        assert!(e.contains("holds `many`"), "{e}");
    }

    #[test]
    fn an_open_that_the_card_does_not_hold_is_refused() {
        let before = CardMemory {
            vram: 1000,
            resident: 1000,
        };
        let held = CardMemory {
            vram: 5000,
            resident: 5000,
        };
        assert_eq!(check_open("0000:03:00.0", 4000, before, held), Ok(()));
        let e = check_open("0000:03:00.0", 4001, before, held).unwrap_err();
        assert!(
            e.contains("asked for 4001 device bytes") && e.contains("only 4000"),
            "{e}"
        );
        let evicted = CardMemory {
            vram: 5000,
            resident: 4900,
        };
        let e = check_open("0000:03:00.0", 4000, before, evicted).unwrap_err();
        assert!(e.contains("100 bytes") && e.contains("in system memory now"), "{e}");
    }
}
