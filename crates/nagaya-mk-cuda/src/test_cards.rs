//! PCI address selection for GPU tests via `NAGAYA_TEST_CARDS` (spec 3.1).

/// Environment variable naming free GPU PCI addresses separated by `,`. It has no default.
pub const VARIABLE: &str = "NAGAYA_TEST_CARDS";

/// True for a PCI address of the form `dddd:bb:dd.f` in hex.
fn is_pci_address(text: &str) -> bool {
    let b = text.as_bytes();
    b.len() == 12
        && b.iter().enumerate().all(|(i, c)| match i {
            4 | 7 => *c == b':',
            10 => *c == b'.',
            _ => c.is_ascii_hexdigit(),
        })
}

/// Parse the first `count` PCI addresses from `value`. An error names what failed and what to set.
pub fn parse(value: Option<&str>, count: usize) -> Result<Vec<String>, String> {
    let value = value.map(str::trim).unwrap_or("");
    if value.is_empty() {
        return Err(format!(
            "{VARIABLE} is not set, so the GPU test has no device. Set it to the PCI address of a free GPU, for \
             example {VARIABLE}=0000:0a:00.0; put `,` between multiple addresses"
        ));
    }
    let addresses: Vec<String> = value.split(',').map(|a| a.trim().to_ascii_lowercase()).collect();
    for (i, address) in addresses.iter().enumerate() {
        if !is_pci_address(address) {
            return Err(format!(
                "{VARIABLE} holds `{address}`, which is not a PCI address; write it as `dddd:bb:dd.f` in hex"
            ));
        }
        if addresses[..i].contains(address) {
            return Err(format!(
                "{VARIABLE} names the GPU {address} two times; name each GPU one time"
            ));
        }
    }
    if addresses.len() < count {
        return Err(format!(
            "the test needs {count} GPUs, and {VARIABLE} holds {}; add the address of another free GPU",
            addresses.len()
        ));
    }
    Ok(addresses[..count].to_vec())
}

/// Return the first `count` GPUs named in `NAGAYA_TEST_CARDS`, or panic with the cause.
pub fn cards(count: usize) -> Vec<String> {
    parse(std::env::var(VARIABLE).ok().as_deref(), count).unwrap_or_else(|error| panic!("{error}"))
}

/// Return the first GPU named in `NAGAYA_TEST_CARDS`.
pub fn card() -> String {
    cards(1).remove(0)
}
