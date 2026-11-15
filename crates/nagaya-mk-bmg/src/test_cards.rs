//! The cards of a card test. A card test opens only the cards that `NAGAYA_TEST_CARDS` names, so it runs beside a
//! nagaya that serves from the other cards (spec 3.1: the nagaya is the only compute client of its cards).

/// The variable that names the cards of a card test: PCI addresses with `,` between them. It has no default.
pub const VARIABLE: &str = "NAGAYA_TEST_CARDS";

/// True for an address of the form `dddd:bb:dd.f` in hex, the form of `ignore_devices` in the bootstrap config.
fn is_pci_address(text: &str) -> bool {
    let b = text.as_bytes();
    b.len() == 12
        && b.iter().enumerate().all(|(i, c)| match i {
            4 | 7 => *c == b':',
            10 => *c == b'.',
            _ => c.is_ascii_hexdigit(),
        })
}

/// The first `count` addresses of the value of the variable. An error says what failed and what to set.
pub fn parse(value: Option<&str>, count: usize) -> Result<Vec<String>, String> {
    let value = value.map(str::trim).unwrap_or("");
    if value.is_empty() {
        return Err(format!(
            "{VARIABLE} is not set, so the card test has no card. Set it to the PCI address of a card that no nagaya \
             serves from, for example {VARIABLE}=0000:03:00.0; put `,` between two addresses"
        ));
    }
    let addresses: Vec<String> = value.split(',').map(|a| a.trim().to_owned()).collect();
    for (i, address) in addresses.iter().enumerate() {
        if !is_pci_address(address) {
            return Err(format!(
                "{VARIABLE} holds `{address}`, which is not a PCI address; write it as `dddd:bb:dd.f` in hex"
            ));
        }
        if addresses[..i].contains(address) {
            return Err(format!(
                "{VARIABLE} names the card {address} two times; name each card one time"
            ));
        }
    }
    if addresses.len() < count {
        return Err(format!(
            "the test needs {count} cards, and {VARIABLE} holds {}; add the address of another free card",
            addresses.len()
        ));
    }
    Ok(addresses[..count].to_vec())
}

/// The first `count` cards of `NAGAYA_TEST_CARDS`. It panics with the cause when the variable does not give them.
pub fn cards(count: usize) -> Vec<String> {
    parse(std::env::var(VARIABLE).ok().as_deref(), count).unwrap_or_else(|error| panic!("{error}"))
}

/// The first card of `NAGAYA_TEST_CARDS`.
pub fn card() -> String {
    cards(1).remove(0)
}
