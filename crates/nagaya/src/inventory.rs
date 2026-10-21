//! The machine's inventory and its probe (spec 2.1).

use nagaya_abi::DeviceId;

use crate::exit::ExitCode;

/// One device of the machine. A CPU is a device when an engine runs work on it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    /// The device index in the machine.
    pub id: DeviceId,
    /// The device name from the probe.
    pub name: String,
    /// The PCI address, for example `0000:03:00.0`. A made-up device has a made-up address.
    pub pci_address: String,
    /// The device memory in bytes.
    pub memory_bytes: u64,
    /// True when device memory is the host's memory (gfx1151host, sm121host).
    pub unified: bool,
}

/// True when `text` is a PCI address in the form `dddd:bb:dd.f`, with hex digits (spec 2.1).
pub fn is_pci_address(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() == 12
        && bytes.iter().enumerate().all(|(index, byte)| match index {
            4 | 7 => *byte == b':',
            10 => *byte == b'.',
            _ => byte.is_ascii_hexdigit(),
        })
}

/// The inventory of one machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inventory {
    /// The devices.
    pub devices: Vec<Device>,
    /// The host memory in bytes.
    pub host_bytes: u64,
}

impl Inventory {
    /// The device with the given id.
    pub fn device(&self, id: DeviceId) -> Option<&Device> {
        self.devices.iter().find(|device| device.id == id)
    }
}

/// An error in the probe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeError {
    /// What the probe could not find.
    pub message: String,
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "the probe failed: {}", self.message)
    }
}

impl std::error::Error for ProbeError {}

impl ProbeError {
    /// The exit code of a probe error at start (spec 10.4).
    ///
    /// The machine is wrong, and a restart cannot change it. Thus the start is refused: exit 3.
    pub fn exit_code(&self) -> ExitCode {
        ExitCode::RefusedStart
    }
}

/// A probe of the machine's devices and host memory.
pub trait Probe {
    /// Read the inventory.
    fn probe(&self) -> Result<Inventory, ProbeError>;
}

/// A probe that returns a fixed result. The real probes (KFD, Xe, NVML) come later.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FakeProbe {
    result: Result<Inventory, ProbeError>,
}

impl FakeProbe {
    /// Make a probe that returns `inventory`.
    pub fn new(inventory: Inventory) -> Self {
        Self { result: Ok(inventory) }
    }

    /// Make a probe that fails.
    pub fn failing(message: &str) -> Self {
        Self {
            result: Err(ProbeError {
                message: message.to_owned(),
            }),
        }
    }
}

impl Probe for FakeProbe {
    fn probe(&self) -> Result<Inventory, ProbeError> {
        self.result.clone()
    }
}
