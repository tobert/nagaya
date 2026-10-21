//! The file probe: it reads the machine's inventory from files under a root path (spec 2.1).

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::inventory::{Device, Inventory, Probe, ProbeError};

/// The vendor of a GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vendor {
    /// AMD, found through the KFD topology.
    Amd,
    /// Intel, found through the Xe driver.
    Intel,
    /// NVIDIA, found through `/proc/driver/nvidia`.
    Nvidia,
}

/// A device and what the probe knows about its place on the bus.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceFacts {
    /// The device as the ledger sees it.
    pub device: Device,
    /// The vendor.
    pub vendor: Vendor,
    /// The PCI address, for example `0000:c2:00.0`.
    pub pci_address: String,
    /// The PCI vendor and device ids in lower case hex, for example `1002:1586`.
    pub pci_id: String,
}

/// One NVMe namespace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nvme {
    /// The block device name, for example `nvme0n1`.
    pub name: String,
    /// The model string, with no padding.
    pub model: String,
    /// The size in bytes.
    pub bytes: u64,
}

/// One physical network interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nic {
    /// The interface name.
    pub name: String,
    /// The kernel driver.
    pub driver: String,
    /// The PCI vendor and device ids in lower case hex.
    pub pci_id: String,
    /// The link speed in megabits per second, or `None` when the link is down.
    pub speed_mbps: Option<u32>,
}

/// Everything that the file probe finds on one machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Machine {
    /// The GPU devices.
    pub devices: Vec<DeviceFacts>,
    /// The host memory in bytes.
    pub host_bytes: u64,
    /// The ids of the online CPUs.
    pub cpus: Vec<u32>,
    /// The NVMe namespaces.
    pub nvme: Vec<Nvme>,
    /// The physical network interfaces.
    pub nics: Vec<Nic>,
}

impl Machine {
    /// The part of the machine that the ledger and placement use.
    pub fn inventory(&self) -> Inventory {
        Inventory {
            devices: self.devices.iter().map(|facts| facts.device.clone()).collect(),
            host_bytes: self.host_bytes,
        }
    }
}

/// A probe that reads files under a root path.
///
/// The probe opens regular files only. It never opens a device node or a GPU runtime (spec 2.1).
#[derive(Clone, Debug)]
pub struct FileProbe {
    root: PathBuf,
}

impl FileProbe {
    /// Make a probe for the tree under `root`. The machine's own root is `/`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The root path of the tree.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Read the machine.
    ///
    /// A file that the probe needs but cannot read is an error. A directory that lists devices of one kind can be
    /// absent: then the machine has no device of that kind.
    pub fn machine(&self) -> Result<Machine, ProbeError> {
        let host_bytes = self.host_bytes()?;
        let mut devices = self.amd_devices()?;
        devices.extend(self.xe_devices()?);
        devices.extend(self.nvidia_devices(host_bytes)?);
        for (id, facts) in devices.iter_mut().enumerate() {
            facts.device.id = u32::try_from(id).expect("a machine has fewer than 2^32 devices");
        }
        Ok(Machine {
            devices,
            host_bytes,
            cpus: self.online_cpus()?,
            nvme: self.nvme()?,
            nics: self.nics()?,
        })
    }

    fn path(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }

    fn read(&self, relative: &str) -> Result<String, ProbeError> {
        let path = self.path(relative);
        fs::read_to_string(&path).map_err(|error| {
            fail(format!(
                "cannot read `{}`: {error}. Check that the file exists on this machine and that its driver is loaded",
                path.display()
            ))
        })
    }

    /// The names in a directory, sorted. `None` means that the directory does not exist.
    fn list_optional(&self, relative: &str) -> Result<Option<Vec<String>>, ProbeError> {
        let path = self.path(relative);
        let entries = match fs::read_dir(&path) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(fail(format!("cannot list `{}`: {error}", path.display()))),
        };
        let mut names = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| fail(format!("cannot list `{}`: {error}", path.display())))?;
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        names.sort();
        Ok(Some(names))
    }

    fn list(&self, relative: &str) -> Result<Vec<String>, ProbeError> {
        self.list_optional(relative)?.ok_or_else(|| {
            fail(format!(
                "cannot list `{}`: the directory does not exist. Check that this is a Linux machine",
                self.path(relative).display()
            ))
        })
    }

    fn decimal(&self, relative: &str) -> Result<u64, ProbeError> {
        let text = self.read(relative)?;
        text.trim()
            .parse()
            .map_err(|_| malformed(&self.path(relative), "a decimal number", &text))
    }

    fn hex(&self, relative: &str) -> Result<u64, ProbeError> {
        let text = self.read(relative)?;
        parse_hex(text.trim()).ok_or_else(|| malformed(&self.path(relative), "a hex number", &text))
    }

    fn pci_id(&self, directory: &str) -> Result<(u64, u64), ProbeError> {
        Ok((
            self.hex(&format!("{directory}/vendor"))?,
            self.hex(&format!("{directory}/device"))?,
        ))
    }

    /// The `MemTotal` line of `/proc/meminfo`, in bytes.
    fn host_bytes(&self) -> Result<u64, ProbeError> {
        let relative = "proc/meminfo";
        let text = self.read(relative)?;
        let line = text
            .lines()
            .find(|line| line.starts_with("MemTotal:"))
            .ok_or_else(|| fail(format!("`{}` has no MemTotal line", self.path(relative).display())))?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        let kilobytes = match fields.as_slice() {
            ["MemTotal:", number, "kB"] => number.parse::<u64>().ok(),
            _ => None,
        };
        kilobytes
            .and_then(|number| number.checked_mul(1024))
            .ok_or_else(|| malformed(&self.path(relative), "a MemTotal line like `MemTotal: 100 kB`", line))
    }

    /// The online CPUs, from the list in `/sys/devices/system/cpu/online`.
    ///
    /// The list keeps the order of the file. The CPU sets (spec 3.2a) check the order.
    pub fn online_cpus(&self) -> Result<Vec<u32>, ProbeError> {
        let relative = "sys/devices/system/cpu/online";
        let text = self.read(relative)?;
        parse_cpu_list(text.trim()).ok_or_else(|| malformed(&self.path(relative), "a CPU list like `0-7,9`", &text))
    }

    fn nvme(&self) -> Result<Vec<Nvme>, ProbeError> {
        let mut found = Vec::new();
        for name in self.list("sys/block")? {
            let Some(order) = nvme_order(&name) else { continue };
            let size = format!("sys/block/{name}/size");
            let bytes = self
                .decimal(&size)?
                .checked_mul(SECTOR_BYTES)
                .ok_or_else(|| malformed(&self.path(&size), "a sector count that fits 64 bits", ""))?;
            let model = self.read(&format!("sys/block/{name}/device/model"))?.trim().to_owned();
            found.push((order, Nvme { name, model, bytes }));
        }
        found.sort_by_key(|(order, _)| *order);
        Ok(found.into_iter().map(|(_, nvme)| nvme).collect())
    }

    /// The network interfaces that sit on a PCI device. Virtual interfaces have no `device` directory.
    fn nics(&self) -> Result<Vec<Nic>, ProbeError> {
        let mut nics = Vec::new();
        for name in self.list("sys/class/net")? {
            let device = format!("sys/class/net/{name}/device");
            if !self.path(&device).is_dir() {
                continue;
            }
            let driver = self.uevent(&device)?.driver()?;
            let (vendor, id) = self.pci_id(&device)?;
            let speed_mbps = self.link_speed(&format!("sys/class/net/{name}/speed"))?;
            nics.push(Nic {
                name,
                driver,
                pci_id: format!("{vendor:04x}:{id:04x}"),
                speed_mbps,
            });
        }
        Ok(nics)
    }

    /// The link speed. A down link gives `-1` or the error `EINVAL`, and then the speed is `None`.
    fn link_speed(&self, relative: &str) -> Result<Option<u32>, ProbeError> {
        let path = self.path(relative);
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::InvalidInput => return Ok(None),
            Err(error) => return Err(fail(format!("cannot read `{}`: {error}", path.display()))),
        };
        match text.trim() {
            "-1" => Ok(None),
            number => number
                .parse()
                .map(Some)
                .map_err(|_| malformed(&path, "a speed in megabits per second", &text)),
        }
    }

    fn uevent(&self, device: &str) -> Result<Uevent, ProbeError> {
        let relative = format!("{device}/uevent");
        let text = self.read(&relative)?;
        let pairs = text
            .lines()
            .filter_map(|line| line.split_once('='))
            .map(|(key, value)| (key.to_owned(), value.to_owned()))
            .collect();
        Ok(Uevent {
            path: self.path(&relative),
            pairs,
        })
    }

    /// The AMD GPUs, from the KFD topology. A node with `gpu_id` 0 is a CPU node.
    fn amd_devices(&self) -> Result<Vec<DeviceFacts>, ProbeError> {
        let nodes_dir = "sys/class/kfd/kfd/topology/nodes";
        let Some(names) = self.list_optional(nodes_dir)? else {
            return Ok(Vec::new());
        };
        let mut nodes = Vec::new();
        for name in names {
            let index: u32 = name
                .parse()
                .map_err(|_| malformed(&self.path(nodes_dir), "node directories named with numbers", &name))?;
            nodes.push((index, name));
        }
        nodes.sort();
        let mut devices = Vec::new();
        for (_, name) in nodes {
            let node = format!("{nodes_dir}/{name}");
            if self.decimal(&format!("{node}/gpu_id"))? == 0 {
                continue;
            }
            devices.push(self.amd_device(&node)?);
        }
        Ok(devices)
    }

    fn amd_device(&self, node: &str) -> Result<DeviceFacts, ProbeError> {
        let properties = Numbers::read(self, &format!("{node}/properties"))?;
        let version = properties.get("gfx_target_version")?;
        let name = format!("gfx{}{:x}{:x}", version / 10_000, version / 100 % 100, version % 100);
        let location = properties.get("location_id")?;
        let pci_address = format!(
            "{:04x}:{:02x}:{:02x}.{}",
            properties.get("domain")?,
            location >> 8 & 0xff,
            location >> 3 & 0x1f,
            location & 7
        );
        let pci_id = format!(
            "{:04x}:{:04x}",
            properties.get("vendor_id")?,
            properties.get("device_id")?
        );

        let banks_dir = format!("{node}/mem_banks");
        let mut bank_bytes: u64 = 0;
        for bank in self.list(&banks_dir)? {
            let bank = Numbers::read(self, &format!("{banks_dir}/{bank}/properties"))?;
            // Heap type 1 is public framebuffer memory. Heap type 2 is private framebuffer memory.
            if matches!(bank.get("heap_type")?, 1 | 2) {
                bank_bytes += bank.get("size_in_bytes")?;
            }
        }
        let pci = format!("sys/bus/pci/devices/{pci_address}");
        let gtt = self.decimal(&format!("{pci}/mem_info_gtt_total"))?;
        let vram = self.decimal(&format!("{pci}/mem_info_vram_total"))?;
        let unified = if bank_bytes > 0 && bank_bytes == gtt {
            true
        } else if bank_bytes > 0 && bank_bytes == vram {
            false
        } else {
            return Err(fail(format!(
                "{name} at {pci_address}: the KFD framebuffer memory is {bank_bytes} bytes. It matches neither \
                 mem_info_gtt_total ({gtt}) nor mem_info_vram_total ({vram}) under `{}`. \
                 The probe cannot tell if this memory is unified",
                self.path(&pci).display()
            )));
        };
        Ok(DeviceFacts {
            device: Device {
                id: 0,
                name,
                memory_bytes: bank_bytes,
                unified,
            },
            vendor: Vendor::Amd,
            pci_address,
            pci_id,
        })
    }

    /// The Intel GPUs that the Xe driver owns, from the DRM cards. Other drivers are not devices of the nagaya.
    fn xe_devices(&self) -> Result<Vec<DeviceFacts>, ProbeError> {
        let Some(names) = self.list_optional("sys/class/drm")? else {
            return Ok(Vec::new());
        };
        let mut cards: Vec<(u32, String)> = names
            .into_iter()
            .filter_map(|name| Some((name.strip_prefix("card")?.parse().ok()?, name)))
            .collect();
        cards.sort();
        let mut devices = Vec::new();
        for (_, card) in cards {
            let device = format!("sys/class/drm/{card}/device");
            let uevent = self.uevent(&device)?;
            if uevent.driver()? != "xe" {
                continue;
            }
            let pci_address = uevent.get("PCI_SLOT_NAME")?.to_owned();
            let (vendor, id) = self.pci_id(&device)?;
            let resource = format!("{device}/resource");
            let memory_bytes = self.largest_memory_bar(&resource)?;
            if memory_bytes < MIN_DEVICE_BAR_BYTES {
                return Err(fail(format!(
                    "Xe card {pci_address}: the largest memory BAR in `{}` holds {memory_bytes} bytes. \
                     The probe reads the device memory from this BAR, so turn on resizable BAR in the firmware",
                    self.path(&resource).display()
                )));
            }
            devices.push(DeviceFacts {
                device: Device {
                    id: 0,
                    name: format!("xe-{id:04x}"),
                    memory_bytes,
                    unified: false,
                },
                vendor: Vendor::Intel,
                pci_address,
                pci_id: format!("{vendor:04x}:{id:04x}"),
            });
        }
        Ok(devices)
    }

    /// The size of the largest prefetchable memory range in a PCI `resource` file.
    fn largest_memory_bar(&self, relative: &str) -> Result<u64, ProbeError> {
        let text = self.read(relative)?;
        let mut largest = 0;
        for line in text.lines() {
            let fields: Vec<Option<u64>> = line.split_whitespace().map(parse_hex).collect();
            let [Some(start), Some(end), Some(flags)] = fields[..] else {
                return Err(malformed(&self.path(relative), "lines of three hex numbers", line));
            };
            if start == 0 && end == 0 {
                continue;
            }
            if flags & IORESOURCE_MEM != 0 && flags & IORESOURCE_PREFETCH != 0 && end >= start {
                largest = largest.max(end - start + 1);
            }
        }
        Ok(largest)
    }

    /// The NVIDIA GPUs, from `/proc/driver/nvidia/gpus`.
    fn nvidia_devices(&self, host_bytes: u64) -> Result<Vec<DeviceFacts>, ProbeError> {
        let Some(addresses) = self.list_optional("proc/driver/nvidia/gpus")? else {
            return Ok(Vec::new());
        };
        let mut devices = Vec::new();
        for address in addresses {
            let information = format!("proc/driver/nvidia/gpus/{address}/information");
            let text = self.read(&information)?;
            let model = text
                .lines()
                .find_map(|line| line.strip_prefix("Model:"))
                .map(|model| model.trim().to_owned())
                .ok_or_else(|| fail(format!("`{}` has no Model line", self.path(&information).display())))?;
            if !UNIFIED_NVIDIA_MODELS.contains(&model.as_str()) {
                return Err(fail(format!(
                    "NVIDIA GPU {address} has the model `{model}`. The file probe does not know how much memory \
                     this model has: it needs NVML. Add the model to UNIFIED_NVIDIA_MODELS if its memory is unified"
                )));
            }
            let address = address.to_lowercase();
            let (vendor, id) = self.pci_id(&format!("sys/bus/pci/devices/{address}"))?;
            devices.push(DeviceFacts {
                device: Device {
                    id: 0,
                    name: model,
                    memory_bytes: host_bytes,
                    unified: true,
                },
                vendor: Vendor::Nvidia,
                pci_address: address,
                pci_id: format!("{vendor:04x}:{id:04x}"),
            });
        }
        Ok(devices)
    }
}

impl Probe for FileProbe {
    fn probe(&self) -> Result<Inventory, ProbeError> {
        Ok(self.machine()?.inventory())
    }
}

/// The size of a sector in `/sys/block/*/size`. The kernel counts it in units of 512 bytes, whatever the device.
const SECTOR_BYTES: u64 = 512;

/// A memory BAR smaller than this is a window and not the device memory.
const MIN_DEVICE_BAR_BYTES: u64 = 1 << 30;

/// The flag of a memory range in a PCI `resource` file (`IORESOURCE_MEM` in the kernel).
const IORESOURCE_MEM: u64 = 0x200;

/// The flag of a prefetchable range in a PCI `resource` file (`IORESOURCE_PREFETCH` in the kernel).
const IORESOURCE_PREFETCH: u64 = 0x2000;

/// The NVIDIA models whose memory is the host memory. The file probe cannot read the memory of any other model.
const UNIFIED_NVIDIA_MODELS: [&str; 1] = ["NVIDIA GB10"];

/// The `KEY=value` lines of a `uevent` file.
struct Uevent {
    path: PathBuf,
    pairs: Vec<(String, String)>,
}

impl Uevent {
    fn get(&self, key: &str) -> Result<&str, ProbeError> {
        self.pairs
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.as_str())
            .ok_or_else(|| fail(format!("`{}` has no {key} line", self.path.display())))
    }

    fn driver(&self) -> Result<String, ProbeError> {
        self.get("DRIVER").map(str::to_owned)
    }
}

/// The `key number` lines of a KFD file.
struct Numbers {
    path: PathBuf,
    values: HashMap<String, u64>,
}

impl Numbers {
    fn read(probe: &FileProbe, relative: &str) -> Result<Self, ProbeError> {
        let text = probe.read(relative)?;
        let path = probe.path(relative);
        let mut values = HashMap::new();
        for line in text.lines() {
            let parsed = line
                .split_once(' ')
                .and_then(|(key, value)| Some((key, value.trim().parse::<u64>().ok()?)));
            let (key, value) = parsed.ok_or_else(|| malformed(&path, "lines like `key 123`", line))?;
            values.insert(key.to_owned(), value);
        }
        Ok(Self { path, values })
    }

    fn get(&self, key: &str) -> Result<u64, ProbeError> {
        self.values
            .get(key)
            .copied()
            .ok_or_else(|| fail(format!("`{}` has no `{key}` line", self.path.display())))
    }
}

fn fail(message: String) -> ProbeError {
    ProbeError { message }
}

fn malformed(path: &Path, expected: &str, found: &str) -> ProbeError {
    fail(format!(
        "`{}` is malformed: expected {expected}, found `{}`",
        path.display(),
        found.trim()
    ))
}

fn parse_hex(text: &str) -> Option<u64> {
    u64::from_str_radix(text.strip_prefix("0x")?, 16).ok()
}

/// Parse a kernel CPU list like `0-3,8,10-11`.
fn parse_cpu_list(text: &str) -> Option<Vec<u32>> {
    let mut cpus = Vec::new();
    for part in text.split(',') {
        match part.split_once('-') {
            Some((first, last)) => {
                let (first, last): (u32, u32) = (first.parse().ok()?, last.parse().ok()?);
                if first > last {
                    return None;
                }
                cpus.extend(first..=last);
            }
            None => cpus.push(part.parse().ok()?),
        }
    }
    Some(cpus)
}

/// The sort key of a name like `nvme1n2`: the controller number, then the namespace number.
fn nvme_order(name: &str) -> Option<(u32, u32)> {
    let (controller, namespace) = name.strip_prefix("nvme")?.split_once('n')?;
    Some((controller.parse().ok()?, namespace.parse().ok()?))
}
