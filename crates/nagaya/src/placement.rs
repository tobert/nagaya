//! Device groups and the device check (spec 3.2).

use nagaya_abi::DeviceId;

use crate::admission::AdmittedEngine;
use crate::inventory::Inventory;
use crate::profile::PlacementProfile;

/// A set of devices that one or more engines use. One thread owns the group (spec 3.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceGroup {
    /// The devices in the group, in the order of the first engine that names them.
    pub devices: Vec<DeviceId>,
    /// The engines in the group, by their index in the profile.
    pub engines: Vec<usize>,
}

/// Group the profile's engines by their devices. Two engines that use one device share a group (spec 3.2).
///
/// The overlap is transitive: an engine that uses a device of two groups joins them into one. The groups keep the
/// order of their first engine. The devices of a group are the union, in the order of first appearance.
pub fn device_groups(profile: &PlacementProfile) -> Vec<DeviceGroup> {
    let mut groups: Vec<DeviceGroup> = Vec::new();
    for (index, engine) in profile.engines.iter().enumerate() {
        let mut merged = DeviceGroup {
            devices: Vec::new(),
            engines: Vec::new(),
        };
        let mut rest = Vec::with_capacity(groups.len());
        let mut slot = None;
        for group in groups {
            if group.devices.iter().any(|device| engine.devices.contains(device)) {
                slot.get_or_insert(rest.len());
                merge_into(&mut merged, group);
            } else {
                rest.push(group);
            }
        }
        merge_into(
            &mut merged,
            DeviceGroup {
                devices: engine.devices.clone(),
                engines: vec![index],
            },
        );
        rest.insert(slot.unwrap_or(rest.len()), merged);
        groups = rest;
    }
    groups
}

fn merge_into(target: &mut DeviceGroup, other: DeviceGroup) {
    for device in other.devices {
        if !target.devices.contains(&device) {
            target.devices.push(device);
        }
    }
    target.engines.extend(other.engines);
    target.engines.sort_unstable();
}

/// Why an engine's devices cannot open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceError {
    /// The inventory has no device with this id.
    Unknown {
        /// The engine name.
        engine: String,
        /// The device id.
        device: DeviceId,
    },
    /// The plan's `devices` are not the group indexes 0 to n - 1 in order, for the n devices of the profile (spec 2.4).
    GroupIndexes {
        /// The engine name.
        engine: String,
        /// The devices of the plan.
        planned: Vec<DeviceId>,
        /// The count of devices in the profile.
        count: usize,
    },
}

impl std::fmt::Display for DeviceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceError::Unknown { engine, device } => {
                write!(
                    formatter,
                    "engine `{engine}` names device {device}, but the inventory does not have it"
                )
            }
            DeviceError::GroupIndexes { engine, planned, count } => {
                write!(
                    formatter,
                    "engine `{engine}` plans the devices {planned:?}, but its profile names {count} devices; \
                     the plan must give the group indexes 0 to {} in order",
                    count.saturating_sub(1)
                )
            }
        }
    }
}

impl std::error::Error for DeviceError {}

/// Check each engine's devices against the inventory, and check that its plan gives one group index for each device.
pub fn check_devices(
    profile: &PlacementProfile,
    planned: &[AdmittedEngine],
    inventory: &Inventory,
) -> Result<(), DeviceError> {
    assert_eq!(
        profile.engines.len(),
        planned.len(),
        "the profile has {} engines but the plan has {}",
        profile.engines.len(),
        planned.len()
    );
    for (engine, planned) in profile.engines.iter().zip(planned) {
        assert_eq!(
            engine.name, planned.name,
            "the profile and the plan name different engines at one position"
        );
        for device in &engine.devices {
            if inventory.device(*device).is_none() {
                return Err(DeviceError::Unknown {
                    engine: engine.name.clone(),
                    device: *device,
                });
            }
        }
        // The engine declares group indexes, so it never sees a machine device id (spec 2.4).
        let indexes: Vec<DeviceId> = (0..engine.devices.len() as DeviceId).collect();
        if planned.limits.devices != indexes {
            return Err(DeviceError::GroupIndexes {
                engine: engine.name.clone(),
                planned: planned.limits.devices.clone(),
                count: engine.devices.len(),
            });
        }
    }
    Ok(())
}
