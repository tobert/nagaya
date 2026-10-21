//! The resource of the telemetry: the host and the hardware of the inventory (spec 9.1).

use std::io;

use nagaya_telemetry::ResourceValue;

use crate::inventory::Inventory;

/// The name of the host, from the kernel.
pub fn host_name() -> io::Result<String> {
    let text = std::fs::read_to_string("/proc/sys/kernel/hostname")?;
    Ok(text.trim().to_owned())
}

/// The resource attributes: `host.name`, and the hardware from the inventory.
///
/// The hardware attributes are the count, the names and the memory of the devices, and the host memory.
/// Spec 9.1 asks for `host.name` and the hardware of the inventory.
pub fn resource(host_name: Option<&str>, inventory: Option<&Inventory>) -> Vec<(String, ResourceValue)> {
    let mut resource = Vec::new();
    if let Some(name) = host_name {
        resource.push(("host.name".to_owned(), ResourceValue::Text(name.to_owned())));
    }
    if let Some(inventory) = inventory {
        let names: Vec<&str> = inventory.devices.iter().map(|device| device.name.as_str()).collect();
        let memory: u64 = inventory.devices.iter().map(|device| device.memory_bytes).sum();
        resource.push((
            "nagaya.device.count".to_owned(),
            ResourceValue::Count(inventory.devices.len() as i64),
        ));
        resource.push(("nagaya.device.names".to_owned(), ResourceValue::Text(names.join(","))));
        resource.push((
            "nagaya.device.memory_bytes".to_owned(),
            ResourceValue::Count(i64::try_from(memory).unwrap_or(i64::MAX)),
        ));
        resource.push((
            "nagaya.host.memory_bytes".to_owned(),
            ResourceValue::Count(i64::try_from(inventory.host_bytes).unwrap_or(i64::MAX)),
        ));
    }
    resource
}
