//! The rule of the available bytes (`nagaya_mk_cuda::memory`). It needs no GPU.

use nagaya_mk_cuda::memory::{Count, available_bytes, parse_mem_available};

const GIB: u64 = 1 << 30;
const MEMINFO: &str =
    "MemTotal:       127000000 kB\nMemFree:         7700000 kB\nMemAvailable:   119100000 kB\nBuffers:  1 kB\n";

#[test]
fn the_meminfo_line_parses_to_bytes() {
    assert_eq!(parse_mem_available(MEMINFO), Ok(119_100_000 * 1024));
}

#[test]
fn memfree_is_not_memavailable() {
    assert_ne!(parse_mem_available(MEMINFO), Ok(7_700_000 * 1024));
}

#[test]
fn a_missing_line_or_a_bad_number_is_an_error() {
    let missing = parse_mem_available("MemTotal: 1 kB\nMemFree: 2 kB\n").unwrap_err();
    assert!(
        missing.contains("MemAvailable") && missing.contains("kernel"),
        "{missing}"
    );
    for bad in [
        "MemAvailable: lots kB\n",
        "MemAvailable: kB\n",
        "MemAvailable: 12\n",
        "MemAvailable: 12 MB\n",
        "MemAvailable: 12 kB extra\n",
        "MemAvailable: 99999999999999999999 kB\n",
        "MemAvailable: 18446744073709551615 kB\n",
    ] {
        let message = parse_mem_available(bad).unwrap_err();
        assert!(message.contains("MemAvailable"), "{bad}: {message}");
    }
}

#[test]
fn an_integrated_gpu_takes_memavailable() {
    let got = available_bytes(true, 16 * GIB, 121 * GIB, "MemAvailable: 62914560 kB\n").unwrap();
    assert_eq!((got.bytes, got.count), (60 * GIB, Count::MemAvailable));
}

#[test]
fn memavailable_is_capped_at_the_total_of_the_driver() {
    let got = available_bytes(true, 16 * GIB, 100 * GIB, "MemAvailable: 188743680 kB\n").unwrap();
    assert_eq!(got.bytes, 100 * GIB);
}

#[test]
fn a_discrete_gpu_takes_the_free_bytes_and_reads_no_meminfo() {
    // The text is bad on purpose: a discrete GPU must not read it.
    let got = available_bytes(false, 16 * GIB, 24 * GIB, "no such line").unwrap();
    assert_eq!((got.bytes, got.count), (16 * GIB, Count::GpuFree));
}

#[test]
fn an_integrated_gpu_with_a_bad_text_is_an_error() {
    assert!(available_bytes(true, 16 * GIB, 24 * GIB, "no such line").is_err());
}
