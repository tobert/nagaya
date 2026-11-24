//! The rules of the engine that need no GPU: which driver errors are sticky, and which rows of logits a pass takes.

use nagaya_mk_cuda::memory::Count;

use crate::spec::MAX_LOGIT_ROWS;

/// The available bytes that the machine must still hold after an open: 2 GiB. On a GPU with unified memory the
/// available bytes are the memory of the whole machine, and an open that leaves less starves the system. The count is
/// `MemAvailable` of the host there, and the free bytes of the GPU on a discrete GPU (`nagaya_mk_cuda::memory`).
pub const MIN_FREE_AFTER_BYTES: u64 = 2 << 30;

/// The refusal of an open that left too little memory, or `None`. `counted` is the bytes of weights and slots that the
/// open allocated. `count` says which count the two available figures are. They are not compared: another process
/// frees and takes memory while the open reads the checkpoint, and a measured open saw the free bytes rise by 34 GB.
pub fn after_open_problem(counted: u64, available_before: u64, available_after: u64, count: Count) -> Option<String> {
    let label = count.label();
    (available_after < MIN_FREE_AFTER_BYTES).then(|| {
        format!(
            "the open took {counted} bytes of GPU memory for weights and slots, and the {label} is {available_after} \
             bytes after it, from {available_before} bytes before. The machine needs {MIN_FREE_AFTER_BYTES} bytes \
             available. Name fewer slots or slots of fewer positions in `slots`, or stop the programs that hold GPU \
             memory"
        )
    })
}

/// The refusal of an open whose weights take more device bytes than the plan counted, or `None`. The ledger admits the
/// engine from the plan, so a larger load is an admission hole (crash before you corrupt).
pub fn weights_problem(loaded: u64, planned: u64) -> Option<String> {
    (loaded > planned).then(|| {
        format!(
            "the weights take {loaded} bytes of device memory, and the plan counted {planned} bytes. The admission of \
             the engine would be too small by {} bytes. This is a defect of the plan: report it",
            loaded - planned
        )
    })
}

/// The codes of the CUDA driver that leave the context unusable (`nagaya_mk_cuda::driver`: "the context cannot run
/// again"). After one of them no call of the device can succeed.
pub const STICKY_CODES: [u32; 6] = [700, 701, 702, 715, 719, 999];

/// The code of the driver error in `message`, or `None` when the message names none. A driver message reads
/// `<call> failed with CUDA driver error <code> (<name>): ...`.
pub fn driver_code(message: &str) -> Option<u32> {
    const MARK: &str = "CUDA driver error ";
    let rest = &message[message.find(MARK)? + MARK.len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// True when `message` names a sticky error of the driver.
pub fn is_sticky(message: &str) -> bool {
    driver_code(message).is_some_and(|code| STICKY_CODES.contains(&code))
}

/// The rows of the logits that a block asks for must rise one by one and lie inside `MAX_LOGIT_ROWS` rows. The pass
/// checks the same rule, and the engine checks it first, so that a refusal comes before the engine changes the slot.
pub fn check_logit_rows(rows: &[usize]) -> Result<(), String> {
    if let Some(pair) = rows.windows(2).find(|pair| pair[0] >= pair[1]) {
        return Err(format!(
            "the rows for logits must rise one by one, and {} comes before {}. Sort the rows and drop a repeat",
            pair[0], pair[1]
        ));
    }
    if let (Some(first), Some(last)) = (rows.first(), rows.last())
        && last - first + 1 > MAX_LOGIT_ROWS
    {
        return Err(format!(
            "the rows for logits lie in {} rows, from {first} to {last}. They must lie inside {MAX_LOGIT_ROWS} rows",
            last - first + 1
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_open_that_leaves_less_than_2_gib_free_is_refused_with_the_bytes() {
        let host = Count::MemAvailable;
        assert_eq!(
            after_open_problem(20_000_000_000, 60_000_000_000, 40_000_000_000, host),
            None
        );
        // The free bytes may rise during an open: that is not a refusal.
        assert_eq!(
            after_open_problem(20_000_000_000, 27_000_000_000, 61_000_000_000, host),
            None
        );
        assert_eq!(after_open_problem(1, 3 << 30, 2 << 30, host), None);
        let message = after_open_problem(20_000_000_000, 22_000_000_000, (2 << 30) - 1, host).unwrap();
        assert!(
            message.contains("20000000000") && message.contains("2147483647"),
            "{message}"
        );
        assert!(message.contains("MemAvailable of the host"), "{message}");
        let message = after_open_problem(1, 5, 0, Count::GpuFree).unwrap();
        assert!(message.contains("free bytes of the GPU"), "{message}");
        assert!(message.contains("`slots`"), "{message}");
    }

    #[test]
    fn weights_that_pass_the_plan_are_refused_with_the_bytes() {
        assert_eq!(weights_problem(100, 100), None);
        assert_eq!(weights_problem(99, 100), None);
        let message = weights_problem(101, 100).unwrap();
        assert!(
            message.contains("101") && message.contains("100") && message.contains("by 1 bytes"),
            "{message}"
        );
    }

    #[test]
    fn a_driver_message_gives_its_code() {
        let message = "cuLaunchKernel failed with CUDA driver error 700 (CUDA_ERROR_ILLEGAL_ADDRESS): an illegal \
                       memory access was encountered. Restart the process";
        assert_eq!(driver_code(message), Some(700));
        assert_eq!(
            driver_code("cuMemAlloc failed with CUDA driver error 2 (X): out of memory"),
            Some(2)
        );
        assert_eq!(driver_code("the slot is poisoned"), None);
        assert_eq!(driver_code("CUDA driver error x"), None);
    }

    #[test]
    fn only_the_sticky_codes_kill_the_engine() {
        for code in [700, 701, 702, 715, 719, 999] {
            assert!(
                is_sticky(&format!("a call failed with CUDA driver error {code} (X): y")),
                "{code}"
            );
        }
        // An out of memory error and an invalid value leave the context usable.
        for code in [2, 1, 201, 400, 720] {
            assert!(
                !is_sticky(&format!("a call failed with CUDA driver error {code} (X): y")),
                "{code}"
            );
        }
        assert!(!is_sticky("the launch of the pass failed"));
    }

    #[test]
    fn the_rows_of_the_logits_rise_and_lie_inside_the_span() {
        assert!(check_logit_rows(&[]).is_ok());
        assert!(check_logit_rows(&[3]).is_ok());
        assert!(check_logit_rows(&[0, 15]).is_ok());
        assert!(check_logit_rows(&[2, 17]).is_ok(), "a span of 16 rows is the most");
        assert!(check_logit_rows(&[2, 18]).is_err(), "a span of 17 rows is too much");
        assert!(check_logit_rows(&[1, 18]).is_err());
        assert!(check_logit_rows(&[4, 4]).unwrap_err().contains("rise one by one"));
        assert!(check_logit_rows(&[5, 3]).is_err());
    }
}
