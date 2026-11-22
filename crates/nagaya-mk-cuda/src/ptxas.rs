//! The report of `ptxas -v`. `build.rs` reads it, and it refuses a kernel that spills registers: a spill is a read and
//! a write of local memory in the hot loop of the interpreter.

/// Return an error that names the function and the bytes, if `report` shows a spill. Return an error too if the
/// report has no spill line, because then the check has not looked at anything.
pub fn spill_problem(report: &str) -> Result<(), String> {
    let mut seen = false;
    let mut function = "an unnamed function";
    for line in report.lines() {
        if let Some(name) = line.split("Function properties for ").nth(1) {
            function = name.trim();
        }
        let (Some(stores), Some(loads)) = (
            bytes_before(line, "bytes spill stores"),
            bytes_before(line, "bytes spill loads"),
        ) else {
            continue;
        };
        seen = true;
        if stores != 0 || loads != 0 {
            return Err(format!(
                "ptxas reports a spill in {function}: {stores} bytes of spill stores and {loads} bytes of spill \
                 loads. Lower the registers that the kernel holds, or split the code that holds them"
            ));
        }
    }
    if seen {
        Ok(())
    } else {
        Err(
            "the ptxas report has no line about spills. Pass `-Xptxas -v` to nvcc, so that the build can check them"
                .to_owned(),
        )
    }
}

/// The number that stands before `what` in `line`, after the last `,` or the start.
fn bytes_before(line: &str, what: &str) -> Option<u64> {
    let head = line.split(what).next().filter(|head| head.len() < line.len())?;
    head.rsplit(',').next()?.split_whitespace().next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLEAN: &str = "ptxas info    : Compiling entry function 'mk_persistent_interpreter' for 'sm_121a'\n\
        ptxas info    : Function properties for mk_persistent_interpreter\n\
        \x20   0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n\
        ptxas info    : Used 114 registers, used 16 barriers, 1536 bytes smem\n";

    #[test]
    fn a_report_with_no_spill_passes() {
        assert_eq!(spill_problem(CLEAN), Ok(()));
    }

    #[test]
    fn a_spill_store_or_a_spill_load_fails_and_names_the_function() {
        for (stores, loads) in [(4, 0), (0, 8), (16, 16)] {
            let report = CLEAN.replace(
                "0 bytes spill stores, 0 bytes spill loads",
                &format!("{stores} bytes spill stores, {loads} bytes spill loads"),
            );
            let error = spill_problem(&report).unwrap_err();
            assert!(error.contains("mk_persistent_interpreter"), "{error}");
            assert!(error.contains(&format!("{stores} bytes of spill stores")), "{error}");
            assert!(error.contains(&format!("{loads} bytes of spill loads")), "{error}");
        }
    }

    #[test]
    fn a_report_with_no_spill_line_fails() {
        let error = spill_problem("ptxas info    : 0 bytes gmem\n").unwrap_err();
        assert!(error.contains("-Xptxas -v"), "{error}");
    }

    #[test]
    fn the_second_function_is_checked_too() {
        let report = format!(
            "{CLEAN}ptxas info    : Function properties for other\n    0 bytes stack frame, 12 bytes spill stores, 0 bytes spill loads\n"
        );
        let error = spill_problem(&report).unwrap_err();
        assert!(error.contains("other") && error.contains("12 bytes"), "{error}");
    }
}
