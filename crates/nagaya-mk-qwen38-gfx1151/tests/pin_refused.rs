//! The library pin: open refuses a library whose SHA-256 is not `MK_ENGINE_LIB_SHA256`.

#[path = "fixtures/pin.rs"]
mod fixture;

use fixture::FakeInstall;

#[test]
fn open_refuses_a_library_whose_digest_is_not_the_pin() {
    let install = FakeInstall::new();
    let wrong = "0".repeat(64);
    // SAFETY: this binary holds one test, so no other test reads the environment.
    unsafe { install.set_env(Some(&wrong)) };
    let error = match nagaya_mk_qwen38_gfx1151::open() {
        Ok(_) => panic!("open accepted a library whose digest is not the pin"),
        Err(error) => error,
    };
    let message = error.message();
    assert!(message.contains("has SHA-256"), "{message}");
    assert!(message.contains(&wrong), "{message}");
    assert!(message.contains("MK_ENGINE_LIB_SHA256"), "{message}");
    assert!(
        !message.contains("load "),
        "the binding loaded a library that the pin refused: {message}"
    );
}
