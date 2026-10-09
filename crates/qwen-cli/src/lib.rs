pub mod bounded_file;
pub mod shutdown;
pub mod tracing_init;

#[cfg(test)]
mod tests {
    use std::process::Command;

    const CHILD_ENV: &str = "QWEN_CLI_SIGNAL_TEST_CHILD";

    /// The signal flag is process-wide and never clears, so raising SIGTERM
    /// in the library's test process would fail every later test that
    /// calls `checkpoint()`. The signal runs in a child test process.
    #[test]
    fn installed_signal_is_observed_by_public_checkpoint() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "tests::signal_child",
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "child failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "child test did not run: {}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    #[test]
    #[ignore = "subprocess fixture; run by installed_signal_is_observed_by_public_checkpoint"]
    fn signal_child() {
        assert_eq!(std::env::var(CHILD_ENV).as_deref(), Ok("1"));
        super::shutdown::install().unwrap();
        assert!(super::shutdown::checkpoint().is_ok());
        // The installed handler records the signal in the same library
        // static that checkpoint reads.
        assert_eq!(unsafe { libc::raise(libc::SIGTERM) }, 0);
        let error = super::shutdown::checkpoint().unwrap_err();
        assert!(error.to_string().contains("termination signal"));
    }
}
