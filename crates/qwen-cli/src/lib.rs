pub mod bounded_file;
pub mod shutdown;
pub mod tracing_init;

#[cfg(test)]
mod tests {
    /// The signal flag is process-wide and never clears, so the clear and
    /// signalled checks share one test: a separate clear-state test would
    /// race this one in the same test process.
    #[test]
    fn installed_signal_is_observed_by_public_checkpoint() {
        super::shutdown::install().unwrap();
        assert!(super::shutdown::checkpoint().is_ok());

        // The installed handler records this process signal in the same
        // library static that checkpoint reads.
        assert_eq!(unsafe { libc::raise(libc::SIGTERM) }, 0);
        let error = super::shutdown::checkpoint().unwrap_err();
        assert!(error.to_string().contains("termination signal"));
    }
}
