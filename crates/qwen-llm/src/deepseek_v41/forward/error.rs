use crate::metal::MemoryAdmissionDenied;

#[derive(Debug)]
pub enum StateError {
    MemoryAdmissionDenied(MemoryAdmissionDenied),
    Invalid(String),
}

impl StateError {
    pub fn memory_admission(&self) -> Option<&MemoryAdmissionDenied> {
        match self {
            Self::MemoryAdmissionDenied(reason) => Some(reason),
            Self::Invalid(_) => None,
        }
    }
}

impl From<String> for StateError {
    fn from(value: String) -> Self {
        Self::Invalid(value)
    }
}

impl From<&str> for StateError {
    fn from(value: &str) -> Self {
        Self::Invalid(value.to_owned())
    }
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryAdmissionDenied(reason) => reason.fmt(f),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for StateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.memory_admission()
            .map(|reason| reason as &dyn std::error::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusal_keeps_typed_reason_and_signals() {
        let signals = crate::metal::MetalMemorySignals {
            recommended_max_bytes: 100,
            current_allocated_bytes: 90,
            process_limit_remaining_bytes: Some(5),
        };
        let refusal =
            crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(20, 10, 0, signals, true)
                .refusal()
                .unwrap();
        let error = StateError::MemoryAdmissionDenied(refusal);
        assert_eq!(error.memory_admission(), Some(&refusal));
        assert!(
            std::error::Error::source(&error)
                .unwrap()
                .is::<MemoryAdmissionDenied>()
        );
        assert!(
            StateError::from("invalid shape".to_string())
                .memory_admission()
                .is_none()
        );
    }
}
