use crate::gguf::GgufFile;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFamily {
    Qwen35,
    Qwen35Moe,
    Qwen4Exp,
    DeepSeek4,
    MuseGlimmer,
    K2Horizon,
}

impl ModelFamily {
    /// Every recognised family; the single source for "what does qwen
    /// support" messages and closedness tests.
    pub const ALL: &'static [Self] = &[
        Self::Qwen35,
        Self::Qwen35Moe,
        Self::Qwen4Exp,
        Self::DeepSeek4,
        Self::MuseGlimmer,
        Self::K2Horizon,
    ];

    /// Bump when adding a family; the exhaustive match in the closedness test
    /// is what tells you to update this value and `ALL`.
    pub const FAMILY_COUNT: usize = 6;

    pub fn from_architecture_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|family| family.architecture_name() == name)
    }

    pub fn detect(gguf: &GgufFile) -> Option<Self> {
        gguf.architecture()
            .as_deref()
            .and_then(Self::from_architecture_name)
    }

    /// Stable family label for records and machine-readable surfaces
    /// (request-stats `model.family`, `qwen info --json`). Dense and MoE
    /// ordinary Qwen share one label because they share every contract the
    /// records describe; `architecture_name` remains the GGUF spelling.
    pub fn record_label(self) -> &'static str {
        match self {
            Self::Qwen35 | Self::Qwen35Moe => "qwen",
            Self::Qwen4Exp => "qwen4exp",
            Self::DeepSeek4 => "deepseek_v4",
            Self::MuseGlimmer => "muse_glimmer",
            Self::K2Horizon => "k2_horizon",
        }
    }

    pub fn architecture_name(self) -> &'static str {
        match self {
            Self::Qwen35 => "qwen35",
            Self::Qwen35Moe => "qwen35moe",
            Self::Qwen4Exp => "qwen4exp",
            Self::DeepSeek4 => "deepseek4",
            Self::MuseGlimmer => crate::muse_glimmer::ARCHITECTURE_NAME,
            Self::K2Horizon => crate::k2_horizon::ARCHITECTURE_NAME,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ModelFamily;

    #[test]
    fn architecture_names_are_closed() {
        assert_eq!(
            ModelFamily::from_architecture_name("deepseek4"),
            Some(ModelFamily::DeepSeek4)
        );
        assert_eq!(ModelFamily::DeepSeek4.architecture_name(), "deepseek4");
        assert_eq!(
            ModelFamily::from_architecture_name("qwen4exp"),
            Some(ModelFamily::Qwen4Exp)
        );
        assert_eq!(ModelFamily::Qwen4Exp.architecture_name(), "qwen4exp");
        assert_eq!(
            ModelFamily::from_architecture_name("muse-glimmer"),
            Some(ModelFamily::MuseGlimmer)
        );
        assert_eq!(ModelFamily::from_architecture_name("deepseek2"), None);
        assert_eq!(
            ModelFamily::from_architecture_name("k2-horizon"),
            Some(ModelFamily::K2Horizon)
        );
        assert_eq!(ModelFamily::K2Horizon.record_label(), "k2_horizon");
    }

    /// Families in `ALL` round-trip through their architecture names, and
    /// their names and record labels are unique.
    #[test]
    fn all_families_round_trip_with_unique_names() {
        let mut names = std::collections::BTreeSet::new();
        let mut labels = std::collections::BTreeSet::new();
        for family in ModelFamily::ALL {
            assert_eq!(
                ModelFamily::from_architecture_name(family.architecture_name()),
                Some(*family)
            );
            assert!(names.insert(family.architecture_name()), "{family:?}");
            labels.insert(family.record_label());
        }
        // Dense and MoE ordinary Qwen share one record label by design.
        assert_eq!(labels.len(), ModelFamily::ALL.len() - 1);
        // The match forces this test to acknowledge every enum variant; it
        // does not prove that every variant appears in ALL.
        for family in ModelFamily::ALL {
            match family {
                ModelFamily::Qwen35
                | ModelFamily::Qwen35Moe
                | ModelFamily::Qwen4Exp
                | ModelFamily::DeepSeek4
                | ModelFamily::MuseGlimmer
                | ModelFamily::K2Horizon => {}
            }
        }
        assert_eq!(ModelFamily::ALL.len(), ModelFamily::FAMILY_COUNT);
    }
}
