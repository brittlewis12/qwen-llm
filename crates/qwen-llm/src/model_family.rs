use crate::gguf::GgufFile;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFamily {
    Qwen35,
    Qwen35Moe,
    DeepSeek4,
}

impl ModelFamily {
    pub fn from_architecture_name(name: &str) -> Option<Self> {
        match name {
            "qwen35" => Some(Self::Qwen35),
            "qwen35moe" => Some(Self::Qwen35Moe),
            "deepseek4" => Some(Self::DeepSeek4),
            _ => None,
        }
    }

    pub fn detect(gguf: &GgufFile) -> Option<Self> {
        gguf.architecture()
            .as_deref()
            .and_then(Self::from_architecture_name)
    }

    pub fn architecture_name(self) -> &'static str {
        match self {
            Self::Qwen35 => "qwen35",
            Self::Qwen35Moe => "qwen35moe",
            Self::DeepSeek4 => "deepseek4",
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
        assert_eq!(ModelFamily::from_architecture_name("deepseek2"), None);
    }
}
