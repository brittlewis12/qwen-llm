//! Shape-complete prompt dispatch. Small independent decode batches stay out.

crate::env_flag!(
    /// Default-on Flash-Next append row completion on the tensor tier: dense/expert/bucket routes at 128..511 rows and bounded BF16 HC/F32 narrow projection tails at 128..2048. Qualified Q5 projections extend at 128..2047; N12288 and aligned M2048 keep their incumbents. No padded model timesteps or new buffers. Zero restores prior row/alignment gates; decode batches below 128 rows are unchanged.
    default_on configured,
    "QWEN_FN_APPEND_SHAPES"
);

crate::env_flag!(
    /// Default-on Flash-Next IQ4_NL/Q2_0 tensor down at 128..2047 append rows under QWEN_FN_APPEND_SHAPES. Preserves existing half operand values and F32 accumulation; zero restores legacy down selection. Original 2048-row qualification and its controls are unchanged.
    default_on configured_tensor_down,
    "QWEN_FN_APPEND_TENSOR_DOWN"
);

crate::env_flag!(
    /// Flash-Next newly admitted HC rows under QWEN_FN_APPEND_SHAPES: four BF16 K partitions, balanced F32 merge and 32-query tiles. Original operand conversion, no extra buffers or launches; zero restores the prior HC row/alignment gates.
    default_on configured_hc_split,
    "QWEN_FN_APPEND_HC_SPLIT"
);

pub(crate) fn tensor_down() -> bool {
    configured_tensor_down()
}

pub(crate) fn hc_split() -> bool {
    configured_hc_split()
}

#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

fn enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = OVERRIDE.with(|s| s.get()) {
        return value;
    }
    configured()
}

pub(crate) fn selected(rows: usize) -> bool {
    enabled() && (128..=2048).contains(&rows)
}

#[cfg(test)]
pub(crate) struct Scope(Option<bool>);

#[cfg(test)]
impl Drop for Scope {
    fn drop(&mut self) {
        OVERRIDE.with(|s| s.set(self.0));
    }
}

#[cfg(test)]
pub(crate) fn scope(value: bool) -> Scope {
    Scope(OVERRIDE.with(|s| s.replace(Some(value))))
}

#[cfg(test)]
pub(crate) fn with_enabled<R>(value: bool, work: impl FnOnce() -> R) -> R {
    let _restore = scope(value);
    work()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_prompt_rows_exclude_decode_and_oversize_commands() {
        for on in [false, true] {
            with_enabled(on, || {
                for rows in [0, 1, 2, 3, 4, 8, 16, 32, 127, 2049, 4096, usize::MAX] {
                    assert!(!selected(rows));
                }
                for rows in [128, 129, 399, 400, 401, 511, 512, 513, 2000, 2047, 2048] {
                    assert_eq!(selected(rows), on);
                }
            });
        }
    }
}
