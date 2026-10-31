use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
struct BoundLinear {
    input: usize,
    output: usize,
    dtype: GgmlType,
    cache_bytes: u64,
}

/// Frozen active binding inventory. Cache buffers are distinct from native
/// controls, retained quantized sources and per-request preweight scratch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BindingMemoryPlan {
    cache_enabled: bool,
    linears: BTreeMap<String, BoundLinear>,
    control_metal_bytes: u64,
    cache_metal_bytes: u64,
}

impl BindingMemoryPlan {
    pub fn cache_enabled(&self) -> bool {
        self.cache_enabled
    }

    pub fn control_metal_bytes(&self) -> u64 {
        self.control_metal_bytes
    }

    pub fn cache_metal_bytes(&self) -> u64 {
        self.cache_metal_bytes
    }

    pub fn cache_count(&self) -> usize {
        self.linears.values().filter(|l| l.cache_bytes != 0).count()
    }

    pub(in crate::deepseek_v41) fn new(
        specs: impl IntoIterator<Item = LinearSpec>,
        dtype: impl Fn(&LinearSpec) -> Result<GgmlType>,
        cache_enabled: bool,
        price: &impl Fn(u64) -> Result<u64>,
    ) -> Result<Self> {
        let mut plan = Self {
            cache_enabled,
            ..Self::default()
        };
        for spec in specs {
            let dtype = dtype(&spec)?;
            let control = native_control_bytes(dtype)?;
            if control != 0 {
                plan.control_metal_bytes = add(plan.control_metal_bytes, price(control)?)?;
            }
            let cache_bytes = if cache_enabled && dtype == GgmlType::Q5_K {
                u64::try_from(spec.input)
                    .ok()
                    .and_then(|input| input.checked_mul(spec.output as u64))
                    .and_then(|elements| elements.checked_mul(2))
                    .filter(|&bytes| bytes != 0 && bytes <= isize::MAX as u64)
                    .ok_or("dense preweight cache byte extent overflow")?
            } else {
                0
            };
            if cache_bytes != 0 {
                plan.cache_metal_bytes = add(plan.cache_metal_bytes, price(cache_bytes)?)?;
            }
            op::require(
                plan.linears
                    .insert(
                        spec.name,
                        BoundLinear {
                            input: spec.input,
                            output: spec.output,
                            dtype,
                            cache_bytes,
                        },
                    )
                    .is_none(),
                "duplicate binding linear",
            )?;
        }
        add(plan.control_metal_bytes, plan.cache_metal_bytes)?;
        Ok(plan)
    }

    pub(in crate::deepseek_v41) fn cache_for(
        &self,
        name: &str,
        input: usize,
        output: usize,
        dtype: GgmlType,
    ) -> Result<bool> {
        let linear = self.linears.get(name).ok_or("unplanned binding linear")?;
        op::require(
            (linear.input, linear.output, linear.dtype) == (input, output, dtype),
            "binding linear differs from frozen inventory",
        )?;
        Ok(linear.cache_bytes != 0)
    }

    pub(in crate::deepseek_v41) fn with_admission<T>(
        self,
        binding_cpu_bytes: u64,
        future_metal_bytes: u64,
        future_cpu_bytes: u64,
        signals: metal::MetalMemorySignals,
        factory: impl FnOnce(Self) -> Result<T>,
    ) -> Result<T> {
        let metal = add(
            add(self.control_metal_bytes, self.cache_metal_bytes)?,
            future_metal_bytes,
        )?;
        let cpu = add(binding_cpu_bytes, future_cpu_bytes)?;
        let admission =
            metal::evaluate_metal_memory_admission_with_cpu_bytes(metal, cpu, 0, signals, true);
        if let Some(denied) = admission.refusal() {
            return Err(denied.to_string());
        }
        factory(self)
    }
}

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| "binding memory overflow".into())
}

#[cfg(test)]
mod tests;
