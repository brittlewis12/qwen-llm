use super::*;
use objc2::rc::Retained;
use objc2_metal::MTLBuffer;
use sha2::{Digest, Sha256};
use std::{fmt::Write, mem::size_of};

const DESCRIPTOR_CPU_BYTES: u64 = 1024;
const WINDOW_CPU_BYTES: u64 = 4096;
const HEX_DIGEST_BYTES: usize = 64;
const BANK_NAME_BYTES: usize = 5 + usize::MAX.ilog10() as usize + 1 + 3;
// One descriptor-sized workspace covers the digest, final hash, formatting
// arguments, closure/control state and stack/allocation slack, not tensor data.
const DIGEST_FORMAT_CPU_BYTES: u64 = DESCRIPTOR_CPU_BYTES;
const _: () = assert!(
    size_of::<Sha256>()
        + 32
        + size_of::<std::fmt::Arguments<'static>>()
        + size_of::<ResourceInventory>()
        + size_of::<RetainedSummaryBookkeeping>()
        < DIGEST_FORMAT_CPU_BYTES as usize
);

/// CPU-only summary bookkeeping within the ordinary descriptor/window category.
/// The retained vector/string and temporary identity vector, digest and reused
/// bank-name buffer coexist at peak. No tensor payload or GPU length is priced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RetainedSummaryBookkeeping {
    pub bank_descriptors: u64,
    pub dense_descriptors: u64,
    pub window_count: u64,
    pub resource_capacity: usize,
    pub retained_cpu_bytes: u64,
    pub temporary_cpu_bytes: u64,
    pub cpu_bytes: u64,
}

impl RetainedSummaryBookkeeping {
    /// The caller's descriptor/window charge and this loader-local recheck cover
    /// the SAME allocations: take max, not sum. Add disjoint GSQ controls/source
    /// bookkeeping, private shadows and other caller allocations separately.
    pub(crate) fn shared_cpu_envelope(self, caller_descriptor_window_bytes: u64) -> u64 {
        self.cpu_bytes.max(caller_descriptor_window_bytes)
    }

    pub(crate) fn load_cpu_bytes(
        self,
        controls: &ControlAccounting,
        private_shadow_bytes: u64,
    ) -> Result<u64, LoadError> {
        add(
            private_shadow_bytes,
            add(
                controls.retained_cpu_bytes,
                add(
                    controls.source_bookkeeping_bytes,
                    self.shared_cpu_envelope(0),
                )?,
            )?,
        )
    }
}

#[cfg(test)]
mod tests;

/// Use with the validated metadata/storage plan, before importing any buffers.
/// Counts exactly the bank/dense views the loader creates; excluded tensors and
/// companion lookup tables do not become summary entries. Oversized singletons
/// cost one descriptor, not their serialized or Metal buffer byte length.
pub(crate) fn retained_summary_bookkeeping(
    g: &GgufFile,
    p: &Plan,
    storage: &ResidentStoragePlan,
) -> Result<RetainedSummaryBookkeeping, LoadError> {
    let mut banks = 0;
    let mut dense = 0;
    for t in g.tensors.iter().filter(|t| resident_tensor(t, p.layers)) {
        if names::bank(&t.name).is_some() {
            banks = add(banks, 1)?;
        } else {
            dense = add(dense, 1)?;
        }
    }
    for_counts(banks, dense, storage.windows.len() as u64)
}

fn mul(a: u64, b: u64) -> Result<u64, LoadError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("summary bookkeeping byte count overflow"))
}

fn for_counts(
    banks: u64,
    dense: u64,
    windows: u64,
) -> Result<RetainedSummaryBookkeeping, LoadError> {
    let descriptors = add(banks, dense)?;
    if descriptors != 0 && windows == 0 {
        return Err(invalid(
            "resident summary descriptors have no storage windows",
        ));
    }
    let capacity = descriptors.min(windows);
    let resource_bytes = mul(capacity, size_of::<RetainedWeightResource>() as u64)?;
    let identity_bytes = mul(capacity, size_of::<usize>() as u64)?;
    if resource_bytes > isize::MAX as u64 || identity_bytes > isize::MAX as u64 {
        return Err(invalid("summary bookkeeping allocation exceeds isize"));
    }
    let retained_cpu_bytes = add(
        size_of::<RetainedWeightResourceSummary>() as u64,
        add(resource_bytes, HEX_DIGEST_BYTES as u64)?,
    )?;
    let temporary_cpu_bytes = add(
        add(size_of::<Vec<usize>>() as u64, size_of::<String>() as u64)?,
        add(
            identity_bytes,
            add(
                DIGEST_FORMAT_CPU_BYTES,
                if banks == 0 {
                    0
                } else {
                    BANK_NAME_BYTES as u64
                },
            )?,
        )?,
    )?;
    let peak = add(retained_cpu_bytes, temporary_cpu_bytes)?;
    let descriptor_window = add(
        mul(descriptors, DESCRIPTOR_CPU_BYTES)?,
        mul(windows, WINDOW_CPU_BYTES)?,
    )?;
    Ok(RetainedSummaryBookkeeping {
        bank_descriptors: banks,
        dense_descriptors: dense,
        window_count: windows,
        resource_capacity: usize::try_from(capacity)
            .map_err(|_| invalid("summary resource capacity exceeds usize"))?,
        retained_cpu_bytes,
        temporary_cpu_bytes,
        // Even an empty summary retains a digest and needs fixed workspace.
        cpu_bytes: descriptor_window.max(peak),
    })
}

struct ResourceInventory {
    identities: Vec<usize>,
    resources: Vec<RetainedWeightResource>,
    limit: usize,
}

impl ResourceInventory {
    fn new(limit: usize) -> Result<Self, LoadError> {
        let mut result = Self {
            identities: Vec::new(),
            resources: Vec::new(),
            limit,
        };
        result
            .identities
            .try_reserve_exact(limit)
            .map_err(|e| invalid(format!("summary identity allocation: {e}")))?;
        result
            .resources
            .try_reserve_exact(limit)
            .map_err(|e| invalid(format!("summary resource allocation: {e}")))?;
        Ok(result)
    }

    fn record(
        &mut self,
        identity: usize,
        length: impl FnOnce() -> u64,
    ) -> Result<usize, LoadError> {
        if let Some(ordinal) = self.identities.iter().position(|&key| key == identity) {
            return Ok(ordinal);
        }
        let ordinal = self.resources.len();
        if ordinal >= self.limit {
            return Err(invalid("live resources exceed admitted summary capacity"));
        }
        self.identities.push(identity);
        self.resources.push(RetainedWeightResource {
            ordinal,
            length: length(),
            tensor_view_count: 0,
        });
        Ok(ordinal)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct RetainedWeightResource {
    pub ordinal: usize,
    pub length: u64,
    pub tensor_view_count: usize,
}

/// Live MTLBuffer identity/length inventory, not traffic or physical residency.
/// Identities become first-seen ordinals; no addresses or retained handles escape.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct RetainedWeightResourceSummary {
    pub resources: Vec<RetainedWeightResource>,
    pub tensor_view_count: usize,
    pub total_unique_length: u64,
    pub max_unique_length: u64,
    pub binding_layout_sha256: String,
}

pub(super) fn summarize(
    banks: &[PackedBank],
    dense: &BTreeMap<String, MetalTensor>,
    bookkeeping: RetainedSummaryBookkeeping,
) -> Result<RetainedWeightResourceSummary, LoadError> {
    if banks.len() as u64 != bookkeeping.bank_descriptors
        || dense.len() as u64 != bookkeeping.dense_descriptors
    {
        return Err(invalid(
            "resident summary descriptor counts changed after admission",
        ));
    }
    // Fixed-capacity vectors avoid growth peaks and allocator-specific B-tree
    // node assumptions. Identities remain temporary and never leave this call.
    let mut inventory = ResourceInventory::new(bookkeeping.resource_capacity)?;
    let mut digest = Sha256::new();
    digest.update(b"v41-live-retained-weight-bindings-v1\0");
    let mut total = 0;
    let mut maximum = 0;
    let mut views = 0;
    let mut visit = |name: &str, tensor: &MetalTensor| -> Result<(), LoadError> {
        if tensor.provenance() != metal::MetalTensorProvenance::RetainedGgufReadOnly {
            return Err(invalid("resident weight is not a retained read-only view"));
        }
        let identity = Retained::as_ptr(&tensor.buffer) as *const () as usize;
        let ordinal = inventory.record(identity, || {
            // Exactly one length query per unique LIVE object, after view creation.
            tensor.buffer.length() as u64
        })?;
        let resource = &mut inventory.resources[ordinal];
        if resource.tensor_view_count == 0 {
            total = add(total, resource.length)?;
            maximum = maximum.max(resource.length);
        }
        resource.tensor_view_count += 1;
        views += 1;
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((ordinal as u64).to_le_bytes());
        digest.update((tensor.offset as u64).to_le_bytes());
        Ok(())
    };
    let mut bank_name = String::new();
    bank_name
        .try_reserve_exact(if banks.is_empty() { 0 } else { BANK_NAME_BYTES })
        .map_err(|e| invalid(format!("summary bank-name allocation: {e}")))?;
    for bank in banks {
        if !matches!(bank.projection, "w1" | "w2" | "w3") {
            return Err(invalid("invalid summary bank projection"));
        }
        bank_name.clear();
        write!(&mut bank_name, "bank/{}/{}", bank.layer, bank.projection)
            .map_err(|e| invalid(e.to_string()))?;
        visit(&bank_name, &bank.tensor)?;
    }
    for (name, tensor) in dense {
        visit(name, tensor)?;
    }
    let mut binding_layout_sha256 = String::new();
    binding_layout_sha256
        .try_reserve_exact(HEX_DIGEST_BYTES)
        .map_err(|e| invalid(format!("summary digest allocation: {e}")))?;
    write!(&mut binding_layout_sha256, "{:x}", digest.finalize())
        .map_err(|e| invalid(e.to_string()))?;
    Ok(RetainedWeightResourceSummary {
        resources: inventory.resources,
        tensor_view_count: views,
        total_unique_length: total,
        max_unique_length: maximum,
        binding_layout_sha256,
    })
}
