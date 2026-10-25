use super::*;
use objc2::rc::Retained;
use objc2_metal::MTLBuffer;
use sha2::{Digest, Sha256};

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
) -> Result<RetainedWeightResourceSummary, LoadError> {
    // Temporary O(bindings + windows) CPU inventory fits the existing 1024-byte
    // per-descriptor / 4096-byte per-window preflight reserve. No buffer clones.
    let mut identities = BTreeMap::new();
    let mut resources = Vec::<RetainedWeightResource>::new();
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
        let next = resources.len();
        let ordinal = *identities.entry(identity).or_insert_with(|| {
            // Exactly one length query per unique LIVE object, after view creation.
            let length = tensor.buffer.length() as u64;
            resources.push(RetainedWeightResource {
                ordinal: next,
                length,
                tensor_view_count: 0,
            });
            next
        });
        let resource = &mut resources[ordinal];
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
    for bank in banks {
        visit(
            &format!("bank/{}/{}", bank.layer, bank.projection),
            &bank.tensor,
        )?;
    }
    for (name, tensor) in dense {
        visit(name, tensor)?;
    }
    Ok(RetainedWeightResourceSummary {
        resources,
        tensor_view_count: views,
        total_unique_length: total,
        max_unique_length: maximum,
        binding_layout_sha256: format!("{:x}", digest.finalize()),
    })
}
