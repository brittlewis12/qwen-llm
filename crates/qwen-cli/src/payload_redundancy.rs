use anyhow::{Context, Result, anyhow, bail};
use qwen_llm::gguf::{GgufFile, GgufShardStamp};
use qwen_llm::loader::{Block, Model};
use qwen_llm::tensor::TensorDesc;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

const HASH_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const VERIFY_BUFFER_BYTES: usize = 1024 * 1024;
const MAX_PROJECTION_ROWS: usize = 10_000_000;
const MAX_DUPLICATE_ROW_CLASSES: usize = 10_000;
const MAX_REPORTED_ROW_OCCURRENCES: usize = 1024;

struct ProjectionGroup<'a> {
    id: String,
    kind: &'static str,
    layer: u32,
    members: Vec<&'a TensorDesc>,
}

#[derive(Clone, Copy, Debug)]
struct RowSpec {
    group_idx: u32,
    dtype_tag: i32,
    input_elements: u64,
    row_bytes: u64,
    row_count: u32,
}

#[derive(Clone, Copy, Debug)]
struct RowFingerprint {
    digest: [u8; 32],
    tensor_idx: u32,
    row_idx: u32,
}

struct ExactRowClass {
    representative: Vec<u8>,
    count: u64,
    reported: Vec<RowFingerprint>,
    group_counts: BTreeMap<u32, u64>,
}

pub(crate) fn census(gguf: &GgufFile, model: &Model<'_>) -> Result<Value> {
    let stamps_before = gguf
        .revalidate_retained_shard_stamps()
        .context("revalidate GGUF before payload scan")?;
    let groups = projection_groups(model);
    let (row_specs, group_report, selected_rows) = build_row_specs(gguf, &groups)?;
    let tensor_payload_bytes = gguf.tensors.iter().try_fold(0_u64, |sum, desc| {
        sum.checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("tensor payload byte sum overflow"))
    })?;

    eprintln!(
        "[payload-redundancy] hashing {} tensors ({} bytes) and {} projection rows",
        gguf.tensors.len(),
        tensor_payload_bytes,
        selected_rows,
    );
    let (tensor_digests, mut row_fingerprints) = hash_payloads(gguf, &row_specs, selected_rows)?;
    let stamps_after_hash = gguf
        .revalidate_retained_shard_stamps()
        .context("revalidate GGUF after payload hashing")?;
    if stamps_before != stamps_after_hash {
        bail!("GGUF shard stamps changed during payload hashing");
    }

    let (tensor_duplicate_classes, full_tensor_redundant_bytes) =
        tensor_duplicate_classes(gguf, &tensor_digests)?;
    let (row_duplicate_classes, duplicate_row_occurrences, row_redundant_bytes) =
        row_duplicate_classes(gguf, &row_specs, &groups, &mut row_fingerprints)?;

    let stamps_after_verify = gguf
        .revalidate_retained_shard_stamps()
        .context("revalidate GGUF after duplicate verification")?;
    if stamps_before != stamps_after_verify {
        bail!("GGUF shard stamps changed during duplicate verification");
    }

    let tensors = gguf
        .tensors
        .iter()
        .zip(&tensor_digests)
        .enumerate()
        .map(|(tensor_ordinal, (desc, digest))| {
            json!({
                "tensor_ordinal": tensor_ordinal,
                "name": desc.name,
                "shape": desc.shape,
                "dtype": desc.dtype.wire_name(),
                "dtype_tag": desc.dtype as i32,
                "shard_idx": desc.shard_idx,
                "data_offset": desc.data_offset,
                "n_bytes": desc.n_bytes,
                "sha256": digest_hex(digest),
            })
        })
        .collect::<Vec<_>>();

    eprintln!(
        "[payload-redundancy] exact duplicates: tensors={} rows={} tensor_redundant_bytes={} row_redundant_bytes={} (scopes may overlap)",
        tensor_duplicate_classes.len(),
        row_duplicate_classes.len(),
        full_tensor_redundant_bytes,
        row_redundant_bytes,
    );

    Ok(json!({
        "schema_version": 1,
        "semantics": {
            "payload_hash": "sha256",
            "duplicate_proof": "sha256_candidate_then_exact_byte_compare",
            "tensor_scope": "all_descriptors",
            "row_scope": "typed_same_input_projection_groups",
            "row_layout": "ggml_shape0_contiguous_storage_blocks",
            "source_ranges": "half_open",
            "bounded_read_buffer_bytes": HASH_BUFFER_BYTES,
            "maximum_projection_rows": MAX_PROJECTION_ROWS,
            "maximum_duplicate_row_classes": MAX_DUPLICATE_ROW_CLASSES,
            "maximum_reported_occurrences_per_row_class": MAX_REPORTED_ROW_OCCURRENCES,
            "redundant_byte_scopes": "full_tensor_and_row_totals_are_independent_and_may_overlap",
            "source_revalidated_before_and_after": true,
        },
        "shards": stamps_before.iter().map(stamp_json).collect::<Vec<_>>(),
        "tensors": tensors,
        "tensor_duplicate_classes": tensor_duplicate_classes,
        "projection_groups": group_report,
        "row_duplicate_classes": row_duplicate_classes,
        "totals": {
            "tensor_count": gguf.tensors.len(),
            "tensor_payload_bytes_hashed": tensor_payload_bytes,
            "full_tensor_duplicate_classes": tensor_duplicate_classes.len(),
            "full_tensor_redundant_bytes": full_tensor_redundant_bytes,
            "projection_group_count": groups.len(),
            "projection_tensor_count": row_specs.iter().filter(|spec| spec.is_some()).count(),
            "projection_rows_hashed": row_fingerprints.len(),
            "duplicate_row_classes": row_duplicate_classes.len(),
            "duplicate_row_occurrences": duplicate_row_occurrences,
            "row_redundant_bytes": row_redundant_bytes,
        },
    }))
}

fn projection_groups<'a>(model: &Model<'a>) -> Vec<ProjectionGroup<'a>> {
    let mut groups = Vec::new();
    for (layer, block) in model.blocks.iter().enumerate() {
        let layer = layer as u32;
        match block {
            Block::Gdn(block) => {
                groups.push(ProjectionGroup {
                    id: format!("blk.{layer}/gdn_front"),
                    kind: "gdn_front",
                    layer,
                    members: vec![
                        block.in_proj_qkv,
                        block.in_proj_z,
                        block.beta_proj,
                        block.alpha_proj,
                    ],
                });
                groups.push(ProjectionGroup {
                    id: format!("blk.{layer}/ffn_front"),
                    kind: "ffn_gate_up",
                    layer,
                    members: vec![block.ffn_gate, block.ffn_up],
                });
            }
            Block::Attn(block) => {
                groups.push(ProjectionGroup {
                    id: format!("blk.{layer}/attn_front"),
                    kind: "attn_qkv",
                    layer,
                    members: vec![block.q, block.k, block.v],
                });
                groups.push(ProjectionGroup {
                    id: format!("blk.{layer}/ffn_front"),
                    kind: "ffn_gate_up",
                    layer,
                    members: vec![block.ffn_gate, block.ffn_up],
                });
            }
        }
    }
    if let Some(mtp) = &model.mtp {
        groups.push(ProjectionGroup {
            id: format!("blk.{}/mtp_attn_front", mtp.block_idx),
            kind: "mtp_attn_qkv",
            layer: mtp.block_idx,
            members: vec![mtp.attn.q, mtp.attn.k, mtp.attn.v],
        });
        groups.push(ProjectionGroup {
            id: format!("blk.{}/mtp_ffn_front", mtp.block_idx),
            kind: "mtp_ffn_gate_up",
            layer: mtp.block_idx,
            members: vec![mtp.attn.ffn_gate, mtp.attn.ffn_up],
        });
    }
    groups
}

fn build_row_specs(
    gguf: &GgufFile,
    groups: &[ProjectionGroup<'_>],
) -> Result<(Vec<Option<RowSpec>>, Vec<Value>, usize)> {
    let tensor_indices = gguf
        .tensors
        .iter()
        .enumerate()
        .map(|(idx, desc)| (desc.name.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let mut specs = vec![None; gguf.tensors.len()];
    let mut report = Vec::with_capacity(groups.len());
    let mut selected_rows = 0usize;

    for (group_idx, group) in groups.iter().enumerate() {
        let group_idx = u32::try_from(group_idx).context("projection group count exceeds u32")?;
        let mut member_names = Vec::with_capacity(group.members.len());
        let mut input_elements = None;
        for desc in &group.members {
            let tensor_idx = *tensor_indices.get(desc.name.as_str()).ok_or_else(|| {
                anyhow!("projection tensor {:?} missing from GGUF table", desc.name)
            })?;
            if specs[tensor_idx].is_some() {
                bail!(
                    "projection tensor {:?} belongs to multiple groups",
                    desc.name
                );
            }
            let (input, row_count, row_bytes) = projection_row_layout(desc)?;
            if let Some(expected) = input_elements {
                if input != expected {
                    bail!(
                        "projection group {:?} mixes input widths {expected} and {input}",
                        group.id
                    );
                }
            } else {
                input_elements = Some(input);
            }
            let row_count_u32 = u32::try_from(row_count)
                .with_context(|| format!("row count exceeds u32 for {:?}", desc.name))?;
            selected_rows = selected_rows
                .checked_add(usize::try_from(row_count).context("row count exceeds usize")?)
                .ok_or_else(|| anyhow!("selected row count overflow"))?;
            if selected_rows > MAX_PROJECTION_ROWS {
                bail!(
                    "projection row count {selected_rows} exceeds bounded scanner limit {MAX_PROJECTION_ROWS}; use an external-sort scanner for this asset"
                );
            }
            specs[tensor_idx] = Some(RowSpec {
                group_idx,
                dtype_tag: desc.dtype as i32,
                input_elements: input,
                row_bytes,
                row_count: row_count_u32,
            });
            member_names.push(desc.name.clone());
        }
        report.push(json!({
            "group_id": group.id,
            "kind": group.kind,
            "layer": group.layer,
            "input_elements": input_elements,
            "members": member_names,
        }));
    }
    Ok((specs, report, selected_rows))
}

fn projection_row_layout(desc: &TensorDesc) -> Result<(u64, u64, u64)> {
    if desc.shape.len() < 2 {
        bail!("projection tensor {:?} must have rank >= 2", desc.name);
    }
    let (block_elements, block_bytes) = desc.dtype.storage_layout().ok_or_else(|| {
        anyhow!(
            "unsupported storage type {:?} for {:?}",
            desc.dtype,
            desc.name
        )
    })?;
    if block_elements == 0 || block_bytes == 0 {
        bail!("zero storage block geometry for {:?}", desc.name);
    }
    let input_elements = desc.shape[0];
    if !input_elements.is_multiple_of(block_elements) {
        bail!(
            "projection tensor {:?} input width {} is not divisible by {:?} block size {}",
            desc.name,
            input_elements,
            desc.dtype,
            block_elements,
        );
    }
    let row_count = desc.shape[1..]
        .iter()
        .try_fold(1_u64, |product, dim| product.checked_mul(*dim))
        .ok_or_else(|| anyhow!("row count overflow for {:?}", desc.name))?;
    let row_bytes = input_elements
        .checked_div(block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| anyhow!("row byte count overflow for {:?}", desc.name))?;
    if row_bytes > HASH_BUFFER_BYTES as u64 {
        bail!(
            "projection tensor {:?} row size {} exceeds bounded scanner limit {}",
            desc.name,
            row_bytes,
            HASH_BUFFER_BYTES,
        );
    }
    let expected_bytes = row_bytes
        .checked_mul(row_count)
        .ok_or_else(|| anyhow!("tensor byte count overflow for {:?}", desc.name))?;
    if expected_bytes != desc.n_bytes {
        bail!(
            "projection tensor {:?} row geometry gives {} bytes, descriptor declares {}",
            desc.name,
            expected_bytes,
            desc.n_bytes,
        );
    }
    Ok((input_elements, row_count, row_bytes))
}

fn hash_payloads(
    gguf: &GgufFile,
    row_specs: &[Option<RowSpec>],
    selected_rows: usize,
) -> Result<(Vec<[u8; 32]>, Vec<RowFingerprint>)> {
    let mut tensor_digests = vec![None; gguf.tensors.len()];
    let mut rows = Vec::new();
    rows.try_reserve_exact(selected_rows)
        .context("reserve bounded projection fingerprint table")?;
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    let mut order = (0..gguf.tensors.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&idx| {
        let desc = &gguf.tensors[idx];
        (desc.shard_idx, desc.data_offset)
    });

    for tensor_idx in order {
        let desc = &gguf.tensors[tensor_idx];
        let mut hasher = Sha256::new();
        if let Some(spec) = row_specs[tensor_idx] {
            let row_bytes = usize::try_from(spec.row_bytes)
                .with_context(|| format!("row bytes exceed usize for {:?}", desc.name))?;
            let rows_per_chunk = (HASH_BUFFER_BYTES / row_bytes).max(1);
            let chunk_capacity = rows_per_chunk
                .checked_mul(row_bytes)
                .ok_or_else(|| anyhow!("row chunk size overflow for {:?}", desc.name))?;
            if buffer.len() < chunk_capacity {
                buffer.resize(chunk_capacity, 0);
            }
            let total_rows = spec.row_count as usize;
            let mut row_start = 0usize;
            while row_start < total_rows {
                let chunk_rows = rows_per_chunk.min(total_rows - row_start);
                let chunk_bytes = chunk_rows
                    .checked_mul(row_bytes)
                    .ok_or_else(|| anyhow!("row chunk byte overflow for {:?}", desc.name))?;
                let byte_offset = row_start
                    .checked_mul(row_bytes)
                    .and_then(|offset| u64::try_from(offset).ok())
                    .and_then(|offset| desc.data_offset.checked_add(offset))
                    .ok_or_else(|| anyhow!("row offset overflow for {:?}", desc.name))?;
                gguf.read_shard_exact_at(desc.shard_idx, byte_offset, &mut buffer[..chunk_bytes])
                    .with_context(|| format!("read projection tensor {:?}", desc.name))?;
                hasher.update(&buffer[..chunk_bytes]);
                for (row_in_chunk, bytes) in
                    buffer[..chunk_bytes].chunks_exact(row_bytes).enumerate()
                {
                    let row_idx =
                        u32::try_from(row_start + row_in_chunk).context("row index exceeds u32")?;
                    rows.push(RowFingerprint {
                        digest: Sha256::digest(bytes).into(),
                        tensor_idx: u32::try_from(tensor_idx)
                            .context("tensor index exceeds u32")?,
                        row_idx,
                    });
                }
                row_start += chunk_rows;
            }
        } else {
            let mut offset = 0_u64;
            while offset < desc.n_bytes {
                let remaining = desc.n_bytes - offset;
                let chunk_bytes = usize::try_from(remaining.min(HASH_BUFFER_BYTES as u64))
                    .context("hash chunk exceeds usize")?;
                let absolute_offset = desc
                    .data_offset
                    .checked_add(offset)
                    .ok_or_else(|| anyhow!("tensor offset overflow for {:?}", desc.name))?;
                gguf.read_shard_exact_at(
                    desc.shard_idx,
                    absolute_offset,
                    &mut buffer[..chunk_bytes],
                )
                .with_context(|| format!("read tensor {:?}", desc.name))?;
                hasher.update(&buffer[..chunk_bytes]);
                offset = offset
                    .checked_add(chunk_bytes as u64)
                    .ok_or_else(|| anyhow!("tensor hash offset overflow"))?;
            }
        }
        tensor_digests[tensor_idx] = Some(hasher.finalize().into());
    }

    let tensor_digests = tensor_digests
        .into_iter()
        .enumerate()
        .map(|(idx, digest)| {
            digest.ok_or_else(|| anyhow!("tensor {idx} was not hashed by the scan order"))
        })
        .collect::<Result<Vec<_>>>()?;
    if rows.len() != selected_rows {
        bail!(
            "hashed {} projection rows, expected {selected_rows}",
            rows.len()
        );
    }
    Ok((tensor_digests, rows))
}

fn tensor_duplicate_classes(gguf: &GgufFile, digests: &[[u8; 32]]) -> Result<(Vec<Value>, u64)> {
    let mut order = (0..gguf.tensors.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        let left_desc = &gguf.tensors[left];
        let right_desc = &gguf.tensors[right];
        left_desc
            .n_bytes
            .cmp(&right_desc.n_bytes)
            .then_with(|| digests[left].cmp(&digests[right]))
            .then_with(|| left.cmp(&right))
    });

    let mut report = Vec::new();
    let mut redundant_bytes = 0_u64;
    let mut start = 0usize;
    while start < order.len() {
        let first = order[start];
        let mut end = start + 1;
        while end < order.len()
            && gguf.tensors[order[end]].n_bytes == gguf.tensors[first].n_bytes
            && digests[order[end]] == digests[first]
        {
            end += 1;
        }
        if end - start > 1 {
            let mut exact_classes: Vec<Vec<usize>> = Vec::new();
            for &tensor_idx in &order[start..end] {
                let mut matched = false;
                for class in &mut exact_classes {
                    if tensor_payloads_equal(gguf, class[0], tensor_idx)? {
                        class.push(tensor_idx);
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    exact_classes.push(vec![tensor_idx]);
                }
            }
            for class in exact_classes.into_iter().filter(|class| class.len() > 1) {
                let representative = &gguf.tensors[class[0]];
                let class_redundant = representative
                    .n_bytes
                    .checked_mul((class.len() - 1) as u64)
                    .ok_or_else(|| anyhow!("full-tensor redundant byte count overflow"))?;
                redundant_bytes = redundant_bytes
                    .checked_add(class_redundant)
                    .ok_or_else(|| anyhow!("full-tensor redundant byte sum overflow"))?;
                let metadata_compatible = class.iter().all(|&idx| {
                    let desc = &gguf.tensors[idx];
                    desc.dtype == representative.dtype && desc.shape == representative.shape
                });
                let members = class
                    .iter()
                    .map(|&idx| {
                        let desc = &gguf.tensors[idx];
                        json!({
                            "tensor_ordinal": idx,
                            "name": desc.name,
                            "shard_idx": desc.shard_idx,
                            "data_offset": desc.data_offset,
                        })
                    })
                    .collect::<Vec<_>>();
                report.push(json!({
                    "class_id": report.len(),
                    "sha256": digest_hex(&digests[class[0]]),
                    "n_bytes": representative.n_bytes,
                    "byte_verified": true,
                    "metadata_compatible": metadata_compatible,
                    "members": members,
                    "physical_bytes_redundant": class_redundant,
                }));
            }
        }
        start = end;
    }
    Ok((report, redundant_bytes))
}

fn tensor_payloads_equal(gguf: &GgufFile, left_idx: usize, right_idx: usize) -> Result<bool> {
    let left = &gguf.tensors[left_idx];
    let right = &gguf.tensors[right_idx];
    if left.n_bytes != right.n_bytes {
        return Ok(false);
    }
    let mut left_buffer = vec![0_u8; VERIFY_BUFFER_BYTES];
    let mut right_buffer = vec![0_u8; VERIFY_BUFFER_BYTES];
    let mut offset = 0_u64;
    while offset < left.n_bytes {
        let chunk_bytes = usize::try_from((left.n_bytes - offset).min(VERIFY_BUFFER_BYTES as u64))
            .context("verification chunk exceeds usize")?;
        gguf.read_shard_exact_at(
            left.shard_idx,
            left.data_offset
                .checked_add(offset)
                .ok_or_else(|| anyhow!("left tensor verification offset overflow"))?,
            &mut left_buffer[..chunk_bytes],
        )?;
        gguf.read_shard_exact_at(
            right.shard_idx,
            right
                .data_offset
                .checked_add(offset)
                .ok_or_else(|| anyhow!("right tensor verification offset overflow"))?,
            &mut right_buffer[..chunk_bytes],
        )?;
        if left_buffer[..chunk_bytes] != right_buffer[..chunk_bytes] {
            return Ok(false);
        }
        offset = offset
            .checked_add(chunk_bytes as u64)
            .ok_or_else(|| anyhow!("tensor verification offset overflow"))?;
    }
    Ok(true)
}

fn row_duplicate_classes(
    gguf: &GgufFile,
    specs: &[Option<RowSpec>],
    groups: &[ProjectionGroup<'_>],
    rows: &mut [RowFingerprint],
) -> Result<(Vec<Value>, u64, u64)> {
    rows.sort_unstable_by(|left, right| row_fingerprint_cmp(left, right, specs));
    let mut report = Vec::new();
    report
        .try_reserve_exact(MAX_DUPLICATE_ROW_CLASSES)
        .context("reserve bounded duplicate-row report")?;
    let mut duplicate_occurrences = 0_u64;
    let mut redundant_bytes = 0_u64;
    let mut start = 0usize;
    while start < rows.len() {
        let mut end = start + 1;
        while end < rows.len() && same_row_hash_class(&rows[start], &rows[end], specs) {
            end += 1;
        }
        if end - start > 1 {
            let mut exact_classes: Vec<ExactRowClass> = Vec::new();
            for row in &rows[start..end] {
                let bytes = read_row(gguf, specs, row)?;
                let spec = specs[row.tensor_idx as usize]
                    .ok_or_else(|| anyhow!("duplicate row lacks row spec"))?;
                if let Some(class) = exact_classes
                    .iter_mut()
                    .find(|class| class.representative == bytes)
                {
                    class.count = class
                        .count
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("exact row class count overflow"))?;
                    *class.group_counts.entry(spec.group_idx).or_default() += 1;
                    if class.reported.len() < MAX_REPORTED_ROW_OCCURRENCES {
                        class.reported.push(*row);
                    }
                } else {
                    exact_classes.push(ExactRowClass {
                        representative: bytes,
                        count: 1,
                        reported: vec![*row],
                        group_counts: BTreeMap::from([(spec.group_idx, 1)]),
                    });
                }
            }
            for class in exact_classes.into_iter().filter(|class| class.count > 1) {
                let first = class.reported[0];
                let first_spec = specs[first.tensor_idx as usize]
                    .ok_or_else(|| anyhow!("duplicate row lacks row spec"))?;
                let class_occurrences = class.count;
                let class_redundant = first_spec
                    .row_bytes
                    .checked_mul(class_occurrences - 1)
                    .ok_or_else(|| anyhow!("row redundant byte count overflow"))?;
                duplicate_occurrences = duplicate_occurrences
                    .checked_add(class_occurrences)
                    .ok_or_else(|| anyhow!("duplicate row occurrence count overflow"))?;
                redundant_bytes = redundant_bytes
                    .checked_add(class_redundant)
                    .ok_or_else(|| anyhow!("row redundant byte sum overflow"))?;

                let occurrences = class
                    .reported
                    .iter()
                    .map(|row| {
                        let tensor_idx = row.tensor_idx as usize;
                        let desc = &gguf.tensors[tensor_idx];
                        let spec = specs[tensor_idx].expect("fingerprint row spec");
                        let offset = desc
                            .data_offset
                            .checked_add(spec.row_bytes * row.row_idx as u64)
                            .expect("validated row offset");
                        json!({
                            "group_id": groups[spec.group_idx as usize].id,
                            "tensor_ordinal": tensor_idx,
                            "tensor": desc.name,
                            "row": row.row_idx,
                            "shard_idx": desc.shard_idx,
                            "offset": offset,
                        })
                    })
                    .collect::<Vec<_>>();
                let within_group = class.group_counts.values().any(|count| *count > 1);
                let dtype = gguf.tensors[first.tensor_idx as usize].dtype;
                if report.len() == MAX_DUPLICATE_ROW_CLASSES {
                    bail!(
                        "duplicate row class count exceeds bounded report limit {MAX_DUPLICATE_ROW_CLASSES}; use a streaming reporter for this asset"
                    );
                }
                report.push(json!({
                    "class_id": report.len(),
                    "sha256": digest_hex(&first.digest),
                    "dtype": dtype.wire_name(),
                    "dtype_tag": dtype as i32,
                    "input_elements": first_spec.input_elements,
                    "row_bytes": first_spec.row_bytes,
                    "byte_verified": true,
                    "occurrence_count": class_occurrences,
                    "occurrences_reported": occurrences.len(),
                    "occurrences_truncated": class_occurrences > occurrences.len() as u64,
                    "within_group": within_group,
                    "cross_group": class.group_counts.len() > 1,
                    "occurrences": occurrences,
                    "physical_bytes_redundant": class_redundant,
                }));
            }
        }
        start = end;
    }
    Ok((report, duplicate_occurrences, redundant_bytes))
}

fn row_fingerprint_cmp(
    left: &RowFingerprint,
    right: &RowFingerprint,
    specs: &[Option<RowSpec>],
) -> Ordering {
    let left_spec = specs[left.tensor_idx as usize].expect("fingerprint row spec");
    let right_spec = specs[right.tensor_idx as usize].expect("fingerprint row spec");
    left_spec
        .dtype_tag
        .cmp(&right_spec.dtype_tag)
        .then_with(|| left_spec.input_elements.cmp(&right_spec.input_elements))
        .then_with(|| left_spec.row_bytes.cmp(&right_spec.row_bytes))
        .then_with(|| left.digest.cmp(&right.digest))
        .then_with(|| left.tensor_idx.cmp(&right.tensor_idx))
        .then_with(|| left.row_idx.cmp(&right.row_idx))
}

fn same_row_hash_class(
    left: &RowFingerprint,
    right: &RowFingerprint,
    specs: &[Option<RowSpec>],
) -> bool {
    let left_spec = specs[left.tensor_idx as usize].expect("fingerprint row spec");
    let right_spec = specs[right.tensor_idx as usize].expect("fingerprint row spec");
    left_spec.dtype_tag == right_spec.dtype_tag
        && left_spec.input_elements == right_spec.input_elements
        && left_spec.row_bytes == right_spec.row_bytes
        && left.digest == right.digest
}

fn read_row(gguf: &GgufFile, specs: &[Option<RowSpec>], row: &RowFingerprint) -> Result<Vec<u8>> {
    let tensor_idx = row.tensor_idx as usize;
    let desc = &gguf.tensors[tensor_idx];
    let spec = specs[tensor_idx].ok_or_else(|| anyhow!("fingerprint row lacks row spec"))?;
    if row.row_idx >= spec.row_count {
        bail!(
            "row {} exceeds tensor {:?} row count {}",
            row.row_idx,
            desc.name,
            spec.row_count,
        );
    }
    let offset = spec
        .row_bytes
        .checked_mul(row.row_idx as u64)
        .and_then(|offset| desc.data_offset.checked_add(offset))
        .ok_or_else(|| anyhow!("row read offset overflow for {:?}", desc.name))?;
    let mut bytes = vec![0_u8; usize::try_from(spec.row_bytes).context("row bytes exceed usize")?];
    gguf.read_shard_exact_at(desc.shard_idx, offset, &mut bytes)
        .with_context(|| format!("verify row {} of {:?}", row.row_idx, desc.name))?;
    Ok(bytes)
}

fn stamp_json(stamp: &GgufShardStamp) -> Value {
    json!({
        "shard_idx": stamp.shard_idx,
        "path": stamp.path,
        "device": stamp.device,
        "inode": stamp.inode,
        "size": stamp.size,
        "mtime_sec": stamp.mtime_sec,
        "mtime_nsec": stamp.mtime_nsec,
        "ctime_sec": stamp.ctime_sec,
        "ctime_nsec": stamp.ctime_nsec,
    })
}

fn digest_hex(digest: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen_llm::tensor::GgmlType;

    fn desc(shape: Vec<u64>, dtype: GgmlType, n_bytes: u64) -> TensorDesc {
        TensorDesc {
            name: "fixture.weight".to_string(),
            shape,
            dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes,
        }
    }

    #[test]
    fn q4_k_projection_rows_use_shape_zero_as_contiguous_input() {
        let tensor = desc(vec![5120, 17_408], GgmlType::Q4_K, 50_135_040);
        assert_eq!(
            projection_row_layout(&tensor).expect("row layout"),
            (5120, 17_408, 2880)
        );
    }

    #[test]
    fn projection_rows_reject_total_only_block_alignment() {
        let tensor = desc(vec![128, 2], GgmlType::Q4_K, 144);
        let error = projection_row_layout(&tensor).expect_err("shape[0] must align");
        assert!(error.to_string().contains("input width 128"));
    }

    #[test]
    fn projection_rows_reject_rows_above_memory_bound() {
        let input = 16_777_216_u64;
        let row_bytes = input / 256 * 144;
        let tensor = desc(vec![input, 1], GgmlType::Q4_K, row_bytes);
        let error = projection_row_layout(&tensor).expect_err("row must be bounded");
        assert!(error.to_string().contains("bounded scanner limit"));
    }
}
