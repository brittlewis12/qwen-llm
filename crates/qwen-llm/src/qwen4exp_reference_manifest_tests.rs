//! Independent baseline captured from main cc3e1029 and the actual GSQ IQ3_S
//! reference shards. Expected descriptors/views are frozen data, never rebuilt
//! from the current schema. The capture used main's native weight planner and
//! a temporary read-only accessor; it allocated no weight buffers.
use super::*;
use serde_json::{Value, json};

fn captured() -> Value {
    serde_json::from_str(include_str!(
        "../tests/fixtures/qwen4exp_reference_main_cc3e1029.json"
    ))
    .map(|manifest: Value| {
        assert_eq!(
            manifest["capture"]["checkout_revision"], "cc3e1029249867241cbc74b66e15c9bccb164b57",
            "capture must establish the independently pinned main checkout"
        );
        assert_eq!(
            manifest["capture"]["checkout_tree"],
            "43689b7f777572ac7f126b537569aca90f1fe197"
        );
        assert_eq!(manifest["baseline"], "cc3e1029");
        manifest
    })
    .unwrap()
}
fn descriptor(value: &Value) -> TensorDesc {
    TensorDesc {
        name: value["name"].as_str().unwrap().into(),
        shape: value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect(),
        dtype: GgmlType::from_raw(value["dtype"].as_u64().unwrap() as u32),
        shard_idx: value["shard"].as_u64().unwrap() as usize,
        data_offset: value["offset"].as_u64().unwrap(),
        n_bytes: value["bytes"].as_u64().unwrap(),
    }
}
fn descriptor_json(d: &TensorDesc) -> Value {
    json!({"name":d.name,"shape":d.shape,"dtype":d.dtype as u32,"shard":d.shard_idx,"offset":d.data_offset,"bytes":d.n_bytes})
}
fn retained_json(plan: &RetainedStoragePlan) -> Value {
    let entries:Vec<_>=plan.entries.iter().map(|e| {
        let disposition=match e.disposition {
            RetainedStorageDisposition::View{window_index,buffer_offset}=>json!({"view":window_index,"offset":buffer_offset}),
            RetainedStorageDisposition::CopyFallback{reason}=>json!({"copy":format!("{reason:?}")}),
            RetainedStorageDisposition::Alias{source_request_index}=>json!({"alias":source_request_index}),
        };
        json!({"name":e.name,"shard":e.shard_idx,"offset":e.data_offset,"bytes":e.n_bytes,"disposition":disposition})
    }).collect();
    let windows: Vec<_> = plan
        .windows
        .iter()
        .map(|w| json!({"shard":w.shard_idx,"offset":w.mmap_offset,"length":w.length}))
        .collect();
    json!({"page_size":plan.page_size,"max_buffer_length":plan.max_buffer_length,"entries":entries,"windows":windows})
}
#[test]
fn captured_reference_main_inventory_and_views_are_unchanged() {
    let expected = captured();
    let config = Qwen4ExpConfig::flash_next_reference();
    config.validate_artifact().unwrap();
    assert_eq!(
        format!("{config:?}"),
        expected["configuration_debug"].as_str().unwrap()
    );
    let source: Vec<_> = expected["ordered_tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(descriptor)
        .collect();
    let model = Qwen4ExpModel::from_inventory(config, &source, 0).unwrap();
    let actual: Vec<_> = model.tensors.iter().map(descriptor_json).collect();
    assert_eq!(
        json!(actual),
        expected["ordered_tensors"],
        "main's actual descriptor order, shapes, dtypes, shards and offsets"
    );
    validate_native_dtypes(&model.tensors).unwrap();
    let normalized = normalization_descriptors(&model.tensors).unwrap();
    assert_eq!(
        json!({"buffer_count":normalized.len(),"bytes":normalized.iter().map(|d|d.n_bytes).sum::<u64>()}),
        expected["normalization"]
    );
    let lengths: Vec<_> = expected["shards"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["length"].as_u64().unwrap() as usize)
        .collect();
    let mut plan = plan_retained_storage(
        &lengths,
        &model.residency_sources(),
        expected["retained"]["page_size"].as_u64().unwrap() as usize,
        expected["retained"]["max_buffer_length"].as_u64().unwrap() as usize,
        QWEN4EXP_GGUF_BINDING_ALIGNMENT,
    )
    .unwrap();
    exclude_nonresident_pages(&mut plan, &[model.ple_embedding.as_ref().unwrap()]).unwrap();
    validate_retained_policy(&plan).unwrap();
    assert_eq!(
        retained_json(&plan),
        expected["retained"],
        "actual main planned windows and views"
    );
}
#[test]
#[ignore = "set QWEN4EXP_REFERENCE_MANIFEST_GGUF to the captured GSQ reference first shard; normal Metal lease, no weight realization"]
fn captured_reference_manifest_matches_live_artifact_without_realization() {
    use sha2::{Digest, Sha256};
    let expected = captured();
    let gguf = GgufFile::open(std::env::var("QWEN4EXP_REFERENCE_MANIFEST_GGUF").unwrap()).unwrap();
    assert_eq!(
        json!(gguf.tensors.iter().map(descriptor_json).collect::<Vec<_>>()),
        expected["ordered_tensors"]
    );
    for (shard, manifest) in gguf
        .shards
        .iter()
        .zip(expected["shards"].as_array().unwrap())
    {
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(&shard.mmap_bytes()[..shard.tensor_data_start as usize])
            ),
            manifest["header_sha256"].as_str().unwrap()
        );
    }
    let ctx = crate::test_fixtures::metal_context_or_skip().unwrap();
    let before = ctx.current_allocated_size();
    let plan = Qwen4ExpMetalWeightPlan::for_native(&ctx, &gguf).unwrap();
    assert_eq!(before, ctx.current_allocated_size());
    assert_eq!(retained_json(&plan.retained), expected["retained"]);
}
