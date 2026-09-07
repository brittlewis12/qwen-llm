use objc2::rc::Retained;
use objc2_metal::MTLBuffer;
use qwen_llm::gguf::GgufFile;
use qwen_llm::qwen4exp::Qwen4ExpConfig;
use qwen_llm::qwen4exp_loader::{MixerWeights, Qwen4ExpModel};
use qwen_llm::qwen4exp_ple::PleIq4NlTable;
use qwen_llm::qwen4exp_residency::{
    QWEN4EXP_METAL_TENSOR_COUNT, QWEN4EXP_PLE_SOURCE_BYTES, QWEN4EXP_RELEASE_TENSOR_COUNT,
    QWEN4EXP_RETAINED_WINDOW_CEILING_BYTES, Qwen4ExpDtypeCensus, Qwen4ExpMetalWeightPlan,
    Qwen4ExpMetalWeights,
};
use qwen_llm::tensor::GgmlType;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

const PINNED_8BDC666_Q3_PLE_ROWS_SHA256: &str =
    "ccc9aefbb25cc77a515e8cde2dab9bdff314283f760ec4300fddf95d475dcf5d";

#[test]
#[ignore = "set QWEN4EXP_METADATA_GGUF to a standalone metadata shard"]
fn parses_live_qwen4exp_metadata_when_available() {
    let path = qwen_llm::test_fixtures::QWEN4EXP_Q3_K_XL.required();
    let gguf = GgufFile::open(path).expect("open Qwen4Exp metadata GGUF");
    let actual = Qwen4ExpConfig::from_gguf(&gguf).expect("parse Qwen4Exp metadata");
    assert_eq!(actual, Qwen4ExpConfig::flash_next_reference());

    let model = Qwen4ExpModel::from_gguf(&gguf).expect("bind Qwen4Exp tensors");
    assert_eq!(gguf.tensors.len(), 1_224);
    assert_eq!(model.blocks.len(), 48);
    let ple_embedding = model.ple_embedding.unwrap();
    assert_eq!(ple_embedding.dtype, GgmlType::IQ4_NL);
    assert!(model.blocks[1].ple.is_some());
    assert_eq!(
        model
            .blocks
            .iter()
            .filter(|block| block.ple.is_some())
            .count(),
        1
    );
    assert_eq!(
        model
            .blocks
            .iter()
            .filter(|block| matches!(block.mixer, MixerWeights::QwenSparseAttention(_)))
            .count(),
        12
    );

    let table = PleIq4NlTable::new(
        ple_embedding,
        gguf.try_slice(ple_embedding).expect("slice PLE embedding"),
        actual
            .ple
            .as_ref()
            .unwrap()
            .logical_row_count()
            .expect("count logical PLE rows"),
    )
    .expect("bind IQ4_NL PLE table");
    let row_ids = actual
        .ple
        .as_ref()
        .unwrap()
        .row_ids(42, &[])
        .expect("hash PLE rows");
    assert_eq!(row_ids.len(), 16);
    assert_eq!(table.row_width(), 160);
    let mut packed = vec![0; table.packed_staging_bytes(row_ids.len()).unwrap()];
    table
        .gather_packed_into(&row_ids, &mut packed)
        .expect("gather released PLE rows");
    assert_eq!(packed.len(), 1_440);
    let mut decoded = vec![0.0; table.f32_staging_elements(row_ids.len()).unwrap()];
    table
        .gather_f32_into(&row_ids, &mut decoded)
        .expect("decode released PLE rows");
    assert_eq!(decoded.len(), 2_560);
    assert!(decoded.iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "set QWEN4EXP_Q3_K_XL_GGUF to the first released UD-Q3_K_XL shard"]
fn plans_released_q3_k_xl_without_residing_the_ple_table() {
    let path = qwen_llm::test_fixtures::QWEN4EXP_Q3_K_XL.required();
    let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
    let ctx = qwen_llm::metal::MetalContext::new().expect("initialize Metal");
    let allocated_before = ctx.current_allocated_size();
    let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf)
        .expect("plan released UD-Q3_K_XL weights");
    plan.revalidate(&ctx, &gguf)
        .expect("revalidate released weight plan");
    assert_eq!(ctx.current_allocated_size(), allocated_before);

    let report = plan.report();
    assert_eq!(report.source_tensor_count, QWEN4EXP_RELEASE_TENSOR_COUNT);
    assert_eq!(report.metal_tensor_count, QWEN4EXP_METAL_TENSOR_COUNT);
    assert_eq!(report.cpu_ple_tensor_count, 1);
    assert_eq!(report.cpu_ple_source_bytes, QWEN4EXP_PLE_SOURCE_BYTES);
    assert_eq!(
        report.source_bytes - report.cpu_ple_source_bytes,
        report.metal_source_bytes
    );
    assert_eq!(report.dtype_census, Qwen4ExpDtypeCensus::UD_Q3_K_XL);
    assert_eq!(
        report.view_count + report.fallback_count,
        QWEN4EXP_METAL_TENSOR_COUNT
    );
    assert!(report.planned_window_count > 0);
    assert!(report.ple_boundary_overlap_bytes < 2 * report.page_size as u64);
    assert_eq!(
        report.planning_max_buffer_length,
        report
            .device_max_buffer_length
            .min(QWEN4EXP_RETAINED_WINDOW_CEILING_BYTES)
    );

    let table = plan
        .ple_source()
        .bind(&gguf)
        .expect("bind row-addressed PLE source");
    let row_ids = plan
        .config()
        .ple
        .as_ref()
        .unwrap()
        .row_ids(42, &[])
        .expect("hash released PLE rows");
    let mut packed = vec![0; table.packed_staging_bytes(row_ids.len()).unwrap()];
    table
        .gather_packed_into(&row_ids, &mut packed)
        .expect("gather real packed PLE rows");
    assert_eq!(
        format!("{:x}", Sha256::digest(&packed)),
        PINNED_8BDC666_Q3_PLE_ROWS_SHA256
    );
    let mut decoded = vec![0.0; table.f32_staging_elements(row_ids.len()).unwrap()];
    table
        .gather_f32_into(&row_ids, &mut decoded)
        .expect("gather real released PLE rows");
    assert!(decoded.iter().all(|value| value.is_finite()));
    assert!(decoded.iter().any(|&value| value != 0.0));
    assert_eq!(ctx.current_allocated_size(), allocated_before);
}

#[test]
#[ignore = "set QWEN4EXP_Q3_K_XL_REALIZE_GGUF to opt into full weight realization"]
fn realizes_released_q3_k_xl_as_read_only_metal_views() {
    let path = qwen_llm::test_fixtures::QWEN4EXP_Q3_K_XL.required();
    let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
    let ctx = qwen_llm::metal::MetalContext::new().expect("initialize Metal");
    let baseline = ctx.current_allocated_size();
    let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf)
        .expect("plan released UD-Q3_K_XL weights");
    let priced_upper = plan.memory_plan().priced_upper_bytes();
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit released Metal weights");
    assert!(admitted.admission().admitted);

    let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted)
        .expect("realize released Metal weights");
    assert!(realized.admission().admitted);
    assert_eq!(realized.allocated_before(), baseline);
    assert!(realized.observed_allocation_delta() <= priced_upper);
    assert!(realized.allocated_after() >= realized.allocated_before());

    let weights = realized.weights();
    weights.validate_context(&ctx).unwrap();
    assert_eq!(weights.len(), QWEN4EXP_METAL_TENSOR_COUNT);
    assert!(!weights.is_empty());
    assert!(weights.tensor("per_layer_token_embd.weight").is_none());
    assert_eq!(
        weights.require_tensor("token_embd.weight").unwrap().dtype,
        GgmlType::Q8_0
    );
    assert_eq!(
        weights.require_tensor("output.weight").unwrap().dtype,
        GgmlType::Q6_K
    );
    assert_eq!(
        weights
            .require_tensor("blk.2.ffn_gate_exps.weight")
            .unwrap()
            .dtype,
        GgmlType::IQ4_XS
    );
    assert!(weights.tensors().all(|(_, tensor)| {
        !tensor.is_writable()
            && matches!(
                tensor.provenance(),
                qwen_llm::metal::MetalTensorProvenance::RetainedGgufReadOnly
                    | qwen_llm::metal::MetalTensorProvenance::OwnedWeightReadOnly
            )
    }));

    let mut seen_buffers = HashSet::new();
    let mut retained_samples = Vec::new();
    let mut retained_window_buffers = 0_usize;
    let mut copied_fallback_buffers = 0_usize;
    for (name, tensor) in weights.tensors() {
        let identity = Retained::as_ptr(&tensor.buffer) as *const () as usize;
        if !seen_buffers.insert(identity) {
            continue;
        }
        match tensor.provenance() {
            qwen_llm::metal::MetalTensorProvenance::RetainedGgufReadOnly => {
                retained_window_buffers += 1;
            }
            qwen_llm::metal::MetalTensorProvenance::OwnedWeightReadOnly => {
                copied_fallback_buffers += 1;
            }
            qwen_llm::metal::MetalTensorProvenance::OwnedWritable => unreachable!(),
        }
        let desc = gguf.find(name).unwrap();
        let sample_len = usize::try_from(desc.n_bytes.min(32)).unwrap();
        let expected = gguf.try_slice(desc).unwrap()[..sample_len].to_vec();
        retained_samples.push((name.to_string(), tensor.clone(), expected));
    }
    assert_eq!(
        retained_window_buffers,
        weights.report().planned_window_count
    );
    assert_eq!(copied_fallback_buffers, weights.report().fallback_count);
    assert!(copied_fallback_buffers > 0);
    assert_eq!(retained_samples.len(), weights.memory_plan().buffer_count());

    let table = weights
        .ple_source()
        .bind(&gguf)
        .expect("bind CPU PLE source after Metal realization");
    let row_ids = weights
        .config()
        .ple
        .as_ref()
        .unwrap()
        .row_ids(42, &[])
        .expect("hash released PLE rows");
    let mut packed = vec![0; table.packed_staging_bytes(row_ids.len()).unwrap()];
    table.gather_packed_into(&row_ids, &mut packed).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&packed)),
        PINNED_8BDC666_Q3_PLE_ROWS_SHA256
    );
    drop(gguf);
    for (name, tensor, expected) in retained_samples {
        // SAFETY: realization validated this tensor's complete byte range
        // against the retained Metal buffer.
        let actual = unsafe {
            std::slice::from_raw_parts(
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize),
                expected.len(),
            )
        };
        assert_eq!(actual, expected, "post-drop sample mismatch for {name}");
    }
}
