use super::*;
use sha2::{Digest, Sha256};

fn digest(session: &K2Session<'_, '_>) -> String {
    let cache = &session.buffers.cache;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            cache.buffer.contents().as_ptr().cast::<u8>(),
            cache.n_bytes() as usize,
        )
    };
    format!("{:x}", Sha256::digest(bytes))
}

fn equal(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter()
            .zip(b)
            .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits())
    );
}

#[test]
#[ignore = "K2_GGUF; production lease/wired gate; 1024-token and high-position runtime controls, not full-context qualification"]
fn gpu_checkpoint_context_boundaries_and_memory_admission() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let source = GgufFile::open(std::env::var("K2_GGUF").unwrap()).unwrap();
    let config = K2HorizonConfig::from_gguf(&source).unwrap();
    let ctx = MetalContext::new().unwrap();
    let before = ctx.current_allocated_size();
    assert!(
        K2LoadedModel::load_with_attention_unqualified(
            &ctx,
            &source,
            7169,
            AttentionBackend::Materialized
        )
        .is_err()
    );
    assert!(
        K2RuntimePlan::inspect(
            &source,
            config.context_length,
            host_page_size_bytes().unwrap(),
            1024 * 1024
        )
        .is_err()
    );
    // Refuse the full cache only when actual device admission says it cannot fit;
    // a larger machine is allowed to admit it, and this test never allocates it.
    let price = ctx.price_shared_buffer_upper(
        config
            .kv_storage_bytes(u64::from(config.context_length), K2KvStorage::F16)
            .unwrap(),
    );
    eprintln!("K2 full-context cache price-only admission: {price:?}");
    if config
        .kv_storage_bytes(u64::from(config.context_length), K2KvStorage::F16)
        .unwrap()
        > ctx.max_buffer_length() as u64
    {
        assert!(
            K2RuntimePlan::inspect(
                &source,
                config.context_length,
                host_page_size_bytes().unwrap(),
                ctx.max_buffer_length()
            )
            .is_err()
        );
    }
    assert_eq!(ctx.current_allocated_size(), before);
    let tokens = crate::tokenizer::NativeTokenizer::from_gguf(&source)
        .unwrap()
        .encode(
            &vec!["The archive records each observation without assuming its interpretation."; 130]
                .join(" "),
            true,
        )
        .unwrap()
        .into_iter()
        .take(1024)
        .map(|id| u32::try_from(id).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(tokens.len(), 1024);
    let sites = [0, 11, 23, 35];
    let mut model = K2LoadedModel::load(&ctx, &source, 1025).unwrap();
    for base in [0, config.context_length - 1025] {
        model.prefill = PrefillMode::Serial;
        let mut serial = model.create_session(base).unwrap();
        let mut checkpoints = Vec::new();
        let boundaries = [257usize, 1024];
        let mut captures = Vec::new();
        for (index, &token) in tokens.iter().enumerate() {
            let result = serial
                .append_with_interventions(&[token], if index == 1023 { &sites } else { &[] }, &[])
                .unwrap();
            if index == 1023 {
                captures = result.residuals;
            }
            if boundaries.contains(&(index + 1)) {
                checkpoints.push((result.logits, digest(&serial)));
            }
        }
        let final_logits = checkpoints.last().unwrap().0.clone();
        equal(
            &serial.readout(&captures[3 * 4096..]).unwrap(),
            &final_logits,
        );
        let continued = serial.append(&[42]).unwrap();
        assert_eq!(serial.committed_len(), 1025);
        assert!(serial.append(&[42]).is_err());
        assert!(!serial.is_poisoned());
        drop(serial);
        model.prefill = PrefillMode::BatchQ8;
        let mut split = model.create_session(base).unwrap();
        let mut start = 0;
        for (&end, (logits, cache)) in boundaries.iter().zip(&checkpoints) {
            let result = split
                .append_with_interventions(
                    &tokens[start..end],
                    if end == 1024 { &sites } else { &[] },
                    &[],
                )
                .unwrap();
            equal(&result.logits, logits);
            assert_eq!(&digest(&split), cache);
            if end == 1024 {
                equal(&result.residuals, &captures);
            }
            start = end;
        }
        equal(&split.append(&[42]).unwrap(), &continued);
        drop(split);
        let mut whole = model.create_session(base).unwrap();
        let result = whole.append_with_captures(&tokens, &sites).unwrap();
        equal(&result.logits, &final_logits);
        equal(&result.residuals, &captures);
        assert_eq!(digest(&whole), checkpoints.last().unwrap().1);
        equal(&whole.append(&[42]).unwrap(), &continued);
        eprintln!(
            "K2 context controls: base={base}, 257/1024 boundaries, serial/split/whole cache/captures/readout/continuation bitwise pass"
        );
    }
}
