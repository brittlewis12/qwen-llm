//! Full-verifier instrument; no production selector or scratch-mode changes.

use super::*;
use crate::gguf::GgufFile;
use crate::loader::Model;
use crate::metal::{attn_matrix_ml_elems, read_back_f32};
use crate::metal_forward::MetalModel;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::Instant;

const N: usize = 16;
const BASE: usize = 32752;
const END: usize = BASE + N;
const NQ: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const CAPTURE: &[u32] = &[1, 16, 31, 46, 61];

pub(super) struct Workspace {
    vt: MetalTensor,
    scores: MetalTensor,
    ml: MetalTensor,
    calls: std::cell::Cell<usize>,
}

impl Workspace {
    fn new(ctx: &MetalContext) -> Self {
        Self {
            vt: MetalTensor::zeros_f16(ctx, vec![(END * NKV * HD) as u64]).unwrap(),
            scores: MetalTensor::zeros_f16(ctx, vec![(N * NQ * END) as u64]).unwrap(),
            ml: MetalTensor::zeros_f32(ctx, vec![attn_matrix_ml_elems(N, NQ, END) as u64]).unwrap(),
            calls: std::cell::Cell::new(0),
        }
    }

    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        q: &MetalTensor,
        k: &MetalTensor,
        v: &MetalTensor,
        out: &MetalTensor,
        base: usize,
    ) -> Result<(), MetalError> {
        assert_eq!(base, BASE);
        crate::metal::encode_attn_matrix_transpose_v_f16(
            ctx,
            enc,
            v,
            &self.vt,
            0,
            END,
            END,
            NKV * HD,
            END,
            NKV,
            HD,
        )?;
        crate::metal::encode_attn_matrix_kq_online_f32(
            ctx,
            enc,
            q,
            k,
            &self.scores,
            &self.ml,
            N,
            base,
            END,
            NKV * HD,
            NQ,
            NKV,
            NQ / NKV,
            HD,
            true,
        )?;
        crate::metal::encode_attn_matrix_kqv_norm_f32(
            ctx,
            enc,
            &self.scores,
            &self.ml,
            &self.vt,
            out,
            N,
            base,
            END,
            END,
            NQ,
            NKV,
            NQ / NKV,
            HD,
            true,
        )?;
        self.calls.set(self.calls.get() + 1);
        Ok(())
    }
}

fn bytes(t: &MetalTensor) -> &[u8] {
    assert_eq!(t.buffer.storageMode(), objc2_metal::MTLStorageMode::Shared);
    assert!(t.offset as usize + t.n_bytes() as usize <= t.buffer.length());
    // Every caller runs after the owning synchronous forward has completed.
    unsafe {
        std::slice::from_raw_parts(
            (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize),
            t.n_bytes() as usize,
        )
    }
}

fn floats(t: &MetalTensor) -> Vec<f32> {
    match t.dtype {
        GgmlType::F32 => bytes(t)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        GgmlType::F16 => bytes(t)
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        _ => panic!("unexpected state dtype"),
    }
}

fn prefix_hash(session: &MetalSession) -> Vec<String> {
    session
        .kv_k
        .iter()
        .chain(&session.kv_v)
        .map(|t| format!("{:x}", Sha256::digest(&bytes(t)[..BASE * NKV * HD * 2])))
        .collect()
}

struct Seed {
    gdn: Vec<Vec<u8>>,
    conv: Vec<Vec<u8>>,
}

impl Seed {
    fn new(session: &MetalSession) -> Self {
        Self {
            gdn: session
                .gdn_state
                .iter()
                .map(|t| bytes(t).to_vec())
                .collect(),
            conv: session.gdn_conv.iter().map(|t| bytes(t).to_vec()).collect(),
        }
    }

    fn restore(&self, session: &mut MetalSession) {
        for (src, dst) in self
            .gdn
            .iter()
            .chain(&self.conv)
            .zip(session.gdn_state.iter().chain(&session.gdn_conv))
        {
            assert_eq!(src.len(), bytes(dst).len());
            // Synchronous test-only reset; rejected KV suffix is unreachable.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    (dst.buffer.contents().as_ptr() as *mut u8).add(dst.offset as usize),
                    src.len(),
                );
            }
        }
        session.kv_n_pos.fill(BASE);
    }
}

struct Witness {
    argmax: Vec<i32>,
    logits: Vec<f32>,
    hidden: Vec<f32>,
    state: Vec<Vec<f32>>,
    gaps: Vec<f32>,
}

fn state(session: &MetalSession, keep: usize) -> Vec<Vec<f32>> {
    let mut values: Vec<_> = session
        .gdn_state
        .iter()
        .chain(&session.gdn_conv)
        .map(floats)
        .collect();
    for t in session.kv_k.iter().chain(&session.kv_v) {
        let tail = t.view_subrange((BASE * NKV * HD) as u64, vec![(keep * NKV * HD) as u64]);
        values.push(floats(&tail));
    }
    assert!(session.kv_n_pos.iter().all(|&pos| pos == BASE + keep));
    values
}

fn compare(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|x| x.is_finite()));
    let (mut dot, mut aa, mut bb, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for (&a, &b) in a.iter().zip(b) {
        dot += a as f64 * b as f64;
        aa += (a as f64).powi(2);
        bb += (b as f64).powi(2);
        max_abs = max_abs.max((a - b).abs());
    }
    let cos = if aa == 0.0 && bb == 0.0 {
        1.0
    } else {
        dot / (aa * bb).sqrt()
    };
    println!(
        "FULL_VERIFY_JSON {}",
        json!({"kind":"numerical", "label":label, "cosine":cos, "max_abs":max_abs})
    );
    assert!(cos >= 0.999, "{label}: cosine {cos}");
}

#[test]
#[ignore = "CPU-only fixture identity; no Metal context"]
fn online_n16_fixture_identity() {
    let path = std::env::var("QWEN_FULL_VERIFY_MODEL").unwrap();
    let tokens = fixture_tokens(&path);
    println!(
        "FULL_VERIFY_JSON {}",
        json!({"kind":"fixture_identity",
        "prefix_sha256":format!("{:x}", Sha256::digest(bytemuck::cast_slice(&tokens[..BASE]))),
        "verify_sha256":format!("{:x}", Sha256::digest(bytemuck::cast_slice(&tokens[BASE..END]))),
        "token_count":tokens.len()})
    );
}

fn fixture_tokens(path: &str) -> Vec<i32> {
    let fixture = std::env::var("QWEN_FULL_VERIFY_FIXTURE").unwrap();
    let text = std::fs::read_to_string(fixture).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(text.as_bytes())),
        std::env::var("QWEN_FULL_VERIFY_FIXTURE_SHA256").unwrap()
    );
    let tokenizer = crate::tokenizer::Tokenizer::open(path).unwrap();
    let tokens = tokenizer.encode(&text, false).unwrap();
    assert!(tokens.len() >= END, "fixture needs at least {END} tokens");
    tokens
}

#[test]
#[ignore = "serial Metal, real 27B and 32752-token fixture prefix"]
fn online_n16_full_verifier_witness() {
    for (key, _) in std::env::vars_os() {
        let key = key.to_string_lossy();
        assert!(!key.starts_with("QWEN_ATTN_") && !key.starts_with("QWEN_MTP_"));
    }
    let path = std::env::var("QWEN_FULL_VERIFY_MODEL").unwrap();
    let all_tokens = fixture_tokens(&path);
    let tokens = &all_tokens[..BASE];
    let verify_tokens = &all_tokens[BASE..END];
    assert_eq!(
        format!("{:x}", Sha256::digest(bytemuck::cast_slice(tokens))),
        std::env::var("QWEN_FULL_VERIFY_PREFIX_SHA256").unwrap()
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytemuck::cast_slice(verify_tokens))),
        std::env::var("QWEN_FULL_VERIFY_TOKENS_SHA256").unwrap()
    );
    println!(
        "FULL_VERIFY_JSON {}",
        json!({"kind":"fixture", "prefix_tokens":BASE,
        "prefix_ids_sha256":format!("{:x}", Sha256::digest(bytemuck::cast_slice(tokens))),
        "verify_tokens":verify_tokens, "fixture_total_tokens":all_tokens.len()})
    );
    let gguf = GgufFile::open(&path).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();
    let ctx = MetalContext::new().unwrap();
    let initial = ctx.current_allocated_size();
    let metal = MetalModel::load(&ctx, &gguf, &model).unwrap();
    assert_eq!(metal.arch.kind, crate::model::ArchKind::Dense);
    assert_eq!(metal.arch.n_layer, 64);
    assert_eq!(
        metal
            .blocks
            .iter()
            .filter(|b| matches!(b, MetalBlock::Attn(_)))
            .count(),
        16
    );
    let forward = MetalForward::new(&ctx, &metal);
    let model_bytes = ctx.current_allocated_size() - initial;
    let before_session = ctx.current_allocated_size();
    let mut session = MetalSession::fresh(&ctx, &metal, END + 16).unwrap();
    let session_bytes = ctx.current_allocated_size() - before_session;
    let mut prefill = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        &ctx,
        &metal,
        1024,
        END + 16,
    )
    .unwrap();
    let start = Instant::now();
    prefill_tokens_with_multi_hidden(&forward, tokens, 0, &mut session, &mut prefill, &[], None)
        .unwrap();
    println!(
        "FULL_VERIFY_JSON {}",
        json!({"kind":"prefill", "wall_ms":start.elapsed().as_secs_f64()*1e3,
        "allocated_bytes":ctx.current_allocated_size()-initial})
    );
    drop(prefill);
    let prefix = prefix_hash(&session);
    let seed = Seed::new(&session);
    let before_scratch = ctx.current_allocated_size();
    let mut verify =
        MetalDFlashVerifyScratch::fresh(&ctx, &metal, N as u32, CAPTURE.len() as u32).unwrap();
    let mut layer = MetalDFlashLayerMajorScratch::fresh(&ctx, &metal, N as u32).unwrap();
    let scratch_bytes = ctx.current_allocated_size() - before_scratch;
    let before_online = ctx.current_allocated_size();
    let mut workspace = Some(Workspace::new(&ctx));
    let online_bytes = ctx.current_allocated_size() - before_online;
    assert!(online_bytes <= 128 * 1024 * 1024);
    println!(
        "FULL_VERIFY_JSON {}",
        json!({"kind":"allocation", "model_bytes":model_bytes,
        "session_bytes":session_bytes, "verify_plus_layer_scratch_bytes":scratch_bytes,
        "online_increment_bytes":online_bytes, "actual_total_bytes":ctx.current_allocated_size()-initial,
        "timing_residency":"workspace retained in both arms; one model and session"})
    );
    let debug =
        MetalTensor::zeros_f32(&ctx, vec![(N * metal.arch.vocab_size as usize) as u64]).unwrap();
    let mut witnesses = Vec::new();
    for online in [false, true] {
        seed.restore(&mut session);
        if online {
            layer.online_verify_pilot = workspace.take();
        }
        let argmax = encode_packed_verify_layer_major_inner(
            &forward,
            CAPTURE,
            verify_tokens,
            BASE as u32,
            &mut verify,
            &mut layer,
            &mut session,
            Some(&debug),
            None,
        )
        .unwrap();
        witnesses.push(Witness {
            argmax,
            logits: floats(&debug),
            hidden: floats(&verify.hidden_capture),
            state: state(&session, N),
            gaps: read_back_f32(&verify.verify_gap.buffer, N),
        });
        assert_eq!(prefix_hash(&session), prefix);
        if online {
            assert_eq!(layer.online_verify_pilot.as_ref().unwrap().calls.get(), 16);
            workspace = layer.online_verify_pilot.take();
        }
    }
    for row in 0..N {
        let vocab = metal.arch.vocab_size as usize;
        compare(
            &format!("logits-{row}"),
            &witnesses[0].logits[row * vocab..(row + 1) * vocab],
            &witnesses[1].logits[row * vocab..(row + 1) * vocab],
        );
    }
    compare("hidden", &witnesses[0].hidden, &witnesses[1].hidden);
    for (i, (a, b)) in witnesses[0]
        .state
        .iter()
        .zip(&witnesses[1].state)
        .enumerate()
    {
        compare(&format!("state-{i}"), a, b);
    }
    println!(
        "FULL_VERIFY_JSON {}",
        json!({"kind":"argmax", "a":witnesses[0].argmax,
        "b":witnesses[1].argmax, "gaps_a":witnesses[0].gaps, "gaps_b":witnesses[1].gaps})
    );
    assert_eq!(witnesses[0].argmax, witnesses[1].argmax);
    drop(witnesses);
    drop(debug);
    let replay_hidden = MetalTensor::zeros_f32(
        &ctx,
        vec![(CAPTURE.len() * metal.arch.hidden_size as usize) as u64],
    )
    .unwrap();
    for mode in ["full", "partial8", "replay8"] {
        let mut states = Vec::new();
        for (index, online) in [false, true, true, false, false, true, true, false]
            .into_iter()
            .enumerate()
        {
            seed.restore(&mut session);
            if online {
                layer.online_verify_pilot = workspace.take();
            }
            let start = Instant::now();
            encode_packed_verify_layer_major_inner(
                &forward,
                CAPTURE,
                verify_tokens,
                BASE as u32,
                &mut verify,
                &mut layer,
                &mut session,
                None,
                None,
            )
            .unwrap();
            let verify_ms = start.elapsed().as_secs_f64() * 1e3;
            let restore_start = Instant::now();
            if mode == "replay8" {
                encode_restore_to_pre_block(&forward, &verify, BASE as u32, &mut session, None)
                    .unwrap();
            } else {
                encode_restore_after_partial_accept_inner(
                    &forward,
                    &verify,
                    if mode == "full" { 16 } else { 8 },
                    BASE as u32,
                    &mut session,
                    None,
                )
                .unwrap();
            }
            let restore_ms = restore_start.elapsed().as_secs_f64() * 1e3;
            let replay_start = Instant::now();
            if mode == "replay8" {
                for (row, &token) in verify_tokens[..8].iter().enumerate() {
                    forward
                        .single_token_with_multi_hidden(
                            token,
                            (BASE + row) as u32,
                            &mut session,
                            CAPTURE,
                            &replay_hidden,
                        )
                        .unwrap();
                }
            }
            let replay_ms = replay_start.elapsed().as_secs_f64() * 1e3;
            let total_ms = start.elapsed().as_secs_f64() * 1e3;
            if online {
                workspace = layer.online_verify_pilot.take();
            }
            println!(
                "FULL_VERIFY_JSON {}",
                json!({"kind":if index<4 {"warmup"} else {"sample"},
                "mode":mode, "index":index, "arm":if online {"B"} else {"A"},
                "verify_ms":verify_ms, "restore_ms":restore_ms, "forced_replay_ms":replay_ms, "total_ms":total_ms})
            );
            if index == 0 || index == 1 {
                states.push(state(&session, if mode == "full" { 16 } else { 8 }));
            }
        }
        for (i, (a, b)) in states[0].iter().zip(&states[1]).enumerate() {
            if mode == "replay8" {
                assert_eq!(a, b, "replay state {i}");
            } else {
                compare(&format!("{mode}-state-{i}"), a, b);
            }
        }
        assert_eq!(prefix_hash(&session), prefix);
    }
}
