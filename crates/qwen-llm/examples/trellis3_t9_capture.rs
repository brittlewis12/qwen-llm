//! T9 Stage A activation capture (docs/bench/2026-07-20-trellis3-t9-ldlq-pilot/).
//!
//! Runs the 0.8B F32 model on the Metal engine over a document-split
//! calibration corpus, capturing per-token FFN inputs (post-norm `h`)
//! and SwiGLU intermediates for the declared block set, producing:
//!   - f64 Gram matrices (train split) per captured space,
//!   - raw f32 activation chunks (test split) per space for r_H scoring,
//!   - a manifest (doc split, token counts, corpus sha recorded by
//!     the runner script).
//!
//! Env: T9_MODEL (gguf), T9_CORPUS (wikitext txt), T9_OUT (dir),
//! T9_TRAIN_TOKENS (default 32768), T9_TEST_TOKENS (32768),
//! T9_DOC_MAX_TOKENS (512), T9_LAYERS (default "0,3"),
//! T9_GRAM_THREADS (default 8).

use objc2_metal::MTLBuffer;
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::Model;
use qwen_llm::metal::{MetalContext, MetalTensor};
use qwen_llm::metal_forward::{
    MetalForward, MetalModel, MetalSession, t9_ffn_capture_install, t9_ffn_capture_reset_token,
    t9_ffn_capture_uninstall,
};
use qwen_llm::tokenizer::Tokenizer;
use qwen_llm::trellis_ldlq::GramAccumulator;
use std::io::Write;

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn read_f32_tensor(t: &MetalTensor) -> Vec<f32> {
    let n = t.n_elements() as usize;
    let mut out = vec![0f32; n];
    unsafe {
        let p = (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize) as *const f32;
        std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), n);
    }
    out
}

/// Split wikitext into documents at top-level ` = Title = ` headers
/// (exactly one leading `=`; subsections have two or more).
fn split_documents(text: &str) -> Vec<String> {
    let mut docs: Vec<String> = Vec::new();
    let mut cur = String::new();
    for line in text.lines() {
        let t = line.trim();
        let is_title = t.starts_with("= ") && t.ends_with(" =") && !t.starts_with("= =");
        if is_title && !cur.trim().is_empty() {
            docs.push(std::mem::take(&mut cur));
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if !cur.trim().is_empty() {
        docs.push(cur);
    }
    docs
}

fn main() {
    let model_path = std::env::var("T9_MODEL")
        .unwrap_or_else(|_| qwen_llm::test_fixtures::QWEN35_0_8B_F32.path().into());
    let corpus_path = std::env::var("T9_CORPUS").expect("T9_CORPUS required");
    let out_dir = std::env::var("T9_OUT").expect("T9_OUT required");
    let train_budget = env_or("T9_TRAIN_TOKENS", 32_768);
    let test_budget = env_or("T9_TEST_TOKENS", 32_768);
    let doc_max = env_or("T9_DOC_MAX_TOKENS", 512);
    let gram_threads = env_or("T9_GRAM_THREADS", 8);
    let layers: Vec<usize> = std::env::var("T9_LAYERS")
        .unwrap_or_else(|_| "0,3".into())
        .split(',')
        .map(|s| s.trim().parse().expect("layer index"))
        .collect();

    std::fs::create_dir_all(&out_dir).expect("mkdir out");
    let corpus = std::fs::read_to_string(&corpus_path).expect("read corpus");
    let docs = split_documents(&corpus);
    eprintln!("[t9-capture] corpus: {} documents", docs.len());

    let g = GgufFile::open(&model_path).expect("open gguf");
    let m = Model::from_gguf(&g).expect("model");
    let arch = m.arch;
    let h_dim = arch.hidden_size as usize;
    let f_dim = arch.intermediate_size as usize;
    let tok = Tokenizer::from_gguf(&g).expect("tokenizer");
    let ctx = MetalContext::new().expect("metal");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal model");
    let mf = MetalForward::new(&ctx, &mm);
    eprintln!(
        "[t9-capture] model {} h={h_dim} f={f_dim} layers={:?}",
        model_path, layers
    );

    // Capture buffers + install.
    let mut slots = Vec::new();
    let mut bufs: Vec<(usize, MetalTensor, MetalTensor)> = Vec::new();
    for &l in &layers {
        let hb = MetalTensor::zeros_f32(&ctx, vec![h_dim as u64]).expect("hbuf");
        let ib = MetalTensor::zeros_f32(&ctx, vec![f_dim as u64]).expect("ibuf");
        slots.push((l, hb.clone(), ib.clone()));
        bufs.push((l, hb, ib));
    }
    t9_ffn_capture_install(slots);

    // Per (layer, space) train accumulators + test writers.
    let mut grams: Vec<(String, GramAccumulator)> = Vec::new();
    let mut testers: Vec<(String, std::io::BufWriter<std::fs::File>)> = Vec::new();
    for &l in &layers {
        for (space, d) in [("h", h_dim), ("inner", f_dim)] {
            let name = format!("blk{l}_{space}");
            grams.push((name.clone(), GramAccumulator::new(d)));
            let f = std::fs::File::create(format!("{out_dir}/test_{name}.f32")).expect("test file");
            testers.push((name, std::io::BufWriter::new(f)));
        }
    }
    // Batch staging for the Gram updates.
    let mut stage: Vec<Vec<f32>> = grams.iter().map(|_| Vec::new()).collect();
    let stage_rows = 256usize;

    let mut train_tokens = 0usize;
    let mut test_tokens = 0usize;
    let mut train_docs = 0usize;
    let mut test_docs = 0usize;
    let t0 = std::time::Instant::now();

    'docs: for (di, doc) in docs.iter().enumerate() {
        let in_train = train_tokens < train_budget;
        if !in_train && test_tokens >= test_budget {
            break 'docs;
        }
        let ids = tok.encode(doc, false).expect("tokenize");
        if ids.len() < 64 {
            continue;
        }
        let ids = &ids[..ids.len().min(doc_max)];
        let mut session = MetalSession::fresh(&ctx, &mm, doc_max + 8).expect("session");
        for (p, &t) in ids.iter().enumerate() {
            t9_ffn_capture_reset_token();
            let _ = mf.single_token(t, p as u32, &mut session).expect("fw");
            let mut k = 0usize;
            for (_, hb, ib) in &bufs {
                let hv = read_f32_tensor(hb);
                let iv = read_f32_tensor(ib);
                if in_train {
                    stage[k].extend_from_slice(&hv);
                    stage[k + 1].extend_from_slice(&iv);
                } else {
                    let hbytes: Vec<u8> = hv.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let ibytes: Vec<u8> = iv.iter().flat_map(|v| v.to_le_bytes()).collect();
                    testers[k].1.write_all(&hbytes).expect("w");
                    testers[k + 1].1.write_all(&ibytes).expect("w");
                }
                k += 2;
            }
            if in_train {
                train_tokens += 1;
                for (si, st) in stage.iter_mut().enumerate() {
                    let d = grams[si].1.d;
                    if st.len() >= stage_rows * d {
                        grams[si].1.add_batch(st, gram_threads);
                        st.clear();
                    }
                }
            } else {
                test_tokens += 1;
                if test_tokens >= test_budget {
                    break;
                }
            }
        }
        if in_train {
            train_docs += 1;
        } else {
            test_docs += 1;
        }
        if di % 50 == 0 {
            eprintln!(
                "[t9-capture] doc {di}: train {train_tokens}/{train_budget} \
                 test {test_tokens}/{test_budget} ({:.1}s)",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    // Flush stages.
    for (si, st) in stage.iter_mut().enumerate() {
        if !st.is_empty() {
            grams[si].1.add_batch(st, gram_threads);
            st.clear();
        }
    }
    t9_ffn_capture_uninstall();

    for (name, acc) in &grams {
        let path = format!("{out_dir}/gram_{name}.f64");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("gram file"));
        for v in &acc.h {
            f.write_all(&v.to_le_bytes()).expect("w");
        }
        eprintln!(
            "[t9-capture] {name}: d={} n_samples={} -> {path}",
            acc.d, acc.n_samples
        );
    }
    for (_, w) in testers.iter_mut() {
        w.flush().expect("flush");
    }
    let manifest = format!(
        "{{\"model\":{:?},\"corpus\":{:?},\"layers\":{:?},\"train_tokens\":{},\
         \"test_tokens\":{},\"train_docs\":{},\"test_docs\":{},\"doc_max\":{},\
         \"h_dim\":{},\"f_dim\":{},\"wall_s\":{:.1}}}",
        model_path,
        corpus_path,
        layers,
        train_tokens,
        test_tokens,
        train_docs,
        test_docs,
        doc_max,
        h_dim,
        f_dim,
        t0.elapsed().as_secs_f64()
    );
    std::fs::write(format!("{out_dir}/manifest.json"), &manifest).expect("manifest");
    eprintln!("[t9-capture] done: {manifest}");
}
