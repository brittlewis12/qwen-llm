use anyhow::{Context, Result, ensure};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use qwen_llm::metal::*;
use qwen_llm::metal_dflash::{MetalDFlashLayerMajorScratch, prefill_tokens_with_multi_hidden};
use qwen_llm::metal_forward::{MetalForward, MetalModel, MetalSession, encode_mat_vec_dispatch};
use qwen_llm::qwen4exp::Qwen4ExpConfig;
use qwen_llm::qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity};
use qwen_llm::{gguf::GgufFile, tokenizer::Tokenizer};
use sha2::{Digest, Sha256};
use std::io::{BufReader, BufWriter, Read, Write};

struct Score {
    writer: Option<BufWriter<std::fs::File>>,
    reader: Option<BufReader<std::fs::File>>,
    refrow: Vec<f32>,
    n: usize,
    nll: f64,
    refnll: f64,
    kls: Vec<f64>,
    agree: usize,
}
impl Score {
    fn new(path: &str, on: bool, ids: &[i32], v: usize) -> Result<Self> {
        let hash = Sha256::digest(bytemuck::cast_slice(ids));
        let mut header = Vec::from(&b"GDNQUAL1"[..]);
        header.extend_from_slice(&(ids.len() as u64).to_le_bytes());
        header.extend_from_slice(&(v as u64).to_le_bytes());
        header.extend_from_slice(&hash);
        let (writer, reader) = if on {
            let mut r = BufReader::new(std::fs::File::open(path)?);
            let mut old = vec![0; header.len()];
            r.read_exact(&mut old)?;
            ensure!(old == header, "baseline tokens/geometry differ");
            (None, Some(r))
        } else {
            let mut w = BufWriter::new(std::fs::File::create(path)?);
            w.write_all(&header)?;
            (Some(w), None)
        };
        eprintln!("token_sha256={hash:x}");
        Ok(Self {
            writer,
            reader,
            refrow: vec![0.0; v],
            n: 0,
            nll: 0.0,
            refnll: 0.0,
            kls: vec![],
            agree: 0,
        })
    }
    fn row(&mut self, x: &[f32], target: usize) -> Result<()> {
        ensure!(x.iter().all(|x| x.is_finite()), "nonfinite logits");
        fn lse(x: &[f32]) -> f64 {
            let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            m + x.iter().map(|v| (*v as f64 - m).exp()).sum::<f64>().ln()
        }
        let z = lse(x);
        self.nll += z - x[target] as f64;
        if let Some(w) = self.writer.as_mut() {
            w.write_all(bytemuck::cast_slice(x))?;
        }
        if let Some(r) = self.reader.as_mut() {
            r.read_exact(bytemuck::cast_slice_mut(&mut self.refrow))?;
            ensure!(self.refrow.iter().all(|v| v.is_finite()));
            let pz = lse(&self.refrow);
            self.refnll += pz - self.refrow[target] as f64;
            let kl = self
                .refrow
                .iter()
                .zip(x)
                .map(|(p, q)| {
                    let lp = *p as f64 - pz;
                    lp.exp() * (lp - (*q as f64 - z))
                })
                .sum::<f64>();
            self.kls.push(kl.max(0.0));
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                    .unwrap()
                    .0
            };
            self.agree += usize::from(argmax(&self.refrow) == argmax(x));
        }
        self.n += 1;
        Ok(())
    }
    fn finish(mut self) -> Result<()> {
        if let Some(w) = self.writer.as_mut() {
            w.flush()?;
        }
        if let Some(r) = self.reader.as_mut() {
            let mut b = [0];
            ensure!(r.read(&mut b)? == 0, "baseline has trailing rows");
        }
        let n = self.n as f64;
        let kl = self.kls.iter().sum::<f64>() / n;
        self.kls.sort_by(f64::total_cmp);
        let p99 = self
            .kls
            .get((self.kls.len() as f64 * 0.99).floor() as usize)
            .copied()
            .unwrap_or(0.0);
        let result = serde_json::json!({"rows":self.n,"nll":self.nll/n,"nll_off":self.refnll/n,"dnll":(self.nll-self.refnll)/n,"mean_kl":kl,"p99_kl":p99,"max_kl":self.kls.last(),"argmax_agreement":self.agree,"on":self.reader.is_some()});
        println!("{result}");
        if self.reader.is_some() {
            ensure!(
                ((self.nll - self.refnll) / n).abs() <= 0.005 && kl <= 1e-2,
                "distribution gate failed"
            );
        }
        Ok(())
    }
}
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 7, "family model text baseline rows on");
    let family = &args[1];
    let gguf = GgufFile::open(&args[2])?;
    let text = std::fs::read_to_string(&args[3])?;
    let n: usize = args[5].parse()?;
    let on = args[6] == "on";
    ensure!(std::env::var("QWEN_GDN_CHUNKED_PREFILL").as_deref() == Ok(if on { "1" } else { "0" }));
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let ids = tokenizer.encode(&text, false)?;
    ensure!(ids.len() > n, "not enough text tokens: {}", ids.len());
    let ids = &ids[..n + 1];
    let ctx = MetalContext::new()?;
    if family == "fn" {
        let cfg = Qwen4ExpConfig::from_gguf(&gguf)?;
        let cap = Qwen4ExpSessionCapacity::for_forward_limit(&cfg, n + 1)?;
        let mut loaded = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, cap, n)?;
        ensure!(
            loaded.packed_prefill_capacity() == Some(2048),
            "unexpected chunk {:?}",
            loaded.packed_prefill_capacity()
        );
        let mut runner = loaded.create_runner(&ctx)?;
        let mut captures = runner.allocate_capture_bank(vec![cfg.layer_count - 1], 2048)?;
        let mut logits = runner.allocate_row_logits_bank(2048)?;
        let mut score = Score::new(&args[4], on, ids, cfg.vocab_size as usize)?;
        for start in (0..n).step_by(2048) {
            let end = (start + 2048).min(n);
            let tokens: Vec<u32> = ids[start..end].iter().map(|x| *x as u32).collect();
            let output = runner.forward_rows_with_captures_and_logits(
                &tokens,
                &mut captures,
                &mut logits,
            )?;
            for j in 0..end - start {
                score.row(
                    output.logits().row(j).context("logit row")?,
                    ids[start + j + 1] as usize,
                )?;
            }
        }
        score.finish()?;
    } else {
        let model = qwen_llm::loader::Model::from_gguf(&gguf)?;
        let weights = MetalModel::load(&ctx, &gguf, &model)?;
        let forward = MetalForward::new(&ctx, &weights);
        let mut session = MetalSession::fresh(&ctx, &weights, n + 1)?;
        let mut scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
            &ctx,
            &weights,
            2048,
            n + 1,
        )?;
        let h = model.arch.hidden_size as usize;
        let v = model.arch.vocab_size as usize;
        let hidden = MetalTensor::zeros_f32(&ctx, vec![n as u64, h as u64])?;
        let endpoint = prefill_tokens_with_multi_hidden(
            &forward,
            &ids[..n],
            0,
            &mut session,
            &mut scratch,
            &[model.arch.n_layer - 1],
            Some(&hidden),
        )?;
        let bank = MetalTensor::zeros_f32(&ctx, vec![256, v as u64])?;
        let norm = MetalTensor::zeros_f32(&ctx, vec![256, h as u64])?;
        let mut score = Score::new(&args[4], on, ids, v)?;
        for start in (0..n).step_by(256) {
            let count = (n - start).min(256);
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            for j in 0..count {
                let x = hidden.view_subrange(((start + j) * h) as u64, vec![h as u64]);
                let nr = norm.view_subrange((j * h) as u64, vec![h as u64]);
                let y = bank.view_subrange((j * v) as u64, vec![v as u64]);
                encode_rms_norm_mul_f32(&ctx, &enc, &x, &weights.output_norm, &nr, 1e-6)?;
                encode_mat_vec_dispatch(&ctx, &enc, &weights.lm_head, &nr, &y, h, v)?;
            }
            enc.end();
            cmd.commit();
            wait_completed(&cmd)?;
            let values = unsafe {
                std::slice::from_raw_parts(bank.buffer.contents().as_ptr().cast::<f32>(), count * v)
            };
            if start + count == n {
                ensure!(
                    values[(count - 1) * v..]
                        .iter()
                        .zip(&endpoint)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "native endpoint mismatch"
                );
            }
            for j in 0..count {
                score.row(&values[j * v..(j + 1) * v], ids[start + j + 1] as usize)?;
            }
        }
        score.finish()?;
    }
    Ok(())
}
