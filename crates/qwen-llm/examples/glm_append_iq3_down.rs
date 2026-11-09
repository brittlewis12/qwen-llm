use qwen_llm::{gguf::GgufFile, metal::*, tokenizer::Tokenizer};
use std::time::Instant;
fn emit(v: serde_json::Value) {
    println!("{v}");
}
fn pressure() -> i32 {
    let x = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .unwrap();
    String::from_utf8(x.stdout)
        .unwrap()
        .split_whitespace()
        .last()
        .unwrap()
        .parse()
        .unwrap()
}
fn compilers() -> String {
    let x = std::process::Command::new("ps")
        .args(["-axo", "comm="])
        .output()
        .unwrap();
    String::from_utf8(x.stdout)
        .unwrap()
        .lines()
        .filter(|s| {
            [
                "rustc", "cargo", "metal", "metalfe", "metallib", "clang", "clang++", "air-lld",
            ]
            .contains(&s.rsplit('/').next().unwrap_or(s))
        })
        .map(|s| s.rsplit('/').next().unwrap_or(s))
        .collect::<Vec<_>>()
        .join(";")
}
fn native(corpus: &str) -> anyhow::Result<()> {
    use qwen_llm::{
        glm_append_price as probe,
        glm5_next_metal::{Glm5NextSession, Glm5NextWeights},
    };
    let c = MetalContext::new()?;
    let path = qwen_llm::test_fixtures::GLM53_FLASH_UD_IQ3_XXS
        .path_or_skip()
        .ok_or_else(|| anyhow::anyhow!("resolve GLM via QWEN_MODEL_SEARCH_PATH"))?;
    let g = GgufFile::open(path)?;
    let t = Tokenizer::from_gguf(&g)?;
    let text = std::fs::read_to_string(corpus)?
        .replace("<|", "‹|")
        .replace("<｜", "‹｜");
    let ids: Vec<_> = t
        .encode(&text, false)?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let depth = 65536;
    assert!(ids.len() > depth + 4096);
    let w = Glm5NextWeights::load(&c, &g)?;
    probe::down_mode(0);
    let mut s = Glm5NextSession::with_prefill_rows(&c, &w, 98304, 2048)?;
    for part in ids[..depth].chunks(2048) {
        while pressure() > 1 {
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
        s.prefill_packed(&c, part)?;
    }
    let snap = s.capture_snapshot()?;
    emit(serde_json::json!({"kind":"header","depth":depth,"ledger":s.ledger().terms()}));
    let modes = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "2".into())
        .split(',')
        .map(|s| s.parse::<u8>().unwrap())
        .collect::<Vec<_>>();
    for mode in modes {
        for n in [400, 2000] {
            for (rep, on) in [
                false, true, false, true, true, false, true, false, false, true,
            ]
            .into_iter()
            .enumerate()
            {
                // mode 0 = force incumbent, 2 = force adaptive, 3 = no
                // override (exercise the production env default).
                match (on, mode) {
                    (true, 3) => probe::down_mode_default(),
                    (true, m) => probe::down_mode(m),
                    (false, _) => probe::down_mode(0),
                }
                s.restore_snapshot(&snap)?;
                while pressure() > 1 {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
                let p0 = pressure();
                let compile0 = compilers();
                probe::reset_hits();
                let start = Instant::now();
                let v = s.prefill_packed(&c, &ids[depth..depth + n])?;
                let ms = start.elapsed().as_secs_f64() * 1e3;
                let hits = probe::hits();
                anyhow::ensure!((hits > 0) == on, "route hits {hits}");
                let p1 = pressure();
                let compile1 = compilers();
                anyhow::ensure!(p0 <= 1 && p1 <= 1, "thermal pressure {p0}/{p1}");
                assert_eq!(s.position(), depth + n);
                assert!(!s.is_poisoned());
                assert!(v.iter().all(|x| x.is_finite()));
                emit(
                    serde_json::json!({"kind":"native","mode":mode,"hits":hits,"rows":n,"on":on,"rep":rep,"warm":rep<2,"ms":ms,"pressure_before":p0,"pressure_after":p1,"compile_before":compile0,"compile_after":compile1,"hash":blake3::hash(bytemuck::cast_slice(&v)).to_hex().to_string()}),
                );
            }
        }
    }
    Ok(())
}
fn log_z(v: &[f32]) -> f64 {
    let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    mx + v.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln()
}
fn quality(root: &str) -> anyhow::Result<()> {
    use qwen_llm::{
        glm_append_price as probe,
        glm5_next_metal::{Glm5NextSession, Glm5NextWeights},
    };
    let c = MetalContext::new()?;
    let model = qwen_llm::test_fixtures::GLM53_FLASH_UD_IQ3_XXS
        .path_or_skip()
        .ok_or_else(|| anyhow::anyhow!("resolve UD-IQ3_XXS with QWEN_MODEL_SEARCH_PATH"))?;
    let gg = GgufFile::open(model)?;
    let tok = Tokenizer::from_gguf(&gg)?;
    let w = Glm5NextWeights::load(&c, &gg)?;
    let root = std::path::Path::new(root);
    let ids = |p: &std::path::Path| -> anyhow::Result<Vec<u32>> {
        Ok(tok
            .encode(&std::fs::read_to_string(p)?, false)?
            .into_iter()
            .map(|x| x as u32)
            .collect())
    };
    let prefix = ids(&root.join("prefix.txt"))?;
    let prefix: Vec<_> = prefix.iter().copied().cycle().take(65536).collect();
    let mut s = Glm5NextSession::with_prefill_rows(&c, &w, 98304, 2048)?;
    probe::down_mode(0);
    for part in prefix.chunks(2048) {
        s.prefill_packed(&c, part)?;
    }
    let snap = s.capture_snapshot()?;
    let ad = evaluate_metal_memory_admission_with_cpu_bytes(
        0,
        256 << 20,
        512 << 20,
        c.memory_signals(),
        true,
    );
    anyhow::ensure!(ad.is_admitted(), "quality reference admission {ad:?}");
    emit(
        serde_json::json!({"kind":"quality_header","depth":65536,"documents":8,"targets":128,"noise":"independent incumbent repeat","widths":[400,2000]}),
    );
    for doc in 0..8 {
        let seq = ids(&root.join(format!("doc{doc}.txt")))?;
        anyhow::ensure!(seq.len() >= 2128);
        for rows in [400usize, 2000] {
            let mut reference = Vec::<Vec<f32>>::new();
            let mut rz = Vec::<f64>::new();
            let mut rn = Vec::<f64>::new();
            for (name, mode) in [("incumbent", 0u8), ("repeat", 0), ("adaptive", 2)] {
                probe::down_mode(mode);
                probe::reset_hits();
                s.restore_snapshot(&snap)?;
                let p0 = pressure();
                let t = Instant::now();
                let mut v = s.prefill_packed(&c, &seq[..rows])?;
                let ms = t.elapsed().as_secs_f64() * 1000.;
                let hits = probe::hits();
                anyhow::ensure!((hits > 0) == (mode == 2), "route hits {hits} mode{mode}");
                let mut nll = Vec::new();
                let mut delta = Vec::new();
                let mut kl = Vec::new();
                let mut bitwise = true;
                for target in 0..128 {
                    anyhow::ensure!(v.iter().all(|v| v.is_finite()));
                    let zv = log_z(&v);
                    let loss = zv - v[seq[rows + target] as usize] as f64;
                    nll.push(loss);
                    if name == "incumbent" {
                        rn.push(loss);
                        rz.push(zv);
                        reference.push(v.clone());
                        delta.push(0.);
                        kl.push(0.);
                    } else {
                        delta.push(loss - rn[target]);
                        let old = &reference[target];
                        kl.push(
                            old.iter()
                                .zip(&v)
                                .map(|(&a, &b)| {
                                    let lp = a as f64 - rz[target];
                                    lp.exp() * (lp - (b as f64 - zv))
                                })
                                .sum::<f64>(),
                        );
                        bitwise &= old.iter().zip(&v).all(|(a, b)| a.to_bits() == b.to_bits());
                    }
                    if target < 127 {
                        v = s.forward(&c, seq[rows + target])?;
                    }
                }
                emit(
                    serde_json::json!({"kind":"quality","doc":doc,"rows":rows,"mode":name,"nll":nll,"delta":delta,"kl":kl,"bitwise":bitwise,"hits":hits,"prefill_ms_descriptive":ms,"pressure_before":p0,"pressure_after":pressure()}),
                );
            }
        }
    }
    Ok(())
}
fn main() -> anyhow::Result<()> {
    emit(
        serde_json::json!({"kind":"binary","blake3":blake3::hash(&std::fs::read(std::env::current_exe()?)?).to_hex().to_string()}),
    );
    if std::env::args().nth(1).as_deref() == Some("witness") {
        anyhow::bail!("operator witness is in the IQ3 panel probe")
    } else if std::env::args().nth(1).as_deref() == Some("quality") {
        quality(&std::env::args().nth(2).expect("corpus directory"))
    } else {
        native(&std::env::args().nth(1).expect("corpus"))
    }
}
