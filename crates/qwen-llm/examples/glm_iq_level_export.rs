//! Extract selected native expert matrices from the stock GLM GGUF for the
//! limiter-harness level-code experiment. It copies only the requested expert
//! slices; no model/runtime or Metal buffers are created.
use qwen_llm::{gguf::GgufFile, tensor::GgmlType};
use std::{env, error::Error, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err("usage: glm_iq_level_export <first-shard.gguf> <out-dir> <layer:expert-ids-csv;layer:expert-ids-csv>".into());
    }
    let model = &args[1];
    let out = PathBuf::from(&args[2]);
    let selections: Vec<(usize, Vec<usize>)> = args[3]
        .split(';')
        .map(|part| {
            let (layer, experts) = part.split_once(':').ok_or("selection needs layer:expert,expert")?;
            let experts = experts.split(',').map(str::parse).collect::<Result<Vec<usize>, _>>()?;
            Ok((layer.parse::<usize>()?, experts))
        })
        .collect::<Result<_, Box<dyn Error>>>()?;
    fs::create_dir_all(&out)?;
    let gguf = GgufFile::open(model)?;
    if gguf.architecture().as_deref() != Some("glm5-next") {
        return Err(format!("unexpected GGUF architecture {:?}", gguf.architecture()).into());
    }
    let roles = [
        ("gate", "ffn_gate_exps.weight", GgmlType::IQ2_S, 4096usize, 2048usize),
        ("up", "ffn_up_exps.weight", GgmlType::IQ2_S, 4096usize, 2048usize),
        ("down", "ffn_down_exps.weight", GgmlType::IQ3_S, 2048usize, 4096usize),
    ];
    let mut manifest = String::from("layer,expert,role,dtype,k,n,bytes,path\n");
    for (layer, experts) in selections {
      for expert in experts {
        for (role, suffix, expected, k, n) in roles {
            let name = format!("blk.{layer}.{suffix}");
            let desc = gguf.find(&name).ok_or_else(|| format!("missing tensor {name}"))?;
            if desc.dtype != expected || desc.shape.as_slice() != [k as u64, n as u64, 288] {
                return Err(format!("{name}: dtype/shape {:?}/{:?}, expected {:?}/[{k},{n},288]", desc.dtype, desc.shape, expected).into());
            }
            if expert >= 288 || desc.n_bytes % 288 != 0 {
                return Err(format!("invalid expert slice {expert} for {name}").into());
            }
            let bytes = usize::try_from(desc.n_bytes / 288)?;
            let offset = desc.data_offset.checked_add((expert * bytes) as u64).ok_or("offset overflow")?;
            let mut raw = vec![0u8; bytes];
            gguf.read_shard_exact_at(desc.shard_idx, offset, &mut raw)?;
            let filename = format!("layer{layer}-expert{expert}-{role}.iq{}.raw", if expected == GgmlType::IQ2_S { "2s" } else { "3s" });
            fs::write(out.join(&filename), &raw)?;
            manifest.push_str(&format!("{layer},{expert},{role},{expected:?},{k},{n},{bytes},{}\n", filename));
            println!("export layer={layer} expert={expert} role={role} dtype={expected:?} k={k} n={n} bytes={bytes} sha_pending");
        }
      }
    }
    fs::write(out.join("manifest.csv"), manifest)?;
    println!("shards={} tensors={} output={}", gguf.shards.len(), gguf.tensors.len(), out.display());
    Ok(())
}
