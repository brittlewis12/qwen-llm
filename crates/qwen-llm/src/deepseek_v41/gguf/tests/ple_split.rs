use super::*;
use std::io::{Seek, SeekFrom};

struct SplitFixture(std::path::PathBuf);

impl SplitFixture {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "qwen-v41-ple-split-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn write(&self, name: &str, mut file: std::fs::File) {
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut output = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(self.0.join(name))
            .unwrap();
        std::io::copy(&mut file, &mut output).unwrap();
    }
}

impl Drop for SplitFixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn split_keys(index: u32, count: u32, total: u32) -> Vec<(&'static str, Value<'static>)> {
    vec![
        ("split.no", Value::U32(index)),
        ("split.count", Value::U32(count)),
        ("split.tensors.count", Value::U32(total)),
    ]
}

#[test]
fn mixed_quant_table_shard_uses_its_own_descriptor_without_weakening_split_checks() {
    for case in [
        "q5",
        "q6",
        "index",
        "count",
        "total",
        "missing",
        "duplicate",
        "encoding",
    ] {
        let fixture = SplitFixture::new();
        let (mut metadata, mut tensors) = ple_engram::parts(true, true, GgmlType::Q5_K);
        let total = tensors.len() as u32;
        let table = tensors.pop().unwrap();
        assert_eq!(table.0, "per_layer_token_embd.weight");
        metadata.extend(split_keys(0, 3, total));
        if case == "encoding" {
            metadata.push(("deepseek41.engram.encoding", Value::Str("Q5_K")));
        }
        fixture.write(
            "model-00001-of-00003.gguf",
            fixture_file(metadata, vec![], true, |_, _, _| None),
        );
        fixture.write(
            "model-00002-of-00003.gguf",
            fixture_file(split_keys(1, 3, total), tensors.clone(), true, |_, _, _| {
                None
            }),
        );
        let dtype = if case == "q5" {
            GgmlType::Q5_K
        } else {
            GgmlType::Q6_K
        };
        let donor = if case == "missing" {
            vec![]
        } else if case == "duplicate" {
            vec![tensors[0].clone()]
        } else {
            vec![(table.0, table.1, dtype)]
        };
        fixture.write(
            "donor.gguf",
            fixture_file(
                split_keys(
                    if case == "index" { 1 } else { 2 },
                    if case == "count" { 4 } else { 3 },
                    if case == "total" { total + 1 } else { total },
                ),
                donor,
                false,
                |name, shape, kind| {
                    engram::is_table(name)
                        .then(|| ple_engram::packed_rows(kind, 0, shape[1] as usize))
                },
            ),
        );
        std::fs::hard_link(
            fixture.0.join("donor.gguf"),
            fixture.0.join("model-00003-of-00003.gguf"),
        )
        .unwrap();
        let opened = GgufFile::open(fixture.0.join("model-00001-of-00003.gguf"));
        if ["index", "count", "total", "missing", "duplicate"].contains(&case) {
            assert!(opened.is_err(), "{case}");
            continue;
        }
        let g = Arc::new(opened.unwrap());
        if case == "encoding" {
            assert!(plan(&g).unwrap_err().to_string().contains("conflicts"));
            continue;
        }
        assert_eq!(g.find("per_layer_token_embd.weight").unwrap().dtype, dtype);
        assert_eq!(g.find("per_layer_token_embd.weight").unwrap().shard_idx, 2);
        assert_eq!(
            plan(&g).unwrap().bytes.disk_engram,
            120 * dtype.storage_layout().unwrap().1
        );
        let disk = DiskEngram::new(g).unwrap();
        let expected = if dtype == GgmlType::Q5_K {
            17.0 * 49.0 / 8.0
        } else {
            49.0 / 8.0
        };
        assert_eq!(disk.read_rows(1, &[0]).unwrap()[0], expected);
    }
}
