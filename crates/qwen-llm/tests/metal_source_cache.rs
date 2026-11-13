#[path = "../metal_source_cache.rs"]
mod metal_source_cache;

use metal_source_cache::source_cache_key;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> io::Result<Self> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "metal-include-cache-{}-{nanos}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(path.join("nested"))?;
        Ok(Self(path))
    }
    fn key(&self) -> io::Result<metal_source_cache::SourceCacheKey> {
        source_cache_key(
            &self.0.join("source.metal"),
            "metal4.0",
            "26.0",
            "test-metal",
            "26.0",
        )
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn transitive_header_edits_invalidate_air_even_with_unchanged_mtime() -> io::Result<()> {
    let fixture = Fixture::new()?;
    let source = "#include <metal_stdlib>\n#include \"outer.metalh\"\n";
    std::fs::write(fixture.0.join("source.metal"), source)?;
    std::fs::write(
        fixture.0.join("outer.metalh"),
        "# include \"nested/inner.h\"\n",
    )?;
    let inner = fixture.0.join("nested/inner.h");
    std::fs::write(&inner, "constant int value = 1;\n")?;
    let before = fixture.key()?;
    assert_eq!(before.dependencies.len(), 3);
    std::fs::write(fixture.0.join("unrelated.h"), "unreferenced\n")?;
    assert_eq!(before.hash, fixture.key()?.hash);
    let modified = std::fs::metadata(&inner)?.modified()?;
    std::fs::write(&inner, "constant int value = 2;\n")?;
    std::fs::File::options()
        .write(true)
        .open(&inner)?
        .set_modified(modified)?;
    assert_eq!(std::fs::metadata(&inner)?.modified()?, modified);
    assert_eq!(
        std::fs::read_to_string(fixture.0.join("source.metal"))?,
        source
    );
    assert_ne!(before.hash, fixture.key()?.hash);
    Ok(())
}

#[test]
fn include_cycles_are_bounded_and_unresolved_dependencies_refuse_cache_keys() -> io::Result<()> {
    let fixture = Fixture::new()?;
    std::fs::write(
        fixture.0.join("source.metal"),
        "#pragma once\n#include \"nested/inner.h\"\n",
    )?;
    std::fs::write(
        fixture.0.join("nested/inner.h"),
        "#pragma once\n#include \"../source.metal\"\n",
    )?;
    assert_eq!(fixture.key()?.dependencies.len(), 2);
    std::fs::write(fixture.0.join("source.metal"), "#include \"missing.h\"\n")?;
    assert_eq!(fixture.key().err().unwrap().kind(), io::ErrorKind::NotFound);
    std::fs::write(fixture.0.join("source.metal"), "#include HEADER_MACRO\n")?;
    assert_eq!(
        fixture.key().err().unwrap().kind(),
        io::ErrorKind::InvalidData
    );
    Ok(())
}

#[test]
fn real_kernel_tiers_track_the_sorted_expert_header_without_a_metal_compiler() -> io::Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    let mut found_sorted_header = false;
    for tier in ["", "tensor", "tensor41", "research"] {
        let dir = root.join(tier);
        if !dir.exists() {
            continue;
        }
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("metal") {
                continue;
            }
            let key = source_cache_key(
                &path,
                "test-language",
                "test-target",
                "test-metal",
                "test-sdk",
            )?;
            found_sorted_header |= key.dependencies.iter().any(|p| {
                p.file_name().and_then(|s| s.to_str()) == Some("v41_sorted_experts.metalh")
            });
        }
    }
    assert!(found_sorted_header);
    Ok(())
}
