use std::time::Instant;

use qwen_llm::{
    gguf::GgufFile,
    loader::Model,
    metal_forward::MetalModel,
    test_fixtures::{self, metal_context_or_skip},
};

#[test]
fn a3b_load_wall_ms() {
    let Some(path) = test_fixtures::A3B_Q4_K_M.path_or_skip() else {
        return;
    };
    let Some(ctx) = metal_context_or_skip() else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open A3B GGUF");
    let model = Model::from_gguf(&gguf).expect("bind A3B model descriptors");
    let started = Instant::now();
    let _metal_model = MetalModel::load(&ctx, &gguf, &model).expect("load A3B weights");
    println!("LOAD_WALL_MS={:.3}", started.elapsed().as_secs_f64() * 1e3);
}
