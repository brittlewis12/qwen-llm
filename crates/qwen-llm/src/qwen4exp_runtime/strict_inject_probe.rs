use super::*;
use sha2::{Digest, Sha256};

pub(super) fn state(runner: &Qwen4ExpTextRunner<'_, '_, '_>) -> Vec<String> {
    assert!(!runner.workspace.is_poisoned());
    let position = runner.next_position();
    let mut active_qsa = Vec::new();
    for (layer, tensors) in runner.workspace.qsa_persistent_state_tensors() {
        assert_eq!(tensors.len(), 4);
        let ratio = runner
            .weights
            .geometry
            .post_ple()
            .iter()
            .find(|block| block.layer() == layer)
            .unwrap()
            .mixer()
            .qsa()
            .unwrap()
            .compression_ratio();
        for (part, tensor) in tensors.into_iter().enumerate() {
            let bytes = match part {
                0 => tensor.n_bytes() as usize,
                1 => tensor.shape[0] as usize * (position / ratio) * 2,
                2 | 3 => tensor.shape[0] as usize * tensor.shape[1] as usize * position * 2,
                _ => unreachable!(),
            };
            active_qsa.push((tensor, bytes));
        }
    }
    let mut result = vec![
        format!("{}", runner.next_position()),
        format!("{:?}", runner.workspace.qsa_committed_lengths()),
        format!("{:?}", runner.workspace.ple_prior_tokens()),
    ];
    let mut total_bytes = 0;
    for tensor in runner.workspace.persistent_state_tensors() {
        let active_bytes = active_qsa
            .iter()
            .find(|(q, _)| {
                std::ptr::addr_eq(
                    Retained::as_ptr(&q.buffer),
                    Retained::as_ptr(&tensor.buffer),
                ) && q.offset == tensor.offset
            })
            .map_or(tensor.n_bytes() as usize, |(_, bytes)| *bytes);
        assert!(active_bytes <= tensor.n_bytes() as usize);
        total_bytes += active_bytes;
        // Called only after synchronous commands have completed and released.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize),
                active_bytes,
            )
        };
        result.push(format!("{:x}", Sha256::digest(bytes)));
    }
    assert_eq!(total_bytes as u64, runner.snapshot_bytes().unwrap());
    result
}
