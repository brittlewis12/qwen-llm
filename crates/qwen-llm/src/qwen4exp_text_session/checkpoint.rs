use super::*;

pub(crate) struct Checkpoint {
    owner: usize,
    length: usize,
    history: PleHistory,
    storage: Vec<Vec<u8>>,
}

impl Qwen4ExpTextSessionMetalWorkspace {
    fn checkpoint_tensors(&self) -> Vec<MetalTensor> {
        let mut tensors = self.persistent_state_tensors();
        tensors.push(self.logits.clone());
        tensors.push(self.hyper_residual.clone());
        tensors
    }

    pub(crate) fn checkpoint_for_tests(&self) -> Checkpoint {
        self.require_idle().unwrap();
        assert!(self.pending_length.is_none() && !self.state_poisoned && !self.encode_failed);
        assert!(self.logits_ready);
        assert!(
            self.qsa_committed_lengths()
                .iter()
                .all(|(_, n)| *n == self.committed_length)
        );
        let history = self.zero_one.checkpoint_history_for_tests();
        assert_eq!(history.next_position(), Some(self.committed_length as u64));
        Checkpoint {
            owner: self.logits.buffer.contents().as_ptr() as usize,
            length: self.committed_length,
            history,
            storage: self
                .checkpoint_tensors()
                .iter()
                .map(|tensor| unsafe {
                    let source = tensor
                        .buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(tensor.offset as usize);
                    std::slice::from_raw_parts(source, tensor.n_bytes() as usize).to_vec()
                })
                .collect(),
        }
    }

    pub(crate) fn restore_checkpoint_for_tests(&mut self, checkpoint: &Checkpoint) {
        self.require_idle().unwrap();
        assert!(self.pending_length.is_none() && !self.state_poisoned && !self.encode_failed);
        assert_eq!(
            checkpoint.owner,
            self.logits.buffer.contents().as_ptr() as usize
        );
        assert!(checkpoint.length <= self.committed_length);
        let tensors = self.checkpoint_tensors();
        assert_eq!(tensors.len(), checkpoint.storage.len());
        for (tensor, bytes) in tensors.iter().zip(&checkpoint.storage) {
            assert!(tensor.is_writable());
            assert_eq!(tensor.n_bytes() as usize, bytes.len());
        }
        self.zero_one.restore_history_for_tests(&checkpoint.history);
        for block in &mut self.post_ple {
            block.restore_mixer_length_for_tests(checkpoint.length);
        }
        for (tensor, bytes) in tensors.iter().zip(&checkpoint.storage) {
            unsafe {
                let destination = tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
            }
        }
        self.committed_length = checkpoint.length;
        self.logits_ready = true;
    }

    pub(crate) fn final_hyper_for_tests(&self) -> Vec<f32> {
        self.require_idle().unwrap();
        assert!(self.logits_ready && !self.state_poisoned);
        unsafe {
            let source = self
                .hyper_residual
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(self.hyper_residual.offset as usize)
                .cast::<f32>();
            std::slice::from_raw_parts(source, self.hyper_residual.n_elements() as usize).to_vec()
        }
    }
}
