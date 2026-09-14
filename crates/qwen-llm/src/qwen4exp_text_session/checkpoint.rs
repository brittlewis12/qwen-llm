use super::*;

#[test]
fn split_decode_plan_and_session_custody() {
    let _benchmark_lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    use objc2_metal::MTLCommandQueue;
    let ctx = MetalContext::new().unwrap();
    let geometry =
        Qwen4ExpTextSessionMetalGeometry::from_config(&Qwen4ExpConfig::flash_next_reference(), 4)
            .unwrap();
    let off = Qwen4ExpTextSessionMemoryPlan::for_geometry_with_residency_bytes(&ctx, &geometry, 0)
        .unwrap();
    let plan = Qwen4ExpTextSessionPlan {
        geometry: geometry.clone(),
        memory: off.clone(),
        device_registry_id: ctx.device.registryID(),
    };
    let plan = plan.with_split_decode(&ctx, true).unwrap();
    let plan = plan.with_split_decode(&ctx, true).unwrap();
    let on = plan.memory.clone();
    assert_eq!(on.allocations().len(), off.allocations().len() + 1);
    assert_eq!(
        on.session_logical_bytes() - off.session_logical_bytes(),
        1_585_152
    );
    assert_eq!(
        on.allocations()
            .iter()
            .filter(|a| a.name == "session.qsa_split")
            .count(),
        1
    );
    assert_eq!(plan.with_split_decode(&ctx, false).unwrap().memory, off);
    let admission = on.admission_after_residency(ctx.memory_signals());
    assert!(admission.admitted);
    let mut first =
        Qwen4ExpTextSessionMetalWorkspace::allocate(&ctx, geometry.clone(), on.clone(), admission)
            .unwrap();
    let mut second = Qwen4ExpTextSessionMetalWorkspace::allocate(
        &ctx,
        geometry,
        on.clone(),
        on.admission_after_residency(ctx.memory_signals()),
    )
    .unwrap();
    let identity = |w: &Qwen4ExpTextSessionMetalWorkspace| {
        w.split_decode_scratch
            .as_ref()
            .unwrap()
            .buffer
            .contents()
            .as_ptr() as usize
    };
    let ids = |w: &Qwen4ExpTextSessionMetalWorkspace| {
        w.post_ple
            .iter()
            .filter_map(|b| b.split_scratch_identity())
            .collect::<Vec<_>>()
    };
    assert_ne!(identity(&first), identity(&second));
    assert_eq!(ids(&first), vec![identity(&first); 12]);
    assert_eq!(ids(&second), vec![identity(&second); 12]);
    let unchanged = ids(&first);
    first.active_command = Some(ctx.queue.commandBuffer().unwrap());
    assert!(first.set_split_decode_for_tests(&ctx, false).is_err());
    assert_eq!(ids(&first), unchanged);
    first.active_command = None;
    first.pending_length = Some(1);
    assert!(first.set_split_decode_for_tests(&ctx, false).is_err());
    assert_eq!(ids(&first), unchanged);
    first.pending_length = None;
    first.set_split_decode_for_tests(&ctx, false).unwrap();
    assert!(ids(&first).is_empty());
    first.set_split_decode_for_tests(&ctx, true).unwrap();
    assert_eq!(ids(&first), unchanged);
    first.reset().unwrap();
    assert_eq!(ids(&first), unchanged);
    second.reset().unwrap();
    assert_eq!(ids(&second), vec![identity(&second); 12]);
}

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
