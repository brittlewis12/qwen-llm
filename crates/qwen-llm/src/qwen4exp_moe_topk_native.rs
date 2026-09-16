use super::*;

#[derive(Clone)]
pub(crate) struct Bank {
    logits: MetalTensor,
    ids: MetalTensor,
    weights: MetalTensor,
    rows: usize,
}

impl Bank {
    pub(crate) fn new(ctx: &MetalContext, rows: usize) -> Self {
        Self {
            logits: MetalTensor::zeros_f32(ctx, vec![(rows * 512) as u64]).unwrap(),
            ids: MetalTensor::zeros_i32(ctx, vec![(rows * 10) as u64]).unwrap(),
            weights: MetalTensor::zeros_f32(ctx, vec![(rows * 10) as u64]).unwrap(),
            rows,
        }
    }

    pub(crate) fn save(&self, path: &std::path::Path, name: &str) {
        for (label, t) in [
            ("logits", &self.logits),
            ("ids", &self.ids),
            ("weights", &self.weights),
        ] {
            std::fs::write(path.join(format!("{name}-routes-{label}.bin")), bytes(t)).unwrap();
        }
    }

    pub(crate) fn assert_equivalent(&self, other: &Self) {
        assert_eq!(self.rows, other.rows);
        for (a, b) in [
            (&self.logits, &other.logits),
            (&self.ids, &other.ids),
            (&self.weights, &other.weights),
        ] {
            assert_eq!(bytes(a), bytes(b));
        }
        assert_finite(&self.logits);
        assert_finite(&self.weights);
        let values: Vec<f32> = bytes(&self.logits)
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let ids: Vec<i32> = bytes(&self.ids)
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        for (scores, selected) in values.chunks_exact(512).zip(ids.chunks_exact(10)) {
            let mut ordered: Vec<usize> = (0..512).collect();
            ordered.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
            assert_eq!(
                selected,
                ordered[..10].iter().map(|&i| i as i32).collect::<Vec<_>>()
            );
        }
    }
}

struct Mode {
    parallel: bool,
    bank: Option<Bank>,
    calls: usize,
}
thread_local! {static MODE:RefCell<Option<Mode>>=const{RefCell::new(None)};}

pub(crate) fn with_mode<T>(
    parallel: bool,
    bank: Option<&Bank>,
    f: impl FnOnce() -> T,
) -> (T, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            MODE.with(|m| {
                m.borrow_mut().take();
            });
        }
    }
    MODE.with(|m| {
        assert!(m.borrow().is_none());
        *m.borrow_mut() = Some(Mode {
            parallel,
            bank: bank.cloned(),
            calls: 0,
        });
    });
    let _reset = Reset;
    let result = f();
    let mode = MODE.with(|m| m.borrow_mut().take().unwrap());
    if let Some(bank) = mode.bank {
        assert_eq!(mode.calls, bank.rows);
    }
    (result, mode.calls)
}

fn view(t: &MetalTensor, row: usize, width: usize) -> MetalTensor {
    let mut v = t.clone();
    v.offset += (row * width * 4) as u64;
    v.shape = vec![width as u64];
    v
}

pub(crate) fn preflight(ctx: &MetalContext) {
    let p = ctx
        .pipeline("kernel_topk_logits_softmax_parallel_f32")
        .unwrap();
    assert!(p.maxTotalThreadsPerThreadgroup() >= 512);
    assert!(p.staticThreadgroupMemoryLength() + 6144 <= ctx.device.maxThreadgroupMemoryLength());
}

pub(in crate::qwen4exp_moe) fn encode_if_active(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    buffers: Qwen4ExpMoeSingletonBuffers<'_>,
    g: Qwen4ExpMoeMetalGeometry,
) -> bool {
    MODE.with(|m| {
        let mut m = m.borrow_mut();
        let Some(mode) = m.as_mut() else {
            return false;
        };
        assert_eq!((g.expert_count, g.experts_per_token), (512, 10));
        let row = mode.calls;
        mode.calls += 1;
        if let Some(bank) = &mode.bank {
            assert!(row < bank.rows);
            encode_copy_offset_f32(
                ctx,
                enc,
                buffers.router_logits,
                0,
                &view(&bank.logits, row, 512),
                512,
            )
            .unwrap();
        }
        topk_screen::select(
            ctx,
            enc,
            buffers.router_logits,
            buffers.topk_ids,
            buffers.topk_weights,
            mode.parallel,
        );
        if let Some(bank) = &mode.bank {
            encode_copy_offset_i32(ctx, enc, buffers.topk_ids, 0, &view(&bank.ids, row, 10), 10)
                .unwrap();
            encode_copy_offset_f32(
                ctx,
                enc,
                buffers.topk_weights,
                0,
                &view(&bank.weights, row, 10),
                10,
            )
            .unwrap();
        }
        true
    })
}
