//! Experimental owned active banks. No decoded/requantized weights.
use super::*;

pub(super) struct Staging {
    banks: [MetalTensor; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_native_production_width_bit_parity() {
        let ctx = MetalContext::new().unwrap();
        let c = Config {
            experts: 8,
            ..Config::default()
        };
        let make = |input: usize, output: usize, seed: usize| {
            let mut bytes = vec![0u8; input / 32 * output * c.experts * 17];
            for (i, block) in bytes.chunks_exact_mut(17).enumerate() {
                block[0] = 120 + (i % 5) as u8;
                for (j, byte) in block[1..].iter_mut().enumerate() {
                    *byte = (i.wrapping_mul(7) + j * 13 + seed) as u8;
                }
            }
            MetalTensor::from_bytes(
                &ctx,
                &bytes,
                vec![input as u64, output as u64, c.experts as u64],
                GgmlType::MXFP4,
            )
            .unwrap()
        };
        let source = [
            make(c.hidden, c.intermediate, 1),
            make(c.hidden, c.intermediate, 3),
            make(c.intermediate, c.hidden, 7),
        ];
        let mut staging = Staging::new(&ctx, &c).unwrap();
        let ids = [0i32, 2, 3, 4, 6, 7];
        let banks = staging.copy(&c, source.each_ref(), &ids).unwrap();
        let x: Vec<_> = (0..c.hidden)
            .map(|i| ((i % 31) as f32 - 15.0) / 32.0)
            .collect();
        let input = upload(&ctx, &x, vec![c.hidden as u64]).unwrap();
        let status = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&[metal::ROUTE_STATUS_READY]),
            vec![1],
            GgmlType::I32,
        )
        .unwrap();
        let run = |banks: [&MetalTensor; 3], ids: &[i32], experts| {
            let ids =
                MetalTensor::from_bytes(&ctx, bytemuck::cast_slice(ids), vec![6], GgmlType::I32)
                    .unwrap();
            let act = MetalTensor::zeros_f32(&ctx, vec![c.intermediate as u64, 6]).unwrap();
            submit(&ctx, "test_gate", [0; 4], |enc| {
                metal::encode_all_slots_gate_up_swiglu_native(
                    &ctx,
                    enc,
                    banks[0],
                    banks[1],
                    &input,
                    &ids,
                    &status,
                    &act,
                    c.hidden,
                    c.intermediate,
                    experts,
                    6,
                    c.clamp,
                )
            })
            .unwrap();
            let mut weighted = f32_slice(&act).unwrap().to_vec();
            for (i, slot) in weighted.chunks_exact_mut(c.intermediate).enumerate() {
                for v in slot {
                    *v *= (i + 1) as f32 / 21.0;
                }
            }
            let weighted = upload(&ctx, &weighted, vec![c.intermediate as u64, 6]).unwrap();
            let down = MetalTensor::zeros_f32(&ctx, vec![c.hidden as u64, 6]).unwrap();
            submit(&ctx, "test_down", [0; 4], |enc| {
                metal::encode_all_slots_down_native(
                    &ctx,
                    enc,
                    banks[2],
                    &weighted,
                    &ids,
                    &status,
                    &down,
                    c.intermediate,
                    c.hidden,
                    experts,
                    6,
                )
            })
            .unwrap();
            (
                f32_slice(&act)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                f32_slice(&down)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(
            run(source.each_ref(), &ids, 8),
            run(banks, &[0, 1, 2, 3, 4, 5], 6)
        );
    }

    #[test]
    fn staging_memory_api_validation() {
        let c = Config::default();
        assert_eq!(bytes(&c).unwrap(), 112_803_840);
        let legacy = c.estimate_request_memory(14).unwrap();
        let staged = c.estimate_request_memory_with_expert_staging(14).unwrap();
        assert_eq!(staged.total_bytes - legacy.total_bytes, 112_803_840);
        assert_eq!(staged.metal_bytes - legacy.metal_bytes, 112_803_840);
        assert_eq!(staged.cpu_bytes, legacy.cpu_bytes);
        for bad in [
            Config {
                hidden: 31,
                ..c.clone()
            },
            Config {
                topk: 0,
                ..c.clone()
            },
            Config {
                topk: 273,
                ..c.clone()
            },
            Config {
                intermediate: usize::MAX,
                ..c.clone()
            },
        ] {
            assert!(bytes(&bad).is_err());
        }
    }

    #[test]
    fn staging_checked_copy_reuses_allocation_and_rejects_before_mutation() {
        let ctx = MetalContext::new().unwrap();
        let c = Config {
            hidden: 32,
            intermediate: 64,
            experts: 8,
            topk: 6,
            ..Config::default()
        };
        let alloc = |i, o| MetalTensor::zeros_dtype(&ctx, vec![i, o, 8], GgmlType::MXFP4).unwrap();
        let banks = [alloc(32, 64), alloc(32, 64), alloc(64, 32)];
        let mut stage = Staging::new(&ctx, &c).unwrap();
        let pointers = stage.banks.each_ref().map(|b| b.buffer.contents());
        let src = banks.each_ref();
        stage.copy(&c, src, &[0, 1, 2, 3, 4, 5]).unwrap();
        // Distinct new source bytes expose even a partial copy on rejection.
        for bank in &banks {
            unsafe {
                std::ptr::write_bytes(
                    bank.buffer.contents().as_ptr().cast::<u8>(),
                    0x5a,
                    bank.buffer.length(),
                );
            }
        }
        let unchanged = |stage: &Staging| {
            for bank in &stage.banks {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        bank.buffer.contents().as_ptr().cast::<u8>(),
                        bank.buffer.length(),
                    )
                };
                assert!(bytes.iter().all(|&b| b == 0));
            }
        };
        for ids in [
            [0, 1, 2, 3, 4, 8],
            [-1, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 3, 5],
            [1, 0, 2, 3, 4, 5],
        ] {
            assert!(stage.copy(&c, src, &ids).is_err());
            unchanged(&stage);
        }
        let mut invalid = banks[2].clone();
        invalid.offset = u64::MAX;
        assert!(
            stage
                .copy(&c, [&banks[0], &banks[1], &invalid], &[0, 1, 2, 3, 4, 5])
                .is_err()
        );
        invalid.offset = 0;
        unchanged(&stage);
        invalid.shape[0] = 32;
        assert!(
            stage
                .copy(&c, [&banks[0], &banks[1], &invalid], &[0, 1, 2, 3, 4, 5])
                .is_err()
        );
        unchanged(&stage);
        stage.copy(&c, src, &[2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(
            pointers,
            stage.banks.each_ref().map(|b| b.buffer.contents())
        );
    }
}

pub(super) fn bytes(c: &Config) -> Result<u64> {
    op::require(
        c.hidden > 0
            && c.intermediate > 0
            && c.hidden % 32 == 0
            && c.intermediate % 32 == 0
            && c.topk > 0
            && c.topk <= 16
            && c.topk <= c.experts,
        "expert staging geometry",
    )?;
    (c.hidden as u64)
        .checked_mul(c.intermediate as u64)
        .and_then(|n| n.checked_div(32))
        .and_then(|n| n.checked_mul(17))
        .and_then(|n| n.checked_mul(c.topk as u64))
        .and_then(|n| n.checked_mul(3))
        .filter(|&n| n <= isize::MAX as u64)
        .ok_or_else(|| "expert staging extent overflow".into())
}

impl Staging {
    pub(super) fn new(ctx: &MetalContext, c: &Config) -> Result<Self> {
        let size = bytes(c)?;
        op::require(
            size / 3 <= ctx.max_buffer_length() as u64,
            "expert staging buffer limit",
        )?;
        let priced = ctx
            .price_shared_buffer_upper(size / 3)
            .map_err(|e| e.to_string())?
            .priced_upper_bytes
            .checked_mul(3)
            .ok_or("expert staging pricing overflow")?;
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            priced,
            0,
            0,
            ctx.memory_signals(),
            true,
        );
        if let Some(reason) = admission.refusal() {
            return Err(format!("expert staging admission: {reason:?}"));
        }
        let alloc = |input, output| {
            MetalTensor::zeros_dtype_unstaged(
                ctx,
                vec![input as u64, output as u64, c.topk as u64],
                GgmlType::MXFP4,
            )
            .map_err(|e| e.to_string())
        };
        Ok(Self {
            banks: [
                alloc(c.hidden, c.intermediate)?,
                alloc(c.hidden, c.intermediate)?,
                alloc(c.intermediate, c.hidden)?,
            ],
        })
    }

    pub(super) fn copy(
        &mut self,
        c: &Config,
        source: [&MetalTensor; 3],
        ids: &[i32],
    ) -> Result<[&MetalTensor; 3]> {
        let _wall = profile::Wall(
            "expert_staging",
            [c.hidden, c.intermediate, c.experts, c.topk],
            profile::start(),
        );
        let row = usize::try_from(bytes(c)? / 3 / c.topk as u64).map_err(|e| e.to_string())?;
        op::require(
            ids.len() == c.topk
                && ids.iter().all(|&id| id >= 0 && (id as usize) < c.experts)
                && ids.windows(2).all(|w| w[0] < w[1]),
            "expert staging IDs must be ascending and in range",
        )?;
        // Validate all sources and destinations before touching any scratch byte.
        for (part, (src, dst)) in source.iter().zip(&self.banks).enumerate() {
            let (input, output) = if part == 2 {
                (c.intermediate, c.hidden)
            } else {
                (c.hidden, c.intermediate)
            };
            for (tensor, experts) in [(*src, c.experts), (dst, c.topk)] {
                let extent = row
                    .checked_mul(experts)
                    .and_then(|n| n.checked_add(usize::try_from(tensor.offset).ok()?));
                op::require(
                    tensor.dtype == GgmlType::MXFP4
                        && tensor.shape == [input as u64, output as u64, experts as u64]
                        && tensor.buffer.storageMode() == objc2_metal::MTLStorageMode::Shared
                        && extent.is_some_and(|n| {
                            n <= tensor.buffer.length() && n <= isize::MAX as usize
                        }),
                    "expert staging bank shape/extent",
                )?;
            }
        }
        for (src, dst) in source.iter().zip(&self.banks) {
            for (slot, &id) in ids.iter().enumerate() {
                // Fresh private owned buffers cannot alias immutable source banks.
                // All prior native submits completed synchronously before reuse.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.buffer
                            .contents()
                            .as_ptr()
                            .cast::<u8>()
                            .add(src.offset as usize + id as usize * row),
                        dst.buffer
                            .contents()
                            .as_ptr()
                            .cast::<u8>()
                            .add(dst.offset as usize + slot * row),
                        row,
                    );
                }
            }
        }
        Ok([&self.banks[0], &self.banks[1], &self.banks[2]])
    }
}
