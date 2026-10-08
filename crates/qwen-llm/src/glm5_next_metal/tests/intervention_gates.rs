//! Gates for GLM-5.3 module-site interventions and captures
//! (`glm5_next_metal::interventions`).

use super::*;
use crate::metal::PostBlockIntervention;

fn unit(values: Vec<f32>) -> Vec<f32> {
    let norm = values
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    values
        .iter()
        .map(|v| (f64::from(*v) / norm) as f32)
        .collect()
}

fn f32_tensor(ctx: &MetalContext, values: &[f32]) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(values),
        vec![values.len() as u64],
        GgmlType::F32,
    )
    .unwrap()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

#[test]
#[ignore = "GLM intervention gates: loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF and an idle GPU"]
fn module_interventions_are_inert_when_empty_and_exact_where_applied() {
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let h = weights.config.hidden_size as usize;
    let tokens: Vec<u32> = checkpoint_tokens();
    let (prefix, last) = tokens.split_at(tokens.len() - 1);
    let fresh = || {
        let mut s = Glm5NextSession::new(&ctx, &weights, tokens.len() + 4).unwrap();
        for &t in prefix {
            s.advance(&ctx, t).unwrap();
        }
        s
    };
    // Reference: plain forward.
    let mut plain = fresh();
    let reference = plain.forward(&ctx, last[0]).unwrap();
    let reference_state = state_bits(&plain);
    drop(plain);

    // 1. No operations, captures at every site of blocks 0, 3, 20: bitwise
    //    equal to the plain forward; captures finite.
    let mut captures = vec![Glm5NextSiteCapture {
        site: Glm5NextSite::Embedding,
        block: 0,
        point: Glm5NextCapturePoint::BeforeOperations,
    }];
    for block in [0usize, 3, 20] {
        for site in [
            Glm5NextSite::MixerOutput,
            Glm5NextSite::RoutedExpertsOutput,
            Glm5NextSite::SharedExpertOutput,
            Glm5NextSite::FfnOutput,
        ] {
            if block == 0 && site != Glm5NextSite::MixerOutput && site != Glm5NextSite::FfnOutput {
                continue;
            }
            for point in [
                Glm5NextCapturePoint::BeforeOperations,
                Glm5NextCapturePoint::AfterOperations,
            ] {
                captures.push(Glm5NextSiteCapture { site, block, point });
            }
        }
    }
    let mut seen = 0usize;
    let mut s = fresh();
    let logits = s
        .forward_with_interventions(&ctx, last[0], &[], &captures, &mut |_, values| {
            assert_eq!(values.len(), h);
            assert_finite("capture", values);
            seen += 1;
        })
        .unwrap();
    assert_eq!(seen, captures.len());
    assert_eq!(
        logit_bits(&[logits]),
        logit_bits(std::slice::from_ref(&reference)),
        "captures changed logits"
    );
    assert_eq!(state_bits(&s), reference_state, "captures changed state");
    drop(s);

    // 2. A projection then a fixed add at one site, in caller order: the
    //    after-capture equals the formula applied to the before-capture.
    let r = unit(
        (0..h)
            .map(|i| ((i * 7919 % 1013) as f32 / 1013.0) - 0.5)
            .collect(),
    );
    let e = unit(
        (0..h)
            .map(|i| if i % 97 == 0 { 1.0 } else { 0.0 })
            .collect(),
    );
    let r_tensor = f32_tensor(&ctx, &r);
    let e_tensor = f32_tensor(&ctx, &e);
    let (alpha, beta) = (2.397f32, 0.5f32);
    for (site, block) in [
        (Glm5NextSite::MixerOutput, 20usize),
        (Glm5NextSite::SharedExpertOutput, 20),
        (Glm5NextSite::RoutedExpertsOutput, 15),
        (Glm5NextSite::FfnOutput, 1),
        (Glm5NextSite::Embedding, 0),
    ] {
        let ops = [
            Glm5NextModuleIntervention {
                site,
                op: PostBlockIntervention::Projection {
                    layer: block as u32,
                    direction: &r_tensor,
                    coefficient: alpha,
                },
            },
            Glm5NextModuleIntervention {
                site,
                op: PostBlockIntervention::Fixed {
                    layer: block as u32,
                    direction: &e_tensor,
                    coefficient: beta,
                },
            },
        ];
        let captures = [
            Glm5NextSiteCapture {
                site,
                block,
                point: Glm5NextCapturePoint::BeforeOperations,
            },
            Glm5NextSiteCapture {
                site,
                block,
                point: Glm5NextCapturePoint::AfterOperations,
            },
        ];
        let (mut before, mut after) = (Vec::new(), Vec::new());
        let mut s = fresh();
        let edited = s
            .forward_with_interventions(&ctx, last[0], &ops, &captures, &mut |capture, values| {
                match capture.point {
                    Glm5NextCapturePoint::BeforeOperations => before = values.to_vec(),
                    Glm5NextCapturePoint::AfterOperations => after = values.to_vec(),
                }
            })
            .unwrap();
        let rb = dot(&r, &before);
        let worst = before
            .iter()
            .zip(&after)
            .enumerate()
            .map(|(i, (b, a))| {
                let expect = f64::from(*b) - f64::from(alpha) * rb * f64::from(r[i])
                    + f64::from(beta) * f64::from(e[i]);
                (f64::from(*a) - expect).abs()
            })
            .fold(0.0f64, f64::max);
        let scale = before
            .iter()
            .map(|v| f64::from(*v).abs())
            .fold(0.0f64, f64::max)
            .max(1.0);
        eprintln!(
            "{site:?} block {block}: r.before {rb:.4}, worst |after - formula| {worst:.3e} (scale {scale:.3e})"
        );
        assert!(
            worst <= 1e-5 * scale + 1e-6,
            "{site:?} block {block}: formula mismatch {worst:.3e}"
        );
        assert!(
            kl_divergence(&reference, &edited) > 0.0,
            "{site:?} block {block}: the edit changed nothing"
        );
        assert_eq!(s.position(), tokens.len());
    }

    // 3. Refusals before any work: the session does not move.
    let mut s = fresh();
    let position = s.position();
    let bad: [(&str, Glm5NextModuleIntervention<'_>); 3] = [
        (
            "shared expert in a dense block",
            Glm5NextModuleIntervention {
                site: Glm5NextSite::SharedExpertOutput,
                op: PostBlockIntervention::Projection {
                    layer: 1,
                    direction: &r_tensor,
                    coefficient: 1.0,
                },
            },
        ),
        (
            "embedding past block 0",
            Glm5NextModuleIntervention {
                site: Glm5NextSite::Embedding,
                op: PostBlockIntervention::Projection {
                    layer: 3,
                    direction: &r_tensor,
                    coefficient: 1.0,
                },
            },
        ),
        (
            "zero dose",
            Glm5NextModuleIntervention {
                site: Glm5NextSite::MixerOutput,
                op: PostBlockIntervention::Projection {
                    layer: 3,
                    direction: &r_tensor,
                    coefficient: 0.0,
                },
            },
        ),
    ];
    for (label, op) in bad {
        let refused = s.forward_with_interventions(&ctx, last[0], &[op], &[], &mut |_, _| {});
        assert!(refused.is_err(), "{label} was accepted");
        assert_eq!(s.position(), position, "{label} moved the session");
    }
    let short = f32_tensor(&ctx, &r[..h - 1]);
    let refused = s.forward_with_interventions(
        &ctx,
        last[0],
        &[Glm5NextModuleIntervention {
            site: Glm5NextSite::MixerOutput,
            op: PostBlockIntervention::Projection {
                layer: 3,
                direction: &short,
                coefficient: 1.0,
            },
        }],
        &[],
        &mut |_, _| {},
    );
    assert!(refused.is_err(), "a short direction was accepted");
    assert_eq!(s.position(), position);
    // A misaligned direction in a late block is refused with an earlier
    // capture requested: validation precedes any submitted work.
    let wide = f32_tensor(&ctx, &[r.as_slice(), &[0.0]].concat());
    let mut misaligned = wide.view_subrange(0, vec![h as u64]);
    misaligned.offset += 2;
    let refused = s.forward_with_interventions(
        &ctx,
        last[0],
        &[Glm5NextModuleIntervention {
            site: Glm5NextSite::MixerOutput,
            op: PostBlockIntervention::Projection {
                layer: 40,
                direction: &misaligned,
                coefficient: 1.0,
            },
        }],
        &[Glm5NextSiteCapture {
            site: Glm5NextSite::MixerOutput,
            block: 3,
            point: Glm5NextCapturePoint::BeforeOperations,
        }],
        &mut |_, _| panic!("a capture ran before the refusal"),
    );
    assert!(refused.is_err(), "a misaligned direction was accepted");
    assert_eq!(s.position(), position);
    assert!(!s.is_poisoned(), "a refusal poisoned the session");
    assert_eq!(
        logit_bits(&[s.forward(&ctx, last[0]).unwrap()]),
        logit_bits(&[reference]),
        "a refusal changed state"
    );
}
