//! Same-artifact numerical diagnosis, not a new grouped-prefill acceptance bound.
use super::*;

const MODEL: crate::test_fixtures::Fixture = crate::test_fixtures::Fixture {
    id: "deepseek-v4.1-reap-272e-mxfp4-q8engram",
    env: &[],
    filename: "DeepSeek-V4.1-Flash-REAP-272E-MXFP4-Q8Engram.gguf",
};
const TOKENS: [u32; 7] = [0, 5, 223, 939, 23, 695, 22];

fn metrics(label: &str, reference: &[f32], actual: &[f32]) -> usize {
    assert_eq!(reference.len(), actual.len());
    assert!(!reference.is_empty());
    assert!(reference.iter().chain(actual).all(|v| v.is_finite()));
    let mut max_abs = 0.0_f64;
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut violations = 0;
    for (&r, &a) in reference.iter().zip(actual) {
        let (r, a) = (f64::from(r), f64::from(a));
        let delta = (a - r).abs();
        max_abs = max_abs.max(delta);
        squared_error += delta * delta;
        squared_reference += r * r;
        violations += usize::from(delta > 3e-5 + 3e-5 * r.abs());
    }
    let log_normalizer = |values: &[f32]| {
        let max = values
            .iter()
            .map(|&v| f64::from(v))
            .fold(f64::NEG_INFINITY, f64::max);
        let log_sum = values
            .iter()
            .map(|&v| (f64::from(v) - max).exp())
            .sum::<f64>()
            .ln();
        (max, log_sum)
    };
    let (rmax, rsum) = log_normalizer(reference);
    let (amax, asum) = log_normalizer(actual);
    let kl = reference
        .iter()
        .zip(actual)
        .map(|(&r, &a)| {
            let log_p = f64::from(r) - rmax - rsum;
            let log_q = f64::from(a) - amax - asum;
            log_p.exp() * (log_p - log_q)
        })
        .sum::<f64>();
    let top1 = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .fold(0, |best, (i, &v)| if v > values[best] { i } else { best })
    };
    eprintln!(
        "{}",
        serde_json::json!({
            "comparison": label, "max_abs": max_abs,
            "rms": (squared_error / reference.len() as f64).sqrt(),
            "relative_l2": (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt(),
            "kl_reference_to_actual_f64": kl,
            "reference_top1": top1(reference), "actual_top1": top1(actual),
            "atol": 3e-5, "rtol": 3e-5, "violations": violations,
        })
    );
    violations
}

#[test]
#[ignore = "model witness: deepseek-v4.1-reap-272e-mxfp4-q8engram"]
fn gguf_scalar_rowwise_grouped_prefill_diagnostic() {
    let ctx = MetalContext::new().unwrap();
    let gguf = Arc::new(crate::gguf::GgufFile::open(MODEL.required()).unwrap());
    let weights = ResidentWeights::load(&ctx, gguf).unwrap();
    let model =
        Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad).unwrap();
    let prefix = TOKENS.into_iter().cycle().take(32).collect::<Vec<_>>();
    let endpoint = TOKENS[32 % TOKENS.len()];
    eprintln!(
        "S=scalar; L=layer-major rowwise GEMV; G=layer-major grouped; prefix_rows=32; endpoint={endpoint}"
    );
    let scalar = {
        let mut state = model.new_state(33).unwrap();
        for &token in &prefix {
            model.advance(&mut state, token, None).unwrap();
        }
        model.step(&mut state, endpoint, None).unwrap()
    };
    let rowwise = {
        let mut state = model.new_state(33).unwrap();
        with_rowwise_linears(|| model.advance_prefill(&mut state, &prefix, None)).unwrap();
        model.step(&mut state, endpoint, None).unwrap()
    };
    let mut packed = model.new_state(36).unwrap();
    model.advance_prefill(&mut packed, &prefix, None).unwrap();
    let snapshot = model.capture_snapshot(&packed).unwrap();
    let grouped = model.step(&mut packed, endpoint, None).unwrap();
    let sl_violations = metrics("S-L", &scalar, &rowwise);
    metrics("S-G (characterization only)", &scalar, &grouped);
    metrics("L-G (characterization only)", &rowwise, &grouped);

    // All branches start from the identical packed-prefix snapshot and use
    // scalar steps. Growth/restore is tested independently of S/L/G reductions.
    let mut grown = model.new_state(32).unwrap();
    model.restore_snapshot(&mut grown, &snapshot).unwrap();
    model.ensure_capacity(&mut grown, 36).unwrap();
    let mut restored = model.new_state(36).unwrap();
    model.restore_snapshot(&mut restored, &snapshot).unwrap();
    let mut bitwise = true;
    for (i, token) in (32..35).map(|i| TOKENS[i % TOKENS.len()]).enumerate() {
        let expected = if i == 0 {
            grouped.clone()
        } else {
            model.step(&mut packed, token, None).unwrap()
        };
        for (label, state) in [("grown", &mut grown), ("restored", &mut restored)] {
            let actual = model.step(state, token, None).unwrap();
            assert_eq!(expected.len(), actual.len());
            let mismatches = expected
                .iter()
                .zip(&actual)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "packed-prefix {label} position={} bitwise_mismatches={mismatches}",
                32 + i
            );
            bitwise &= mismatches == 0;
        }
    }
    assert!(
        bitwise,
        "same-arithmetic packed snapshot/growth continuation differs"
    );
    assert_eq!(
        sl_violations, 0,
        "S-L exceeds existing atol=rtol=3e-5; G remains characterization only"
    );
}
