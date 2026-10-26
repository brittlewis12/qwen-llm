use anyhow::{Result, ensure};
use serde_json::{Value, json};

#[derive(Default)]
pub(super) struct Difference {
    elements: u64,
    mismatches: u64,
    max_abs: f64,
    sum_sq: f64,
    first: Option<Value>,
}

impl Difference {
    pub fn add(
        &mut self,
        reference: &[u8],
        candidate: &[u8],
        float: bool,
        context: &Value,
        strict: bool,
    ) -> Result<()> {
        ensure!(
            reference.len() == candidate.len() && reference.len().is_multiple_of(4),
            "comparison extent: {context}"
        );
        for (i, (a, b)) in reference
            .as_chunks::<4>()
            .0
            .iter()
            .zip(candidate.as_chunks::<4>().0)
            .enumerate()
        {
            let a = u32::from_le_bytes(*a);
            let b = u32::from_le_bytes(*b);
            let (x, y) = if float {
                (f32::from_bits(a) as f64, f32::from_bits(b) as f64)
            } else {
                (a as i32 as f64, b as i32 as f64)
            };
            ensure!(
                x.is_finite() && y.is_finite(),
                "nonfinite comparison {context} element {i}"
            );
            self.elements += 1;
            let delta = (x - y).abs();
            self.max_abs = self.max_abs.max(delta);
            self.sum_sq += delta * delta;
            if a != b {
                self.mismatches += 1;
                if self.first.is_none() {
                    self.first = Some(json!({"context": context, "element": i,
                    "reference_bits": a, "candidate_bits": b, "reference": x, "candidate": y}));
                }
                ensure!(
                    !strict,
                    "strict bit mismatch {context}, element={i}, reference=0x{a:08x} ({x}), candidate=0x{b:08x} ({y})"
                );
            }
        }
        Ok(())
    }
    pub fn json(&self) -> Value {
        json!({"elements": self.elements, "bit_mismatches": self.mismatches, "max_abs": self.max_abs,
            "rmse": if self.elements == 0 { 0.0 } else { (self.sum_sq / self.elements as f64).sqrt() }, "first_difference": self.first})
    }
}

fn partition(logits: &[f32]) -> Result<(f64, f64, usize)> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
        "nonfinite/empty logits"
    );
    let mut top = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[top] {
            top = i;
        }
    }
    let max = logits[top] as f64;
    Ok((
        max,
        logits
            .iter()
            .map(|&x| (x as f64 - max).exp())
            .sum::<f64>()
            .ln(),
        top,
    ))
}

pub(super) fn prediction(reference: &[f32], candidate: &[f32], target: u32) -> Result<Value> {
    ensure!(
        reference.len() == candidate.len() && (target as usize) < reference.len(),
        "prediction extent/token"
    );
    let (mr, zr, tr) = partition(reference)?;
    let (mc, zc, tc) = partition(candidate)?;
    let kl: f64 = reference
        .iter()
        .zip(candidate)
        .map(|(&r, &c)| {
            let lr = (r as f64 - mr) - zr;
            lr.exp() * (lr - ((c as f64 - mc) - zc))
        })
        .sum();
    let rnll = (mr - reference[target as usize] as f64) + zr;
    let cnll = (mc - candidate[target as usize] as f64) + zc;
    ensure!(
        kl.is_finite() && rnll.is_finite() && cnll.is_finite(),
        "nonfinite quality diagnostic"
    );
    Ok(
        json!({"target": target, "reference_nll": rnll, "candidate_nll": cnll,
        "kl_reference_candidate": kl, "reference_top1": tr, "candidate_top1": tc,
        "reference_hit": tr == target as usize, "candidate_hit": tc == target as usize, "top1_agrees": tr == tc}),
    )
}

pub(super) fn held_summary(rows: &[Value]) -> Result<Value> {
    ensure!(
        rows.len() == 256,
        "exactly 256 held-token predictions required"
    );
    let sum = |key: &str| rows.iter().map(|r| r[key].as_f64().unwrap()).sum::<f64>();
    let hits = |key: &str| rows.iter().filter(|r| r[key] == true).count();
    Ok(
        json!({"predictions": rows.len(), "reference_mean_nll": sum("reference_nll") / 256.0,
        "candidate_mean_nll": sum("candidate_nll") / 256.0, "reference_total_nll": sum("reference_nll"),
        "candidate_total_nll": sum("candidate_nll"), "reference_hits": hits("reference_hit"),
        "candidate_hits": hits("candidate_hit"), "top1_agreements": hits("top1_agrees"),
        "mean_kl_reference_candidate": sum("kl_reference_candidate") / 256.0, "formal_quality_verdict": Value::Null}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_large_logits_and_extra_row_not_a_prediction() {
        let r = prediction(&[10000.0, 9999.0], &[10000.0, 9999.0], 1).unwrap();
        assert!((r["reference_nll"].as_f64().unwrap() - 1.3132616875).abs() < 1e-8);
        assert_eq!(r["kl_reference_candidate"], 0.0);
        let huge = prediction(&[f32::MAX; 2], &[f32::MAX; 2], 1).unwrap();
        assert_eq!(huge["reference_nll"].as_f64().unwrap(), 2f64.ln());
        assert!(prediction(&[f32::NAN], &[0.0], 0).is_err());
        assert!(held_summary(&vec![r.clone(); 257]).is_err());
        assert_eq!(held_summary(&vec![r; 256]).unwrap()["predictions"], 256);
    }
    #[test]
    fn strict_bits_never_relax_even_signed_zero() {
        let mut d = Difference::default();
        assert!(
            d.add(
                &0f32.to_le_bytes(),
                &(-0f32).to_le_bytes(),
                true,
                &json!({"row": 7}),
                true
            )
            .is_err()
        );
        assert_eq!(d.json()["bit_mismatches"], 1);
        assert_eq!(d.json()["max_abs"], 0.0);
    }
}
