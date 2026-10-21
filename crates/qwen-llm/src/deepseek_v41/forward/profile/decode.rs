//! PRIVATE coarse decode views. Raw serial encoder intervals, not kernels.
//! Serial encoders may have overlapping timestamp envelopes. These remain valid
//! inclusive observations, not an exclusive partition or proof of concurrency.
//! Dimensions are [position, layer (layers for global spans), 1, 0].
use super::*;
use crate::metal::{KernelEncoder, MetalContext, MetalTimestampSampleBuffer};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLCommandBuffer;

pub const MAX_LAYERS: usize = 40;
pub const MAX_SAMPLES: usize = 2 * (5 * MAX_LAYERS + 4);
pub const MAX_TOKENS: usize = 32;
pub const FFN_LAYERS: [usize; 10] = [0, 2, 5, 10, 15, 20, 25, 30, 35, 38];
pub const FFN_STAGES: [&str; 4] = [
    "decode_ffn_mhc_pre",
    "decode_ffn_router_shared",
    "decode_ffn_routed",
    "decode_ffn_mhc_post",
];

#[derive(Debug, Serialize)]
pub struct Interval {
    pub operation: &'static str,
    pub position: usize,
    pub layer: usize,
    pub start_ns: u64,
    pub end_ns: u64,
}

/// Geometry of sampled envelopes only. No command endpoint containment claim.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct EnvelopeSummary {
    pub first_sample_ns: u64,
    pub last_sample_ns: u64,
    pub sum_inclusive_durations_ns: u64,
    pub interval_union_ns: u64,
    pub sampled_envelope_ns: u64,
    pub overlap_multiplicity_ns: u64,
    pub covered_by_two_or_more_ns: u64,
    pub uncovered_sampled_envelope_ns: u64,
    pub exclusive: bool,
}

pub fn summarize_envelopes(intervals: &[(u64, u64)]) -> Result<EnvelopeSummary, String> {
    if intervals.is_empty() || intervals.len() > MAX_SAMPLES / 2 {
        return Err("private decode interval count".into());
    }
    let mut events = std::collections::BTreeMap::<u64, i32>::new();
    let mut sum = 0u64;
    for &(start, end) in intervals {
        if start == 0 || end <= start || end == u64::MAX {
            return Err("private decode invalid sample endpoints".into());
        }
        sum = sum
            .checked_add(end - start)
            .ok_or("private decode duration sum overflow")?;
        *events.entry(start).or_default() += 1;
        *events.entry(end).or_default() -= 1;
    }
    let first = *events.first_key_value().unwrap().0;
    let last = *events.last_key_value().unwrap().0;
    let mut previous = first;
    let mut active = 0;
    let mut union = 0;
    let mut multiple = 0;
    // Group equal-time endpoints before advancing: touching intervals do not overlap.
    for (time, delta) in events {
        let duration = time - previous;
        if active > 0 {
            union += duration;
        }
        if active >= 2 {
            multiple += duration;
        }
        active += delta;
        previous = time;
    }
    debug_assert_eq!(active, 0);
    Ok(EnvelopeSummary {
        first_sample_ns: first,
        last_sample_ns: last,
        sum_inclusive_durations_ns: sum,
        interval_union_ns: union,
        sampled_envelope_ns: last - first,
        overlap_multiplicity_ns: sum - union,
        covered_by_two_or_more_ns: multiple,
        uncovered_sampled_envelope_ns: last - first - union,
        exclusive: multiple == 0,
    })
}

pub fn layout(
    layers: usize,
    engrams: &[usize],
    prepared: bool,
    logits: bool,
) -> Result<Vec<(&'static str, usize)>, String> {
    if !(1..=MAX_LAYERS).contains(&layers)
        || engrams.iter().any(|&l| l >= layers)
        || engrams.windows(2).any(|w| w[0] >= w[1])
    {
        return Err("private decode layer/Engram extent".into());
    }
    let mut spans = Vec::with_capacity(5 * layers + 4);
    if !prepared && !engrams.is_empty() {
        spans.push(("decode_engram_projections", layers));
    }
    for layer in 0..layers {
        if engrams.contains(&layer) {
            spans.push(("decode_engram_gate", layer));
        }
        for name in [
            "decode_attention_mhc_pre",
            "decode_attention_operator_composite",
            "decode_attention_mhc_post",
        ] {
            spans.push((name, layer));
        }
        if FFN_LAYERS.contains(&layer) {
            spans.extend(FFN_STAGES.map(|name| (name, layer)));
        } else {
            spans.push(("decode_ffn_composite_including_mhc", layer));
        }
    }
    if logits {
        for name in [
            "decode_head_collapse",
            "decode_head_norm",
            "decode_head_projection",
        ] {
            spans.push((name, layers));
        }
    }
    sample_count(spans.len())?;
    Ok(spans)
}

fn sample_count(spans: usize) -> Result<usize, String> {
    spans
        .checked_mul(2)
        .filter(|&n| n > 0 && n <= MAX_SAMPLES)
        .ok_or_else(|| "private decode sample count overflow/limit".into())
}

pub(crate) struct Encoder<'a> {
    command: &'a Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    samples: Option<Rc<MetalTimestampSampleBuffer>>,
    expected: Vec<(&'static str, usize)>,
    next: usize,
    current: Option<KernelEncoder>,
}

impl<'a> Encoder<'a> {
    pub(crate) fn new(
        command: &'a Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        samples: Option<Rc<MetalTimestampSampleBuffer>>,
        expected: Vec<(&'static str, usize)>,
    ) -> Result<Self, String> {
        if let Some(samples) = &samples {
            if sample_count(expected.len())? > samples.sample_count() {
                return Err("private decode insufficient samples".into());
            }
            let used = ACTIVE
                .with(|active| active.borrow().as_ref().map(|a| a.decode_intervals.len()))
                .ok_or("private decode scope missing")?;
            if used + expected.len() > MAX_TOKENS * MAX_SAMPLES / 2 {
                return Err("private decode bounded interval limit exceeded".into());
            }
        }
        let current = samples.is_none().then(|| KernelEncoder::begin(command));
        Ok(Self {
            command,
            samples,
            expected,
            next: 0,
            current,
        })
    }

    pub(crate) fn stage(&mut self, name: &'static str, layer: usize) -> Result<(), String> {
        let Some(samples) = &self.samples else {
            return Ok(());
        };
        if self.expected.get(self.next) != Some(&(name, layer)) {
            return Err(format!("private decode unexpected stage {name}/{layer}"));
        }
        if let Some(enc) = self.current.take() {
            enc.end();
        }
        self.current = Some(
            KernelEncoder::try_begin_sampled(
                self.command,
                samples,
                self.next * 2,
                self.next * 2 + 1,
                false,
            )
            .map_err(|e| e.to_string())?,
        );
        self.next += 1;
        Ok(())
    }

    pub(crate) fn sampled(&self) -> bool {
        self.samples.is_some()
    }

    pub(crate) fn end(&mut self) -> Result<(), String> {
        if let Some(enc) = self.current.take() {
            enc.end();
        }
        if self.samples.is_some() && self.next != self.expected.len() {
            return Err("private decode incomplete stage coverage".into());
        }
        Ok(())
    }

    pub(crate) fn resolve(&self, ctx: &MetalContext, position: usize) -> Result<(), String> {
        let Some(samples) = &self.samples else {
            return Ok(());
        };
        let count = sample_count(self.expected.len())?;
        let times = ctx
            .resolve_timestamp_samples(samples, count)
            .map_err(|e| e.to_string())?;
        if times.len() != count {
            return Err("private decode resolved sample count".into());
        }
        for (&(name, layer), pair) in self.expected.iter().zip(times.chunks_exact(2)) {
            gpu_interval_dimensions(name, [position, layer, 1, 0], pair[0], pair[1]);
        }
        ACTIVE.with(|active| -> Result<(), String> {
            let mut active = active.borrow_mut();
            let active = active.as_mut().ok_or("private decode scope missing")?;
            if active.decode_intervals.len() + self.expected.len() > MAX_TOKENS * MAX_SAMPLES / 2 {
                return Err("private decode bounded interval limit exceeded".into());
            }
            active
                .decode_intervals
                .extend(self.expected.iter().zip(times.chunks_exact(2)).map(
                    |(&(operation, layer), pair)| Interval {
                        operation,
                        position,
                        layer,
                        start_ns: pair[0],
                        end_ns: pair[1],
                    },
                ));
            Ok(())
        })?;
        Ok(())
    }
}

impl std::ops::Deref for Encoder<'_> {
    type Target = KernelEncoder;
    fn deref(&self) -> &Self::Target {
        self.current.as_ref().expect("decode stage before encoding")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_events_measure_union_multiplicity_and_sampled_gaps_exactly() {
        for (intervals, expected) in [
            (vec![(10, 20)], (10, 10, 10, 0, 0, 0)),
            (vec![(10, 20), (30, 40)], (20, 20, 30, 0, 0, 10)),
            (vec![(10, 30), (20, 40)], (40, 30, 30, 10, 10, 0)),
            (vec![(10, 50), (20, 40), (25, 35)], (70, 40, 40, 30, 20, 0)),
            (vec![(10, 20); 3], (30, 10, 10, 20, 10, 0)),
            (
                vec![(10, 100), (20, 30), (40, 50)],
                (110, 90, 90, 20, 20, 0),
            ),
            (vec![(10, 20), (20, 30)], (20, 20, 20, 0, 0, 0)),
            (vec![(10, 30), (15, 20), (40, 45)], (30, 25, 35, 5, 5, 10)),
        ] {
            for offset in [0, u64::MAX - 1000] {
                let shifted: Vec<_> = intervals
                    .iter()
                    .map(|(s, e)| (s + offset, e + offset))
                    .collect();
                let s = summarize_envelopes(&shifted).unwrap();
                assert_eq!(
                    (
                        s.sum_inclusive_durations_ns,
                        s.interval_union_ns,
                        s.sampled_envelope_ns,
                        s.overlap_multiplicity_ns,
                        s.covered_by_two_or_more_ns,
                        s.uncovered_sampled_envelope_ns
                    ),
                    expected
                );
                assert_eq!(s.exclusive, expected.3 == 0);
                assert!(
                    shifted
                        .iter()
                        .all(|&(start, end)| start >= s.first_sample_ns && end <= s.last_sample_ns)
                );
                assert_eq!(
                    s.interval_union_ns + s.uncovered_sampled_envelope_ns,
                    s.last_sample_ns - s.first_sample_ns
                );
                let mut reversed = shifted;
                reversed.reverse();
                assert_eq!(s, summarize_envelopes(&reversed).unwrap());
            }
        }
    }

    #[test]
    fn corrupt_samples_and_overflow_are_rejected_without_normalization() {
        for intervals in [
            vec![],
            vec![(0, 1)],
            vec![(10, 10)],
            vec![(11, 10)],
            vec![(1, u64::MAX)],
            vec![(1, u64::MAX - 1); 2],
            vec![(1, 2); MAX_SAMPLES / 2 + 1],
        ] {
            assert!(summarize_envelopes(&intervals).is_err(), "{intervals:?}");
        }
        assert_eq!(
            summarize_envelopes(&[(1, 2); MAX_SAMPLES / 2])
                .unwrap()
                .interval_union_ns,
            1
        );
    }

    #[test]
    fn bounded_layout_covers_presence_absence_and_head() {
        for engrams in [vec![], vec![0, 2], vec![1, 14]] {
            for prepared in [false, true] {
                for logits in [false, true] {
                    let spans = layout(40, &engrams, prepared, logits).unwrap();
                    let initial = usize::from(!prepared && !engrams.is_empty());
                    assert_eq!(
                        spans.len(),
                        190 + engrams.len() + initial + 3 * usize::from(logits)
                    );
                    assert_eq!(
                        spans
                            .iter()
                            .filter(|s| s.0 == "decode_engram_gate")
                            .map(|s| s.1)
                            .collect::<Vec<_>>(),
                        engrams
                    );
                    assert!(sample_count(spans.len()).unwrap() <= MAX_SAMPLES);
                }
            }
        }
        let spans = layout(40, &[1, 14], false, true).unwrap();
        assert_eq!(spans.len(), 196);
        assert_eq!(sample_count(spans.len()).unwrap(), 392);
        assert_eq!(spans.len() * 2 - 1, 391);
        for layer in 0..40 {
            for name in FFN_STAGES {
                assert_eq!(spans.contains(&(name, layer)), FFN_LAYERS.contains(&layer));
            }
            assert_eq!(
                spans.contains(&("decode_ffn_composite_including_mhc", layer)),
                !FFN_LAYERS.contains(&layer)
            );
        }
        assert!(layout(40, &(0..40).collect::<Vec<_>>(), false, true).is_err());
        for (layers, engrams) in [
            (0, vec![]),
            (41, vec![]),
            (2, vec![2]),
            (2, vec![1, 1]),
            (2, vec![1, 0]),
        ] {
            assert!(layout(layers, &engrams, false, true).is_err());
        }
        for spans in [0, MAX_SAMPLES, usize::MAX] {
            assert!(sample_count(spans).is_err());
        }
    }
}
