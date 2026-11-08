use super::*;
use execution::Engine;
use qwen_llm::deepseek_v41::forward::{ColdPrefillSlice, PrefillGeometry};
use reuse::{Live, Restored, Reuse};
use serde_json::json;

struct Seq {
    position: usize,
    encoded: usize,
    target: usize,
    ready: bool,
    report: PrefillGeometry,
}
impl Default for Seq {
    fn default() -> Self {
        Self {
            position: 0,
            encoded: 0,
            target: 0,
            ready: false,
            report: PrefillGeometry {
                start: 0,
                tokens: 0,
                selected_rows: 4096,
            },
        }
    }
}
#[derive(Default)]
struct Probe {
    cold_calls: usize,
    full_calls: Vec<usize>,
    decline: bool,
}
impl Engine<Seq> for Probe {
    fn admit(&mut self, _: usize, _: usize, _: u64) -> Result<(), StateError> {
        Ok(())
    }
    fn new_state(&mut self, _: usize) -> Result<Seq, ServeError> {
        Ok(Seq {
            ready: true,
            ..Seq::default()
        })
    }
    fn prefill(
        &mut self,
        state: &mut Seq,
        tokens: &[u32],
        _: &AtomicBool,
        logits: bool,
    ) -> Result<Vec<f32>, ServeError> {
        assert!(state.ready);
        state.report = PrefillGeometry {
            start: state.position,
            tokens: tokens.len(),
            selected_rows: 4096,
        };
        state.position += tokens.len();
        self.full_calls.push(tokens.len());
        Ok(if logits { heads() } else { Vec::new() })
    }
    fn cold_prefill(
        &mut self,
        state: &mut Seq,
        tokens: &[u32],
        _: &AtomicBool,
        begin: bool,
        max_rows: usize,
    ) -> Result<Option<ColdPrefillSlice>, ServeError> {
        if self.decline {
            assert!(begin);
            return Ok(None);
        }
        if begin {
            assert!(state.ready);
            state.ready = false;
            state.target = state.position + tokens.len();
            state.encoded = state.position;
            state.report = PrefillGeometry {
                start: state.position,
                tokens: tokens.len(),
                selected_rows: 4096,
            };
        } else {
            assert!(!state.ready);
        }
        assert_eq!(tokens.len(), state.target - state.position);
        let end = (state.encoded + max_rows).min(state.target);
        let rows = end - state.encoded;
        state.encoded = end;
        self.cold_calls += 1;
        let complete = end == state.target;
        if complete {
            state.position = end;
            state.ready = true;
        }
        Ok(Some(ColdPrefillSlice {
            encoder_rows: rows,
            complete,
            logits: if complete { heads() } else { vec![] },
        }))
    }
    fn step(&mut self, _: &mut Seq, _: u32, _: &AtomicBool) -> Result<Vec<f32>, String> {
        panic!("EOS")
    }
    fn now(&mut self) -> Duration {
        Duration::ZERO
    }
}
fn heads() -> Vec<f32> {
    let mut h = vec![0.; 257];
    h[tokenizer_fixture::EOS as usize] = 100.;
    h
}
#[derive(Default)]
struct Cache {
    cuts: Vec<usize>,
    interior: Vec<usize>,
    committed: bool,
}
impl Reuse<Seq> for Cache {
    fn take(&mut self, _: &[u32], _: usize, _: u64) -> Result<Option<Restored<Seq>>, ServeError> {
        Ok(None)
    }
    fn prefill_report(&self, s: &Seq, _: usize, _: usize) -> PrefillGeometry {
        assert!(s.ready);
        s.report
    }
    fn checkpoint(&mut self, s: &Seq, ids: &[u32], _: &[f32], _: u64) {
        assert!(s.ready);
        assert_eq!(s.position, ids.len());
        self.cuts.push(ids.len());
    }
    fn ladder_enabled(&self) -> bool {
        true
    }
    fn ladder_checkpoint(
        &mut self,
        _: super::super::checkpoint_ladder::Conversation,
        s: &Seq,
        ids: &[u32],
        _: u64,
    ) {
        assert!(s.ready);
        assert_eq!(s.position, ids.len());
        self.interior.push(ids.len());
    }
    fn commit(&mut self, live: Live<Seq>) {
        assert!(live.state.ready);
        self.committed = true;
    }
    fn capacity(&self, _: &Seq, required: usize) -> usize {
        required
    }
}
struct Sink;
impl GenerationSink for Sink {
    fn piece(&mut self, _: &[u8]) -> io::Result<()> {
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn whole_cold_plan_yields_without_snapshots_then_preserves_shared_interior_and_body() {
    for (enabled, decline) in [(true, false), (false, false), (true, true)] {
        let mut settings = Settings::new(
            tokenizer_fixture::tokenizer(None),
            32768,
            1,
            SamplingConfig::default(),
            tokenizer_fixture::EOS,
            "test".into(),
        )
        .unwrap();
        settings.headroom = || finite_headroom(1 << 40);
        settings.cold_ced = enabled;
        let request = settings
            .profile()
            .parse(&json!({"model":"test","input":"root","max_output_tokens":1}))
            .unwrap();
        let body = 343 + 4 * 4096 + 7;
        let prompt = "p".repeat(body + 2);
        let bounds = Some(super::super::http::PromptBoundaries {
            shared_prefix_end: Some(343),
            generation_header_start: Some(body),
        });
        let ceiling = super::super::output_memory::OutputByteCeiling::from_tokens_and_piece(
            1,
            settings.limits.max_piece_bytes,
        )
        .unwrap();
        let mut probe = Probe {
            decline,
            ..Probe::default()
        };
        let mut cache = Cache::default();
        let result = settings
            .execute_with_engine(
                ceiling, &request, &prompt, bounds, &mut Sink, &mut probe, &mut cache,
            )
            .unwrap();
        assert!(cache.committed);
        assert_eq!(cache.cuts, vec![343, body]);
        assert_eq!(cache.interior, vec![343 + 4 * 4096]);
        if enabled && !decline {
            assert_eq!(probe.cold_calls, 4);
            assert_eq!(probe.full_calls, vec![343, 7, 2]);
            assert_eq!(result.stats.unwrap().phases.unwrap()["ced_episodes"], 1);
        } else {
            assert_eq!(probe.cold_calls, 0);
            assert_eq!(probe.full_calls, vec![343, 4096, 4096, 4096, 4096, 7, 2]);
        }
    }
}
