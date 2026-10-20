use super::*;
use qwen_llm::glm5_next_metal::{Glm5NextDFlashCarry, Glm5NextDFlashCosts};
use qwen_llm::tokenizer::TokError;

mod admission;

struct Bytes;
impl Tokenize for Bytes {
    fn encode(&self, text: &str, _: bool) -> Result<Vec<i32>, TokError> {
        Ok(text.bytes().map(i32::from).collect())
    }
    fn try_decode_piece_bytes(&self, token: i32) -> Result<Vec<u8>, TokError> {
        u8::try_from(token)
            .map(|byte| vec![byte])
            .map_err(|_| TokError::InvalidToken(token))
    }
    fn try_decode_piece(&self, token: i32) -> Result<String, TokError> {
        Ok(String::from_utf8(self.try_decode_piece_bytes(token)?).unwrap())
    }
    fn try_decode(&self, tokens: &[i32]) -> Result<String, TokError> {
        tokens
            .iter()
            .map(|&token| self.try_decode_piece(token))
            .collect()
    }
    fn decode(&self, tokens: &[i32]) -> String {
        self.try_decode(tokens).unwrap()
    }
    fn n_vocab(&self) -> u32 {
        *CHAT_STOPS.iter().max().unwrap() as u32 + 1
    }
    fn bos(&self) -> Option<i32> {
        None
    }
    fn eos(&self) -> Option<i32> {
        Some(CHAT_STOPS[0])
    }
}

#[derive(Default)]
struct Sink {
    events: Vec<String>,
    pieces: Vec<u8>,
    ticks: usize,
    fail_tick: Option<usize>,
    fail_piece: Option<usize>,
}
impl GenerationSink for Sink {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        if self.fail_piece == Some(self.pieces.len()) {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "piece disconnected",
            ));
        }
        self.events
            .push(format!("piece:{}", String::from_utf8_lossy(bytes)));
        self.pieces.extend_from_slice(bytes);
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        self.events.push("tick".into());
        self.ticks += 1;
        if self.fail_tick == Some(self.ticks) {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "tick disconnected",
            ));
        }
        Ok(())
    }
}

fn packet(
    visible: &[u8],
    retained: usize,
    stop: Option<Glm5NextDFlashStop>,
    eos: Option<i32>,
) -> Glm5NextDFlashPacket {
    let tokens: Vec<i32> = visible.iter().copied().map(i32::from).collect();
    let emitted_tokens = tokens.iter().copied().chain(eos).collect();
    Glm5NextDFlashPacket {
        tokens,
        emitted_tokens,
        next_carry: visible
            .last()
            .map(|&token| Glm5NextDFlashCarry::Bonus(token.into())),
        stop_reason: stop,
        accepted_drafts: visible.len().saturating_sub(1),
        retain_input_count: retained,
        verification_arithmetic: None,
        retained_arithmetic: None,
        anchor_only: false,
        costs: Glm5NextDFlashCosts::default(),
    }
}

fn no_step(_: i32, _: &mut dyn GenerationSink) -> Result<Glm5NextDFlashPacket, BackendFailure> {
    panic!("terminal packet must not be stepped")
}

#[test]
fn packets_emit_once_piece_then_tick_and_preserve_actual_eos() {
    let eos = CHAT_STOPS[1];
    let mut packets = [
        packet(b"bc", 2, None, None),
        packet(b"d", 2, Some(Glm5NextDFlashStop::Eos), Some(eos)),
    ]
    .into_iter();
    let mut carries = Vec::new();
    let mut sink = Sink::default();
    let generation = collect_packets(
        packet(b"a", 0, None, None),
        8,
        &Bytes,
        &mut sink,
        |carry, _| {
            carries.push(carry);
            Ok(packets.next().unwrap())
        },
    )
    .unwrap();
    assert_eq!(carries, [b'a' as i32, b'c' as i32]);
    assert_eq!(generation.tokens, [97, 98, 99, 100, eos]);
    assert_eq!(generation.transitions, 4);
    assert_eq!(sink.pieces, b"abcd");
    assert_eq!(
        sink.events,
        [
            "piece:a", "tick", "piece:b", "tick", "piece:c", "tick", "piece:d", "tick", "tick"
        ]
    );
    let outcome = crate::serve::outcome::finish_generation(12, &generation, 0, 0.0);
    assert_eq!(outcome.usage.output_tokens, 5);
    assert_eq!(outcome.usage.cached_tokens, 0);
    assert_eq!(outcome.stats.unwrap().matched_tokens, 0);
    assert_eq!(
        outcome.end,
        crate::serve::output_partition::GenerationEnd::StopToken(eos)
    );
}

#[test]
fn every_first_eos_is_counted_without_emission_or_transition() {
    for &eos in &CHAT_STOPS {
        let mut sink = Sink::default();
        let generation = collect_packets(
            packet(b"", 0, Some(Glm5NextDFlashStop::Eos), Some(eos)),
            1,
            &Bytes,
            &mut sink,
            no_step,
        )
        .unwrap();
        assert_eq!(generation.tokens, [eos]);
        assert_eq!(generation.transitions, 0);
        assert_eq!(generation.stop_reason, StopReason::Eos);
        assert!(generation.first_token_callback_ms.is_none());
        assert!(sink.pieces.is_empty());
    }
}

#[test]
fn accepted_eos_needs_no_carry_and_bonus_eos_is_not_emitted() {
    for carry in [None, Some(Glm5NextDFlashCarry::Bonus(CHAT_STOPS[0]))] {
        let mut last = packet(b"b", 2, Some(Glm5NextDFlashStop::Eos), Some(CHAT_STOPS[0]));
        last.next_carry = carry;
        let mut last = Some(last);
        let mut sink = Sink::default();
        let generation =
            collect_packets(packet(b"a", 0, None, None), 3, &Bytes, &mut sink, |_, _| {
                Ok(last.take().unwrap())
            })
            .unwrap();
        assert_eq!(generation.tokens, [97, 98, CHAT_STOPS[0]]);
        assert_eq!(generation.transitions, 2);
        assert_eq!(sink.pieces, b"ab");
    }
}

#[test]
fn token_limit_leaves_last_selected_token_unconsumed() {
    let mut sink = Sink::default();
    let generation = collect_packets(
        packet(b"a", 0, Some(Glm5NextDFlashStop::TokenLimit), None),
        1,
        &Bytes,
        &mut sink,
        no_step,
    )
    .unwrap();
    assert_eq!(generation.tokens, [97]);
    assert_eq!(generation.transitions, 0);
    assert_eq!(generation.stop_reason, StopReason::TokenLimit);
}

#[test]
fn halfway_packet_disconnect_is_typed_and_never_steps_again() {
    for fail_piece in [false, true] {
        let mut sink = Sink {
            fail_piece: fail_piece.then_some(2),
            fail_tick: (!fail_piece).then_some(2),
            ..Default::default()
        };
        let mut steps = 0;
        let result = collect_packets(packet(b"a", 0, None, None), 8, &Bytes, &mut sink, |_, _| {
            steps += 1;
            assert_eq!(steps, 1);
            Ok(packet(b"bcd", 3, None, None))
        });
        assert!(
            matches!(result, Err(BackendFailure::Aborted(error)) if error.kind() == io::ErrorKind::BrokenPipe)
        );
        assert_eq!(sink.pieces, b"ab");
        assert_eq!(steps, 1);
    }
}

#[test]
fn prefill_polls_heartbeat_and_preserves_original_sink_error() {
    let mut sink = Sink {
        fail_tick: Some(3),
        ..Default::default()
    };
    let result: Result<(), BackendFailure> = with_ticks(&mut sink, |cancelled| {
        assert!(!cancelled());
        assert!(cancelled());
        assert!(cancelled());
        Err(Glm5NextDFlashError::Closed)
    });
    assert!(
        matches!(result, Err(BackendFailure::Aborted(error)) if error.to_string() == "tick disconnected")
    );
    assert_eq!(sink.ticks, 3);
}

#[test]
fn prefill_abort_before_execution_and_after_success_are_typed() {
    let mut sink = Sink {
        fail_tick: Some(1),
        ..Default::default()
    };
    let result: Result<(), BackendFailure> = with_ticks(&mut sink, |_| panic!("already cancelled"));
    assert!(matches!(result, Err(BackendFailure::Aborted(_))));
    let mut sink = Sink {
        fail_tick: Some(2),
        ..Default::default()
    };
    let result = with_ticks(&mut sink, |_| Ok(()));
    assert!(matches!(result, Err(BackendFailure::Aborted(_))));
}

#[test]
fn malformed_packets_fail_before_publication_without_serial_fallback() {
    for packet in [
        packet(b"a", 0, Some(Glm5NextDFlashStop::Eos), None),
        packet(b"a", 0, Some(Glm5NextDFlashStop::Capacity), None),
        packet(b"a", 0, Some(Glm5NextDFlashStop::TokenLimit), None),
        packet(b"a", 1, None, None),
        packet(b"", 0, None, None),
        packet(b"abcd", 0, None, None),
    ] {
        let mut sink = Sink::default();
        assert!(collect_packets(packet, 3, &Bytes, &mut sink, no_step).is_err());
        assert!(sink.pieces.is_empty());
    }
}

#[test]
fn complete_transport_and_output_future_reserves_are_checked_sums() {
    assert_eq!(cpu_reserve(300, 7).unwrap(), 328);
    assert!(cpu_reserve(u64::MAX, 1).is_err());
    assert!(cpu_reserve(0, usize::MAX).is_err());
}

#[test]
fn inconsistent_history_and_callback_tokens_fail_before_emission() {
    for damage in 0..6 {
        let mut packet = packet(b"ab", 1, Some(Glm5NextDFlashStop::TokenLimit), None);
        match damage {
            0 => packet.emitted_tokens.reverse(),
            1 => {
                packet.emitted_tokens.pop();
            }
            2 => {
                packet.tokens.pop();
            }
            3 => {
                packet.stop_reason = Some(Glm5NextDFlashStop::Eos);
                packet.emitted_tokens[1] = CHAT_STOPS[0];
                packet.tokens[1] = CHAT_STOPS[0];
            }
            4 => {
                packet.emitted_tokens[0] = CHAT_STOPS[0];
                packet.tokens[0] = CHAT_STOPS[0];
            }
            _ => {
                packet.emitted_tokens[0] = -1;
                packet.tokens[0] = -1;
            }
        }
        let mut sink = Sink::default();
        assert!(collect_packets(packet, 2, &Bytes, &mut sink, no_step).is_err());
        assert!(sink.pieces.is_empty());
    }
}

#[test]
fn suppressed_eos_still_consumes_output_budget() {
    let mut sink = Sink::default();
    let packet = packet(b"a", 1, Some(Glm5NextDFlashStop::Eos), Some(CHAT_STOPS[0]));
    assert!(collect_packets(packet, 1, &Bytes, &mut sink, no_step).is_err());
    assert!(sink.pieces.is_empty());
}

#[test]
fn initial_selection_uses_measured_cost_not_total_minus_prefill() {
    use std::time::Duration;
    let mut packet = packet(b"a", 0, Some(Glm5NextDFlashStop::TokenLimit), None);
    packet.costs.total = Duration::from_millis(100);
    packet.costs.prefill = Duration::from_millis(20);
    packet.costs.append = Duration::from_millis(10);
    packet.costs.selection = Duration::from_micros(2500);
    let generation = collect_packets(packet, 1, &Bytes, &mut Sink::default(), no_step).unwrap();
    assert_eq!(generation.first_token_selection_ms, 2.5);
    assert_eq!(generation.first_token_ready_ms, Some(2.5));
    assert!(generation.first_token_callback_ms.unwrap() >= 2.5);
    assert!(generation.wall_ms >= 2.5);
}

#[test]
fn sampled_proposal_is_separate_temperature_only_and_requires_selector() {
    let mut target = SamplingConfig::glm5_next(42);
    assert!(proposal_config(target, false).is_err());
    let proposal = proposal_config(target, true).unwrap().unwrap();
    assert_eq!(proposal.temperature, target.temperature);
    assert_eq!(
        (proposal.top_k, proposal.top_p, proposal.min_p),
        (0, 1.0, 0.0)
    );
    assert_eq!(
        proposal.seed,
        target.seed ^ crate::dflash::DFLASH_PROPOSAL_SEED_XOR
    );
    target.temperature = 0.0;
    assert!(proposal_config(target, false).unwrap().is_none());
}
