//! GLM-5.3-Flash text chat over the serve and run lanes: the output
//! grammar (reasoning pre-opened by `<|assistant|><think>`, closed by the
//! first `</think>`).
use super::partition_preopened::{PreopenedGrammar, PreopenedPartition};
use qwen_llm::glm5_next_chat::{CHAT_STOPS, THINK_CLOSE, THINK_OPEN};

pub(crate) fn partition() -> PreopenedPartition {
    PreopenedPartition::new(PreopenedGrammar {
        family: "GLM-5.3-Flash",
        open: THINK_OPEN.into(),
        closes: &[THINK_CLOSE],
        stops: &CHAT_STOPS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::output_partition::GenerationEnd;
    use crate::serve::partition::PartitionEvent;

    fn collect(events: &[PartitionEvent]) -> (String, String, usize) {
        let (mut r, mut v, mut close) = (String::new(), String::new(), 0);
        for event in events {
            match event {
                PartitionEvent::Reasoning(t) => r.push_str(t),
                PartitionEvent::Visible(t) => v.push_str(t),
                PartitionEvent::ReasoningClosed => close += 1,
                PartitionEvent::FunctionCall(_) => panic!("no-tools grammar parsed a call"),
            }
        }
        (r, v, close)
    }

    #[test]
    fn every_byte_split_closes_once_and_keeps_later_tags_visible() {
        for opener in ["", "<think>"] {
            for reason in ["", "plan \u{1f389} <|user|> <think>nested"] {
                let visible = "answer \u{2192}</think><think>x</think><tool_call>t</tool_call>";
                let all = format!("{opener}{reason}</think>{visible}");
                for split in 0..=all.len() {
                    for stop in CHAT_STOPS {
                        let mut p = partition();
                        let mut events = Vec::new();
                        p.push(&all.as_bytes()[..split], &mut events);
                        p.push(&all.as_bytes()[split..], &mut events);
                        assert!(p.closed());
                        p.finish(GenerationEnd::StopToken(stop), &mut events)
                            .unwrap();
                        assert_eq!(collect(&events), (reason.into(), visible.into(), 1));
                        assert!(matches!(events.first(), Some(PartitionEvent::Reasoning(_))));
                    }
                }
            }
        }
    }

    #[test]
    fn truncated_reasoning_is_incomplete_and_bad_stops_fail() {
        for text in ["", "p</thi", "<thi", "p</think_x>", "<think>"] {
            for end in [
                GenerationEnd::TokenLimit,
                GenerationEnd::StopToken(154_820),
                GenerationEnd::StopToken(154_827),
            ] {
                let mut p = partition();
                let mut events = Vec::new();
                for byte in text.as_bytes() {
                    p.push(&[*byte], &mut events);
                }
                let result = p.finish(end, &mut events);
                assert_eq!(result.is_ok(), end.is_token_limit(), "{text:?} {end:?}");
                if let Err(error) = result {
                    assert!(error.message.contains("GLM-5.3-Flash"), "{}", error.message);
                }
                assert!(collect(&events).1.is_empty());
                assert_eq!(collect(&events).2, 0);
            }
        }
        let mut p = partition();
        let mut events = Vec::new();
        p.push(b"a</think>x", &mut events);
        // <|assistant|> is not a released stop.
        assert!(
            p.finish(GenerationEnd::StopToken(154_828), &mut events)
                .is_err()
        );
    }
}
