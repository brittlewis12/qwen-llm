//! Publish whole-site pairs only after their original forward has completed.

use super::*;

pub(super) fn publish(
    plan: &Plan,
    token: i32,
    position: u32,
    layer: u32,
    counters: &Counters,
    applied: &[&str],
    before: Option<&[f32]>,
    after: &[f32],
    sink: &Sink,
) -> Result<()> {
    let requests = plan.pairs.events.get(&position).and_then(|e| e.get(&layer));
    ensure!(
        before.is_some() == requests.is_some(),
        "paired capture/site mismatch"
    );
    let phase = if u64::from(position) < counters.prompt_tokens {
        Phase::Prefill
    } else {
        Phase::Decode
    };
    let index = if phase == Phase::Prefill {
        u64::from(position)
    } else {
        u64::from(position) - counters.prompt_tokens
    };
    let before_key = format!("before-{position}-{layer}");
    let after_key = format!("source-{position}-{layer}");
    let metrics = before
        .map(|b| super::super::measurements::metrics(b, after))
        .transpose()?;
    if let Some(before) = before {
        sink.array(json!({"key":before_key,"quantity":"source_residual_before","phase":phase,"index":index,"position":position,
            "source_layer":layer,"input_token_id":token,"capture_stage":"post_block_before_operations","applied_operation_ids":[],
            "site_operation_ids":applied,"provenance":"original_forward"}),before,phase,counters)?;
    }
    checkpoint(sink)?;
    publish_source(plan, token, position, layer, counters, applied, after, sink)?;
    if let Some(requests) = requests {
        for &request in requests {
            checkpoint(sink)?;
            sink.record(json!({"kind":"residual_pair","id":plan.pairs.requests[request].id,"before_key":before_key,"after_key":after_key,
                "phase":phase,"index":index,"position":position,"source_layer":layer,"input_token_id":token,
                "capture_stage":"whole_post_block_program","applied_operation_ids":applied,"provenance":"original_forward","metrics":metrics}),phase,counters);
        }
    }
    checkpoint(sink)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::native::{CpuFixture, Outcome, run_cpu_readouts, writer};
    use std::sync::Arc;

    #[test]
    fn pair_only_producer_deduplicates_arrays_and_reopens_without_inference() {
        let mut fixture = CpuFixture::new();
        Arc::get_mut(&mut fixture.profile).unwrap().plain_readouts = true;
        let mut request = fixture.request("pairs");
        let pair = |id| json!({"id":id,"scope":{"layers":{"kind":"all"},"prefill":{"kind":"values","values":[0]},"decode":{"kind":"all"}}});
        request["diagnostics"] = json!({"directions":[],"operations":[],"readouts":[],"residual_pairs":[pair("a"),pair("b")]});
        let prepared = fixture
            .profile
            .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
            .ok()
            .unwrap();
        assert_eq!(prepared.readouts.head_evaluations, 0);
        assert!(prepared.staging.keys.is_empty());
        let id = fixture
            .store
            .accept_with_archive("pairs", &request, true, prepared.readouts.archive_bytes)
            .unwrap()
            .status
            .id;
        let writer = writer::Writer::spawn(
            fixture.store.clone(),
            id.clone(),
            fixture.store.control(&id).unwrap(),
            &prepared,
            Default::default(),
        )
        .unwrap();
        writer.wait_ready().unwrap();
        let outcome = run_cpu_readouts(&prepared, writer.sink(), &fixture.profile.tokenizer);
        writer.finish(outcome).unwrap();
        let records = fixture.store.result(&id, None, 128).unwrap().records;
        let arrays = records
            .iter()
            .filter(|r| r["kind"] == "retained_array")
            .collect::<Vec<_>>();
        let pairs = records
            .iter()
            .filter(|r| r["kind"] == "residual_pair")
            .collect::<Vec<_>>();
        assert_eq!(pairs.len(), prepared.readouts.pairs.rows);
        assert_eq!(arrays.len(), 2 * prepared.readouts.pairs.sites());
        assert_eq!(
            arrays
                .iter()
                .map(|r| r["array"]["byte_length"].as_u64().unwrap())
                .sum::<u64>(),
            prepared.readouts.archive_bytes
        );
        assert!(
            !records
                .iter()
                .any(|r| r["kind"] == "readout" || r["kind"] == "operation_application")
        );
        for pair in pairs {
            assert_eq!(pair["metrics"]["delta_norm"], 0.);
            assert_eq!(pair["applied_operation_ids"], json!([]));
            let before = arrays
                .iter()
                .find(|a| a["key"] == pair["before_key"])
                .unwrap();
            let after = arrays
                .iter()
                .find(|a| a["key"] == pair["after_key"])
                .unwrap();
            assert!(
                before["seq"].as_u64() < after["seq"].as_u64()
                    && after["seq"].as_u64() < pair["seq"].as_u64()
            );
        }
        let replacement = Arc::new(
            crate::serve::jobs::store::JobStore::open(
                &fixture.root.join("replacement"),
                Default::default(),
            )
            .unwrap(),
        );
        drop(std::mem::replace(&mut fixture.store, replacement));
        let reopened = crate::serve::jobs::store::JobStore::open(
            &fixture.root.join("jobs"),
            Default::default(),
        )
        .unwrap();
        assert_eq!(reopened.result(&id, None, 128).unwrap().records, records);
    }

    #[test]
    fn failure_after_committed_before_never_publishes_dangling_pairs() {
        for cancel in [false, true] {
            let mut fixture = CpuFixture::new();
            Arc::get_mut(&mut fixture.profile).unwrap().plain_readouts = true;
            let mut request = fixture.request("prefix");
            request["diagnostics"] = json!({"residual_pairs":[{"id":"p","scope":{"layers":{"kind":"values","values":[0]},"prefill":{"kind":"values","values":[0]}}}],"readouts":[],"operations":[],"directions":[]});
            let prepared = fixture
                .profile
                .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
                .ok()
                .unwrap();
            let id = fixture
                .store
                .accept_with_archive("prefix", &request, true, prepared.readouts.archive_bytes)
                .unwrap()
                .status
                .id;
            let writer = writer::Writer::spawn(
                fixture.store.clone(),
                id.clone(),
                fixture.store.control(&id).unwrap(),
                &prepared,
                Default::default(),
            )
            .unwrap();
            writer.wait_ready().unwrap();
            let counters = Counters {
                prompt_tokens: prepared.prompt.len() as u64,
                consumed_prompt_tokens: 1,
                ..Default::default()
            };
            writer
                .sink()
                .array(
                    json!({"key":"before-0-0","quantity":"source_residual_before"}),
                    &[1., 2.],
                    Phase::Prefill,
                    &counters,
                )
                .unwrap();
            let deadline = Instant::now() + std::time::Duration::from_secs(5);
            while !fixture
                .store
                .result(&id, None, 128)
                .unwrap()
                .records
                .iter()
                .any(|r| r["key"] == "before-0-0")
            {
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            if cancel {
                fixture.store.control(&id).unwrap().cancel();
            } else {
                fixture
                    .store
                    .fail_once(crate::serve::jobs::store::FaultPoint::ArrayPartialWrite);
            }
            // Exercise the real after-array/FIFO dependency path with before already durable.
            let _ = publish_source(
                &prepared.readouts,
                prepared.prompt[0],
                0,
                0,
                &counters,
                &[],
                &[3., 4.],
                writer.sink(),
            );
            if checkpoint(writer.sink()).is_ok() {
                writer.sink().record(json!({"kind":"residual_pair","id":"p","before_key":"before-0-0","after_key":"source-0-0"}),Phase::Prefill,&counters);
            }
            writer.finish(Outcome::interrupted(counters)).unwrap();
            let records = fixture.store.result(&id, None, 128).unwrap().records;
            assert_eq!(
                records
                    .iter()
                    .filter(|r| r["kind"] == "retained_array")
                    .count(),
                1
            );
            assert!(!records.iter().any(|r| r["kind"] == "residual_pair"));
            let replacement = Arc::new(
                crate::serve::jobs::store::JobStore::open(
                    &fixture.root.join("replacement"),
                    Default::default(),
                )
                .unwrap(),
            );
            drop(std::mem::replace(&mut fixture.store, replacement));
            let reopened = crate::serve::jobs::store::JobStore::open(
                &fixture.root.join("jobs"),
                Default::default(),
            )
            .unwrap();
            assert_eq!(reopened.result(&id, None, 128).unwrap().records, records);
            assert_eq!(
                std::fs::metadata(fixture.root.join("jobs").join(id).join("arrays.bin"))
                    .unwrap()
                    .len(),
                8
            );
        }
    }
}
