use super::*;
use serde_json::json;

fn direction(id: &str, normalization: &str) -> Value {
    json!({"id":id,"lens":"fit","row":{"kind":"token_id","token_id":1},"normalization":normalization})
}
fn operation(id: &str, kind: &str, coefficient: f32) -> Value {
    json!({"id":id,"scope":{"layers":{"kind":"values","values":[0]},"prefill":{"kind":"values","values":[0]},"decode":{"kind":"values","values":[0]}},
        "action":{"kind":kind,"direction":"unit","coefficient":coefficient}})
}

#[test]
fn ordered_nonzero_applications_share_projection_but_not_normalized_rows() {
    let (_files, registry) = super::super::registry::tests::fitted_fixture();
    let mut add = operation("add", "fixed_add", 1.0);
    add["action"]["direction"] = json!("raw");
    let plan = Plan::compile(
        &[direction("raw", "as_stored"), direction("unit", "unit_l2")],
        &[
            add,
            operation("zero", "projection_ablate", -0.0),
            operation("ablate", "projection_ablate", 0.5),
            operation("relative", "residual_l2_fraction", 0.1),
        ],
        3,
        2,
        3,
        32,
        2,
        Some(&registry),
    )
    .unwrap();
    assert_eq!(plan.applications, 6);
    assert_eq!(plan.events[&0][&0], [0, 2, 3]);
    assert_eq!(plan.events[&2][&0], [0, 2, 3]);
    assert_eq!(plan.applied_ids(2, 0), ["add", "ablate", "relative"]);
    assert_eq!(plan.direction_rows, 2);
    assert_eq!(plan.projected_rows, 1);
    assert_eq!(plan.matrices.len(), 1);
    let mut buffers = Vec::new();
    let (gpu, cpu) = plan
        .price(2, |n| {
            buffers.push(n);
            Ok(n.div_ceil(256) * 256)
        })
        .unwrap();
    assert_eq!(buffers, [8, 8, 8, 8, 8, 4]);
    assert_eq!(gpu, 7 * 256);
    assert_eq!(cpu, 8 * 8 + 4 * 1024 * 1024);
    assert!(plan.price(u64::MAX, Ok).is_err());
}

#[test]
fn zero_controls_validate_without_staging_or_scope_expansion() {
    let (_files, registry) = super::super::registry::tests::fitted_fixture();
    let dirs = [direction("unit", "unit_l2")];
    let zero = operation("zero", "projection_ablate", 0.0);
    let plan = Plan::compile(&dirs, &[zero.clone()], 3, 2, 3, 32, 2, Some(&registry)).unwrap();
    assert!(plan.events.is_empty() && plan.rows.is_empty() && plan.matrices.is_empty());
    assert_eq!(
        plan.price(u64::MAX, |_| panic!("zero allocation")).unwrap(),
        (0, 0)
    );
    for (field, value) in [("direction", json!("missing")), ("kind", json!("unknown"))] {
        let mut bad = zero.clone();
        bad["action"][field] = value;
        assert!(Plan::compile(&dirs, &[bad], 3, 2, 3, 32, 2, Some(&registry)).is_err());
    }
    assert!(
        Plan::compile(
            &[direction("unit", "as_stored")],
            &[zero.clone()],
            3,
            2,
            3,
            32,
            2,
            Some(&registry)
        )
        .is_err()
    );
    let mut absent = zero.clone();
    absent["scope"]["layers"] = json!({"kind":"values","values":[1]});
    assert!(Plan::compile(&dirs, &[absent], 3, 2, 3, 32, 2, Some(&registry)).is_err());
    let mut huge = zero;
    huge["scope"]["layers"] = json!({"kind":"all"});
    assert!(Plan::compile(&dirs, &[huge], u32::MAX, 2, 3, 32, 2, Some(&registry)).is_err());
    let mut unreachable = operation("decode", "fixed_add", 1.0);
    unreachable["scope"]
        .as_object_mut()
        .unwrap()
        .remove("prefill");
    assert!(Plan::compile(&dirs, &[unreachable], 3, 2, 1, 32, 2, Some(&registry)).is_err());
}

#[test]
fn profile_stages_union_once_and_operation_only_jobs_do_not_require_readouts() {
    let mut fixture = super::super::CpuFixture::new();
    let (_files, registry) = super::super::registry::tests::fitted_fixture();
    let profile = std::sync::Arc::get_mut(&mut fixture.profile).unwrap();
    profile.registry = Some(registry.clone());
    profile.plain_readouts = true;
    let mut request = fixture.request("ops");
    request["preconditions"]["asset_identities"] =
        json!({"fit":registry.asset("fit").unwrap()["identity"]});
    request["diagnostics"] = json!({"directions":[direction("unit","unit_l2")],"operations":[operation("add","fixed_add",1.0)],"readouts":[]});
    let compile = |v: &Value| {
        fixture
            .profile
            .prepare(&crate::serve::lens_http::input::Request::parse(v).unwrap())
            .ok()
            .unwrap()
    };
    let prepared = compile(&request);
    assert!(prepared.readouts.events.is_empty());
    assert!(!prepared.interventions.events.is_empty());
    assert_eq!(prepared.staging.keys.len(), 1);
    assert_eq!(prepared.staging.matrix_bytes, 8);
    assert!(prepared.staging.cpu_bytes().unwrap() > 8);
    request["diagnostics"]["readouts"] = json!([{"id":"read","lens":"fit","mode":"full_vocabulary","top_k":2,
        "scope":{"layers":{"kind":"values","values":[0]},"prefill":{"kind":"values","values":[0]}}}]);
    let combined = compile(&request);
    assert_eq!(combined.staging.keys.len(), 1);
    assert_eq!(combined.staging.matrix_bytes, 8);
    let record: Value = serde_json::from_slice(&combined.record).unwrap();
    assert_eq!(record["effective_operation_ids"], json!(["add"]));
    assert_eq!(record["resolved_scopes"].as_array().unwrap().len(), 2);
}

#[test]
fn application_publication_follows_successful_consumption_without_readout_events() {
    use super::super::{
        execute::{self, TokenEngine},
        writer::Writer,
    };
    use crate::serve::jobs::state::{Counters, JobState};
    use std::sync::Arc;
    let mut fixture = super::super::CpuFixture::new();
    let (_files, registry) = super::super::registry::tests::fitted_fixture();
    let profile = Arc::get_mut(&mut fixture.profile).unwrap();
    profile.registry = Some(registry.clone());
    profile.plain_readouts = true;
    struct Engine<'a> {
        plan: &'a Plan,
        fail: bool,
    }
    impl TokenEngine for Engine<'_> {
        fn forward(&mut self, _: i32, _: u32, _: bool) -> Result<Vec<f32>> {
            ensure!(!self.fail, "synthetic failed forward");
            Ok(vec![0., 1.])
        }
        fn observe(
            &mut self,
            _: i32,
            position: u32,
            _: &[f32],
            counters: &Counters,
            sink: &super::super::Sink,
        ) -> Result<()> {
            assert_eq!(
                u64::from(position) + 1,
                counters.consumed_prompt_tokens + counters.consumed_generated_tokens
            );
            self.plan.publish_applications(position, counters, sink)
        }
    }
    for fail in [false, true] {
        let key = if fail { "failed" } else { "success" };
        let mut request = fixture.request(key);
        request["preconditions"]["asset_identities"] =
            json!({"fit":registry.asset("fit").unwrap()["identity"]});
        request["diagnostics"] = json!({"directions":[direction("unit","unit_l2")],"operations":[operation("add","fixed_add",1.0),operation("ablate","projection_ablate",0.5)],"readouts":[]});
        let prepared = fixture
            .profile
            .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
            .ok()
            .unwrap();
        let id = fixture.store.accept(key, &request, true).unwrap().status.id;
        let writer = Writer::spawn(
            fixture.store.clone(),
            id.clone(),
            fixture.store.control(&id).unwrap(),
            &prepared,
            Default::default(),
        )
        .unwrap();
        writer.wait_ready().unwrap();
        let mut engine = Engine {
            plan: &prepared.interventions,
            fail,
        };
        let outcome = super::super::with_staged(
            &prepared,
            writer.sink(),
            prepared.staging.cpu_bytes().unwrap(),
            |_| {
                execute::run_engine(&prepared, writer.sink(), &[], &mut engine, |_| {
                    Ok(b"x".to_vec())
                })
            },
        );
        writer.finish(outcome).unwrap();
        assert_eq!(
            fixture.store.status(&id).unwrap().state,
            if fail {
                JobState::Failed
            } else {
                JobState::Completed
            }
        );
        let records = fixture.store.result(&id, None, 128).unwrap().records;
        let apps = records
            .iter()
            .filter(|r| r["kind"] == "operation_application")
            .collect::<Vec<_>>();
        assert_eq!(apps.len(), if fail { 0 } else { 4 });
        if !fail {
            assert_eq!(
                apps.iter()
                    .map(|r| r["id"].as_str().unwrap())
                    .collect::<Vec<_>>(),
                ["add", "ablate", "add", "ablate"]
            );
            assert!(
                apps.iter()
                    .all(|r| r["provenance"] == "successful_original_forward")
            );
        }
    }
}
