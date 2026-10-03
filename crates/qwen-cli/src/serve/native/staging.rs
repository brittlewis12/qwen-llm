//! One owner-authorized setup command on the existing joined artifact worker.

use super::registry::{MatrixKey, Registry, STAGING_OVERHEAD_BYTES, Staged};
use anyhow::{Context, Result, ensure};
use std::{
    collections::BTreeSet,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, SyncSender, sync_channel},
    },
};

#[derive(Default)]
pub(crate) struct Plan {
    pub(crate) registry: Option<Arc<Registry>>,
    pub(crate) keys: BTreeSet<MatrixKey>,
    pub(crate) matrix_bytes: u64,
}
impl Plan {
    pub(crate) fn new(registry: Option<Arc<Registry>>, keys: BTreeSet<MatrixKey>) -> Result<Self> {
        let matrix_bytes = if keys.is_empty() {
            0
        } else {
            registry
                .as_ref()
                .context("missing diagnostic registry")?
                .matrix_bytes(&keys)?
        };
        Ok(Self {
            registry,
            keys,
            matrix_bytes,
        })
    }
    pub(crate) fn cpu_bytes(&self) -> Result<u64> {
        if self.keys.is_empty() {
            return Ok(0);
        }
        self.matrix_bytes
            .checked_add(STAGING_OVERHEAD_BYTES)
            .context("staging memory overflow")
    }
}

pub(super) struct Request {
    registry: Arc<Registry>,
    keys: BTreeSet<MatrixKey>,
    matrix_bytes: u64,
    reserve: u64,
    abandoned: Arc<AtomicBool>,
    response: SyncSender<Result<Staged>>,
    #[cfg(test)]
    pub(super) hook: Option<Hook>,
}
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Point {
    Checkpoint,
    Handoff,
}
#[cfg(test)]
pub(super) type Hook = Arc<dyn Fn(Point) -> Result<()> + Send + Sync>;
pub(super) struct WaitGuard(Arc<AtomicBool>);
impl Drop for WaitGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

pub(super) fn request(
    plan: &Plan,
    reserve: u64,
) -> Result<(Request, Receiver<Result<Staged>>, WaitGuard)> {
    ensure!(!plan.keys.is_empty(), "empty staging command");
    let registry = plan
        .registry
        .as_ref()
        .context("missing staging registry")?
        .clone();
    ensure!(
        registry.matrix_bytes(&plan.keys)? == plan.matrix_bytes,
        "staging plan changed"
    );
    ensure!(
        reserve
            >= plan
                .matrix_bytes
                .checked_add(super::registry::STAGING_OVERHEAD_BYTES)
                .context("staging budget overflow")?,
        "staging allowance omits matrices or scratch"
    );
    let abandoned = Arc::new(AtomicBool::new(false));
    let (response, incoming) = sync_channel(1);
    Ok((
        Request {
            registry,
            keys: plan.keys.clone(),
            matrix_bytes: plan.matrix_bytes,
            reserve,
            abandoned: abandoned.clone(),
            response,
            #[cfg(test)]
            hook: None,
        },
        incoming,
        WaitGuard(abandoned),
    ))
}

impl Request {
    pub(super) fn refuse(self, message: &str) {
        let _ = self.response.try_send(Err(anyhow::anyhow!("{message}")));
    }
    pub(super) fn run(
        self,
        mut checkpoint: impl FnMut() -> Result<()>,
        mut remaining: impl FnMut() -> Option<u64>,
    ) {
        let result = self.registry.stage(&self.keys, self.matrix_bytes, || {
            ensure!(
                !self.abandoned.load(Ordering::Acquire),
                "staging owner abandoned setup"
            );
            checkpoint()?;
            #[cfg(test)]
            if let Some(hook) = &self.hook {
                hook(Point::Checkpoint)?;
            }
            // Keep the full original future reserve, even after some staged bytes
            // enter process usage. Conservative over-refusal is preferable to
            // spending the sequence/workspace or pending durable-write allowance.
            super::super::transport_memory::admit_resident_transport(self.reserve, remaining())
                .map_err(|cause| anyhow::anyhow!("{}", cause.message))
        });
        let _ = self.response.try_send(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn plan() -> (crate::linear_transport::tests::Fixture, Plan) {
        let (files, registry) = super::super::registry::tests::fitted_fixture();
        let value = json!({"id":"a","lens":"fit","mode":"full_vocabulary","top_k":1,
            "scope":{"layers":{"kind":"values","values":[0,2]},"prefill":{"kind":"values","values":[0]}}});
        let readouts = super::super::readouts::Plan::compile_with_registry(
            &[value],
            3,
            1,
            1,
            32,
            Some(registry.clone()),
            2,
        )
        .unwrap();
        (files, Plan::new(Some(registry), readouts.matrices).unwrap())
    }

    #[test]
    fn staging_preserves_full_authorized_reserve_at_every_checkpoint() {
        let (_files, plan) = plan();
        let reserve = plan.matrix_bytes + super::super::registry::STAGING_OVERHEAD_BYTES + 12345;
        for remaining in [None, Some(0), Some(reserve - 1), Some(reserve)] {
            let (work, response, _guard) = request(&plan, reserve).unwrap();
            work.run(|| Ok(()), || remaining);
            let result = response.recv().unwrap();
            assert_eq!(
                result.is_ok(),
                matches!(remaining, Some(0)) || remaining == Some(reserve)
            );
        }
        assert!(request(&plan, plan.matrix_bytes).is_err());
        let (work, response, _guard) = request(&plan, reserve).unwrap();
        let mut checks = 0;
        work.run(
            || Ok(()),
            || {
                checks += 1;
                Some(if checks < 7 { reserve } else { reserve - 1 })
            },
        );
        assert!(response.recv().unwrap().is_err());
        assert_eq!(checks, 7);
        let (work, response, _guard) = request(&plan, reserve).unwrap();
        work.run(|| Ok(()), || Some(reserve));
        assert_eq!(response.recv().unwrap().unwrap().matrices.len(), 2);
    }

    #[test]
    fn private_abandonment_stops_scanning_and_disconnected_response_never_blocks() {
        let (_files, plan) = plan();
        let reserve = plan.matrix_bytes + super::super::registry::STAGING_OVERHEAD_BYTES;
        let (work, response, guard) = request(&plan, reserve).unwrap();
        let mut guard = Some(guard);
        let mut checkpoints = 0;
        work.run(
            || {
                checkpoints += 1;
                if checkpoints == 3 {
                    guard.take();
                }
                Ok(())
            },
            || Some(reserve),
        );
        assert_eq!(
            response.recv().unwrap().err().unwrap().to_string(),
            "staging owner abandoned setup"
        );
        assert_eq!(checkpoints, 3);
        let (work, response, _guard) = request(&plan, reserve).unwrap();
        drop(response);
        work.run(|| Ok(()), || Some(reserve));
    }
}
