//! Bounded retention for optional committed-state checkpoints. Payloads, pins,
//! expiry and process-memory admission remain in the existing snapshot cache.
use super::items::ServeRequest;
use crate::model_request::Turn;
use anyhow::{Result, bail};
use qwen_llm::snapshot_policy::EntryId;
use sha2::{Digest, Sha256};
use std::ffi::OsStr;

/// Default-on extra committed checkpoints for tier-1 serve conversations; 0 restores the previous capture and lookup paths.
const ENABLED_ENV: &str = "QWEN_SERVE_CHECKPOINT_LADDER";
/// Optional checkpoint retention ceiling in MiB, or auto (default: 8192 MiB); effective budget is also capped at half the existing snapshot-cache budget. Zero disables extra checkpoints; no additional memory entitlement.
const BUDGET_ENV: &str = "QWEN_SERVE_CHECKPOINT_LADDER_MIB";
const MAX_POINTS: usize = 5;
const MIN_STRIDE: usize = 4096;

#[derive(Clone, Copy, Debug)]
pub(super) struct Config {
    enabled: bool,
    bytes: u64,
}

impl Config {
    pub(super) fn from_env() -> Result<Self> {
        Self::parse(
            std::env::var_os(ENABLED_ENV).as_deref(),
            std::env::var_os(BUDGET_ENV).as_deref(),
        )
    }

    fn parse(enabled: Option<&OsStr>, budget: Option<&OsStr>) -> Result<Self> {
        let enabled = match enabled {
            None => true,
            Some(v) if v == "1" => true,
            Some(v) if v == "0" => false,
            _ => bail!("{ENABLED_ENV} must be 0 or 1"),
        };
        let mib = match budget {
            None => 8192,
            Some(v) if v == "auto" => 8192,
            Some(v) => v
                .to_str()
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or_else(|| {
                    anyhow::anyhow!("{BUDGET_ENV} must be auto or a nonnegative MiB integer")
                })?,
        };
        let bytes = mib
            .checked_mul(1 << 20)
            .ok_or_else(|| anyhow::anyhow!("{BUDGET_ENV} overflows bytes"))?;
        Ok(Self { enabled, bytes })
    }

    #[cfg(test)]
    pub(super) fn testing(bytes: u64) -> Self {
        Self {
            enabled: true,
            bytes,
        }
    }

    pub(super) fn budget(self, parent: u64) -> u64 {
        if self.enabled {
            self.bytes.min(parent / 2)
        } else {
            0
        }
    }
}

/// Stateless Responses has no conversation id. The first user turn identifies
/// a conversation family; forks keep it, independent roots get independent
/// retention. This is a retention hint, NEVER restore compatibility: namespaces
/// and the complete exact token-prefix key still authorize every restore.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Conversation([u8; 32]);

#[cfg(test)]
impl Conversation {
    pub(super) fn testing(n: u8) -> Self {
        Self([n; 32])
    }
}

pub(super) fn conversation(request: &ServeRequest) -> Option<Conversation> {
    let user = request
        .model_request
        .turns
        .iter()
        .find_map(|turn| match turn {
            Turn::User(text) => Some(text),
            _ => None,
        })?;
    Some(Conversation(Sha256::digest(user.as_bytes()).into()))
}

#[derive(Clone)]
struct Point {
    id: Option<EntryId>,
    conversation: Conversation,
    position: usize,
    bytes: u64,
    sequence: u64,
}

#[derive(Default)]
pub(super) struct Ladder {
    budget: u64,
    points: Vec<Point>,
    sequence: u64,
}

impl Ladder {
    pub(super) fn new(config: Config, parent: u64) -> Self {
        Self {
            budget: config.budget(parent),
            ..Self::default()
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.budget > 0
    }
    pub(super) fn budget(&self) -> u64 {
        self.budget
    }
    pub(super) fn bytes(&self) -> u64 {
        self.points.iter().map(|p| p.bytes).sum()
    }
    pub(super) fn len(&self) -> usize {
        self.points.len()
    }
    pub(super) fn contains(&self, id: EntryId) -> bool {
        self.points.iter().any(|p| p.id == Some(id))
    }
    pub(super) fn retain(&mut self, mut live: impl FnMut(EntryId) -> bool) {
        self.points.retain(|point| point.id.is_some_and(&mut live));
    }

    /// Plan without eviction/allocation. A failed capture must not discard old
    /// points. Keep the endpoints and thin the closest interior pair first;
    /// this preserves spaced older history instead of only the newest turns.
    pub(super) fn plan(
        &self,
        conversation: Conversation,
        position: usize,
        bytes: u64,
        parent: usize,
        removable: impl Fn(EntryId) -> bool,
        rank: impl Fn(EntryId) -> usize,
    ) -> Option<Vec<EntryId>> {
        if self.budget == 0
            || bytes > self.budget
            || position < MIN_STRIDE
            || position.checked_sub(parent)? < MIN_STRIDE
        {
            return None;
        }
        let mut points = self.points.clone();
        points.push(Point {
            id: None,
            conversation,
            position,
            bytes,
            sequence: self.sequence + 1,
        });
        let mut removed = Vec::new();
        while points
            .iter()
            .filter(|p| p.conversation == conversation)
            .count()
            > MAX_POINTS
        {
            let mut group: Vec<_> = points
                .iter()
                .enumerate()
                .filter(|(_, p)| p.conversation == conversation)
                .map(|(i, _)| i)
                .collect();
            group.sort_by_key(|&i| (points[i].position, points[i].sequence));
            let interior = group
                .windows(3)
                .filter_map(|w| {
                    let p = &points[w[1]];
                    p.id.filter(|&id| removable(id)).map(|_| {
                        (
                            points[w[2]].position.saturating_sub(points[w[0]].position),
                            p.sequence,
                            w[1],
                        )
                    })
                })
                .min();
            let victim = interior.map(|(_, _, i)| i).or_else(|| {
                group
                    .into_iter()
                    .filter(|&i| points[i].id.is_some_and(&removable))
                    .min_by_key(|&i| points[i].sequence)
            })?;
            removed.push(points.remove(victim).id?);
        }
        while points
            .iter()
            .try_fold(0u64, |sum, p| sum.checked_add(p.bytes))?
            > self.budget
        {
            let victim = points
                .iter()
                .enumerate()
                .filter_map(|(i, p)| {
                    p.id.filter(|&id| removable(id))
                        .map(|id| (rank(id), std::cmp::Reverse(p.sequence), i))
                })
                .max()
                .map(|(_, _, i)| i)?;
            removed.push(points.remove(victim).id?);
        }
        Some(removed)
    }

    pub(super) fn inserted(
        &mut self,
        id: EntryId,
        conversation: Conversation,
        position: usize,
        bytes: u64,
    ) {
        self.sequence += 1;
        self.points.push(Point {
            id: Some(id),
            conversation,
            position,
            bytes,
            sequence: self.sequence,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(n: u8) -> Conversation {
        Conversation([n; 32])
    }

    #[test]
    fn config_is_bounded_by_parent_and_has_explicit_rollback() {
        let config = Config::parse(None, None).unwrap();
        assert_eq!(config.budget(16 << 30), 8 << 30);
        assert_eq!(config.budget(100), 50);
        assert_eq!(
            Config::parse(Some(OsStr::new("0")), None)
                .unwrap()
                .budget(u64::MAX),
            0
        );
        assert_eq!(
            Config::parse(None, Some(OsStr::new("0")))
                .unwrap()
                .budget(u64::MAX),
            0
        );
        for s in ["-1", "garbage", "18446744073709551615"] {
            assert!(Config::parse(None, Some(OsStr::new(s))).is_err());
        }
        assert!(Config::parse(Some(OsStr::new("true")), None).is_err());
    }

    #[test]
    fn spaced_history_survives_small_recent_turns() {
        let mut ladder = Ladder::new(
            Config {
                enabled: true,
                bytes: 1000,
            },
            2000,
        );
        for (i, position) in [14000, 30000, 47000, 63000, 65000, 67000, 69000]
            .into_iter()
            .enumerate()
        {
            // The stride is relative to a matching parent; distinct capture
            // branches can have a nearer coordinate without a matching key.
            let victims = ladder
                .plan(key(1), position, 10, 0, |_| true, |_| 0)
                .unwrap();
            ladder.points.retain(|p| !victims.contains(&p.id.unwrap()));
            ladder.inserted(EntryId::external(i as u64), key(1), position, 10);
        }
        assert_eq!(
            ladder.points.iter().map(|p| p.position).collect::<Vec<_>>(),
            [14000, 30000, 47000, 63000, 69000]
        );
        assert!(
            ladder
                .plan(key(1), 70000, 10, 69000, |_| true, |_| 0)
                .is_none()
        );
    }

    #[test]
    fn byte_limit_pins_and_conversation_limits_are_independent() {
        let mut ladder = Ladder::new(
            Config {
                enabled: true,
                bytes: 30,
            },
            60,
        );
        for i in 0..3 {
            ladder.inserted(EntryId::external(i), key(i as u8), 10000, 10);
        }
        assert!(
            ladder
                .plan(key(9), 20000, 10, 0, |_| false, |_| 0)
                .is_none()
        );
        let victims = ladder
            .plan(
                key(9),
                20000,
                10,
                0,
                |id| id != EntryId::external(0),
                |id| usize::from(id == EntryId::external(2)),
            )
            .unwrap();
        assert_eq!(victims, [EntryId::external(2)]);
        assert_eq!(
            ladder.len(),
            3,
            "planning never evicts before a successful copy"
        );
        assert!(ladder.plan(key(9), 20000, 31, 0, |_| true, |_| 0).is_none());
    }
}
