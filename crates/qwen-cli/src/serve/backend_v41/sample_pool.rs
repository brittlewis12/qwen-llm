//! Persistent threads that run the samplers of one batched decode step.
//!
//! The batch scheduler's owner thread hands each member's [`Sampler`] and
//! logits to the pool by value and gets both back, with the sampled result,
//! before it consumes any member. A sampler is therefore only ever touched by
//! one thread at a time, draws exactly once per job, and the owner remains the
//! only thread that publishes, cancels or fails a request. Sampling is a pure
//! function of the sampler and the logits, so where it runs cannot change a
//! token, a candidate index, an error or the sampler's next state.
use qwen_llm::sampling::{SampledToken, Sampler, SamplingError};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

pub(super) type Sampled = Result<SampledToken, SamplingError>;

/// One member's sampling work: moved to a thread and back.
pub(super) struct SampleJob {
    pub(super) sampler: Sampler,
    pub(super) logits: Vec<f32>,
    pub(super) sampled: Option<Sampled>,
}

struct Ticket {
    slot: usize,
    job: SampleJob,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

impl Ticket {
    fn run(mut self) -> Self {
        let job = &mut self.job;
        match catch_unwind(AssertUnwindSafe(|| job.sampler.sample(&job.logits))) {
            Ok(sampled) => job.sampled = Some(sampled),
            Err(panic) => self.panic = Some(panic),
        }
        self
    }
}

struct Worker {
    jobs: Option<Sender<Ticket>>,
    thread: Option<JoinHandle<()>>,
}

pub(super) struct SamplePool {
    workers: Vec<Worker>,
    done: Receiver<Ticket>,
}

impl SamplePool {
    /// `workers` threads beside the owner, which samples one share itself.
    pub(super) fn new(workers: usize) -> std::io::Result<Self> {
        let (done_tx, done) = channel();
        let mut pool = Self {
            workers: Vec::with_capacity(workers),
            done,
        };
        for index in 0..workers {
            let (jobs, rx) = channel::<Ticket>();
            let done = done_tx.clone();
            let thread = std::thread::Builder::new()
                .name(format!("v41-sample-{index}"))
                .spawn(move || {
                    while let Ok(ticket) = rx.recv() {
                        if done.send(ticket.run()).is_err() {
                            break;
                        }
                    }
                })?;
            pool.workers.push(Worker {
                jobs: Some(jobs),
                thread: Some(thread),
            });
        }
        Ok(pool)
    }

    pub(super) fn workers(&self) -> usize {
        self.workers.len()
    }

    /// Sample every job and return them in input order. Job `i` runs on lane
    /// `i % (workers + 1)`; lane 0 is the calling thread. A panic in any job
    /// resumes on the caller after every job has returned.
    pub(super) fn sample_all(&mut self, jobs: Vec<SampleJob>) -> Vec<SampleJob> {
        let lanes = self.workers.len() + 1;
        let count = jobs.len();
        let mut own = Vec::new();
        let mut sent = 0usize;
        for (slot, job) in jobs.into_iter().enumerate() {
            let ticket = Ticket {
                slot,
                job,
                panic: None,
            };
            let lane = slot % lanes;
            if lane == 0 {
                own.push(ticket);
                continue;
            }
            let sender = self.workers[lane - 1].jobs.as_ref().expect("live worker");
            match sender.send(ticket) {
                Ok(()) => sent += 1,
                Err(returned) => own.push(returned.0),
            }
        }
        let mut finished: Vec<Option<Ticket>> = (0..count).map(|_| None).collect();
        for ticket in own {
            let ticket = ticket.run();
            let slot = ticket.slot;
            finished[slot] = Some(ticket);
        }
        for _ in 0..sent {
            let ticket = self.done.recv().expect("sampling workers outlive the pool");
            let slot = ticket.slot;
            finished[slot] = Some(ticket);
        }
        let mut panic = None;
        let jobs = finished
            .into_iter()
            .map(|ticket| {
                let ticket = ticket.expect("every job returns");
                if panic.is_none() {
                    panic = ticket.panic;
                }
                ticket.job
            })
            .collect();
        if let Some(panic) = panic {
            resume_unwind(panic);
        }
        jobs
    }
}

impl Drop for SamplePool {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            worker.jobs.take();
        }
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen_llm::sampling::SamplingConfig;

    fn row(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 24) as f32 * 8.0
            })
            .collect()
    }

    /// More jobs than lanes, across several steps: each job returns in input
    /// order with the result, logits and next sampler state of sampling it on
    /// the owner, and a failed sample (NaN) leaves its sampler unchanged.
    #[test]
    fn pooled_jobs_return_in_order_with_owner_results_and_states() {
        let mut pool = SamplePool::new(2).unwrap();
        assert_eq!(pool.workers(), 2);
        let config = |seed| SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed,
        };
        let mut pooled: Vec<Sampler> = (0..7).map(|s| Sampler::new(config(s)).unwrap()).collect();
        let mut owner: Vec<Sampler> = (0..7).map(|s| Sampler::new(config(s)).unwrap()).collect();
        for step in 0..4u64 {
            let rows: Vec<Vec<f32>> = (0..7u64)
                .map(|member| {
                    let mut logits = row(step * 16 + member, 4099);
                    if step == 2 && member == 3 {
                        logits[17] = f32::NAN;
                    }
                    logits
                })
                .collect();
            let jobs = pooled
                .drain(..)
                .zip(rows.clone())
                .map(|(sampler, logits)| SampleJob {
                    sampler,
                    logits,
                    sampled: None,
                })
                .collect();
            for ((job, sampler), logits) in
                pool.sample_all(jobs).into_iter().zip(&mut owner).zip(&rows)
            {
                assert_eq!(job.sampled, Some(sampler.sample(logits)));
                assert!(
                    job.logits
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(logits.iter().map(|v| v.to_bits()))
                );
                assert_eq!(job.sampler.draws(), sampler.draws());
                pooled.push(job.sampler);
            }
        }
        // Equal draw counts and equal next samples: equal RNG states.
        let probe = row(99, 4099);
        for (pooled, owner) in pooled.iter_mut().zip(&mut owner) {
            assert_eq!(pooled.sample(&probe), owner.sample(&probe));
        }
    }
}
