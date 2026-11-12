//! Amortized event handoff and join price. Both queues are committed before
//! waiting; every round trips actual dependent data, including a final join
//! consumer on queue zero. Fits remove fixed command-buffer startup costs.
use super::resource_barriers::{dispatch, mix};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLEvent,
};
use qwen_llm::metal::{Buffer, MetalContext, Pipeline};
type Error = Box<dyn std::error::Error>;
const THREADS: usize = 128;
const DEPTHS: [usize; 5] = [1, 4, 16, 32, 64];

#[derive(Clone, Copy)]
enum Mode {
    OneEncoder,
    SplitEncoders,
    Event,
    Shared,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::OneEncoder => "one-queue-one-encoder",
            Self::SplitEncoders => "one-queue-split-encoders",
            Self::Event => "two-queue-event",
            Self::Shared => "two-queue-shared-event",
        }
    }
}
const MODES: [Mode; 4] = [
    Mode::OneEncoder,
    Mode::SplitEncoders,
    Mode::Event,
    Mode::Shared,
];
struct Probe {
    ctx: MetalContext,
    other: MetalContext,
    pso: Pipeline,
    buffers: [Buffer; 2],
}
impl Probe {
    fn event(&self, shared: bool) -> Result<Retained<ProtocolObject<dyn MTLEvent>>, Error> {
        if shared {
            Ok(ProtocolObject::from_retained(
                self.ctx.device.newSharedEvent().ok_or("no shared event")?,
            ))
        } else {
            self.ctx.device.newEvent().ok_or_else(|| "no event".into())
        }
    }
    fn sample(&self, mode: Mode, rounds: usize) -> Result<(f64, f64), Error> {
        unsafe {
            let seed = std::slice::from_raw_parts_mut(
                self.buffers[0].contents().as_ptr().cast::<u32>(),
                THREADS,
            );
            for (i, x) in seed.iter_mut().enumerate() {
                *x = i as u32 + 1;
            }
            self.buffers[1]
                .contents()
                .as_ptr()
                .write_bytes(0, THREADS * 4);
        }
        let started = std::time::Instant::now();
        let a = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or("no first command buffer")?;
        let mut other = None;
        match mode {
            Mode::OneEncoder => {
                let enc = a.computeCommandEncoder().ok_or("no encoder")?;
                for i in 0..2 * rounds + 1 {
                    dispatch(
                        &enc,
                        &self.pso,
                        &self.buffers[i % 2],
                        &self.buffers[(i + 1) % 2],
                        1,
                    );
                }
                enc.endEncoding();
            }
            Mode::SplitEncoders => {
                for i in 0..2 * rounds + 1 {
                    let enc = a.computeCommandEncoder().ok_or("no encoder")?;
                    dispatch(
                        &enc,
                        &self.pso,
                        &self.buffers[i % 2],
                        &self.buffers[(i + 1) % 2],
                        1,
                    );
                    enc.endEncoding();
                }
            }
            Mode::Event | Mode::Shared => {
                let forward = self.event(matches!(mode, Mode::Shared))?;
                let backward = self.event(matches!(mode, Mode::Shared))?;
                let b = self
                    .other
                    .queue
                    .commandBuffer()
                    .ok_or("no second command buffer")?;
                for i in 0..rounds {
                    let value = i as u64 + 1;
                    let enc = a.computeCommandEncoder().ok_or("no producer encoder")?;
                    dispatch(&enc, &self.pso, &self.buffers[0], &self.buffers[1], 1);
                    enc.endEncoding();
                    a.encodeSignalEvent_value(&forward, value);
                    b.encodeWaitForEvent_value(&forward, value);
                    let enc = b.computeCommandEncoder().ok_or("no consumer encoder")?;
                    dispatch(&enc, &self.pso, &self.buffers[1], &self.buffers[0], 1);
                    enc.endEncoding();
                    b.encodeSignalEvent_value(&backward, value);
                    a.encodeWaitForEvent_value(&backward, value);
                }
                // Queue zero consumes the joined result: a timing-only wait
                // without this read would not validate memory visibility.
                let enc = a.computeCommandEncoder().ok_or("no join encoder")?;
                dispatch(&enc, &self.pso, &self.buffers[0], &self.buffers[1], 1);
                enc.endEncoding();
                other = Some(b);
            }
        }
        a.commit();
        if let Some(b) = &other {
            b.commit();
        }
        a.waitUntilCompleted();
        if let Some(b) = &other {
            b.waitUntilCompleted();
        }
        if let Some(e) = a.error() {
            return Err(e.to_string().into());
        }
        let (mut start, mut end) = (a.GPUStartTime(), a.GPUEndTime());
        if let Some(b) = &other {
            if let Some(e) = b.error() {
                return Err(e.to_string().into());
            }
            start = start.min(b.GPUStartTime());
            end = end.max(b.GPUEndTime());
        }
        let wall_us = started.elapsed().as_secs_f64() * 1e6;
        if end <= start {
            return Err("invalid GPU timestamps".into());
        }
        let actual = unsafe {
            std::slice::from_raw_parts(self.buffers[1].contents().as_ptr().cast::<u32>(), THREADS)
        };
        if !actual
            .iter()
            .enumerate()
            .all(|(i, &x)| x == mix(i as u32 + 1, (2 * rounds + 1) as u32))
        {
            return Err(format!("{} rounds {rounds}: join bitwise failure", mode.name()).into());
        }
        Ok(((end - start) * 1e6, wall_us))
    }
}
pub(super) fn run() -> Result<(), Error> {
    let reps: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(61);
    if reps == 0 {
        return Err("positive repetitions required".into());
    }
    let ctx = MetalContext::new()?;
    let selection = std::env::args().nth(3).unwrap_or_else(|| "both".into());
    if !matches!(selection.as_str(), "both" | "event" | "shared") {
        return Err("event kind must be both, event or shared".into());
    }
    let selected: Vec<usize> = (0..4)
        .filter(|&i| {
            i < 2
                || selection == "both"
                || (i == 2 && selection == "event")
                || (i == 3 && selection == "shared")
        })
        .collect();
    println!(
        "{}",
        serde_json::json!({"kind":"event_device", "describe":ctx.describe(), "reps":reps, "roundtrips":DEPTHS, "handoffs_per_roundtrip":2, "iterations":1, "threads":THREADS, "clock":"min GPUStartTime..max GPUEndTime over both command buffers"})
    );
    let probe = Probe {
        other: ctx.with_new_command_queue()?,
        pso: ctx.pipeline("kernel_dispatch_barrier_short")?,
        buffers: [
            ctx.buffer_uninit(THREADS * 4)?,
            ctx.buffer_uninit(THREADS * 4)?,
        ],
        ctx,
    };
    let warm = std::time::Instant::now();
    while warm.elapsed().as_millis() < 500 {
        let command = probe.ctx.queue.commandBuffer().ok_or("no warmup command")?;
        let enc = command.computeCommandEncoder().ok_or("no warmup encoder")?;
        for _ in 0..32 {
            dispatch(
                &enc,
                &probe.pso,
                &probe.buffers[0],
                &probe.buffers[1],
                20000,
            );
        }
        enc.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if let Some(e) = command.error() {
            return Err(e.to_string().into());
        }
    }
    let mut medians = [[0.0; 5]; 4];
    for (d, &rounds) in DEPTHS.iter().enumerate() {
        for &i in &selected {
            probe.sample(MODES[i], rounds)?;
        }
        let mut times: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(reps));
        for rep in 0..reps {
            for step in 0..selected.len() {
                let index = if rep % 2 == 0 {
                    (rep + step) % selected.len()
                } else {
                    (rep + selected.len() - step) % selected.len()
                };
                let i = selected[index];
                let (gpu_us, wall_us) = probe.sample(MODES[i], rounds)?;
                times[i].push(gpu_us);
                println!(
                    "{}",
                    serde_json::json!({"kind":"event_sample", "mode":MODES[i].name(),"roundtrips":rounds,"rep":rep,"gpu_us":gpu_us,"wall_us":wall_us,"bitwise":true})
                );
            }
        }
        for &i in &selected {
            medians[i][d] = super::median(&mut times[i]);
            println!(
                "{}",
                serde_json::json!({"kind":"event_summary", "mode":MODES[i].name(),"roundtrips":rounds,"gpu_us_p50":medians[i][d],"gpu_us_p10":times[i][reps/10],"gpu_us_p90":times[i][reps*9/10],"bitwise":true})
            );
        }
    }
    let xs: Vec<_> = DEPTHS.iter().map(|&n| n as f64).collect();
    let fits: Vec<_> = medians.iter().map(|ys| super::fit(&xs, ys)).collect();
    for &i in &selected {
        println!(
            "{}",
            serde_json::json!({"kind":"event_fit","mode":MODES[i].name(),"gpu_us_per_roundtrip":fits[i].0,"gpu_us_intercept":fits[i].1,"marginal_event_us_per_handoff":(fits[i].0-fits[1].0)/2.0,"total_extra_us_per_handoff_vs_one_encoder":(fits[i].0-fits[0].0)/2.0})
        );
    }
    Ok(())
}
