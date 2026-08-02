use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use num_bigint::{BigInt, Sign};
use num_traits::{One, Signed, ToPrimitive, Zero};
use objc2_metal::{MTLBuffer, MTLResource, MTLStorageMode};
use qwen_llm::{
    loader::Model,
    metal::MetalTensor,
    metal_dflash::{MetalDFlashLayerMajorScratch, prefill_tokens_with_multi_hidden},
    model::{Arch, ArchKind},
    runtime::{Runtime, SequenceConfig},
    sampling::{SAMPLER_ALGORITHM_VERSION, Sampler, SamplingConfig},
    tensor::GgmlType,
    tokenizer::{Tokenizer, token_ids_sha256_i32le},
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::hint::black_box;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use crate::host_validity::HostSnapshot;

const SCHEMA: &str = "qwen-lm-head-screening-oracle/v0664";
const MODEL_PATH: &str = "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
const MODEL_BYTES: u64 = 22_134_528_992;
const MODEL_SHA256: &str = "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61";
const PROMPT_PATH: &str = "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt";
const PROMPT_BYTES: usize = 1_891;
const PROMPT_SHA256: &str = "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474";
const TOKEN_SHA256: &str = "fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f";
const OUTPUT_SHA256: &str = "122386a599833ffec3a266e48dd8a90aedc65e24c31323ce2388f40d7cc7b30c";
const BASE_MODEL_NAME: &str = "Qwen3.6 35B A3B";
const BASENAME: &str = "Qwen3.6-35B-A3B";
const OUTPUT_OFFSET: u64 = 10_990_048;
const HIDDEN: usize = 2_048;
const VOCAB: usize = 248_320;
const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 210;
const BLOCKS: usize = 8;
const ROW_BYTES: usize = 1_680;
const OUTPUT_BYTES: usize = 417_177_600;
const CALLS: [usize; 6] = [0, 1, 7, 31, 63, 127];
const STOP_IDS: [i32; 1] = [248_046];
const TOKENS: usize = 128;
const PROMPT_TOKENS: usize = 419;
const PRUNING_FLOOR: usize = 198_656;
const BYTE_LIMIT: u64 = 125_153_280;
const DOT_EXP: i32 = -173;
const PACKET_PATH: &str = "target/profiles/v0664-generic-lm-head-screening-oracle-a3b-p1";
const PREDECESSOR_DECISION_SHA256: &str =
    "759750cfb5a8af2621e3c8c0dc9a8d120ad7a1df46ddeda223451f61e731dbcc";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ArtifactStatus {
    Complete,
    CompleteGeneration,
    NotRunCaptureCoverage,
}

#[derive(Parser, Debug)]
pub struct LmHeadScreeningOracleArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt_file: PathBuf,
    #[arg(long)]
    tokens: usize,
    #[arg(long, value_delimiter = ',')]
    capture_calls: Vec<usize>,
    #[arg(long)]
    packet_dir: PathBuf,
    #[arg(long)]
    attest_no_other_user_gpu_workload: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ReadinessCommandArguments {
    model: String,
    prompt_file: String,
    tokens: usize,
    capture_calls: Vec<usize>,
    packet_dir: String,
    attest_no_other_user_gpu_workload: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReadinessArtifact {
    schema: String,
    build_identity: super::BuildIdentity,
    executable_identity: ExecutableIdentity,
    operator_attestation: bool,
    command_arguments: ReadinessCommandArguments,
    predecessor_decision_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ModelNameIdentity {
    base_model_name: String,
    basename: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Dyadic {
    coefficient: BigInt,
    exponent: i32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct BigIntPayloadTracker {
    largest_observed_coefficient_payload_bits: u64,
}

impl BigIntPayloadTracker {
    fn observe_coefficient(&mut self, coefficient: &BigInt) {
        let bits = if coefficient.is_zero() {
            1
        } else {
            coefficient.bits() + 1
        };
        self.largest_observed_coefficient_payload_bits =
            self.largest_observed_coefficient_payload_bits.max(bits);
    }

    fn observe_dyadic(&mut self, value: &Dyadic) {
        self.observe_coefficient(&value.coefficient);
    }

    fn merge(&mut self, other: Self) {
        self.largest_observed_coefficient_payload_bits = self
            .largest_observed_coefficient_payload_bits
            .max(other.largest_observed_coefficient_payload_bits);
    }
}

fn tracked_dyadic_mul(left: &Dyadic, right: &Dyadic, tracker: &mut BigIntPayloadTracker) -> Dyadic {
    tracker.observe_dyadic(left);
    tracker.observe_dyadic(right);
    let coefficient = &left.coefficient * &right.coefficient;
    tracker.observe_coefficient(&coefficient);
    let result = Dyadic::new(coefficient, left.exponent + right.exponent);
    tracker.observe_dyadic(&result);
    result
}

fn tracked_dyadic_add(left: &Dyadic, right: &Dyadic, tracker: &mut BigIntPayloadTracker) -> Dyadic {
    tracker.observe_dyadic(left);
    tracker.observe_dyadic(right);
    let exponent = left.exponent.min(right.exponent);
    let left_shifted = &left.coefficient
        << usize::try_from(left.exponent - exponent).expect("nonnegative dyadic shift");
    let right_shifted = &right.coefficient
        << usize::try_from(right.exponent - exponent).expect("nonnegative dyadic shift");
    tracker.observe_coefficient(&left_shifted);
    tracker.observe_coefficient(&right_shifted);
    let sum = left_shifted + right_shifted;
    tracker.observe_coefficient(&sum);
    let result = Dyadic::new(sum, exponent);
    tracker.observe_dyadic(&result);
    result
}

fn tracked_dyadic_cmp(
    left: &Dyadic,
    right: &Dyadic,
    tracker: &mut BigIntPayloadTracker,
) -> Ordering {
    tracker.observe_dyadic(left);
    tracker.observe_dyadic(right);
    let exponent = left.exponent.min(right.exponent);
    let left_shifted = &left.coefficient
        << usize::try_from(left.exponent - exponent).expect("nonnegative dyadic shift");
    let right_shifted = &right.coefficient
        << usize::try_from(right.exponent - exponent).expect("nonnegative dyadic shift");
    tracker.observe_coefficient(&left_shifted);
    tracker.observe_coefficient(&right_shifted);
    left_shifted.cmp(&right_shifted)
}

impl Dyadic {
    fn new(coefficient: BigInt, exponent: i32) -> Self {
        if coefficient.is_zero() {
            return Self {
                coefficient,
                exponent: 0,
            };
        }
        let mut coefficient = coefficient;
        let mut exponent = exponent;
        while (&coefficient & BigInt::one()).is_zero() {
            coefficient >>= 1usize;
            exponent += 1;
        }
        Self {
            coefficient,
            exponent,
        }
    }

    fn zero() -> Self {
        Self::new(BigInt::zero(), 0)
    }

    fn add(&self, other: &Self) -> Self {
        let exponent = self.exponent.min(other.exponent);
        Self::new(
            (&self.coefficient << usize::try_from(self.exponent - exponent).unwrap())
                + (&other.coefficient << usize::try_from(other.exponent - exponent).unwrap()),
            exponent,
        )
    }

    fn mul(&self, other: &Self) -> Self {
        Self::new(
            &self.coefficient * &other.coefficient,
            self.exponent + other.exponent,
        )
    }

    fn square(&self) -> Self {
        self.mul(self)
    }

    fn cmp(&self, other: &Self) -> Ordering {
        let exponent = self.exponent.min(other.exponent);
        (&self.coefficient << usize::try_from(self.exponent - exponent).unwrap())
            .cmp(&(&other.coefficient << usize::try_from(other.exponent - exponent).unwrap()))
    }

    fn signed_bits(&self) -> u64 {
        if self.coefficient.is_zero() {
            1
        } else {
            self.coefficient.abs().bits() + 1
        }
    }

    fn to_f64_rn(&self) -> Result<f64> {
        let c = self
            .coefficient
            .to_f64()
            .context("dyadic coefficient exceeds f64")?;
        let value = c * (2.0f64).powi(self.exponent);
        ensure!(value.is_finite(), "dyadic conversion is nonfinite");
        Ok(value)
    }
}

fn f32_dyadic(value: f32) -> Result<Dyadic> {
    ensure!(value.is_finite(), "nonfinite F32");
    let bits = value.to_bits();
    let sign = if bits >> 31 == 0 { 1i64 } else { -1 };
    let raw_exp = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;
    if raw_exp == 0 {
        Ok(Dyadic::new(BigInt::from(sign * i64::from(fraction)), -149))
    } else {
        Ok(Dyadic::new(
            BigInt::from(sign * i64::from((1 << 23) | fraction)),
            raw_exp as i32 - 127 - 23,
        ))
    }
}

fn f16_dyadic(bits: u16) -> Result<Dyadic> {
    let sign = if bits >> 15 == 0 { 1i64 } else { -1 };
    let raw_exp = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    ensure!(raw_exp != 0x1f, "nonfinite binary16 scale");
    if raw_exp == 0 {
        Ok(Dyadic::new(BigInt::from(sign * i64::from(fraction)), -24))
    } else {
        Ok(Dyadic::new(
            BigInt::from(sign * i64::from(0x400 | fraction)),
            raw_exp as i32 - 15 - 10,
        ))
    }
}

fn f64_dyadic(value: f64) -> Result<Dyadic> {
    ensure!(value.is_finite(), "nonfinite F64");
    let bits = value.to_bits();
    let sign = if bits >> 63 == 0 { 1i64 } else { -1 };
    let raw_exp = (bits >> 52) & 0x7ff;
    let fraction = bits & ((1u64 << 52) - 1);
    if raw_exp == 0 {
        Ok(Dyadic::new(BigInt::from(sign) * fraction, -1074))
    } else {
        Ok(Dyadic::new(
            BigInt::from(sign) * ((1u64 << 52) | fraction),
            raw_exp as i32 - 1023 - 52,
        ))
    }
}

fn next_up_f64(value: f64) -> f64 {
    if value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

fn next_down_f64(value: f64) -> f64 {
    -next_up_f64(-value)
}

fn next_up_f32(value: f32) -> f32 {
    if value == f32::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

fn dyadic_interval(value: &Dyadic) -> Result<(f64, f64)> {
    dyadic_interval_profiled(value, &mut BigIntPayloadTracker::default())
}

fn dyadic_interval_profiled(
    value: &Dyadic,
    tracker: &mut BigIntPayloadTracker,
) -> Result<(f64, f64)> {
    let rn = value.to_f64_rn()?;
    let rn_dyadic = f64_dyadic(rn)?;
    match tracked_dyadic_cmp(&rn_dyadic, value, tracker) {
        Ordering::Less => Ok((rn, next_up_f64(rn))),
        Ordering::Greater => Ok((next_down_f64(rn), rn)),
        Ordering::Equal => Ok((rn, rn)),
    }
}

fn f32_upper(value: f64) -> Result<f32> {
    ensure!(value.is_finite() && value >= 0.0, "invalid norm upper");
    let cast = value as f32;
    ensure!(cast.is_finite(), "F64-to-F32 upper overflow");
    Ok(if (cast as f64) < value {
        next_up_f32(cast)
    } else {
        cast
    })
}

fn gamma_sum_upper(terms: &[f64]) -> Result<f64> {
    ensure!(!terms.is_empty(), "empty positive accumulation");
    let mut sum = 0.0;
    for &term in terms {
        ensure!(term.is_finite() && term >= 0.0, "invalid positive term");
        sum += term;
        ensure!(sum.is_finite(), "positive accumulation overflow");
    }
    let n = terms.len() as f64;
    let eps = 2.0f64.powi(-53);
    let gamma_up = next_up_f64(next_up_f64(n * eps) / next_down_f64(1.0 - n * eps));
    let denom_down = next_down_f64(1.0 - gamma_up);
    ensure!(denom_down > 0.0 && denom_down <= 1.0);
    Ok(next_up_f64(sum / denom_down))
}

fn norm_upper(exact_square: &Dyadic, terms: &[f64]) -> Result<f32> {
    norm_upper_profiled(exact_square, terms, &mut BigIntPayloadTracker::default())
}

fn norm_upper_profiled(
    exact_square: &Dyadic,
    terms: &[f64],
    tracker: &mut BigIntPayloadTracker,
) -> Result<f32> {
    ensure!(exact_square.coefficient.sign() != Sign::Minus);
    let sum_up = gamma_sum_upper(terms)?;
    let root_up = next_up_f64(sum_up.sqrt());
    let upper = f32_upper(root_up)?;
    ensure!(upper.is_finite() && upper >= 0.0);
    let upper_dyadic = f32_dyadic(upper)?;
    let upper_square = tracked_dyadic_mul(&upper_dyadic, &upper_dyadic, tracker);
    ensure!(
        tracked_dyadic_cmp(&upper_square, exact_square, tracker) != Ordering::Less,
        "norm upper does not contain exact sum of squares"
    );
    Ok(upper)
}

#[derive(Clone, Copy)]
struct Q6Block<'a> {
    bytes: &'a [u8],
}

impl<'a> Q6Block<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self> {
        ensure!(bytes.len() == BLOCK_BYTES, "Q6_K block length mismatch");
        Ok(Self { bytes })
    }

    fn d_bits(self) -> u16 {
        u16::from_le_bytes([self.bytes[208], self.bytes[209]])
    }

    fn quant_scale(self, index: usize) -> Result<(i8, i8)> {
        ensure!(index < BLOCK);
        let half = index / 128;
        let local = index % 128;
        let quarter = local / 32;
        let lane = local % 32;
        let ql = self.bytes[half * 64 + lane + (quarter & 1) * 32];
        let low = if quarter < 2 { ql & 0x0f } else { ql >> 4 };
        let high = (self.bytes[128 + half * 32 + lane] >> (quarter * 2)) & 0x03;
        let quant = (low | (high << 4)) as i8 - 32;
        let scale_index = half * 8 + quarter * 2 + lane / 16;
        Ok((quant, self.bytes[192 + scale_index] as i8))
    }

    fn coefficient(self, index: usize) -> Result<i16> {
        let (q, scale) = self.quant_scale(index)?;
        Ok(i16::from(q) * i16::from(scale))
    }

    fn exact_square_profiled(self, tracker: &mut BigIntPayloadTracker) -> Result<Dyadic> {
        let d = f16_dyadic(self.d_bits())?;
        tracker.observe_dyadic(&d);
        let mut integer = BigInt::zero();
        tracker.observe_coefficient(&integer);
        for i in 0..BLOCK {
            let c = i64::from(self.coefficient(i)?);
            integer += c * c;
            tracker.observe_coefficient(&integer);
        }
        let integer = Dyadic::new(integer, 0);
        tracker.observe_dyadic(&integer);
        let d_square = tracked_dyadic_mul(&d, &d, tracker);
        Ok(tracked_dyadic_mul(&d_square, &integer, tracker))
    }

    fn outward_norm_f32(self) -> Result<f32> {
        let d = half::f16::from_bits(self.d_bits()).to_f64();
        ensure!(d.is_finite());
        let mut terms = [0.0f64; BLOCK];
        for i in 0..BLOCK {
            let w = d * f64::from(self.coefficient(i)?);
            let square = w * w;
            ensure!(square.is_finite() && square >= 0.0);
            terms[i] = square;
        }
        let sum_up = gamma_sum_upper(&terms)?;
        f32_upper(next_up_f64(sum_up.sqrt()))
    }
}

fn exact_dot_profiled(
    row: &[u8],
    hidden: &[f32],
    tracker: &mut BigIntPayloadTracker,
) -> Result<Dyadic> {
    ensure!(
        !hidden.is_empty() && hidden.len().is_multiple_of(BLOCK),
        "exact dot hidden width is not Q6_K aligned"
    );
    let blocks = hidden.len() / BLOCK;
    ensure!(
        row.len() == blocks * BLOCK_BYTES,
        "exact dot row length does not match hidden width"
    );
    let mut dot_tracker = BigIntPayloadTracker::default();
    let mut coefficient = BigInt::zero();
    dot_tracker.observe_coefficient(&coefficient);
    for block_index in 0..blocks {
        let block =
            Q6Block::parse(&row[block_index * BLOCK_BYTES..(block_index + 1) * BLOCK_BYTES])?;
        let d = f16_dyadic(block.d_bits())?;
        dot_tracker.observe_dyadic(&d);
        for i in 0..BLOCK {
            let h = f32_dyadic(hidden[block_index * BLOCK + i])?;
            let quantized = Dyadic::new(BigInt::from(block.coefficient(i)?), 0);
            let scaled = tracked_dyadic_mul(&d, &quantized, &mut dot_tracker);
            let term = tracked_dyadic_mul(&scaled, &h, &mut dot_tracker);
            let shift = term.exponent - DOT_EXP;
            ensure!(shift >= 0, "dot term below frozen 2^-173 unit");
            let shifted = &term.coefficient << usize::try_from(shift)?;
            dot_tracker.observe_coefficient(&shifted);
            dot_tracker.observe_coefficient(&coefficient);
            coefficient += shifted;
            dot_tracker.observe_coefficient(&coefficient);
        }
    }
    let fixed = Dyadic {
        coefficient,
        exponent: DOT_EXP,
    };
    dot_tracker.observe_dyadic(&fixed);
    ensure!(
        dot_tracker.largest_observed_coefficient_payload_bits < 384,
        "dot intermediate exceeds 384-bit high-water proof"
    );
    ensure!(
        dot_tracker.largest_observed_coefficient_payload_bits <= 512,
        "dot intermediate exceeds frozen 512-bit resource"
    );
    tracker.merge(dot_tracker);
    Ok(fixed)
}

fn exact_dot(row: &[u8], hidden: &[f32]) -> Result<Dyadic> {
    exact_dot_profiled(row, hidden, &mut BigIntPayloadTracker::default())
}

fn hidden_norms_profiled(
    hidden: &[f32],
    tracker: &mut BigIntPayloadTracker,
) -> Result<[f32; BLOCKS]> {
    ensure!(hidden.len() == HIDDEN);
    let mut whole_from_elements = Dyadic::zero();
    let mut whole_from_blocks = Dyadic::zero();
    let mut norms = [0.0f32; BLOCKS];
    for (block_index, chunk) in hidden.chunks_exact(BLOCK).enumerate() {
        let mut exact = Dyadic::zero();
        let mut terms = [0.0f64; BLOCK];
        for (index, &h) in chunk.iter().enumerate() {
            let h_exact = f32_dyadic(h)?;
            let square_exact = tracked_dyadic_mul(&h_exact, &h_exact, tracker);
            exact = tracked_dyadic_add(&exact, &square_exact, tracker);
            whole_from_elements = tracked_dyadic_add(&whole_from_elements, &square_exact, tracker);
            let square = f64::from(h) * f64::from(h);
            ensure!(square.is_finite() && square >= 0.0);
            terms[index] = square;
        }
        whole_from_blocks = tracked_dyadic_add(&whole_from_blocks, &exact, tracker);
        norms[block_index] = norm_upper_profiled(&exact, &terms, tracker)?;
    }
    ensure!(whole_from_elements == whole_from_blocks);
    Ok(norms)
}

fn total_cmp_winner(logits: &[f32]) -> Result<usize> {
    ensure!(logits.len() == VOCAB);
    ensure!(
        logits.iter().all(|value| value.is_finite()),
        "nonfinite logits"
    );
    Ok(logits
        .iter()
        .enumerate()
        .max_by(|(a_id, a), (b_id, b)| a.total_cmp(b).then(a_id.cmp(b_id)))
        .unwrap()
        .0)
}

fn generated_token_digest(tokens: &[i32]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"qwen-token-ids-i32le/v1\0");
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_lower_hex(value: &str, width: usize, label: &str) -> Result<()> {
    ensure!(
        value.len() == width
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} is not {width}-digit lowercase hexadecimal"
    );
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    validate_lower_hex(value, 64, label)
}

fn validate_f32_bits(value: &str, label: &str) -> Result<()> {
    validate_lower_hex(value, 8, label)
}

fn validate_f64_bits(value: &str, label: &str) -> Result<()> {
    validate_lower_hex(value, 16, label)
}

fn decode_f64_bits(value: &str, label: &str) -> Result<f64> {
    validate_f64_bits(value, label)?;
    let decoded = f64::from_bits(u64::from_str_radix(value, 16)?);
    ensure!(decoded.is_finite(), "{label} encodes a nonfinite F64");
    Ok(decoded)
}

fn validate_exact_coefficient(value: &str, label: &str) -> Result<()> {
    validate_lower_hex(value, 128, label)
}

fn f32_le_bytes(values: &[f32]) -> &[u8] {
    const {
        assert!(cfg!(target_endian = "little"));
    }
    bytemuck::cast_slice(values)
}

fn tensor_copy(tensor: &MetalTensor, expected: usize) -> Result<Vec<f32>> {
    ensure!(tensor.dtype == GgmlType::F32 && tensor.shape.as_slice() == [expected as u64]);
    ensure!(
        tensor.buffer.storageMode() == MTLStorageMode::Shared,
        "capture tensor is not Shared"
    );
    let offset = usize::try_from(tensor.offset)?;
    let bytes = expected.checked_mul(4).context("tensor size overflow")?;
    ensure!(
        offset
            .checked_add(bytes)
            .is_some_and(|end| end <= tensor.buffer.length())
    );
    let pointer = tensor.buffer.contents().as_ptr();
    ensure!(!pointer.is_null(), "capture tensor has null contents");
    let values = unsafe {
        std::slice::from_raw_parts(pointer.cast::<u8>().add(offset).cast::<f32>(), expected)
    };
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "nonfinite capture tensor"
    );
    Ok(values.to_vec())
}

fn validate_args(args: &LmHeadScreeningOracleArgs) -> Result<()> {
    ensure!(
        args.model == Path::new(MODEL_PATH),
        "model argument differs from preregistration"
    );
    ensure!(
        args.prompt_file == Path::new(PROMPT_PATH),
        "prompt argument differs from preregistration"
    );
    ensure!(
        args.tokens == TOKENS && args.capture_calls == CALLS,
        "generation arguments differ from preregistration"
    );
    ensure!(
        args.attest_no_other_user_gpu_workload,
        "operator GPU attestation is required"
    );
    ensure!(SAMPLER_ALGORITHM_VERSION == 1);
    ensure!(
        !args.packet_dir.as_os_str().is_empty() && !args.packet_dir.is_absolute(),
        "packet root must be the preregistered relative path"
    );
    ensure!(args.packet_dir == Path::new(PACKET_PATH));
    ensure!(
        args.packet_dir
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    );
    Ok(())
}

fn validate_build(build: &super::BuildIdentity) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "v0.664 requires a release binary without debug assertions"
    );
    validate_clean_build_identity(build)
}

fn validate_clean_build_identity(build: &super::BuildIdentity) -> Result<()> {
    ensure!(build.schema_version == 2);
    ensure!(super::source_identity::full_object_id(&build.build_commit));
    ensure!(
        build.build_commit == build.build_commit.to_ascii_lowercase()
            && build.build_commit_short == build.build_commit[..9]
    );
    let build_source_state = build
        .build_source_state
        .as_deref()
        .context("clean build identity lacks source state")?;
    ensure!(super::source_identity::valid_source_state(
        build_source_state
    ));
    let runtime_commit = build
        .runtime_commit
        .as_deref()
        .context("clean build identity lacks runtime commit")?;
    let runtime_source_state = build
        .runtime_source_state
        .as_deref()
        .context("clean build identity lacks runtime source state")?;
    ensure!(
        runtime_commit == build.build_commit
            && runtime_source_state == build_source_state
            && super::source_identity::valid_source_state(runtime_source_state)
    );
    ensure!(matches!(
        build.stamp_source.as_str(),
        "git" | "environment-verified"
    ));
    ensure!(build.stamp_error.is_none());
    ensure!(build.status == "match");
    ensure!(build.build_dirty == Some(false));
    ensure!(build.runtime_dirty == Some(false));
    ensure!(build.problems.is_empty());
    ensure!(build.overrides.is_empty());
    Ok(())
}

fn readiness_command_arguments(
    args: &LmHeadScreeningOracleArgs,
) -> Result<ReadinessCommandArguments> {
    Ok(ReadinessCommandArguments {
        model: args
            .model
            .to_str()
            .context("model argument is not UTF-8")?
            .to_string(),
        prompt_file: args
            .prompt_file
            .to_str()
            .context("prompt argument is not UTF-8")?
            .to_string(),
        tokens: args.tokens,
        capture_calls: args.capture_calls.clone(),
        packet_dir: args
            .packet_dir
            .to_str()
            .context("packet argument is not UTF-8")?
            .to_string(),
        attest_no_other_user_gpu_workload: args.attest_no_other_user_gpu_workload,
    })
}

fn readiness_payload(
    args: &LmHeadScreeningOracleArgs,
    build_identity: &super::BuildIdentity,
    executable_identity: &ExecutableIdentity,
) -> Result<ReadinessArtifact> {
    Ok(ReadinessArtifact {
        schema: SCHEMA.to_string(),
        build_identity: build_identity.clone(),
        executable_identity: executable_identity.clone(),
        operator_attestation: args.attest_no_other_user_gpu_workload,
        command_arguments: readiness_command_arguments(args)?,
        predecessor_decision_sha256: PREDECESSOR_DECISION_SHA256.to_string(),
    })
}

fn validate_readiness_constants(readiness: &ReadinessArtifact) -> Result<()> {
    ensure!(readiness.schema == SCHEMA);
    validate_clean_build_identity(&readiness.build_identity)?;
    validate_sha256(
        &readiness.executable_identity.sha256,
        "readiness executable SHA-256",
    )?;
    ensure!(!readiness.executable_identity.debug_assertions);
    ensure!(readiness.operator_attestation);
    ensure!(
        readiness.command_arguments
            == ReadinessCommandArguments {
                model: MODEL_PATH.to_string(),
                prompt_file: PROMPT_PATH.to_string(),
                tokens: TOKENS,
                capture_calls: CALLS.to_vec(),
                packet_dir: PACKET_PATH.to_string(),
                attest_no_other_user_gpu_workload: true,
            },
        "readiness command arguments differ from preregistration"
    );
    validate_sha256(
        &readiness.predecessor_decision_sha256,
        "predecessor decision SHA-256",
    )?;
    ensure!(readiness.predecessor_decision_sha256 == PREDECESSOR_DECISION_SHA256);
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ExecutableIdentity {
    path: String,
    sha256: String,
    stamp: FileStamp,
    debug_assertions: bool,
}

fn executable_identity() -> Result<ExecutableIdentity> {
    let path = std::env::current_exe()?;
    let expected = std::env::current_dir()?.join("target/release/qwen-bench");
    let expected_metadata = std::fs::symlink_metadata(&expected)?;
    ensure!(expected_metadata.file_type().is_file() && !expected_metadata.file_type().is_symlink());
    ensure!(
        std::fs::canonicalize(&path)? == std::fs::canonicalize(&expected)?,
        "current executable is not current_dir/target/release/qwen-bench"
    );
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&path)?;
    let stamp = FileStamp::from_metadata(&file.metadata()?);
    let digest = sha256_file_handle(&file)?;
    ensure!(FileStamp::from_metadata(&file.metadata()?) == stamp);
    Ok(ExecutableIdentity {
        path: path.to_string_lossy().into_owned(),
        sha256: digest,
        stamp,
        debug_assertions: cfg!(debug_assertions),
    })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct HostEvidence {
    label: String,
    float_encoding: String,
    canonical_json: String,
    canonical_json_sha256: String,
}

fn host_evidence(label: &str, snapshot: &HostSnapshot) -> Result<HostEvidence> {
    let encoded_snapshot = encode_json_floats(serde_json::to_value(snapshot)?)?;
    let encoded = serde_json::to_string(&encoded_snapshot)?;
    Ok(HostEvidence {
        label: label.to_string(),
        float_encoding: "f64_ieee754_hex_bits".to_string(),
        canonical_json_sha256: sha256(encoded.as_bytes()),
        canonical_json: encoded,
    })
}

fn validate_host_evidence(evidence: &HostEvidence) -> Result<()> {
    ensure!(evidence.float_encoding == "f64_ieee754_hex_bits");
    validate_sha256(
        &evidence.canonical_json_sha256,
        "host canonical JSON SHA-256",
    )?;
    ensure!(sha256(evidence.canonical_json.as_bytes()) == evidence.canonical_json_sha256);
    let parsed: Value = serde_json::from_str(&evidence.canonical_json)?;
    ensure!(
        serde_json::to_string(&parsed)? == evidence.canonical_json,
        "host evidence canonical JSON changed on typed-tree reserialization"
    );
    Ok(())
}

fn encode_json_floats(value: Value) -> Result<Value> {
    Ok(match value {
        Value::Number(number) if number.is_f64() => {
            let value = number.as_f64().context("invalid JSON float")?;
            ensure!(value.is_finite());
            Value::String(format!("{:016x}", value.to_bits()))
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(encode_json_floats)
                .collect::<Result<Vec<_>>>()?,
        ),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| Ok((key, encode_json_floats(value)?)))
                .collect::<Result<serde_json::Map<_, _>>>()?,
        ),
        other => other,
    })
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FileStamp {
    device: u64,
    inode: u64,
    bytes: u64,
    mtime_seconds: i64,
    mtime_nanoseconds: i64,
}

impl FileStamp {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            bytes: metadata.len(),
            mtime_seconds: metadata.mtime(),
            mtime_nanoseconds: metadata.mtime_nsec(),
        }
    }
}

struct Packet {
    root: PathBuf,
    written: BTreeSet<PathBuf>,
    commitments: BTreeMap<PathBuf, (FileStamp, String)>,
    root_stamp: FileStamp,
    terminalized: bool,
    executable_commitment: Option<ExecutableIdentity>,
}

impl Packet {
    fn reserve(root: &Path) -> Result<Self> {
        check_nofollow_parents(root)?;
        ensure!(
            std::fs::symlink_metadata(root)
                .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound),
            "canonical packet root already exists or cannot be inspected"
        );
        std::fs::create_dir(root).with_context(|| format!("reserve {}", root.display()))?;
        let metadata = std::fs::symlink_metadata(root)?;
        ensure!(metadata.file_type().is_dir() && !metadata.file_type().is_symlink());
        let packet = Self {
            root: root.to_path_buf(),
            written: BTreeSet::new(),
            commitments: BTreeMap::new(),
            root_stamp: FileStamp::from_metadata(&metadata),
            terminalized: false,
            executable_commitment: None,
        };
        packet.sync_root()?;
        open_directory_nofollow(root.parent().context("packet root lacks parent")?)?.sync_all()?;
        Ok(packet)
    }

    fn path(&self, relative: &Path) -> Result<PathBuf> {
        ensure!(normalized_relative(relative));
        Ok(self.root.join(relative))
    }

    fn mkdir(&self, relative: &Path) -> Result<()> {
        self.validate_root()?;
        let path = self.path(relative)?;
        std::fs::create_dir(&path)?;
        let metadata = std::fs::symlink_metadata(&path)?;
        ensure!(metadata.file_type().is_dir() && !metadata.file_type().is_symlink());
        open_directory_nofollow(&path)?.sync_all()?;
        open_directory_nofollow(path.parent().context("directory lacks parent")?)?.sync_all()?;
        Ok(())
    }

    fn write(&mut self, relative: impl AsRef<Path>, bytes: &[u8]) -> Result<()> {
        self.validate_root()?;
        let relative = relative.as_ref();
        ensure!(!self.written.contains(relative), "artifact written twice");
        let path = self.path(relative)?;
        let parent = path.parent().context("artifact lacks parent")?;
        let parent_metadata = std::fs::symlink_metadata(parent)?;
        ensure!(parent_metadata.file_type().is_dir() && !parent_metadata.file_type().is_symlink());
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW);
        let mut file = options.open(&path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        open_directory_nofollow(parent)?.sync_all()?;
        let commitment = hash_nofollow(&path)?;
        self.written.insert(relative.to_path_buf());
        self.commitments.insert(relative.to_path_buf(), commitment);
        Ok(())
    }

    fn json_strict<T: Serialize + DeserializeOwned>(
        &mut self,
        relative: impl AsRef<Path>,
        value: &T,
    ) -> Result<()> {
        let bytes = serde_json::to_vec(value)?;
        let parsed: T = serde_json::from_slice(&bytes)?;
        ensure!(
            serde_json::to_vec(&parsed)? == bytes,
            "strict JSON round trip changed bytes"
        );
        let tree = serde_json::to_value(&parsed)?;
        ensure!(
            tree.get("schema").and_then(Value::as_str) == Some(SCHEMA),
            "strict artifact schema mismatch"
        );
        let mut terminated = bytes;
        terminated.push(b'\n');
        self.write(relative, &terminated)
    }

    fn sync_root(&self) -> Result<()> {
        self.validate_root()?;
        open_directory_nofollow(&self.root)?
            .sync_all()
            .map_err(Into::into)
    }

    fn readiness_published(&self) -> bool {
        self.written.contains(Path::new("readiness.json"))
            && self.commitments.contains_key(Path::new("readiness.json"))
    }

    fn validate_root(&self) -> Result<()> {
        let metadata = std::fs::symlink_metadata(&self.root)?;
        ensure!(
            metadata.file_type().is_dir()
                && metadata.dev() == self.root_stamp.device
                && metadata.ino() == self.root_stamp.inode,
            "reserved packet root identity changed"
        );
        Ok(())
    }
}

fn normalized_relative(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn check_nofollow_parents(path: &Path) -> Result<()> {
    let mut current = PathBuf::new();
    let components: Vec<_> = path.components().collect();
    ensure!(!components.is_empty());
    for component in &components[..components.len() - 1] {
        match component {
            Component::Normal(name) => current.push(name),
            other => return Err(anyhow!("unsupported packet path component {other:?}")),
        }
        let metadata = std::fs::symlink_metadata(&current)
            .with_context(|| format!("inspect packet parent {}", current.display()))?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "packet parent is not a no-follow directory"
        );
    }
    Ok(())
}

fn check_absolute_nofollow_parents(path: &Path) -> Result<()> {
    ensure!(path.is_absolute());
    let mut current = PathBuf::from("/");
    let components: Vec<_> = path.components().collect();
    for component in &components[1..components.len() - 1] {
        let Component::Normal(name) = component else {
            return Err(anyhow!("unsupported absolute path component {component:?}"));
        };
        current.push(name);
        let metadata = std::fs::symlink_metadata(&current)?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "absolute model parent is not a no-follow directory"
        );
    }
    Ok(())
}

fn open_directory_nofollow(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_DIRECTORY)
        .open(path)?;
    ensure!(file.metadata()?.file_type().is_dir());
    Ok(file)
}

#[derive(Clone)]
struct Capture {
    call: usize,
    sequence_position: usize,
    consumed_input_token: i32,
    winner: usize,
    winner_logit_bits: u32,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    generated_prefix_token_sha256: String,
    stop_checked_before_capture: bool,
    callback_completed_before_capture: bool,
    token_limit_checked_before_capture: bool,
    hidden_copy_wall_nanoseconds: u64,
    logits_copy_wall_nanoseconds: u64,
    winner_reconstruction_wall_nanoseconds: u64,
}

#[derive(Clone)]
struct AnalyzerCapture {
    capture_identity: String,
    winner: usize,
    hidden: Vec<f32>,
}

#[derive(Clone)]
struct CaptureLabel {
    capture_identity: String,
    call: usize,
    sequence_position: usize,
}

struct GenerationOutcome {
    captures: Vec<Capture>,
    generated: Vec<i32>,
    transitions: usize,
    coverage_eos_call: Option<usize>,
    generation_wall_nanoseconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RawArtifactRecord {
    dtype: String,
    shape: Vec<usize>,
    bytes: usize,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WinnerEvidenceArtifact {
    schema: String,
    call: usize,
    sequence_position: usize,
    consumed_input_token: i32,
    selected_output_token: usize,
    winner_logit_bits: String,
    prompt_token_sha256: String,
    generated_prefix_token_sha256: String,
    stop_checked_before_capture: bool,
    callback_completed_before_capture: bool,
    token_limit_checked_before_capture: bool,
    hidden_copy_wall_nanoseconds: u64,
    logits_copy_wall_nanoseconds: u64,
    winner_reconstruction_wall_nanoseconds: u64,
    hidden: RawArtifactRecord,
    logits: RawArtifactRecord,
    winner_order: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AnalyzerInputArtifact {
    schema: String,
    capture_identity: String,
    winner_id: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PromptRecord {
    path: String,
    bytes: usize,
    sha256: String,
    tokens: usize,
    token_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RequestRecord {
    temperature_bits: String,
    token_limit: usize,
    stop_ids: Vec<i32>,
    prefill_chunk: usize,
    max_context: usize,
    capture_calls: Vec<usize>,
    generated_tokens: usize,
    generated_token_sha256: String,
    transitions: usize,
    termination: GenerationTermination,
    coverage_eos_call: Option<usize>,
    call_order: ConditionalCallOrder,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ConditionalCallOrder {
    unconditional_prefix: [CallStep; 3],
    producer_eos_frozen_capture_suffix: [CallStep; 2],
    producer_eos_noncapture_suffix: [CallStep; 1],
    non_stop_suffix: [CallStep; 4],
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum GenerationTermination {
    ProducerEos,
    TokenLimit,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CallStep {
    Select,
    Append,
    StopCheck,
    Callback,
    TokenLimitCheck,
    ObservationalCapture,
    Branch,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CaptureTelemetry {
    hidden_copy_logical_bytes: usize,
    logits_copy_logical_bytes: usize,
    hidden_copy_wall_nanoseconds: u64,
    logits_copy_wall_nanoseconds: u64,
    winner_reconstruction_wall_nanoseconds: u64,
    generation_wall_nanoseconds: u64,
    retained_capture_vec_requested_capacity_bytes: usize,
    retained_hidden_vec_requested_capacity_bytes: usize,
    retained_logits_vec_requested_capacity_bytes: usize,
    retained_generated_vec_requested_capacity_bytes: usize,
    capacity_note: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CaptureSummaryRecord {
    call: usize,
    sequence_position: usize,
    consumed_input_token: i32,
    winner_id: usize,
    winner_logit_bits: String,
    hidden_sha256: String,
    logits_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ZeroQwenControls {}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CaptureManifestArtifact {
    schema: String,
    status: ArtifactStatus,
    build_identity: super::BuildIdentity,
    executable_identity: ExecutableIdentity,
    qwen_env: ZeroQwenControls,
    user_gpu_attestation: bool,
    model: ModelIdentityRecord,
    runtime_identity: OpenedGgufRuntimeIdentity,
    model_authentication_telemetry: ModelAuthenticationTelemetry,
    prompt: PromptRecord,
    request: RequestRecord,
    capture_telemetry: CaptureTelemetry,
    captures: Vec<CaptureSummaryRecord>,
    host_before: HostEvidence,
    host_after: HostEvidence,
}

fn validate_capture_manifest_constants(manifest: &CaptureManifestArtifact) -> Result<()> {
    ensure!(manifest.schema == SCHEMA);
    ensure!(manifest.status == ArtifactStatus::CompleteGeneration);
    validate_build(&manifest.build_identity)?;
    validate_sha256(
        &manifest.executable_identity.sha256,
        "executable identity SHA-256",
    )?;
    ensure!(!manifest.executable_identity.debug_assertions);
    ensure!(manifest.qwen_env == (ZeroQwenControls {}));
    ensure!(manifest.user_gpu_attestation);
    let model = &manifest.model;
    validate_sha256(&model.sha256, "model SHA-256")?;
    validate_sha256(&model.output_weight.sha256, "output head SHA-256")?;
    ensure!(
        model.path == MODEL_PATH
            && model.bytes == MODEL_BYTES
            && model.sha256 == MODEL_SHA256
            && model.architecture == "qwen35moe"
            && model.base_model_name == BASE_MODEL_NAME
            && model.basename == BASENAME
            && model.file_type == 15
            && model.layers == 40
            && model.hidden == HIDDEN
            && model.vocab == VOCAB
            && model.untied
            && model.mtp_layers == 0
            && !model.output_bias
            && model.stop_token_ids == STOP_IDS
            && model.output_weight.name == "output.weight"
            && model.output_weight.dtype == "Q6_K"
            && model.output_weight.shape == [HIDDEN, VOCAB]
            && model.output_weight.shard_index == 0
            && model.output_weight.data_offset == OUTPUT_OFFSET
            && model.output_weight.bytes == OUTPUT_BYTES
            && model.output_weight.sha256 == OUTPUT_SHA256,
        "capture manifest model constants changed"
    );
    ensure!(
        manifest.runtime_identity.runtime_load_api == "Runtime::load_opened_gguf"
            && manifest.runtime_identity.exact_opened_gguf_consumed
            && manifest.runtime_identity.pre_runtime_stamp == model.retained_file_stamp
            && manifest.runtime_identity.runtime_retained_stamp == model.retained_file_stamp
            && manifest.runtime_identity.post_runtime_load_stamp == model.retained_file_stamp
            && manifest.runtime_identity.post_capture_stamp.as_ref()
                == Some(&model.retained_file_stamp),
        "capture manifest opened-model identity changed"
    );
    ensure!(
        manifest.prompt.path == PROMPT_PATH
            && manifest.prompt.bytes == PROMPT_BYTES
            && manifest.prompt.sha256 == PROMPT_SHA256
            && manifest.prompt.tokens == PROMPT_TOKENS
            && manifest.prompt.token_sha256 == TOKEN_SHA256,
        "capture manifest prompt constants changed"
    );
    let request = &manifest.request;
    validate_f32_bits(&request.temperature_bits, "temperature bits")?;
    validate_sha256(&request.generated_token_sha256, "generated-token SHA-256")?;
    ensure!(
        request.temperature_bits == "00000000"
            && request.token_limit == TOKENS
            && request.stop_ids == STOP_IDS
            && request.prefill_chunk == 1024
            && request.max_context == 1024
            && request.capture_calls == CALLS
            && request.call_order
                == ConditionalCallOrder {
                    unconditional_prefix: [CallStep::Select, CallStep::Append, CallStep::StopCheck,],
                    producer_eos_frozen_capture_suffix: [
                        CallStep::ObservationalCapture,
                        CallStep::Branch,
                    ],
                    producer_eos_noncapture_suffix: [CallStep::Branch],
                    non_stop_suffix: [
                        CallStep::Callback,
                        CallStep::TokenLimitCheck,
                        CallStep::ObservationalCapture,
                        CallStep::Branch,
                    ],
                },
        "capture manifest request constants changed"
    );
    for capture in &manifest.captures {
        validate_f32_bits(&capture.winner_logit_bits, "capture winner-logit bits")?;
        validate_sha256(&capture.hidden_sha256, "capture hidden SHA-256")?;
        validate_sha256(&capture.logits_sha256, "capture logits SHA-256")?;
        ensure!(capture.winner_id < VOCAB);
    }
    let expected_calls = match request.termination {
        GenerationTermination::TokenLimit => {
            ensure!(
                request.generated_tokens == TOKENS
                    && request.transitions == TOKENS - 1
                    && request.coverage_eos_call.is_none(),
                "token-limit generation semantics changed"
            );
            CALLS.as_slice()
        }
        GenerationTermination::ProducerEos => {
            let call = request
                .coverage_eos_call
                .context("producer EOS call missing")?;
            ensure!(
                call < TOKENS
                    && request.generated_tokens == call + 1
                    && request.transitions == call,
                "producer-EOS generation semantics changed"
            );
            &CALLS[..CALLS.partition_point(|required| *required <= call)]
        }
    };
    ensure!(
        manifest.captures.len() == expected_calls.len()
            && manifest
                .captures
                .iter()
                .zip(expected_calls)
                .all(|(capture, call)| capture.call == *call
                    && capture.sequence_position == PROMPT_TOKENS + call),
        "capture manifest call count or frozen prefix order changed"
    );
    validate_host_evidence(&manifest.host_before)?;
    validate_host_evidence(&manifest.host_after)?;
    Ok(())
}

struct AnalyzerInput<'a> {
    geometry: Geometry,
    captures: Vec<AnalyzerCapture>,
    block_norms: Vec<f32>,
    output_weight: &'a [u8],
    ledger: LedgerConstants,
}

#[derive(Clone, Copy)]
struct Geometry {
    hidden: usize,
    vocab: usize,
    blocks: usize,
    row_bytes: usize,
}

#[derive(Clone, Copy)]
struct LedgerConstants {
    full_head: u64,
    byte_limit: u64,
    pruning_floor: usize,
}

fn analyzer_capabilities(input: &AnalyzerInput<'_>) -> [&'static str; 7] {
    let _ = (input.geometry, input.ledger);
    [
        "profile_geometry",
        "capture_identity",
        "winner_id",
        "owned_hidden_f32_bits",
        "owned_authenticated_block_norm_f32_metadata",
        "readonly_authenticated_output_weight_bytes",
        "frozen_ledger_constants",
    ]
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SelfTestArtifact {
    schema: String,
    suite: String,
    passed: bool,
    cases: Vec<String>,
    observations: SelfTestObservations,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SelfTestObservations {
    fpcr_bits: String,
    fegetround_value: i32,
    gradual_underflow_f64_bits: String,
    f64_to_f32_subnormal_bits: String,
    sqrt_2_f64_bits: String,
    f64_to_f32_midpoint_ties_even_bits: [String; 2],
    near_extreme_dot_signed_bits: u64,
    wide_positive_interval_bits: [String; 2],
    wide_negative_interval_bits: [String; 2],
    ledger_total_charged_bytes_64_128_256: [u64; 3],
    compact_fixture_survivors: Vec<u32>,
    reduce_fixture_less_equal_greater: [usize; 3],
    opened_gguf_rewind: OpenedGgufRewindObservation,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct OpenedGgufRewindObservation {
    fixture_bytes: usize,
    shared_cursor_position_before_parse: u64,
    diagnostic_path_absent_before_parse: bool,
    parsed_tensor_count: usize,
    parsed_tensor_name: String,
    model_names: ModelNameIdentity,
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn read_fpcr() -> u64 {
    let value: u64;
    unsafe {
        std::arch::asm!("mrs {value}, fpcr", value = out(reg) value, options(nomem, nostack, preserves_flags));
    }
    value
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
unsafe extern "C" {
    fn fegetround() -> libc::c_int;
}

fn floating_environment_self_test() -> Result<(u64, i32, u64, u32, u64, [u32; 2])> {
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    return Err(anyhow!(
        "v0.664 floating-environment gate requires aarch64 macOS"
    ));

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        const FPCR_RMODE_MASK: u64 = 0b11 << 22;
        const FPCR_FZ: u64 = 1 << 24;
        const FE_TONEAREST: i32 = 0;

        let fpcr = read_fpcr();
        ensure!(fpcr & FPCR_RMODE_MASK == 0, "FPCR is not round-to-nearest");
        ensure!(fpcr & FPCR_FZ == 0, "FPCR flush-to-zero is enabled");
        let fe_round = unsafe { fegetround() };
        ensure!(fe_round == FE_TONEAREST, "fegetround is not FE_TONEAREST");

        let gradual = black_box(f64::MIN_POSITIVE) * black_box(0.5f64);
        let gradual_bits = gradual.to_bits();
        ensure!(
            gradual_bits == 1u64 << 51,
            "F64 gradual-underflow self-test failed"
        );
        let f32_subnormal = black_box(f64::from_bits((874u64) << 52)) as f32;
        let f32_subnormal_bits = f32_subnormal.to_bits();
        ensure!(
            f32_subnormal_bits == 1,
            "F64-to-F32 subnormal conversion self-test failed"
        );
        let sqrt_2_bits = black_box(2.0f64).sqrt().to_bits();
        ensure!(
            sqrt_2_bits == 0x3ff6_a09e_667f_3bcd,
            "F64 sqrt(2) round-to-nearest self-test failed"
        );
        let midpoint_even_low = (black_box(1.0f64) + black_box(2.0f64.powi(-24))) as f32;
        let midpoint_even_high = (black_box(1.0f64) + black_box(3.0f64 * 2.0f64.powi(-24))) as f32;
        let midpoint_bits = [midpoint_even_low.to_bits(), midpoint_even_high.to_bits()];
        ensure!(
            midpoint_bits == [0x3f80_0000, 0x3f80_0002],
            "F64-to-F32 midpoint ties-to-even self-test failed"
        );
        Ok((
            fpcr,
            fe_round,
            gradual_bits,
            f32_subnormal_bits,
            sqrt_2_bits,
            midpoint_bits,
        ))
    }
}

fn self_tests() -> Result<SelfTestArtifact> {
    let (fpcr, fe_round, gradual_bits, f32_subnormal_bits, sqrt_2_bits, midpoint_bits) =
        floating_environment_self_test()?;
    let mut block = [0u8; BLOCK_BYTES];
    block[208..].copy_from_slice(&0x3c00u16.to_le_bytes());
    block[192..208].fill(1);
    // Hand-place q=31 at logical index 32: low nibble ql[32], qh[0] bits 2..3.
    block[32] = 0x0f;
    block[128] = 0x0c;
    let q6 = Q6Block::parse(&block)?;
    ensure!(q6.quant_scale(32)? == (31, 1));
    ensure!(q6.quant_scale(0)? == (-32, 1));
    let mut nonuniform = [0u8; BLOCK_BYTES];
    nonuniform[208..].copy_from_slice(&0x3c00u16.to_le_bytes());
    for (index, scale) in nonuniform[192..208].iter_mut().enumerate() {
        *scale = (index + 1) as u8;
    }
    nonuniform[17] = 0x0f;
    nonuniform[145] = 0x03;
    ensure!(Q6Block::parse(&nonuniform)?.quant_scale(17)? == (31, 2));
    ensure!(f16_dyadic(0x0001)? == Dyadic::new(BigInt::one(), -24));
    ensure!(f16_dyadic(0x7bff)? == Dyadic::new(BigInt::from(2047), 5));
    ensure!(f32_dyadic(f32::from_bits(1))? == Dyadic::new(BigInt::one(), -149));
    ensure!(f32_dyadic(f32::MAX)?.cmp(&Dyadic::zero()) == Ordering::Greater);
    ensure!(f32_dyadic(-0.0)?.coefficient.is_zero());
    ensure!(next_up_f32(1.0) > 1.0 && f32_upper(1.0 + 2.0f64.powi(-24))? > 1.0);
    ensure!(
        total_cmp_winner(&{
            let mut x = vec![-1.0; VOCAB];
            x[3] = -0.0;
            x[4] = 0.0;
            x
        })? == 4
    );
    let exact = Dyadic::new(BigInt::from(5), -2);
    let upper = norm_upper(&exact, &[1.0, 0.25])?;
    ensure!(f32_dyadic(upper)?.square().cmp(&exact) != Ordering::Less);
    let wide = Dyadic::new((BigInt::one() << 200usize) + 1, -100);
    let wide_interval = dyadic_interval(&wide)?;
    ensure!(wide_interval.0 < wide_interval.1);
    ensure!(f64_dyadic(wide_interval.0)?.cmp(&wide) != Ordering::Greater);
    ensure!(f64_dyadic(wide_interval.1)?.cmp(&wide) != Ordering::Less);
    let negative_wide = Dyadic::new(-((BigInt::one() << 200usize) + BigInt::one()), -100);
    let negative_interval = dyadic_interval(&negative_wide)?;
    ensure!(negative_interval.0 < negative_interval.1);
    ensure!(f64_dyadic(negative_interval.0)?.cmp(&negative_wide) != Ordering::Greater);
    ensure!(f64_dyadic(negative_interval.1)?.cmp(&negative_wide) != Ordering::Less);
    ensure!(dyadic_interval(&Dyadic::new(BigInt::one(), 0))? == (1.0, 1.0));
    let cancelled = Dyadic::new(BigInt::one(), -10).add(&Dyadic::new(BigInt::from(-1), -10));
    ensure!(cancelled == Dyadic::zero());
    let mut dot_row = vec![0u8; 2 * BLOCK_BYTES];
    for raw in dot_row.chunks_exact_mut(BLOCK_BYTES) {
        raw[192..208].fill(1);
        raw[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
    }
    let dot = exact_dot(&dot_row, &vec![1.0; 512])?;
    ensure!(dot.cmp(&Dyadic::new(BigInt::from(-16_384), 0)) == Ordering::Equal);
    let mut extreme_row = vec![0xffu8; ROW_BYTES];
    for raw in extreme_row.chunks_exact_mut(BLOCK_BYTES) {
        raw[192..208].fill(0x80);
        raw[208..210].copy_from_slice(&0x7bffu16.to_le_bytes());
    }
    let extreme_dot = exact_dot(&extreme_row, &vec![f32::MAX; HIDDEN])?;
    ensure!((330..384).contains(&extreme_dot.signed_bits()));
    let huge = Dyadic::new((BigInt::one() << 510usize) - 1, DOT_EXP);
    ensure!(huge.signed_bits() == 511 && huge.signed_bits() <= 512);
    ensure!(rounded_union(&[(1, 65), (63, 129)], 64)? == 192);
    ensure!(rounded_union(&[(1, 65), (63, 129)], 128)? == 256);
    ensure!(rounded_union(&[(1, 65), (63, 129)], 256)? == 256);
    ensure!(validate_survivors(&[0, 2, 4], 5, 3).is_ok());
    ensure!(validate_survivors(&[0, 0], 5, 3).is_err());
    ensure!(Q6Block::parse(&block[..209]).is_err());
    let synthetic = AnalyzerInput {
        geometry: Geometry {
            hidden: 1,
            vocab: 2,
            blocks: 1,
            row_bytes: 1,
        },
        captures: vec![],
        block_norms: vec![],
        output_weight: &[],
        ledger: LedgerConstants {
            full_head: 2,
            byte_limit: 1,
            pruning_floor: 1,
        },
    };
    ensure!(analyzer_capabilities(&synthetic).len() == 7);
    ensure!(bound_survives(2.0, 2.0)?);
    let compacted = compact_active(&[1, 0, 1, 0, 1], 3)?;
    ensure!(compacted == [0, 2, 4]);
    let reduced = reduce_survivors(3, &compacted, &[0, 1, 2], (1.0, 1.0))?;
    ensure!(reduced.less_than_winner_count == 1);
    ensure!(reduced.tie_count == 1);
    ensure!(reduced.greater_than_winner_count == 1);
    runtime_ledger_self_test()?;
    runtime_artifact_self_test()?;
    let opened_gguf_rewind = runtime_opened_gguf_rewind_self_test()?;
    Ok(SelfTestArtifact {
        schema: SCHEMA.to_string(),
        suite: "in-process-pre-model".to_string(),
        passed: true,
        cases: [
            "fpcr_fegetround_gradual_underflow_sqrt_and_conversion_rounding_environment",
            "q6_hand_constructed_index32_and_lane17_nonuniform_scale",
            "dyadic_f16_f32_extrema_subnormals_signed_and_cancelled_zero",
            "dyadic_interval_positive_negative_wide_inexact_exact_and_containment",
            "exact_dot_known_512_and_near_extreme_2048_signed_bit_bound",
            "norm_sqrt_and_directed_f32_upper_containment",
            "production_compact_strict_equality_and_reduce_less_equal_greater",
            "ledger_for_hard_coded_64_128_256_streams_totals_constants_and_overlap",
            "malformed_symlink_hardlink_foreign_traversal_schema_json_and_raw_size",
            "prophecy_firewall_exact_positive_capabilities",
            "opened_gguf_shared_eof_cursor_rewind_and_typed_dual_name_extraction",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        observations: SelfTestObservations {
            fpcr_bits: format!("{fpcr:016x}"),
            fegetround_value: fe_round,
            gradual_underflow_f64_bits: format!("{gradual_bits:016x}"),
            f64_to_f32_subnormal_bits: format!("{f32_subnormal_bits:08x}"),
            sqrt_2_f64_bits: format!("{sqrt_2_bits:016x}"),
            f64_to_f32_midpoint_ties_even_bits: midpoint_bits.map(|bits| format!("{bits:08x}")),
            near_extreme_dot_signed_bits: extreme_dot.signed_bits(),
            wide_positive_interval_bits: [
                format!("{:016x}", wide_interval.0.to_bits()),
                format!("{:016x}", wide_interval.1.to_bits()),
            ],
            wide_negative_interval_bits: [
                format!("{:016x}", negative_interval.0.to_bits()),
                format!("{:016x}", negative_interval.1.to_bits()),
            ],
            ledger_total_charged_bytes_64_128_256: [10_463_488, 10_464_640, 10_466_816],
            compact_fixture_survivors: compacted,
            reduce_fixture_less_equal_greater: [
                reduced.less_than_winner_count,
                reduced.tie_count,
                reduced.greater_than_winner_count,
            ],
            opened_gguf_rewind,
        },
    })
}

fn push_self_test_gguf_string(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value.as_bytes());
}

fn opened_gguf_identity_fixture() -> Vec<u8> {
    const MAGIC: u32 = 0x4655_4747;
    let mut fixture = Vec::new();
    fixture.extend_from_slice(&MAGIC.to_le_bytes());
    fixture.extend_from_slice(&3u32.to_le_bytes());
    fixture.extend_from_slice(&1u64.to_le_bytes());
    fixture.extend_from_slice(&2u64.to_le_bytes());
    for (key, value) in [
        ("general.base_model.0.name", BASE_MODEL_NAME),
        ("general.basename", BASENAME),
    ] {
        push_self_test_gguf_string(&mut fixture, key);
        fixture.extend_from_slice(&8u32.to_le_bytes());
        push_self_test_gguf_string(&mut fixture, value);
    }
    push_self_test_gguf_string(&mut fixture, "t");
    fixture.extend_from_slice(&1u32.to_le_bytes());
    fixture.extend_from_slice(&1u64.to_le_bytes());
    fixture.extend_from_slice(&0u32.to_le_bytes());
    fixture.extend_from_slice(&0u64.to_le_bytes());
    while fixture.len() % 32 != 0 {
        fixture.push(0);
    }
    fixture.extend_from_slice(&1.0f32.to_le_bytes());
    fixture
}

fn runtime_opened_gguf_rewind_self_test() -> Result<OpenedGgufRewindObservation> {
    let fixture = opened_gguf_identity_fixture();

    let root = std::env::temp_dir().join(format!(
        "qwen-v0664-opened-gguf-self-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let diagnostic = root.join("diagnostic.gguf");
    let retained = root.join("retained.gguf");
    std::fs::create_dir(&root)?;
    let result = (|| -> Result<OpenedGgufRewindObservation> {
        let mut writer = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&diagnostic)?;
        writer.write_all(&fixture)?;
        writer.sync_all()?;
        drop(writer);

        let mut file = File::open(&diagnostic)?;
        let mut shared_cursor = file.try_clone()?;
        let eof = shared_cursor.seek(SeekFrom::End(0))?;
        ensure!(eof == fixture.len() as u64 && file.stream_position()? == eof);
        std::fs::rename(&diagnostic, &retained)?;
        ensure!(
            std::fs::symlink_metadata(&diagnostic)
                .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound),
            "diagnostic path remained resolvable during opened-GGUF self-test"
        );

        let gguf = qwen_llm::gguf::GgufFile::from_opened_file(file, &diagnostic)?;
        ensure!(gguf.shard_count() == 1 && gguf.total_mapped_len() == fixture.len());
        let tensor = gguf
            .find("t")
            .context("opened-GGUF fixture tensor missing")?;
        ensure!(
            gguf.tensors.len() == 1
                && tensor.dtype == GgmlType::F32
                && tensor.shape.as_slice() == [1]
                && gguf.try_slice(tensor)? == 1.0f32.to_le_bytes(),
            "opened-GGUF fixture parsed incorrectly"
        );
        let model_names = extract_model_name_identity(&gguf)?;
        ensure!(
            model_names
                == ModelNameIdentity {
                    base_model_name: BASE_MODEL_NAME.to_string(),
                    basename: BASENAME.to_string(),
                },
            "shared model-name extractor selected the wrong GGUF metadata key"
        );
        Ok(OpenedGgufRewindObservation {
            fixture_bytes: fixture.len(),
            shared_cursor_position_before_parse: eof,
            diagnostic_path_absent_before_parse: true,
            parsed_tensor_count: gguf.tensors.len(),
            parsed_tensor_name: tensor.name.clone(),
            model_names,
        })
    })();
    let _ = std::fs::remove_file(&diagnostic);
    let _ = std::fs::remove_file(&retained);
    let _ = std::fs::remove_dir(&root);
    result
}

fn runtime_ledger_self_test() -> Result<()> {
    ensure!(OUTPUT_BYTES == 417_177_600 && BYTE_LIMIT == 125_153_280);
    ensure!(PRUNING_FLOOR == 198_656 && VOCAB - 1 == 248_319);
    let survivors = [0, 2, 4, 7];
    let logical = [
        8192, 32, 8192, 1680, 16, 64, 7_946_208, 32, 1_986_552, 248_319, 248_319, 16, 8192, 16,
        6720, 64, 4, 64, 16, 4, 16, 16,
    ];
    let expected = [
        (
            64,
            10_463_488,
            [
                8192, 64, 8192, 1728, 64, 64, 7_946_240, 64, 1_986_560, 248_320, 248_320, 64, 8192,
                64, 6912, 64, 64, 64, 64, 64, 64, 64,
            ],
        ),
        (
            128,
            10_464_640,
            [
                8192, 128, 8192, 1792, 128, 128, 7_946_240, 128, 1_986_560, 248_320, 248_320, 128,
                8192, 128, 7168, 128, 128, 128, 128, 128, 128, 128,
            ],
        ),
        (
            256,
            10_466_816,
            [
                8192, 256, 8192, 2048, 256, 256, 7_946_240, 256, 1_986_560, 248_320, 248_320, 256,
                8192, 256, 7424, 256, 256, 256, 256, 256, 256, 256,
            ],
        ),
    ];
    for (alignment, total, charged) in expected {
        let ledger = ledger_for(&survivors, 3, alignment)?;
        ensure!(ledger.streams.len() == 22);
        ensure!(ledger.total_charged_bytes == total);
        ensure!(
            ledger
                .streams
                .iter()
                .map(|stream| stream.logical_bytes)
                .eq(logical)
        );
        ensure!(
            ledger
                .streams
                .iter()
                .map(|stream| stream.charged_bytes)
                .eq(charged)
        );
    }
    ensure!(rounded_union(&[(1, 65), (63, 129)], 64)? == 192);
    ensure!(rounded_union(&[(1, 65), (63, 129)], 128)? == 256);
    ensure!(rounded_union(&[(1, 65), (63, 129)], 256)? == 256);
    Ok(())
}

fn runtime_artifact_self_test() -> Result<()> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct NestedDuplicateFixture {
        value: u8,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct DuplicateFixture {
        nested: NestedDuplicateFixture,
    }

    let root = std::env::temp_dir().join(format!(
        "qwen-v0664-self-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&root)?;
    std::fs::write(root.join("plain"), b"synthetic")?;
    let mut files = Vec::new();
    walk_packet(&root, Path::new(""), &mut files)?;
    ensure!(files == [PathBuf::from("plain")]);
    let expected = BTreeSet::from([PathBuf::from("plain")]);
    std::fs::write(root.join("foreign-file"), b"foreign")?;
    let mut foreign = Vec::new();
    walk_packet(&root, Path::new(""), &mut foreign)?;
    ensure!(validate_inventory_set(&foreign.into_iter().collect(), &expected).is_err());
    std::fs::remove_file(root.join("foreign-file"))?;
    std::os::unix::fs::symlink("plain", root.join("link"))?;
    ensure!(walk_packet(&root, Path::new(""), &mut Vec::new()).is_err());
    std::fs::remove_file(root.join("link"))?;
    std::fs::hard_link(root.join("plain"), root.join("alias"))?;
    ensure!(walk_packet(&root, Path::new(""), &mut Vec::new()).is_err());
    std::fs::remove_file(root.join("alias"))?;
    ensure!(!normalized_relative(Path::new("../escape")));
    ensure!(!normalized_relative(Path::new("/absolute")));
    ensure!(decode_f32_raw(&[0u8; 7], 2).is_err());
    let malformed = br#"{"schema":"wrong","suite":"x","passed":true,"cases":[],"extra":1}"#;
    ensure!(serde_json::from_slice::<SelfTestArtifact>(malformed).is_err());
    let nested_valid: DuplicateFixture = serde_json::from_slice(br#"{"nested":{"value":1}}"#)?;
    ensure!(nested_valid.nested.value == 1);
    let nested_duplicate = br#"{"nested":{"value":1,"value":2}}"#;
    ensure!(serde_json::from_slice::<DuplicateFixture>(nested_duplicate).is_err());
    std::fs::remove_file(root.join("plain"))?;
    std::fs::remove_dir(root)?;
    Ok(())
}

fn decode_f32_raw(bytes: &[u8], expected: usize) -> Result<Vec<f32>> {
    ensure!(bytes.len() == expected.checked_mul(4).context("raw F32 size overflow")?);
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let value = f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap()));
            ensure!(value.is_finite(), "raw F32 contains a nonfinite value");
            Ok(value)
        })
        .collect()
}

fn rounded_union(intervals: &[(u64, u64)], alignment: u64) -> Result<u64> {
    ensure!(alignment.is_power_of_two());
    ensure!(
        intervals.len() <= 2,
        "frozen ledger union exceeds two spans"
    );
    let mut rounded = [(0u64, 0u64); 2];
    let mut rounded_len = 0usize;
    for &(start, end) in intervals {
        ensure!(start <= end);
        if start == end {
            continue;
        }
        let a = start / alignment * alignment;
        let b = end.checked_add(alignment - 1).context("round overflow")? / alignment * alignment;
        rounded[rounded_len] = (a, b);
        rounded_len += 1;
    }
    rounded[..rounded_len].sort_unstable();
    let mut total = 0u64;
    let mut current: Option<(u64, u64)> = None;
    for &(a, b) in &rounded[..rounded_len] {
        match current {
            Some((start, end)) if a <= end => current = Some((start, end.max(b))),
            Some((start, end)) => {
                total += end - start;
                current = Some((a, b));
            }
            None => current = Some((a, b)),
        }
    }
    if let Some((start, end)) = current {
        total += end - start;
    }
    Ok(total)
}

fn validate_survivors(ids: &[u32], vocab: usize, winner: usize) -> Result<()> {
    ensure!(ids.windows(2).all(|pair| pair[0] < pair[1]));
    ensure!(
        ids.iter()
            .all(|&id| (id as usize) < vocab && id as usize != winner)
    );
    Ok(())
}

fn bound_survives(upper_bound: f64, winner_lower: f64) -> Result<bool> {
    ensure!(upper_bound.is_finite() && winner_lower.is_finite());
    Ok(!(upper_bound < winner_lower))
}

fn compact_active(active: &[u8], winner: usize) -> Result<Vec<u32>> {
    ensure!(winner < active.len());
    let mut survivors = Vec::new();
    for (row, &is_active) in active.iter().enumerate() {
        ensure!(is_active <= 1, "active mask contains a non-binary entry");
        if row == winner {
            ensure!(is_active == 0, "winner appears in active mask");
        } else if is_active == 1 {
            survivors.push(u32::try_from(row)?);
        }
    }
    Ok(survivors)
}

fn prompt_identity(tokenizer: &Tokenizer, path: &Path) -> Result<(String, Vec<i32>)> {
    check_nofollow_parents(path)?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    let before = FileStamp::from_metadata(&file.metadata()?);
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    ensure!(FileStamp::from_metadata(&file.metadata()?) == before);
    ensure!(bytes.len() == PROMPT_BYTES && sha256(&bytes) == PROMPT_SHA256);
    let prompt = std::str::from_utf8(&bytes).context("prompt is not UTF-8")?;
    let ids = tokenizer.encode(prompt, false)?;
    ensure!(
        ids.len() == PROMPT_TOKENS,
        "prompt token count differs from preregistration"
    );
    ensure!(
        token_ids_sha256_i32le(&ids) == TOKEN_SHA256,
        "prompt token digest differs from preregistration"
    );
    Ok((prompt.to_string(), ids))
}

fn expected_arch() -> Arch {
    Arch {
        kind: ArchKind::Moe,
        n_layer: 40,
        hidden_size: 2048,
        intermediate_size: 0,
        vocab_size: 248_320,
        full_attention_interval: 4,
        n_q_heads: 16,
        n_kv_heads: 2,
        attn_head_dim: 256,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        gdn_n_v_heads: 32,
        gdn_n_k_heads: 16,
        gdn_head_dim: 128,
        gdn_conv_kernel: 4,
        expert_count: 256,
        expert_used_count: 8,
        expert_feed_forward_length: 512,
        expert_shared_feed_forward_length: 512,
        mtp_n_hidden_layers: 0,
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct OutputDescriptorRecord {
    name: String,
    dtype: String,
    shape: [usize; 2],
    shard_index: usize,
    data_offset: u64,
    bytes: usize,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ModelIdentityRecord {
    path: String,
    bytes: u64,
    sha256: String,
    architecture: String,
    base_model_name: String,
    basename: String,
    file_type: u64,
    layers: u32,
    hidden: usize,
    vocab: usize,
    untied: bool,
    mtp_layers: u32,
    output_bias: bool,
    stop_token_ids: Vec<i32>,
    output_weight: OutputDescriptorRecord,
    retained_file_stamp: FileStamp,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct OpenedGgufRuntimeIdentity {
    runtime_load_api: String,
    exact_opened_gguf_consumed: bool,
    pre_runtime_stamp: FileStamp,
    runtime_retained_stamp: FileStamp,
    post_runtime_load_stamp: FileStamp,
    post_capture_stamp: Option<FileStamp>,
}

struct AuthenticatedModel {
    file: File,
    gguf: qwen_llm::gguf::GgufFile,
    analysis_gguf: qwen_llm::gguf::GgufFile,
    stamp: FileStamp,
    identity: ModelIdentityRecord,
    authentication_telemetry: ModelAuthenticationTelemetry,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ModelAuthenticationTelemetry {
    total_wall_nanoseconds: u64,
    complete_file_hash_wall_nanoseconds: u64,
    runtime_gguf_open_profile_and_output_hash_wall_nanoseconds: u64,
    analysis_gguf_open_profile_and_output_hash_wall_nanoseconds: u64,
}

fn stamp_retained_gguf(gguf: &qwen_llm::gguf::GgufFile) -> Result<FileStamp> {
    let stamps = gguf.revalidate_retained_shard_stamps()?;
    ensure!(stamps.len() == 1);
    let stamp = &stamps[0];
    Ok(FileStamp {
        device: stamp.device,
        inode: stamp.inode,
        bytes: stamp.size,
        mtime_seconds: stamp.mtime_sec,
        mtime_nanoseconds: stamp.mtime_nsec,
    })
}

fn extract_model_name_identity(gguf: &qwen_llm::gguf::GgufFile) -> Result<ModelNameIdentity> {
    Ok(ModelNameIdentity {
        base_model_name: gguf
            .get_str("general.base_model.0.name")
            .context("GGUF lacks general.base_model.0.name")?
            .to_string(),
        basename: gguf
            .get_str("general.basename")
            .context("GGUF lacks general.basename")?
            .to_string(),
    })
}

fn validate_raw_gguf(
    gguf: &qwen_llm::gguf::GgufFile,
    expected_stamp: &FileStamp,
) -> Result<ModelIdentityRecord> {
    ensure!(gguf.shard_count() == 1 && gguf.total_mapped_len() as u64 == MODEL_BYTES);
    ensure!(stamp_retained_gguf(gguf)? == *expected_stamp);
    let stop_token_ids = gguf.stop_token_ids()?;
    validate_authenticated_stop_ids(&stop_token_ids)?;
    let model = Model::from_gguf(gguf)?;
    ensure!(model.arch == expected_arch() && !model.tied_embeddings && model.mtp.is_none());
    ensure!(gguf.get_str("general.architecture") == Some("qwen35moe"));
    let model_names = extract_model_name_identity(gguf)?;
    ensure!(
        model_names
            == ModelNameIdentity {
                base_model_name: BASE_MODEL_NAME.to_string(),
                basename: BASENAME.to_string(),
            },
        "GGUF model-name identity differs from the frozen profile"
    );
    ensure!(gguf.get_u64("general.file_type") == Some(15));
    ensure!(gguf.tensors.iter().all(|t| t.name != "output.bias"));
    let desc = model.lm_head;
    ensure!(desc.name == "output.weight" && desc.dtype == GgmlType::Q6_K);
    ensure!(desc.shape.as_slice() == [HIDDEN as u64, VOCAB as u64]);
    ensure!(
        desc.data_offset == OUTPUT_OFFSET
            && desc.n_bytes as usize == OUTPUT_BYTES
            && desc.shard_idx == 0
    );
    let output = gguf.try_slice(desc)?;
    ensure!(sha256(output) == OUTPUT_SHA256);
    Ok(ModelIdentityRecord {
        path: MODEL_PATH.to_string(),
        bytes: MODEL_BYTES,
        sha256: MODEL_SHA256.to_string(),
        architecture: "qwen35moe".to_string(),
        base_model_name: model_names.base_model_name,
        basename: model_names.basename,
        file_type: 15,
        layers: 40,
        hidden: HIDDEN,
        vocab: VOCAB,
        untied: true,
        mtp_layers: 0,
        output_bias: false,
        stop_token_ids,
        output_weight: OutputDescriptorRecord {
            name: "output.weight".to_string(),
            dtype: "Q6_K".to_string(),
            shape: [HIDDEN, VOCAB],
            shard_index: 0,
            data_offset: OUTPUT_OFFSET,
            bytes: OUTPUT_BYTES,
            sha256: OUTPUT_SHA256.to_string(),
        },
        retained_file_stamp: expected_stamp.clone(),
    })
}

fn validate_authenticated_stop_ids(stop_token_ids: &[i32]) -> Result<()> {
    ensure!(
        stop_token_ids == STOP_IDS,
        "authenticated GGUF stop-token vector differs from frozen STOP_IDS"
    );
    Ok(())
}

fn persist_capture(packet: &mut Packet, capture: &Capture, prompt_tokens: &[i32]) -> Result<()> {
    let hidden_bytes = f32_le_bytes(&capture.hidden);
    let logits_bytes = f32_le_bytes(&capture.logits);
    let hidden_sha256 = sha256(hidden_bytes);
    let logits_sha256 = sha256(logits_bytes);
    let directory = PathBuf::from(format!("captures/call-{:03}", capture.call));
    packet.mkdir(&directory)?;
    packet.write(directory.join("hidden.f32le"), hidden_bytes)?;
    packet.write(directory.join("logits.f32le"), logits_bytes)?;
    let evidence_path = PathBuf::from(format!("winner-evidence/call-{:03}.json", capture.call));
    packet.json_strict(
        evidence_path,
        &WinnerEvidenceArtifact {
            schema: SCHEMA.to_string(),
            call: capture.call,
            sequence_position: capture.sequence_position,
            consumed_input_token: capture.consumed_input_token,
            selected_output_token: capture.winner,
            winner_logit_bits: format!("{:08x}", capture.winner_logit_bits),
            prompt_token_sha256: token_ids_sha256_i32le(prompt_tokens),
            generated_prefix_token_sha256: capture.generated_prefix_token_sha256.clone(),
            stop_checked_before_capture: capture.stop_checked_before_capture,
            callback_completed_before_capture: capture.callback_completed_before_capture,
            token_limit_checked_before_capture: capture.token_limit_checked_before_capture,
            hidden_copy_wall_nanoseconds: capture.hidden_copy_wall_nanoseconds,
            logits_copy_wall_nanoseconds: capture.logits_copy_wall_nanoseconds,
            winner_reconstruction_wall_nanoseconds: capture.winner_reconstruction_wall_nanoseconds,
            hidden: RawArtifactRecord {
                dtype: "F32".to_string(),
                shape: vec![HIDDEN],
                bytes: hidden_bytes.len(),
                sha256: hidden_sha256.clone(),
            },
            logits: RawArtifactRecord {
                dtype: "F32".to_string(),
                shape: vec![VOCAB],
                bytes: logits_bytes.len(),
                sha256: logits_sha256,
            },
            winner_order: "rust_f32_total_cmp_highest_token_id_on_equal_bits".to_string(),
        },
    )?;
    packet.json_strict(
        PathBuf::from(format!("analyzer-input/call-{:03}.json", capture.call)),
        &AnalyzerInputArtifact {
            schema: SCHEMA.to_string(),
            capture_identity: hidden_sha256,
            winner_id: capture.winner,
        },
    )?;
    Ok(())
}

fn capture_request(
    loaded: &qwen_llm::runtime::LoadedModel,
    prompt_ids: &[i32],
    authenticated_stop_ids: &[i32],
) -> Result<GenerationOutcome> {
    validate_authenticated_stop_ids(authenticated_stop_ids)?;
    let generation_start = Instant::now();
    let mut sequence = loaded.create_sequence(SequenceConfig::new(1024))?;
    let mut scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        loaded.context(),
        loaded.metal_model(),
        1024,
        1024,
    )?;
    let forward = loaded.forward();
    let prompt_logits = prefill_tokens_with_multi_hidden(
        &forward,
        prompt_ids,
        0,
        unsafe { sequence.metal_session_mut() },
        &mut scratch,
        &[],
        None,
    )?;
    sequence.advance_by(PROMPT_TOKENS)?;
    ensure!(sequence.position() == PROMPT_TOKENS);
    let mut sampler = Sampler::new(SamplingConfig::default())?;
    let mut generated = Vec::with_capacity(TOKENS);
    let mut captures = Vec::with_capacity(CALLS.len());
    let mut pending_prompt_logits = Some(prompt_logits);
    let mut pending_greedy = None;
    let mut transitions = 0usize;
    let mut coverage_eos_call = None;

    macro_rules! capture_observation {
        ($call:expr, $selected:expr, $callback_completed:expr, $token_limit_checked:expr) => {{
            let call = $call;
            let selected = $selected;
            let hidden_copy_start = Instant::now();
            let hidden = tensor_copy(&sequence.metal_session().h, HIDDEN)?;
            let hidden_copy_wall_nanoseconds =
                u64::try_from(hidden_copy_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let logits_copy_start = Instant::now();
            let logits = if call == 0 {
                let copied = tensor_copy(&sequence.metal_session().logits, VOCAB)?;
                let prompt_logits = pending_prompt_logits.as_ref().unwrap();
                ensure!(
                    copied.len() == prompt_logits.len()
                        && copied
                            .iter()
                            .zip(prompt_logits)
                            .all(|(copied, prompt)| copied.to_bits() == prompt.to_bits()),
                    "prompt returned logits differ from session.logits"
                );
                copied
            } else {
                tensor_copy(&sequence.metal_session().logits, VOCAB)?
            };
            let logits_copy_wall_nanoseconds =
                u64::try_from(logits_copy_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let reconstruction_start = Instant::now();
            let reconstructed = total_cmp_winner(&logits)?;
            let winner_reconstruction_wall_nanoseconds =
                u64::try_from(reconstruction_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            ensure!(
                reconstructed == selected as usize,
                "production selection differs from reconstructed GreedyTotal winner"
            );
            captures.push(Capture {
                call,
                sequence_position: sequence.position(),
                consumed_input_token: if call == 0 {
                    prompt_ids[PROMPT_TOKENS - 1]
                } else {
                    generated[call - 1]
                },
                winner: reconstructed,
                winner_logit_bits: logits[reconstructed].to_bits(),
                hidden,
                logits,
                generated_prefix_token_sha256: generated_token_digest(&generated),
                stop_checked_before_capture: true,
                callback_completed_before_capture: $callback_completed,
                token_limit_checked_before_capture: $token_limit_checked,
                hidden_copy_wall_nanoseconds,
                logits_copy_wall_nanoseconds,
                winner_reconstruction_wall_nanoseconds,
            });
        }};
    }

    for call in 0..TOKENS {
        let selected = if call == 0 {
            sampler
                .sample(
                    pending_prompt_logits
                        .as_ref()
                        .context("missing prompt logits")?,
                )?
                .token
        } else {
            pending_greedy
                .take()
                .context("missing GreedyTotal selection")?
        };
        ensure!(selected >= 0 && (selected as usize) < VOCAB);
        generated.push(selected);

        let stop = authenticated_stop_ids.contains(&selected);
        if stop {
            if CALLS.contains(&call) {
                capture_observation!(call, selected, false, false);
            }
            coverage_eos_call = Some(call);
            break;
        }

        // The frozen ordinary callback is a no-op, but reaching this statement
        // is the authoritative completion event.
        let callback_completed = true;
        ensure!(callback_completed);
        let token_limit = generated.len() == TOKENS;
        if CALLS.contains(&call) {
            capture_observation!(call, selected, true, true);
        }
        if token_limit {
            break;
        }
        let position = u32::try_from(sequence.position())?;
        let returned = forward
            .single_token_greedy(selected, position, unsafe { sequence.metal_session_mut() })?
            .into_token()?;
        sequence.advance_by(1)?;
        transitions += 1;
        ensure!(sequence.position() == PROMPT_TOKENS + call + 1);
        pending_greedy = Some(returned);
        pending_prompt_logits = None;
    }
    if coverage_eos_call.is_none() {
        ensure!(generated.len() == TOKENS);
        ensure!(transitions == TOKENS - 1);
        ensure!(
            sequence.position() == PROMPT_TOKENS + TOKENS - 1,
            "capture violated pending N-1 transition semantics"
        );
    }
    Ok(GenerationOutcome {
        captures,
        generated,
        transitions,
        coverage_eos_call,
        generation_wall_nanoseconds: u64::try_from(generation_start.elapsed().as_nanos())
            .unwrap_or(u64::MAX),
    })
}

fn populate_block_norms(output: &[u8]) -> Result<Vec<f32>> {
    ensure!(output.len() == OUTPUT_BYTES);
    let mut norms = Vec::with_capacity(VOCAB * BLOCKS);
    for row in output.chunks_exact(ROW_BYTES) {
        for block in row.chunks_exact(BLOCK_BYTES) {
            let block = Q6Block::parse(block)?;
            norms.push(block.outward_norm_f32()?);
        }
    }
    ensure!(norms.len() == VOCAB * BLOCKS);
    Ok(norms)
}

fn validate_block_norms_exact(output: &[u8], norms: &[f32]) -> Result<u64> {
    ensure!(output.len() == OUTPUT_BYTES && norms.len() == VOCAB * BLOCKS);
    let mut tracker = BigIntPayloadTracker::default();
    for (row_index, row) in output.chunks_exact(ROW_BYTES).enumerate() {
        let mut row_square = Dyadic::zero();
        tracker.observe_dyadic(&row_square);
        for (block_index, raw) in row.chunks_exact(BLOCK_BYTES).enumerate() {
            let block_square = Q6Block::parse(raw)?.exact_square_profiled(&mut tracker)?;
            row_square = tracked_dyadic_add(&row_square, &block_square, &mut tracker);
            let norm = norms[row_index * BLOCKS + block_index];
            ensure!(norm.is_finite() && norm >= 0.0);
            let norm_dyadic = f32_dyadic(norm)?;
            let norm_square = tracked_dyadic_mul(&norm_dyadic, &norm_dyadic, &mut tracker);
            ensure!(
                tracked_dyadic_cmp(&norm_square, &block_square, &mut tracker) != Ordering::Less,
                "stored F32 block norm does not contain exact square"
            );
        }
        ensure!(row_square.coefficient.sign() != Sign::Minus);
        tracker.observe_dyadic(&row_square);
    }
    Ok(tracker.largest_observed_coefficient_payload_bits)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct StreamCharge {
    epoch: String,
    resource: String,
    direction: String,
    logical_bytes: u64,
    charged_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct AlignmentLedger {
    alignment: u64,
    streams: Vec<StreamCharge>,
    total_charged_bytes: u64,
}

fn stream(
    epoch: &'static str,
    resource: &'static str,
    direction: &'static str,
    element_bytes: u64,
    intervals: &[(u64, u64)],
    alignment: u64,
) -> Result<StreamCharge> {
    let logical_bytes = intervals.iter().try_fold(0u64, |sum, &(a, b)| {
        sum.checked_add(b.checked_sub(a).context("interval underflow")?)
            .context("logical overflow")
    })?;
    ensure!(intervals.len() <= 2, "frozen stream exceeds two spans");
    let mut byte_intervals = [(0u64, 0u64); 2];
    for (index, &(a, b)) in intervals.iter().enumerate() {
        byte_intervals[index] = (
            a.checked_mul(element_bytes).context("offset overflow")?,
            b.checked_mul(element_bytes).context("endpoint overflow")?,
        );
    }
    Ok(StreamCharge {
        epoch: epoch.to_string(),
        resource: resource.to_string(),
        direction: direction.to_string(),
        logical_bytes: logical_bytes * element_bytes,
        charged_bytes: rounded_union(&byte_intervals[..intervals.len()], alignment)?,
    })
}

fn survivor_row_stream(survivors: &[u32], alignment: u64) -> Result<StreamCharge> {
    ensure!(alignment.is_power_of_two());
    let mut total = 0u64;
    let mut current: Option<(u64, u64)> = None;
    for &id in survivors {
        let start = u64::from(id)
            .checked_mul(ROW_BYTES as u64)
            .context("survivor row offset overflow")?;
        let end = start
            .checked_add(ROW_BYTES as u64)
            .context("survivor row endpoint overflow")?;
        let a = start / alignment * alignment;
        let b = end.checked_add(alignment - 1).context("round overflow")? / alignment * alignment;
        match current {
            Some((union_start, union_end)) if a <= union_end => {
                current = Some((union_start, union_end.max(b)));
            }
            Some((union_start, union_end)) => {
                total = total
                    .checked_add(union_end - union_start)
                    .context("survivor row union overflow")?;
                current = Some((a, b));
            }
            None => current = Some((a, b)),
        }
    }
    if let Some((start, end)) = current {
        total = total
            .checked_add(end - start)
            .context("survivor row union overflow")?;
    }
    Ok(StreamCharge {
        epoch: "survivor".to_string(),
        resource: "output_weight".to_string(),
        direction: "read".to_string(),
        logical_bytes: (survivors.len() as u64)
            .checked_mul(ROW_BYTES as u64)
            .context("survivor row logical-byte overflow")?,
        charged_bytes: total,
    })
}

fn ledger_for(survivors: &[u32], winner: usize, alignment: u64) -> Result<AlignmentLedger> {
    validate_survivors(survivors, VOCAB, winner)?;
    let n = survivors.len() as u64;
    let competitors = [(0, winner as u64), (winner as u64 + 1, VOCAB as u64)];
    let competitor_blocks = [
        (0, (winner * BLOCKS) as u64),
        (((winner + 1) * BLOCKS) as u64, (VOCAB * BLOCKS) as u64),
    ];
    let hidden = [(0, HIDDEN as u64)];
    let hidden_blocks = [(0, BLOCKS as u64)];
    let compact = [(0, n)];
    let winner_row = [(
        winner as u64 * ROW_BYTES as u64,
        winner as u64 * ROW_BYTES as u64 + ROW_BYTES as u64,
    )];
    let mut streams = vec![
        stream("hidden-norm", "hidden", "read", 4, &hidden, alignment)?,
        stream(
            "hidden-norm",
            "hidden_block_norm",
            "write",
            4,
            &hidden_blocks,
            alignment,
        )?,
        stream("winner-threshold", "hidden", "read", 4, &hidden, alignment)?,
        stream(
            "winner-threshold",
            "output_weight",
            "read",
            1,
            &winner_row,
            alignment,
        )?,
        stream(
            "winner-threshold",
            "winner_interval",
            "write",
            8,
            &[(0, 2)],
            alignment,
        )?,
        stream(
            "winner-threshold",
            "winner_exact",
            "write",
            8,
            &[(0, 8)],
            alignment,
        )?,
        stream(
            "block-bound",
            "block_norm",
            "read",
            4,
            &competitor_blocks,
            alignment,
        )?,
        stream(
            "block-bound",
            "hidden_block_norm",
            "read",
            4,
            &hidden_blocks,
            alignment,
        )?,
        stream("block-bound", "upper", "write", 8, &competitors, alignment)?,
        stream("block-bound", "active", "write", 1, &competitors, alignment)?,
        stream("compact", "active", "read", 1, &competitors, alignment)?,
        stream("compact", "survivor_ids", "write", 4, &compact, alignment)?,
        stream("survivor", "hidden", "read", 4, &hidden, alignment)?,
        stream("survivor", "survivor_ids", "read", 4, &compact, alignment)?,
        survivor_row_stream(survivors, alignment)?,
        stream("survivor", "winner_exact", "read", 8, &[(0, 8)], alignment)?,
        stream("survivor", "survivor_cmp", "write", 1, &compact, alignment)?,
        stream(
            "survivor",
            "exact_accumulator",
            "write",
            8,
            &[(0, 8)],
            alignment,
        )?,
        stream("reduce", "survivor_ids", "read", 4, &compact, alignment)?,
        stream("reduce", "survivor_cmp", "read", 1, &compact, alignment)?,
        stream("reduce", "winner_interval", "read", 8, &[(0, 2)], alignment)?,
        stream(
            "reduce",
            "reduction_record",
            "write",
            8,
            &[(0, 2)],
            alignment,
        )?,
    ];
    let total_charged_bytes = streams.iter().try_fold(0u64, |sum, item| {
        sum.checked_add(item.charged_bytes)
            .context("ledger total overflow")
    })?;
    streams.shrink_to_fit();
    Ok(AlignmentLedger {
        alignment,
        streams,
        total_charged_bytes,
    })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CaptureScreening {
    capture_identity: String,
    winner_id: usize,
    winner_lower_bits: String,
    winner_upper_bits: String,
    winner_exact_coefficient_hex: String,
    winner_exact_exponent: i32,
    competitor_count: usize,
    bound_pruned_count: usize,
    survivor_count: usize,
    less_than_winner_count: usize,
    tie_count: usize,
    greater_than_winner_count: usize,
    unique_ideal_winner: bool,
    pruning_pass: bool,
    bytes_pass: bool,
    charged_64_bytes: u64,
    charged_128_bytes: u64,
    charged_256_bytes: u64,
    unique_q6_footprint_bytes: u64,
    survivor_ids_sha256: String,
    survivor_cmp_sha256: String,
    upper_vec_requested_capacity_bytes: usize,
    active_vec_requested_capacity_bytes: usize,
    survivor_ids_vec_requested_capacity_bytes: usize,
    survivor_cmp_vec_requested_capacity_bytes: usize,
    survivor_ids_digest_vec_requested_capacity_bytes: usize,
    largest_observed_bigint_coefficient_payload_bits: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExactWinnerRecord {
    coefficient_twos_complement_u64x8: String,
    exponent: i32,
    signed_bits: u64,
    lower_f64_bits: String,
    upper_f64_bits: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExactCaptureRecord {
    capture_identity: String,
    winner_id: usize,
    winner: ExactWinnerRecord,
    survivors: usize,
    less: usize,
    ties: usize,
    greater: usize,
    survivor_ids: Vec<u32>,
    survivor_cmp: Vec<u8>,
    comparison_less: u8,
    comparison_equal: u8,
    comparison_greater: u8,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CaptureLedgerRecord {
    capture_identity: String,
    full_head_denominator_bytes: u64,
    gate_limit_bytes: u64,
    winner_excluded: bool,
    logical_survivors: usize,
    alignments: [AlignmentLedger; 3],
    reduction: ReductionRecord,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ReductionRecord {
    competitor_count: usize,
    bound_pruned_count: usize,
    survivor_count: usize,
    less_than_winner_count: usize,
    tie_count: usize,
    greater_than_winner_count: usize,
    winner_interval_lower_bits: String,
    winner_interval_upper_bits: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedCaptureScreening {
    call: usize,
    sequence_position: usize,
    analysis: CaptureScreening,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedExactCaptureRecord {
    call: usize,
    sequence_position: usize,
    analysis: ExactCaptureRecord,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedCaptureLedgerRecord {
    call: usize,
    sequence_position: usize,
    analysis: CaptureLedgerRecord,
}

struct JoinedAnalyzerRecords {
    screening: Vec<PersistedCaptureScreening>,
    exact_records: Vec<PersistedExactCaptureRecord>,
    ledgers: Vec<PersistedCaptureLedgerRecord>,
}

fn reduce_survivors(
    winner: usize,
    survivor_ids: &[u32],
    survivor_cmp: &[u8],
    winner_interval: (f64, f64),
) -> Result<ReductionRecord> {
    ensure!(survivor_ids.len() == survivor_cmp.len());
    validate_survivors(survivor_ids, VOCAB, winner)?;
    ensure!(winner_interval.0.is_finite() && winner_interval.1.is_finite());
    ensure!(winner_interval.0 <= winner_interval.1);
    let mut less = 0usize;
    let mut ties = 0usize;
    let mut greater = 0usize;
    let mut invalid_flag = None;
    for (&id, &flag) in survivor_ids.iter().zip(survivor_cmp) {
        ensure!((id as usize) < VOCAB && id as usize != winner);
        match flag {
            0 => less += 1,
            1 => ties += 1,
            2 => greater += 1,
            _ => {
                invalid_flag.get_or_insert(flag);
            }
        }
    }
    ensure!(
        invalid_flag.is_none(),
        "invalid survivor comparison flag {}",
        invalid_flag.unwrap_or_default()
    );
    let survivor_count = survivor_ids.len();
    let bound_pruned_count = (VOCAB - 1)
        .checked_sub(survivor_count)
        .context("survivor reduction exceeds competitor count")?;
    ensure!(less + ties + greater == survivor_count);
    ensure!(bound_pruned_count + survivor_count == VOCAB - 1);
    Ok(ReductionRecord {
        competitor_count: VOCAB - 1,
        bound_pruned_count,
        survivor_count,
        less_than_winner_count: less,
        tie_count: ties,
        greater_than_winner_count: greater,
        winner_interval_lower_bits: format!("{:016x}", winner_interval.0.to_bits()),
        winner_interval_upper_bits: format!("{:016x}", winner_interval.1.to_bits()),
    })
}

fn bigint_twos_hex(value: &BigInt, words: usize) -> Result<String> {
    ensure!(words == 8, "only the frozen u64x8 resource is supported");
    let mut encoded = [0u64; 8];
    let mut digits = value.iter_u64_digits();
    for word in &mut encoded {
        *word = digits.next().unwrap_or(0);
    }
    ensure!(digits.next().is_none(), "coefficient exceeds u64x8");

    match value.sign() {
        Sign::NoSign => {}
        Sign::Plus => ensure!(encoded[7] >> 63 == 0, "positive coefficient exceeds i512"),
        Sign::Minus => {
            ensure!(
                encoded[7] < (1u64 << 63)
                    || (encoded[7] == (1u64 << 63) && encoded[..7].iter().all(|word| *word == 0)),
                "negative coefficient exceeds i512"
            );
            for word in &mut encoded {
                *word = !*word;
            }
            let mut carry = true;
            for word in &mut encoded {
                if !carry {
                    break;
                }
                let (sum, overflow) = word.overflowing_add(1);
                *word = sum;
                carry = overflow;
            }
        }
    }

    let mut output = String::with_capacity(128);
    for word in encoded.iter().rev() {
        write!(&mut output, "{word:016x}")?;
    }
    ensure!(output.len() == 128);
    Ok(output)
}

fn analyze_capture(
    input: &AnalyzerInput<'_>,
    capture: &AnalyzerCapture,
) -> Result<(CaptureScreening, ExactCaptureRecord, CaptureLedgerRecord)> {
    let mut bigint_tracker = BigIntPayloadTracker::default();
    let hidden_norm = hidden_norms_profiled(&capture.hidden, &mut bigint_tracker)?;
    let winner_row =
        &input.output_weight[capture.winner * ROW_BYTES..(capture.winner + 1) * ROW_BYTES];
    let winner_exact = exact_dot_profiled(winner_row, &capture.hidden, &mut bigint_tracker)?;
    let (lower, upper) = dyadic_interval_profiled(&winner_exact, &mut bigint_tracker)?;
    ensure!(lower.is_finite() && upper.is_finite() && lower <= upper);
    ensure!(
        tracked_dyadic_cmp(&f64_dyadic(lower)?, &winner_exact, &mut bigint_tracker)
            != Ordering::Greater
    );
    ensure!(
        tracked_dyadic_cmp(&f64_dyadic(upper)?, &winner_exact, &mut bigint_tracker)
            != Ordering::Less
    );
    let mut upper_bounds = vec![0.0f64; VOCAB];
    let mut active = vec![0u8; VOCAB];
    for row in 0..VOCAB {
        if row == capture.winner {
            continue;
        }
        let mut terms = [0.0f64; BLOCKS];
        for block in 0..BLOCKS {
            terms[block] =
                f64::from(input.block_norms[row * BLOCKS + block]) * f64::from(hidden_norm[block]);
        }
        ensure!(terms.iter().all(|term| term.is_finite() && *term >= 0.0));
        let ub = gamma_sum_upper(&terms)?;
        upper_bounds[row] = ub;
        if bound_survives(ub, lower)? {
            active[row] = 1;
        }
    }
    // Frozen compact epoch: this pass reads only the completed active mask.
    let survivors = compact_active(&active, capture.winner)?;
    validate_survivors(&survivors, VOCAB, capture.winner)?;
    ensure!(
        survivors.iter().copied().eq((0..VOCAB)
            .filter(|&id| id != capture.winner && upper_bounds[id] >= lower)
            .map(|id| id as u32))
    );
    let bound_pruned = VOCAB - 1 - survivors.len();
    let mut comparisons = Vec::with_capacity(survivors.len());
    for &id in &survivors {
        let start = id as usize * ROW_BYTES;
        let score = exact_dot_profiled(
            &input.output_weight[start..start + ROW_BYTES],
            &capture.hidden,
            &mut bigint_tracker,
        )?;
        let disposition = match tracked_dyadic_cmp(&score, &winner_exact, &mut bigint_tracker) {
            Ordering::Less => 0u8,
            Ordering::Equal => 1u8,
            Ordering::Greater => 2u8,
        };
        comparisons.push(disposition);
    }
    let reduction = reduce_survivors(capture.winner, &survivors, &comparisons, (lower, upper))?;
    ensure!(reduction.bound_pruned_count == bound_pruned);
    let ledger64 = ledger_for(&survivors, capture.winner, 64)?;
    let ledger128 = ledger_for(&survivors, capture.winner, 128)?;
    let ledger256 = ledger_for(&survivors, capture.winner, 256)?;
    let mut ids_bytes = Vec::with_capacity(survivors.len() * 4);
    for id in &survivors {
        ids_bytes.extend_from_slice(&id.to_le_bytes());
    }
    let unique_rows = survivors.len() + 1;
    let result = CaptureScreening {
        capture_identity: capture.capture_identity.clone(),
        winner_id: capture.winner,
        winner_lower_bits: format!("{:016x}", lower.to_bits()),
        winner_upper_bits: format!("{:016x}", upper.to_bits()),
        winner_exact_coefficient_hex: bigint_twos_hex(&winner_exact.coefficient, 8)?,
        winner_exact_exponent: DOT_EXP,
        competitor_count: VOCAB - 1,
        bound_pruned_count: bound_pruned,
        survivor_count: survivors.len(),
        less_than_winner_count: reduction.less_than_winner_count,
        tie_count: reduction.tie_count,
        greater_than_winner_count: reduction.greater_than_winner_count,
        unique_ideal_winner: reduction.tie_count == 0 && reduction.greater_than_winner_count == 0,
        pruning_pass: bound_pruned >= input.ledger.pruning_floor,
        bytes_pass: ledger128.total_charged_bytes <= input.ledger.byte_limit,
        charged_64_bytes: ledger64.total_charged_bytes,
        charged_128_bytes: ledger128.total_charged_bytes,
        charged_256_bytes: ledger256.total_charged_bytes,
        unique_q6_footprint_bytes: (unique_rows * ROW_BYTES) as u64,
        survivor_ids_sha256: sha256(&ids_bytes),
        survivor_cmp_sha256: sha256(&comparisons),
        upper_vec_requested_capacity_bytes: upper_bounds.capacity() * std::mem::size_of::<f64>(),
        active_vec_requested_capacity_bytes: active.capacity() * std::mem::size_of::<u8>(),
        survivor_ids_vec_requested_capacity_bytes: survivors.capacity()
            * std::mem::size_of::<u32>(),
        survivor_cmp_vec_requested_capacity_bytes: comparisons.capacity()
            * std::mem::size_of::<u8>(),
        survivor_ids_digest_vec_requested_capacity_bytes: ids_bytes.capacity(),
        largest_observed_bigint_coefficient_payload_bits: bigint_tracker
            .largest_observed_coefficient_payload_bits,
    };
    let exact = ExactCaptureRecord {
        capture_identity: capture.capture_identity.clone(),
        winner_id: capture.winner,
        winner: ExactWinnerRecord {
            coefficient_twos_complement_u64x8: result.winner_exact_coefficient_hex.clone(),
            exponent: DOT_EXP,
            signed_bits: winner_exact.signed_bits(),
            lower_f64_bits: result.winner_lower_bits.clone(),
            upper_f64_bits: result.winner_upper_bits.clone(),
        },
        survivors: survivors.len(),
        less: reduction.less_than_winner_count,
        ties: reduction.tie_count,
        greater: reduction.greater_than_winner_count,
        survivor_ids: survivors,
        survivor_cmp: comparisons,
        comparison_less: 0,
        comparison_equal: 1,
        comparison_greater: 2,
    };
    let ledger = CaptureLedgerRecord {
        capture_identity: capture.capture_identity.clone(),
        full_head_denominator_bytes: input.ledger.full_head,
        gate_limit_bytes: input.ledger.byte_limit,
        winner_excluded: true,
        logical_survivors: exact.survivors,
        alignments: [ledger64, ledger128, ledger256],
        reduction,
    };
    Ok((result, exact, ledger))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct InventoryEntry {
    path: String,
    bytes: u64,
    sha256: String,
    stamp: FileStamp,
}

fn walk_packet(root: &Path, relative: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    let directory = root.join(relative);
    let metadata = std::fs::symlink_metadata(&directory)?;
    ensure!(metadata.file_type().is_dir() && !metadata.file_type().is_symlink());
    let mut entries = std::fs::read_dir(&directory)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let child = relative.join(entry.file_name());
        let metadata = std::fs::symlink_metadata(entry.path())?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "packet contains a symlink"
        );
        if metadata.file_type().is_dir() {
            ensure!(
                allowed_packet_directory(&child),
                "packet contains a foreign directory"
            );
            walk_packet(root, &child, output)?;
        } else {
            ensure!(
                metadata.file_type().is_file(),
                "packet contains a non-regular file"
            );
            ensure!(
                metadata.nlink() == 1,
                "packet contains a hard-linked artifact"
            );
            output.push(child);
        }
    }
    Ok(())
}

fn allowed_packet_directory(path: &Path) -> bool {
    matches!(
        path.to_str(),
        Some("captures" | "winner-evidence" | "analyzer-input" | "metadata")
    ) || CALLS
        .iter()
        .any(|call| path == Path::new(&format!("captures/call-{call:03}")))
}

fn open_nofollow_regular(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    let metadata = file.metadata()?;
    ensure!(metadata.file_type().is_file() && metadata.nlink() == 1);
    Ok(file)
}

fn hash_nofollow(path: &Path) -> Result<(FileStamp, String)> {
    let mut file = open_nofollow_regular(path)?;
    let before = file.metadata()?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let after = file.metadata()?;
    ensure!(
        before.dev() == after.dev()
            && before.ino() == after.ino()
            && before.len() == after.len()
            && before.mtime() == after.mtime()
            && before.mtime_nsec() == after.mtime_nsec(),
        "artifact changed while hashing"
    );
    Ok((
        FileStamp::from_metadata(&after),
        format!("{:x}", hash.finalize()),
    ))
}

fn validate_raw_record(
    record: &RawArtifactRecord,
    shape: &[usize],
    bytes: usize,
    label: &str,
) -> Result<()> {
    ensure!(
        record.dtype == "F32" && record.shape == shape && record.bytes == bytes,
        "{label} raw descriptor changed"
    );
    validate_sha256(&record.sha256, &format!("{label} SHA-256"))
}

fn validate_winner_evidence(evidence: &WinnerEvidenceArtifact) -> Result<()> {
    ensure!(
        evidence.schema == SCHEMA
            && evidence.call < TOKENS
            && evidence.sequence_position == PROMPT_TOKENS + evidence.call
            && evidence.selected_output_token < VOCAB
            && evidence.prompt_token_sha256 == TOKEN_SHA256
            && evidence.winner_order == "rust_f32_total_cmp_highest_token_id_on_equal_bits"
            && evidence.stop_checked_before_capture,
        "winner evidence constants changed"
    );
    let eos = STOP_IDS.contains(&(evidence.selected_output_token as i32));
    ensure!(
        if eos {
            !evidence.callback_completed_before_capture
                && !evidence.token_limit_checked_before_capture
        } else {
            evidence.callback_completed_before_capture
                && evidence.token_limit_checked_before_capture
        },
        "winner evidence completed-step telemetry is inconsistent"
    );
    validate_f32_bits(&evidence.winner_logit_bits, "winner-logit bits")?;
    validate_sha256(&evidence.prompt_token_sha256, "prompt-token SHA-256")?;
    validate_sha256(
        &evidence.generated_prefix_token_sha256,
        "generated-prefix SHA-256",
    )?;
    validate_raw_record(&evidence.hidden, &[HIDDEN], HIDDEN * 4, "hidden")?;
    validate_raw_record(&evidence.logits, &[VOCAB], VOCAB * 4, "logits")
}

fn validate_screening_record(record: &PersistedCaptureScreening) -> Result<()> {
    let analysis = &record.analysis;
    ensure!(record.sequence_position == PROMPT_TOKENS + record.call);
    validate_sha256(&analysis.capture_identity, "screening capture identity")?;
    let lower = decode_f64_bits(&analysis.winner_lower_bits, "winner lower bits")?;
    let upper = decode_f64_bits(&analysis.winner_upper_bits, "winner upper bits")?;
    ensure!(lower <= upper, "winner interval is reversed");
    validate_exact_coefficient(
        &analysis.winner_exact_coefficient_hex,
        "winner exact coefficient",
    )?;
    validate_sha256(&analysis.survivor_ids_sha256, "survivor-ID SHA-256")?;
    validate_sha256(&analysis.survivor_cmp_sha256, "survivor-comparison SHA-256")
}

fn validate_exact_record(record: &PersistedExactCaptureRecord) -> Result<()> {
    let analysis = &record.analysis;
    ensure!(record.sequence_position == PROMPT_TOKENS + record.call);
    validate_sha256(&analysis.capture_identity, "exact capture identity")?;
    validate_exact_coefficient(
        &analysis.winner.coefficient_twos_complement_u64x8,
        "exact winner coefficient",
    )?;
    let lower = decode_f64_bits(&analysis.winner.lower_f64_bits, "exact winner lower bits")?;
    let upper = decode_f64_bits(&analysis.winner.upper_f64_bits, "exact winner upper bits")?;
    ensure!(lower <= upper, "exact winner interval is reversed");
    Ok(())
}

fn validate_ledger_record(record: &PersistedCaptureLedgerRecord) -> Result<()> {
    ensure!(record.sequence_position == PROMPT_TOKENS + record.call);
    validate_sha256(&record.analysis.capture_identity, "ledger capture identity")?;
    let lower = decode_f64_bits(
        &record.analysis.reduction.winner_interval_lower_bits,
        "ledger winner lower bits",
    )?;
    let upper = decode_f64_bits(
        &record.analysis.reduction.winner_interval_upper_bits,
        "ledger winner upper bits",
    )?;
    ensure!(lower <= upper, "ledger winner interval is reversed");
    Ok(())
}

fn strict_reparse_json(path: &Path, relative: &Path) -> Result<()> {
    if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
        return Ok(());
    }
    let mut bytes = Vec::new();
    open_nofollow_regular(path)?.read_to_end(&mut bytes)?;
    ensure!(
        bytes.last() == Some(&b'\n'),
        "JSON artifact lacks canonical newline"
    );
    bytes.pop();
    let canonical = if relative == Path::new("readiness.json") {
        let parsed: ReadinessArtifact = serde_json::from_slice(&bytes)?;
        validate_readiness_constants(&parsed)?;
        serde_json::to_vec(&parsed)?
    } else if relative == Path::new("self-tests.json") {
        let parsed: SelfTestArtifact = serde_json::from_slice(&bytes)?;
        ensure!(parsed.schema == SCHEMA);
        for bits in parsed
            .observations
            .wide_positive_interval_bits
            .iter()
            .chain(&parsed.observations.wide_negative_interval_bits)
        {
            validate_f64_bits(bits, "self-test interval bits")?;
        }
        validate_lower_hex(&parsed.observations.fpcr_bits, 16, "self-test FPCR bits")?;
        validate_f64_bits(
            &parsed.observations.gradual_underflow_f64_bits,
            "self-test gradual-underflow bits",
        )?;
        validate_f32_bits(
            &parsed.observations.f64_to_f32_subnormal_bits,
            "self-test F64-to-F32 subnormal bits",
        )?;
        validate_f64_bits(
            &parsed.observations.sqrt_2_f64_bits,
            "self-test sqrt(2) bits",
        )?;
        for bits in &parsed.observations.f64_to_f32_midpoint_ties_even_bits {
            validate_f32_bits(bits, "self-test F64-to-F32 midpoint bits")?;
        }
        let opened_fixture_bytes = opened_gguf_identity_fixture().len();
        ensure!(
            parsed.observations.opened_gguf_rewind
                == OpenedGgufRewindObservation {
                    fixture_bytes: opened_fixture_bytes,
                    shared_cursor_position_before_parse: opened_fixture_bytes as u64,
                    diagnostic_path_absent_before_parse: true,
                    parsed_tensor_count: 1,
                    parsed_tensor_name: "t".to_string(),
                    model_names: ModelNameIdentity {
                        base_model_name: BASE_MODEL_NAME.to_string(),
                        basename: BASENAME.to_string(),
                    },
                },
            "opened-GGUF runtime self-test observation changed"
        );
        serde_json::to_vec(&parsed)?
    } else if relative == Path::new("artifact-manifest.json") {
        let parsed: ArtifactManifest = serde_json::from_slice(&bytes)?;
        ensure!(parsed.schema == SCHEMA);
        for entry in &parsed.files {
            validate_sha256(&entry.sha256, "artifact inventory SHA-256")?;
        }
        serde_json::to_vec(&parsed)?
    } else if relative == Path::new("capture-manifest.json") {
        let parsed: CaptureManifestArtifact = serde_json::from_slice(&bytes)?;
        validate_capture_manifest_constants(&parsed)?;
        serde_json::to_vec(&parsed)?
    } else if relative
        .to_str()
        .is_some_and(|path| path.starts_with("winner-evidence/"))
    {
        let parsed: WinnerEvidenceArtifact = serde_json::from_slice(&bytes)?;
        validate_winner_evidence(&parsed)?;
        serde_json::to_vec(&parsed)?
    } else if relative
        .to_str()
        .is_some_and(|path| path.starts_with("analyzer-input/"))
    {
        let parsed: AnalyzerInputArtifact = serde_json::from_slice(&bytes)?;
        ensure!(parsed.schema == SCHEMA && parsed.winner_id < VOCAB);
        validate_sha256(&parsed.capture_identity, "analyzer-input identity")?;
        serde_json::to_vec(&parsed)?
    } else if relative == Path::new("metadata-manifest.json") {
        if let Ok(parsed) = serde_json::from_slice::<MetadataManifestArtifact>(&bytes) {
            ensure!(parsed.schema == SCHEMA);
            validate_sha256(&parsed.sha256, "block-norm SHA-256")?;
            validate_sha256(
                &parsed.source_complete_head_sha256,
                "metadata source-head SHA-256",
            )?;
            serde_json::to_vec(&parsed)?
        } else {
            canonical_skipped_artifact(&bytes)?
        }
    } else if relative == Path::new("exact-validation.json") {
        if let Ok(parsed) = serde_json::from_slice::<ExactValidationArtifact>(&bytes) {
            ensure!(parsed.schema == SCHEMA);
            for record in &parsed.records {
                validate_exact_record(record)?;
            }
            serde_json::to_vec(&parsed)?
        } else {
            canonical_skipped_artifact(&bytes)?
        }
    } else if relative == Path::new("screening-result.json") {
        if let Ok(parsed) = serde_json::from_slice::<ScreeningResultArtifact>(&bytes) {
            ensure!(parsed.schema == SCHEMA);
            for record in &parsed.captures {
                validate_screening_record(record)?;
            }
            serde_json::to_vec(&parsed)?
        } else {
            canonical_skipped_artifact(&bytes)?
        }
    } else if relative == Path::new("byte-ledger.json") {
        if let Ok(parsed) = serde_json::from_slice::<ByteLedgerArtifact>(&bytes) {
            ensure!(parsed.schema == SCHEMA);
            for record in &parsed.ledgers {
                validate_ledger_record(record)?;
            }
            serde_json::to_vec(&parsed)?
        } else {
            canonical_skipped_artifact(&bytes)?
        }
    } else {
        return Err(anyhow!(
            "unexpected JSON artifact path {}",
            relative.display()
        ));
    };
    ensure!(
        canonical == bytes,
        "JSON artifact is not canonical strict encoding"
    );
    Ok(())
}

fn canonical_skipped_artifact(bytes: &[u8]) -> Result<Vec<u8>> {
    let parsed: SkippedAnalysisArtifact = serde_json::from_slice(bytes)?;
    ensure!(parsed.schema == SCHEMA);
    Ok(serde_json::to_vec(&parsed)?)
}

fn validate_raw_artifact_size(relative: &Path, bytes: u64) -> Result<()> {
    let text = relative.to_str().context("non-UTF8 artifact path")?;
    if text.ends_with("/hidden.f32le") {
        ensure!(bytes == (HIDDEN * 4) as u64, "hidden raw-size corruption");
    } else if text.ends_with("/logits.f32le") {
        ensure!(bytes == (VOCAB * 4) as u64, "logits raw-size corruption");
    } else if text == "metadata/block-norm.f32le" {
        ensure!(
            bytes == (VOCAB * BLOCKS * 4) as u64,
            "metadata raw-size corruption"
        );
    }
    Ok(())
}

fn complete_expected_artifacts() -> BTreeSet<PathBuf> {
    let mut expected = BTreeSet::from([
        PathBuf::from("readiness.json"),
        PathBuf::from("self-tests.json"),
        PathBuf::from("capture-manifest.json"),
        PathBuf::from("metadata/block-norm.f32le"),
        PathBuf::from("metadata-manifest.json"),
        PathBuf::from("exact-validation.json"),
        PathBuf::from("screening-result.json"),
        PathBuf::from("byte-ledger.json"),
    ]);
    for call in CALLS {
        expected.insert(PathBuf::from(format!(
            "captures/call-{call:03}/hidden.f32le"
        )));
        expected.insert(PathBuf::from(format!(
            "captures/call-{call:03}/logits.f32le"
        )));
        expected.insert(PathBuf::from(format!(
            "winner-evidence/call-{call:03}.json"
        )));
        expected.insert(PathBuf::from(format!("analyzer-input/call-{call:03}.json")));
    }
    expected
}

fn validate_inventory_set(actual: &BTreeSet<PathBuf>, expected: &BTreeSet<PathBuf>) -> Result<()> {
    ensure!(
        actual == expected,
        "packet contains missing or foreign files"
    );
    Ok(())
}

fn load_json_nofollow<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let mut bytes = Vec::new();
    open_nofollow_regular(path)?.read_to_end(&mut bytes)?;
    ensure!(
        bytes.pop() == Some(b'\n'),
        "persisted JSON lacks final newline"
    );
    Ok(serde_json::from_slice(&bytes)?)
}

fn validate_readiness_and_capture_binding(
    root: &Path,
    capture_manifest_present: bool,
) -> Result<ReadinessArtifact> {
    let readiness: ReadinessArtifact = load_json_nofollow(&root.join("readiness.json"))?;
    validate_readiness_constants(&readiness)?;
    if capture_manifest_present {
        let capture: CaptureManifestArtifact =
            load_json_nofollow(&root.join("capture-manifest.json"))?;
        validate_capture_manifest_constants(&capture)?;
        ensure!(
            serde_json::to_vec(&capture.build_identity)?
                == serde_json::to_vec(&readiness.build_identity)?,
            "capture and readiness build identities differ"
        );
        ensure!(
            capture.executable_identity == readiness.executable_identity,
            "capture and readiness executable identities differ"
        );
    }
    Ok(readiness)
}

fn load_raw_f32_nofollow(path: &Path, elements: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    let mut bytes = Vec::new();
    open_nofollow_regular(path)?.read_to_end(&mut bytes)?;
    let values = decode_f32_raw(&bytes, elements)?;
    Ok((bytes, values))
}

fn semantic_packet_cross_check(
    root: &Path,
    complete_analysis: bool,
) -> Result<CaptureManifestArtifact> {
    validate_readiness_and_capture_binding(root, true)?;
    let manifest: CaptureManifestArtifact =
        load_json_nofollow(&root.join("capture-manifest.json"))?;
    validate_capture_manifest_constants(&manifest)?;
    let mut capture_keys = BTreeMap::new();
    for summary in &manifest.captures {
        let evidence: WinnerEvidenceArtifact = load_json_nofollow(
            &root.join(format!("winner-evidence/call-{:03}.json", summary.call)),
        )?;
        validate_winner_evidence(&evidence)?;
        let analyzer: AnalyzerInputArtifact = load_json_nofollow(
            &root.join(format!("analyzer-input/call-{:03}.json", summary.call)),
        )?;
        ensure!(analyzer.schema == SCHEMA);
        validate_sha256(&analyzer.capture_identity, "analyzer-input identity")?;
        let (hidden_bytes, _) = load_raw_f32_nofollow(
            &root.join(format!("captures/call-{:03}/hidden.f32le", summary.call)),
            HIDDEN,
        )?;
        let (logits_bytes, logits) = load_raw_f32_nofollow(
            &root.join(format!("captures/call-{:03}/logits.f32le", summary.call)),
            VOCAB,
        )?;
        let hidden_digest = sha256(&hidden_bytes);
        let logits_digest = sha256(&logits_bytes);
        let winner = total_cmp_winner(&logits)?;
        ensure!(
            evidence.call == summary.call
                && evidence.sequence_position == summary.sequence_position
                && evidence.consumed_input_token == summary.consumed_input_token
                && evidence.selected_output_token == summary.winner_id
                && evidence.winner_logit_bits == summary.winner_logit_bits
                && evidence.hidden.sha256 == hidden_digest
                && evidence.logits.sha256 == logits_digest
                && summary.hidden_sha256 == hidden_digest
                && summary.logits_sha256 == logits_digest
                && winner == summary.winner_id
                && format!("{:08x}", logits[winner].to_bits()) == summary.winner_logit_bits
                && analyzer.capture_identity == hidden_digest
                && analyzer.winner_id == winner,
            "raw capture, winner evidence, analyzer input, and manifest disagree"
        );
        if summary.call + 1 == manifest.request.generated_tokens {
            ensure!(
                evidence.generated_prefix_token_sha256 == manifest.request.generated_token_sha256,
                "terminal generated-prefix digest disagrees with request digest"
            );
        }
        ensure!(
            capture_keys
                .insert(
                    summary.call,
                    (hidden_digest, winner, summary.sequence_position)
                )
                .is_none(),
            "duplicate capture call"
        );
    }

    if !complete_analysis {
        let eos_call = manifest
            .request
            .coverage_eos_call
            .context("coverage packet lacks EOS call")?;
        let expected_reason = if CALLS.iter().all(|required| *required <= eos_call) {
            SkippedAnalysisReason::ProducerEosInsteadOfTokenLimit
        } else {
            SkippedAnalysisReason::ProducerEosBeforeRequiredCaptureCoverage
        };
        for name in [
            "metadata-manifest.json",
            "exact-validation.json",
            "screening-result.json",
            "byte-ledger.json",
        ] {
            let skipped: SkippedAnalysisArtifact = load_json_nofollow(&root.join(name))?;
            ensure!(
                skipped.schema == SCHEMA
                    && skipped.status == ArtifactStatus::NotRunCaptureCoverage
                    && skipped.reason == expected_reason,
                "coverage skipped-analysis artifact disagrees with termination"
            );
        }
        return Ok(manifest);
    }

    ensure!(manifest.request.termination == GenerationTermination::TokenLimit);
    let metadata: MetadataManifestArtifact =
        load_json_nofollow(&root.join("metadata-manifest.json"))?;
    let exact: ExactValidationArtifact = load_json_nofollow(&root.join("exact-validation.json"))?;
    let screening: ScreeningResultArtifact =
        load_json_nofollow(&root.join("screening-result.json"))?;
    let ledger: ByteLedgerArtifact = load_json_nofollow(&root.join("byte-ledger.json"))?;
    let (norm_bytes, _) =
        load_raw_f32_nofollow(&root.join("metadata/block-norm.f32le"), VOCAB * BLOCKS)?;
    let reported_exact_term_operations = exact
        .bigint_exact_dot_calls
        .checked_mul(HIDDEN as u64)
        .context("reported exact-term operation count overflow")?;
    ensure!(
        metadata.schema == SCHEMA
            && metadata.status == ArtifactStatus::Complete
            && metadata.name == "block_norm"
            && metadata.dtype == "F32"
            && metadata.shape == [VOCAB, BLOCKS]
            && metadata.bytes == norm_bytes.len()
            && metadata.sha256 == sha256(&norm_bytes)
            && metadata.source_complete_head_read_bytes == OUTPUT_BYTES
            && metadata.source_complete_head_sha256 == OUTPUT_SHA256
            && metadata.exhaustive_exact_containment
            && metadata.exact_validation_logical_read_bytes == OUTPUT_BYTES,
        "block-norm metadata and raw artifact disagree"
    );
    ensure!(
        exact.schema == SCHEMA
            && exact.status == ArtifactStatus::Complete
            && exact.dot_unit_exponent == DOT_EXP
            && exact.dot_resource_bits == 512
            && exact.required_high_water_bits_lt == 384
            && exact.norm_square_floor_exponent == -298
            && exact.fixed_dot_resource_bytes == 64
            && exact.bigint_term_operation_proxy == reported_exact_term_operations
            && exact.operation_proxy_note == "logical exact terms, not hardware operations"
            && screening.schema == SCHEMA
            && screening.status == ArtifactStatus::Complete
            && screening.competitors == VOCAB - 1
            && screening.pruning_floor == PRUNING_FLOOR
            && ledger.schema == SCHEMA
            && ledger.status == ArtifactStatus::Complete
            && ledger.competitors == VOCAB - 1
            && ledger.pruning_floor == PRUNING_FLOOR
            && ledger.full_head_denominator_bytes == OUTPUT_BYTES
            && ledger.charged_128_limit_bytes == BYTE_LIMIT
            && ledger.gate_alignment == 128
            && exact.records.len() == CALLS.len()
            && screening.captures.len() == CALLS.len()
            && ledger.ledgers.len() == CALLS.len(),
        "complete analyzer artifact constants changed"
    );
    let mut reconstructed_exact_dot_calls = 0u64;
    for ((screen, exact_record), ledger_record) in screening
        .captures
        .iter()
        .zip(&exact.records)
        .zip(&ledger.ledgers)
    {
        validate_screening_record(screen)?;
        validate_exact_record(exact_record)?;
        validate_ledger_record(ledger_record)?;
        let s = &screen.analysis;
        let e = &exact_record.analysis;
        let l = &ledger_record.analysis;
        let (identity, winner, position) = capture_keys
            .get(&screen.call)
            .context("analysis references an unknown capture call")?;
        let mut ids_bytes = Vec::with_capacity(e.survivor_ids.len() * 4);
        for id in &e.survivor_ids {
            ids_bytes.extend_from_slice(&id.to_le_bytes());
        }
        validate_survivors(&e.survivor_ids, VOCAB, *winner)?;
        ensure!(
            e.survivors == e.survivor_ids.len()
                && e.survivors == e.survivor_cmp.len()
                && e.survivor_cmp.iter().all(|value| *value <= 2)
        );
        let expected_q6_footprint = e
            .survivors
            .checked_add(1)
            .and_then(|rows| rows.checked_mul(ROW_BYTES))
            .context("unique Q6 footprint overflow")? as u64;
        ensure!(
            screen.call == exact_record.call
                && screen.call == ledger_record.call
                && screen.sequence_position == *position
                && exact_record.sequence_position == *position
                && ledger_record.sequence_position == *position
                && s.capture_identity == *identity
                && e.capture_identity == *identity
                && l.capture_identity == *identity
                && s.winner_id == *winner
                && e.winner_id == *winner
                && s.winner_exact_coefficient_hex == e.winner.coefficient_twos_complement_u64x8
                && s.winner_exact_exponent == e.winner.exponent
                && s.winner_lower_bits == e.winner.lower_f64_bits
                && s.winner_upper_bits == e.winner.upper_f64_bits
                && e.winner.signed_bits < 384
                && s.survivor_count == e.survivors
                && s.less_than_winner_count == e.less
                && s.tie_count == e.ties
                && s.greater_than_winner_count == e.greater
                && s.survivor_ids_sha256 == sha256(&ids_bytes)
                && s.survivor_cmp_sha256 == sha256(&e.survivor_cmp)
                && s.unique_q6_footprint_bytes == expected_q6_footprint
                && e.comparison_less == 0
                && e.comparison_equal == 1
                && e.comparison_greater == 2
                && l.full_head_denominator_bytes == OUTPUT_BYTES as u64
                && l.gate_limit_bytes == BYTE_LIMIT
                && l.winner_excluded
                && l.logical_survivors == e.survivors
                && l.reduction.competitor_count == s.competitor_count
                && l.reduction.bound_pruned_count == s.bound_pruned_count
                && l.reduction.survivor_count == s.survivor_count
                && l.reduction.less_than_winner_count == s.less_than_winner_count
                && l.reduction.tie_count == s.tie_count
                && l.reduction.greater_than_winner_count == s.greater_than_winner_count
                && l.reduction.winner_interval_lower_bits == s.winner_lower_bits
                && l.reduction.winner_interval_upper_bits == s.winner_upper_bits
                && s.competitor_count == VOCAB - 1
                && s.bound_pruned_count.checked_add(s.survivor_count) == Some(s.competitor_count)
                && s.less_than_winner_count
                    .checked_add(s.tie_count)
                    .and_then(|count| count.checked_add(s.greater_than_winner_count))
                    == Some(s.survivor_count)
                && s.unique_ideal_winner == (s.tie_count == 0 && s.greater_than_winner_count == 0)
                && s.pruning_pass == (s.bound_pruned_count >= PRUNING_FLOOR),
            "screening, exact, ledger, and capture records disagree"
        );
        let expected_alignments = [64, 128, 256]
            .map(|alignment| ledger_for(&e.survivor_ids, *winner, alignment))
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        ensure!(l.alignments.as_slice() == expected_alignments.as_slice());
        for alignment in &l.alignments {
            ensure!(
                alignment.total_charged_bytes
                    == alignment
                        .streams
                        .iter()
                        .map(|stream| stream.charged_bytes)
                        .sum::<u64>(),
                "ledger charged total is not its stream sum"
            );
        }
        ensure!(
            s.charged_64_bytes == l.alignments[0].total_charged_bytes
                && s.charged_128_bytes == l.alignments[1].total_charged_bytes
                && s.charged_256_bytes == l.alignments[2].total_charged_bytes
                && s.bytes_pass == (s.charged_128_bytes <= BYTE_LIMIT),
            "screening charged totals disagree with gated ledger"
        );
        reconstructed_exact_dot_calls = reconstructed_exact_dot_calls
            .checked_add(1 + e.survivors as u64)
            .context("reconstructed exact-dot call count overflow")?;
    }
    ensure!(exact.bigint_exact_dot_calls == reconstructed_exact_dot_calls);
    Ok(manifest)
}

fn coverage_expected_artifacts(capture_calls: &[usize]) -> BTreeSet<PathBuf> {
    let mut expected = BTreeSet::from([
        PathBuf::from("readiness.json"),
        PathBuf::from("self-tests.json"),
        PathBuf::from("capture-manifest.json"),
        PathBuf::from("metadata-manifest.json"),
        PathBuf::from("exact-validation.json"),
        PathBuf::from("screening-result.json"),
        PathBuf::from("byte-ledger.json"),
    ]);
    for call in capture_calls {
        expected.extend([
            PathBuf::from(format!("captures/call-{call:03}/hidden.f32le")),
            PathBuf::from(format!("captures/call-{call:03}/logits.f32le")),
            PathBuf::from(format!("winner-evidence/call-{call:03}.json")),
            PathBuf::from(format!("analyzer-input/call-{call:03}.json")),
        ]);
    }
    expected
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ArtifactManifest {
    schema: String,
    files: Vec<InventoryEntry>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
enum Disposition {
    #[serde(rename = "INVALID")]
    Invalid,
    #[serde(rename = "KILL_CAPTURE_COVERAGE")]
    KillCaptureCoverage,
    #[serde(rename = "KILL_IDEAL_MISMATCH")]
    KillIdealMismatch,
    #[serde(rename = "KILL_PRUNING_AND_BYTES")]
    KillPruningAndBytes,
    #[serde(rename = "KILL_PRUNING")]
    KillPruning,
    #[serde(rename = "KILL_BYTES")]
    KillBytes,
    #[serde(rename = "GO_OPTIMISTIC_A3B_FIXTURE")]
    GoOptimisticA3bFixture,
}

impl Disposition {
    fn label(self) -> &'static str {
        match self {
            Self::Invalid => "INVALID",
            Self::KillCaptureCoverage => "KILL_CAPTURE_COVERAGE",
            Self::KillIdealMismatch => "KILL_IDEAL_MISMATCH",
            Self::KillPruningAndBytes => "KILL_PRUNING_AND_BYTES",
            Self::KillPruning => "KILL_PRUNING",
            Self::KillBytes => "KILL_BYTES",
            Self::GoOptimisticA3bFixture => "GO_OPTIMISTIC_A3B_FIXTURE",
        }
    }
}

fn mechanism_disposition(
    any_mismatch: bool,
    any_pruning_failure: bool,
    any_byte_failure: bool,
) -> Disposition {
    if any_mismatch {
        Disposition::KillIdealMismatch
    } else if any_pruning_failure && any_byte_failure {
        Disposition::KillPruningAndBytes
    } else if any_pruning_failure {
        Disposition::KillPruning
    } else if any_byte_failure {
        Disposition::KillBytes
    } else {
        Disposition::GoOptimisticA3bFixture
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DecisionArtifact {
    schema: String,
    disposition: Disposition,
    authority: String,
    artifact_manifest_sha256: String,
    any_pruning_failure: bool,
    any_byte_failure: bool,
    any_ideal_mismatch: bool,
    coverage_eos_call: Option<usize>,
    failure: Option<String>,
}

#[derive(Debug)]
struct TerminalOutcome {
    disposition: Disposition,
    any_pruning_failure: bool,
    any_byte_failure: bool,
    any_ideal_mismatch: bool,
    coverage_eos_call: Option<usize>,
    failure: Option<String>,
    require_complete_inventory: bool,
}

fn seal(packet: &mut Packet, outcome: &TerminalOutcome) -> Result<()> {
    ensure!(!packet.terminalized, "packet already terminalized");
    if let Some(expected) = &packet.executable_commitment {
        ensure!(
            executable_identity()? == *expected,
            "executable identity changed before seal"
        );
    }
    let mut paths = Vec::new();
    walk_packet(&packet.root, Path::new(""), &mut paths)?;
    paths.sort();
    let actual: BTreeSet<_> = paths.iter().cloned().collect();
    validate_inventory_set(&actual, &packet.written)?;
    ensure!(
        packet.readiness_published() && actual.contains(Path::new("readiness.json")),
        "terminal packet lacks authenticated readiness"
    );
    let readiness = validate_readiness_and_capture_binding(
        &packet.root,
        actual.contains(Path::new("capture-manifest.json")),
    )?;
    if let Some(expected) = &packet.executable_commitment {
        ensure!(
            readiness.executable_identity == *expected,
            "readiness differs from executable commitment"
        );
    }
    if outcome.require_complete_inventory {
        validate_inventory_set(&actual, &complete_expected_artifacts())?;
    }
    let mut inventory = Vec::with_capacity(paths.len());
    for relative in &paths {
        strict_reparse_json(&packet.root.join(relative), relative)?;
        let (stamp, digest) = hash_nofollow(&packet.root.join(relative))?;
        ensure!(
            packet.commitments.get(relative) == Some(&(stamp.clone(), digest.clone())),
            "artifact identity changed after exclusive publication"
        );
        validate_raw_artifact_size(relative, stamp.bytes)?;
        inventory.push(InventoryEntry {
            path: relative
                .to_str()
                .context("non-UTF8 artifact path")?
                .to_string(),
            bytes: stamp.bytes,
            sha256: digest,
            stamp,
        });
    }
    if outcome.require_complete_inventory {
        semantic_packet_cross_check(&packet.root, true)?;
    } else if outcome.disposition == Disposition::KillCaptureCoverage {
        let capture_manifest = semantic_packet_cross_check(&packet.root, false)?;
        let calls = capture_manifest
            .captures
            .iter()
            .map(|capture| capture.call)
            .collect::<Vec<_>>();
        validate_inventory_set(&actual, &coverage_expected_artifacts(&calls))?;
    }
    packet.json_strict(
        "artifact-manifest.json",
        &ArtifactManifest {
            schema: SCHEMA.to_string(),
            files: inventory.clone(),
        },
    )?;
    let (_, manifest_digest) = hash_nofollow(&packet.root.join("artifact-manifest.json"))?;
    strict_reparse_json(
        &packet.root.join("artifact-manifest.json"),
        Path::new("artifact-manifest.json"),
    )?;
    for entry in &inventory {
        let (stamp, digest) = hash_nofollow(&packet.root.join(&entry.path))?;
        ensure!(stamp == entry.stamp && digest == entry.sha256);
    }
    let (manifest_stamp, second_manifest_digest) =
        hash_nofollow(&packet.root.join("artifact-manifest.json"))?;
    ensure!(manifest_digest == second_manifest_digest);
    let mut final_paths = Vec::new();
    walk_packet(&packet.root, Path::new(""), &mut final_paths)?;
    let mut final_expected = actual;
    final_expected.insert(PathBuf::from("artifact-manifest.json"));
    ensure!(
        final_paths.into_iter().collect::<BTreeSet<_>>() == final_expected,
        "final packet rewalk differs before decision publication"
    );
    let (final_manifest_stamp, final_manifest_digest) =
        hash_nofollow(&packet.root.join("artifact-manifest.json"))?;
    ensure!(manifest_stamp == final_manifest_stamp && manifest_digest == final_manifest_digest);
    let decision = DecisionArtifact {
        schema: SCHEMA.to_string(),
        disposition: outcome.disposition,
        authority: if outcome.disposition == Disposition::GoOptimisticA3bFixture {
            "one_separately_preregistered_dense_27b_guard_only".to_string()
        } else {
            "none".to_string()
        },
        artifact_manifest_sha256: manifest_digest,
        any_pruning_failure: outcome.any_pruning_failure,
        any_byte_failure: outcome.any_byte_failure,
        any_ideal_mismatch: outcome.any_ideal_mismatch,
        coverage_eos_call: outcome.coverage_eos_call,
        failure: outcome.failure.clone(),
    };
    validate_sha256(
        &decision.artifact_manifest_sha256,
        "decision artifact-manifest SHA-256",
    )?;
    let serialized = serde_json::to_vec(&decision)?;
    let parsed: DecisionArtifact = serde_json::from_slice(&serialized)?;
    ensure!(serde_json::to_vec(&parsed)? == serialized);
    let mut bytes = serialized;
    bytes.push(b'\n');
    publish_decision_exclusive(packet, &bytes)?;
    packet.terminalized = true;
    Ok(())
}

fn publish_decision_exclusive(packet: &Packet, bytes: &[u8]) -> Result<()> {
    packet.validate_root()?;
    let temporary = packet.root.join(".decision.json.tmp");
    let terminal = packet.root.join("decision.json");
    ensure!(
        std::fs::symlink_metadata(&terminal)
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound),
        "terminal decision already exists"
    );
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    let old = CString::new(temporary.as_os_str().as_bytes())?;
    let new = CString::new(terminal.as_os_str().as_bytes())?;
    let renamed = unsafe {
        libc::renameatx_np(
            libc::AT_FDCWD,
            old.as_ptr(),
            libc::AT_FDCWD,
            new.as_ptr(),
            libc::RENAME_EXCL,
        )
    };
    if renamed != 0 {
        let error = std::io::Error::last_os_error();
        let _ = std::fs::remove_file(&temporary);
        return Err(error.into());
    }
    packet.sync_root()?;
    let metadata = std::fs::symlink_metadata(&terminal)?;
    ensure!(metadata.file_type().is_file() && metadata.nlink() == 1);
    let mut published = Vec::new();
    open_nofollow_regular(&terminal)?.read_to_end(&mut published)?;
    ensure!(published.last() == Some(&b'\n'));
    let parsed: DecisionArtifact = serde_json::from_slice(&published[..published.len() - 1])?;
    validate_sha256(
        &parsed.artifact_manifest_sha256,
        "published decision artifact-manifest SHA-256",
    )?;
    let mut canonical = serde_json::to_vec(&parsed)?;
    canonical.push(b'\n');
    ensure!(
        canonical == published,
        "published decision failed typed canonical revalidation"
    );
    let _ = hash_nofollow(&terminal)?;
    Ok(())
}

fn sha256_file_handle(file: &File) -> Result<String> {
    let mut reader = BufReader::with_capacity(8 << 20, file.try_clone()?);
    reader.seek(SeekFrom::Start(0))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn authenticate_raw_model(path: &Path) -> Result<AuthenticatedModel> {
    let total_start = Instant::now();
    ensure!(path == Path::new(MODEL_PATH));
    check_absolute_nofollow_parents(path)?;
    let path_metadata = std::fs::symlink_metadata(path)?;
    ensure!(path_metadata.file_type().is_file() && !path_metadata.file_type().is_symlink());
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    let stamp = FileStamp::from_metadata(&file.metadata()?);
    ensure!(stamp == FileStamp::from_metadata(&path_metadata));
    ensure!(stamp.bytes == MODEL_BYTES);
    let hash_start = Instant::now();
    ensure!(sha256_file_handle(&file)? == MODEL_SHA256);
    let complete_file_hash_wall_nanoseconds =
        u64::try_from(hash_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    ensure!(
        FileStamp::from_metadata(&file.metadata()?) == stamp,
        "raw model identity changed during complete hash"
    );
    let runtime_gguf_start = Instant::now();
    let gguf = qwen_llm::gguf::GgufFile::from_opened_file(file.try_clone()?, path)?;
    let identity = validate_raw_gguf(&gguf, &stamp)?;
    let runtime_gguf_open_profile_and_output_hash_wall_nanoseconds =
        u64::try_from(runtime_gguf_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let analysis_gguf_start = Instant::now();
    let analysis_gguf = qwen_llm::gguf::GgufFile::from_opened_file(file.try_clone()?, path)?;
    ensure!(validate_raw_gguf(&analysis_gguf, &stamp)? == identity);
    let analysis_gguf_open_profile_and_output_hash_wall_nanoseconds =
        u64::try_from(analysis_gguf_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    ensure!(
        FileStamp::from_metadata(&file.metadata()?) == stamp,
        "raw model identity changed during descriptor authentication"
    );
    Ok(AuthenticatedModel {
        file,
        gguf,
        analysis_gguf,
        stamp,
        identity,
        authentication_telemetry: ModelAuthenticationTelemetry {
            total_wall_nanoseconds: u64::try_from(total_start.elapsed().as_nanos())
                .unwrap_or(u64::MAX),
            complete_file_hash_wall_nanoseconds,
            runtime_gguf_open_profile_and_output_hash_wall_nanoseconds,
            analysis_gguf_open_profile_and_output_hash_wall_nanoseconds,
        },
    })
}

fn validate_analysis_output<'a>(
    gguf: &'a qwen_llm::gguf::GgufFile,
    expected_stamp: &FileStamp,
) -> Result<&'a [u8]> {
    ensure!(gguf.shard_count() == 1 && gguf.total_mapped_len() as u64 == MODEL_BYTES);
    ensure!(stamp_retained_gguf(gguf)? == *expected_stamp);
    let model = Model::from_gguf(gguf)?;
    ensure!(model.arch == expected_arch() && !model.tied_embeddings && model.mtp.is_none());
    let desc = model.lm_head;
    ensure!(
        desc.name == "output.weight"
            && desc.dtype == GgmlType::Q6_K
            && desc.shape.as_slice() == [HIDDEN as u64, VOCAB as u64]
            && desc.data_offset == OUTPUT_OFFSET
            && desc.n_bytes as usize == OUTPUT_BYTES
    );
    let output = gguf.try_slice(desc)?;
    ensure!(sha256(output) == OUTPUT_SHA256);
    Ok(output)
}

fn capture_manifest_payload(
    build_identity: &super::BuildIdentity,
    executable: &ExecutableIdentity,
    model: &ModelIdentityRecord,
    runtime_identity: &OpenedGgufRuntimeIdentity,
    prompt: &str,
    generation: &GenerationOutcome,
    host_before: &HostEvidence,
    host_after: &HostEvidence,
    model_authentication_telemetry: &ModelAuthenticationTelemetry,
) -> CaptureManifestArtifact {
    let captures = generation
        .captures
        .iter()
        .map(|capture| {
            let hidden = f32_le_bytes(&capture.hidden);
            let logits = f32_le_bytes(&capture.logits);
            CaptureSummaryRecord {
                call: capture.call,
                sequence_position: capture.sequence_position,
                consumed_input_token: capture.consumed_input_token,
                winner_id: capture.winner,
                winner_logit_bits: format!("{:08x}", capture.winner_logit_bits),
                hidden_sha256: sha256(hidden),
                logits_sha256: sha256(logits),
            }
        })
        .collect::<Vec<_>>();
    CaptureManifestArtifact {
        schema: SCHEMA.to_string(),
        status: ArtifactStatus::CompleteGeneration,
        build_identity: build_identity.clone(),
        executable_identity: executable.clone(),
        qwen_env: ZeroQwenControls {},
        user_gpu_attestation: true,
        model: model.clone(),
        runtime_identity: runtime_identity.clone(),
        model_authentication_telemetry: model_authentication_telemetry.clone(),
        prompt: PromptRecord {
            path: PROMPT_PATH.to_string(),
            bytes: prompt.len(),
            sha256: PROMPT_SHA256.to_string(),
            tokens: PROMPT_TOKENS,
            token_sha256: TOKEN_SHA256.to_string(),
        },
        request: RequestRecord {
            temperature_bits: "00000000".to_string(),
            token_limit: TOKENS,
            stop_ids: model.stop_token_ids.clone(),
            prefill_chunk: 1024,
            max_context: 1024,
            capture_calls: CALLS.to_vec(),
            generated_tokens: generation.generated.len(),
            generated_token_sha256: generated_token_digest(&generation.generated),
            transitions: generation.transitions,
            termination: if generation.coverage_eos_call.is_some() {
                GenerationTermination::ProducerEos
            } else {
                GenerationTermination::TokenLimit
            },
            coverage_eos_call: generation.coverage_eos_call,
            call_order: ConditionalCallOrder {
                unconditional_prefix: [CallStep::Select, CallStep::Append, CallStep::StopCheck],
                producer_eos_frozen_capture_suffix: [
                    CallStep::ObservationalCapture,
                    CallStep::Branch,
                ],
                producer_eos_noncapture_suffix: [CallStep::Branch],
                non_stop_suffix: [
                    CallStep::Callback,
                    CallStep::TokenLimitCheck,
                    CallStep::ObservationalCapture,
                    CallStep::Branch,
                ],
            },
        },
        capture_telemetry: CaptureTelemetry {
            hidden_copy_logical_bytes: generation.captures.len() * HIDDEN * 4,
            logits_copy_logical_bytes: generation.captures.len() * VOCAB * 4,
            hidden_copy_wall_nanoseconds: generation
                .captures
                .iter()
                .map(|capture| capture.hidden_copy_wall_nanoseconds)
                .sum(),
            logits_copy_wall_nanoseconds: generation
                .captures
                .iter()
                .map(|capture| capture.logits_copy_wall_nanoseconds)
                .sum(),
            winner_reconstruction_wall_nanoseconds: generation
                .captures
                .iter()
                .map(|capture| capture.winner_reconstruction_wall_nanoseconds)
                .sum(),
            generation_wall_nanoseconds: generation.generation_wall_nanoseconds,
            retained_capture_vec_requested_capacity_bytes: generation.captures.capacity()
                * std::mem::size_of::<Capture>(),
            retained_hidden_vec_requested_capacity_bytes: generation
                .captures
                .iter()
                .map(|capture| capture.hidden.capacity() * 4)
                .sum(),
            retained_logits_vec_requested_capacity_bytes: generation
                .captures
                .iter()
                .map(|capture| capture.logits.capacity() * 4)
                .sum(),
            retained_generated_vec_requested_capacity_bytes: generation.generated.capacity() * 4,
            capacity_note: "requested Vec capacities only; excludes allocator overhead".to_string(),
        },
        captures,
        host_before: host_before.clone(),
        host_after: host_after.clone(),
    }
}

struct AnalyzerResult {
    block_norms: Vec<f32>,
    norm_exact_high_water_bits: u64,
    norm_population_wall_nanoseconds: u64,
    norm_exact_validation_wall_nanoseconds: u64,
    screening_and_exact_survivor_wall_nanoseconds: u64,
    screening: Vec<CaptureScreening>,
    exact_records: Vec<ExactCaptureRecord>,
    ledgers: Vec<CaptureLedgerRecord>,
    exact_dot_calls: u64,
    exact_dot_term_operations: u64,
    allocation_telemetry: AnalyzerAllocationTelemetry,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MetadataManifestArtifact {
    schema: String,
    status: ArtifactStatus,
    name: String,
    dtype: String,
    shape: [usize; 2],
    bytes: usize,
    sha256: String,
    source_complete_head_read_bytes: usize,
    source_complete_head_sha256: String,
    population_wall_nanoseconds: u64,
    exhaustive_exact_containment: bool,
    exact_validation_logical_read_bytes: usize,
    exact_validation_wall_nanoseconds: u64,
    largest_observed_bigint_coefficient_payload_bits: u64,
    bigint_payload_note: String,
    analysis_retained_stamp: FileStamp,
    analysis_retained_output_reauthentication_wall_nanoseconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExactValidationArtifact {
    schema: String,
    status: ArtifactStatus,
    dot_unit_exponent: i32,
    dot_resource_bits: u32,
    required_high_water_bits_lt: u32,
    norm_square_floor_exponent: i32,
    fixed_dot_resource_bytes: usize,
    bigint_exact_dot_calls: u64,
    bigint_term_operation_proxy: u64,
    operation_proxy_note: String,
    norm_validation_wall_nanoseconds: u64,
    screening_and_exact_survivor_wall_nanoseconds: u64,
    allocation_telemetry: AnalyzerAllocationTelemetry,
    records: Vec<PersistedExactCaptureRecord>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AnalyzerAllocationTelemetry {
    analyzer_capture_vec_requested_capacity_bytes: usize,
    analyzer_hidden_vecs_requested_capacity_bytes: usize,
    block_norm_vec_requested_capacity_bytes: usize,
    upper_vec_requested_capacity_bytes_per_capture_high_water: usize,
    active_vec_requested_capacity_bytes_per_capture_high_water: usize,
    survivor_ids_vec_requested_capacity_bytes_per_capture_high_water: usize,
    survivor_cmp_vec_requested_capacity_bytes_per_capture_high_water: usize,
    survivor_ids_digest_vec_requested_capacity_bytes_per_capture_high_water: usize,
    hidden_norm_stack_working_set_bytes_per_capture: usize,
    bound_terms_stack_working_set_bytes_per_row: usize,
    norm_terms_stack_working_set_bytes_per_block: usize,
    screening_vec_requested_capacity_bytes: usize,
    exact_records_vec_requested_capacity_bytes: usize,
    ledgers_vec_requested_capacity_bytes: usize,
    retained_survivor_ids_vec_requested_capacity_bytes: usize,
    retained_survivor_cmp_vec_requested_capacity_bytes: usize,
    retained_ledger_stream_vec_requested_capacity_bytes: usize,
    persistent_before_capture_requested_capacity_high_water_bytes: usize,
    current_capture_transient_requested_capacity_high_water_bytes: usize,
    persistent_after_capture_requested_capacity_high_water_bytes: usize,
    owned_vec_requested_capacity_high_water_bytes: usize,
    largest_observed_bigint_coefficient_payload_bits: u64,
    largest_observed_bigint_coefficient_payload_rounded_limb_bytes: u64,
    note: String,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct OwnedVecPhaseTracker {
    persistent_bytes: usize,
    persistent_before_high_water_bytes: usize,
    transient_high_water_bytes: usize,
    persistent_after_high_water_bytes: usize,
    phase_high_water_bytes: usize,
}

impl OwnedVecPhaseTracker {
    fn new(initial_persistent_bytes: usize) -> Self {
        Self {
            persistent_bytes: initial_persistent_bytes,
            persistent_before_high_water_bytes: initial_persistent_bytes,
            persistent_after_high_water_bytes: initial_persistent_bytes,
            phase_high_water_bytes: initial_persistent_bytes,
            ..Self::default()
        }
    }

    fn observe_capture(&mut self, transient_bytes: usize, retained_bytes: usize) -> Result<()> {
        self.persistent_before_high_water_bytes = self
            .persistent_before_high_water_bytes
            .max(self.persistent_bytes);
        self.transient_high_water_bytes = self.transient_high_water_bytes.max(transient_bytes);
        self.phase_high_water_bytes = self.phase_high_water_bytes.max(
            self.persistent_bytes
                .checked_add(transient_bytes)
                .context("owned Vec transient checkpoint overflow")?,
        );
        self.persistent_bytes = self
            .persistent_bytes
            .checked_add(retained_bytes)
            .context("owned Vec persistent checkpoint overflow")?;
        self.persistent_after_high_water_bytes = self
            .persistent_after_high_water_bytes
            .max(self.persistent_bytes);
        self.phase_high_water_bytes = self.phase_high_water_bytes.max(self.persistent_bytes);
        Ok(())
    }
}

fn ledger_stream_vec_requested_capacity_bytes(ledger: &CaptureLedgerRecord) -> usize {
    ledger
        .alignments
        .iter()
        .map(|alignment| alignment.streams.capacity() * std::mem::size_of::<StreamCharge>())
        .sum()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ScreeningResultArtifact {
    schema: String,
    status: ArtifactStatus,
    competitors: usize,
    pruning_floor: usize,
    captures: Vec<PersistedCaptureScreening>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ByteLedgerArtifact {
    schema: String,
    status: ArtifactStatus,
    competitors: usize,
    pruning_floor: usize,
    full_head_denominator_bytes: usize,
    charged_128_limit_bytes: u64,
    gate_alignment: u64,
    ledgers: Vec<PersistedCaptureLedgerRecord>,
}

fn run_analyzer(output_weight: &[u8], captures: Vec<AnalyzerCapture>) -> Result<AnalyzerResult> {
    ensure!(captures.len() == CALLS.len());
    ensure!(
        captures
            .iter()
            .map(|capture| capture.capture_identity.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            == captures.len(),
        "duplicate analyzer capture identity"
    );
    let norm_population_start = Instant::now();
    let block_norms = populate_block_norms(output_weight)?;
    let norm_population_wall_nanoseconds =
        u64::try_from(norm_population_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let norm_exact_validation_start = Instant::now();
    let norm_exact_high_water_bits = validate_block_norms_exact(output_weight, &block_norms)?;
    let norm_exact_validation_wall_nanoseconds =
        u64::try_from(norm_exact_validation_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let analyzer = AnalyzerInput {
        geometry: Geometry {
            hidden: HIDDEN,
            vocab: VOCAB,
            blocks: BLOCKS,
            row_bytes: ROW_BYTES,
        },
        captures,
        block_norms,
        output_weight,
        ledger: LedgerConstants {
            full_head: OUTPUT_BYTES as u64,
            byte_limit: BYTE_LIMIT,
            pruning_floor: PRUNING_FLOOR,
        },
    };
    let analyzer_capture_vec_requested_capacity_bytes =
        analyzer.captures.capacity() * std::mem::size_of::<AnalyzerCapture>();
    let analyzer_hidden_vecs_requested_capacity_bytes = analyzer
        .captures
        .iter()
        .map(|capture| capture.hidden.capacity() * std::mem::size_of::<f32>())
        .sum::<usize>();
    let block_norm_vec_requested_capacity_bytes =
        analyzer.block_norms.capacity() * std::mem::size_of::<f32>();
    ensure!(analyzer_capabilities(&analyzer).len() == 7);
    ensure!(
        analyzer.geometry.hidden == HIDDEN
            && analyzer.geometry.vocab == VOCAB
            && analyzer.geometry.blocks == BLOCKS
            && analyzer.geometry.row_bytes == ROW_BYTES
    );
    let mut screening = Vec::with_capacity(CALLS.len());
    let mut exact_records = Vec::with_capacity(CALLS.len());
    let mut ledgers = Vec::with_capacity(CALLS.len());
    let screening_vec_requested_capacity_bytes =
        screening.capacity() * std::mem::size_of::<CaptureScreening>();
    let exact_records_vec_requested_capacity_bytes =
        exact_records.capacity() * std::mem::size_of::<ExactCaptureRecord>();
    let ledgers_vec_requested_capacity_bytes =
        ledgers.capacity() * std::mem::size_of::<CaptureLedgerRecord>();
    let initial_persistent_vec_bytes = analyzer_capture_vec_requested_capacity_bytes
        + analyzer_hidden_vecs_requested_capacity_bytes
        + block_norm_vec_requested_capacity_bytes
        + screening_vec_requested_capacity_bytes
        + exact_records_vec_requested_capacity_bytes
        + ledgers_vec_requested_capacity_bytes;
    let mut vec_phase = OwnedVecPhaseTracker::new(initial_persistent_vec_bytes);
    let mut retained_survivor_ids_bytes = 0usize;
    let mut retained_survivor_cmp_bytes = 0usize;
    let mut retained_ledger_stream_bytes = 0usize;
    let mut bigint_tracker = BigIntPayloadTracker::default();
    bigint_tracker.largest_observed_coefficient_payload_bits = norm_exact_high_water_bits;
    let mut exact_dot_calls = 0u64;
    let screening_start = Instant::now();
    for capture in &analyzer.captures {
        let (result, exact, ledger) = analyze_capture(&analyzer, capture)?;
        let survivor_ids_bytes = exact.survivor_ids.capacity() * std::mem::size_of::<u32>();
        let survivor_cmp_bytes = exact.survivor_cmp.capacity() * std::mem::size_of::<u8>();
        let ledger_stream_bytes = ledger_stream_vec_requested_capacity_bytes(&ledger);
        let transient_bytes = result
            .upper_vec_requested_capacity_bytes
            .checked_add(result.active_vec_requested_capacity_bytes)
            .and_then(|bytes| bytes.checked_add(survivor_ids_bytes))
            .and_then(|bytes| bytes.checked_add(survivor_cmp_bytes))
            .and_then(|bytes| {
                bytes.checked_add(result.survivor_ids_digest_vec_requested_capacity_bytes)
            })
            .and_then(|bytes| bytes.checked_add(ledger_stream_bytes))
            .context("current capture transient Vec capacity overflow")?;
        let retained_bytes = survivor_ids_bytes
            .checked_add(survivor_cmp_bytes)
            .and_then(|bytes| bytes.checked_add(ledger_stream_bytes))
            .context("current capture retained Vec capacity overflow")?;
        vec_phase.observe_capture(transient_bytes, retained_bytes)?;
        retained_survivor_ids_bytes += survivor_ids_bytes;
        retained_survivor_cmp_bytes += survivor_cmp_bytes;
        retained_ledger_stream_bytes += ledger_stream_bytes;
        bigint_tracker.largest_observed_coefficient_payload_bits = bigint_tracker
            .largest_observed_coefficient_payload_bits
            .max(result.largest_observed_bigint_coefficient_payload_bits);
        exact_dot_calls = exact_dot_calls
            .checked_add(1 + result.survivor_count as u64)
            .context("exact dot operation count overflow")?;
        screening.push(result);
        exact_records.push(exact);
        ledgers.push(ledger);
    }
    let exact_dot_term_operations = exact_dot_calls
        .checked_mul(HIDDEN as u64)
        .context("exact dot term operation count overflow")?;
    let bigint_bits = bigint_tracker.largest_observed_coefficient_payload_bits;
    let allocation_telemetry = AnalyzerAllocationTelemetry {
        analyzer_capture_vec_requested_capacity_bytes,
        analyzer_hidden_vecs_requested_capacity_bytes,
        block_norm_vec_requested_capacity_bytes,
        upper_vec_requested_capacity_bytes_per_capture_high_water: screening
            .iter()
            .map(|record| record.upper_vec_requested_capacity_bytes)
            .max()
            .unwrap_or(0),
        active_vec_requested_capacity_bytes_per_capture_high_water: screening
            .iter()
            .map(|record| record.active_vec_requested_capacity_bytes)
            .max()
            .unwrap_or(0),
        survivor_ids_vec_requested_capacity_bytes_per_capture_high_water: screening
            .iter()
            .map(|record| record.survivor_ids_vec_requested_capacity_bytes)
            .max()
            .unwrap_or(0),
        survivor_cmp_vec_requested_capacity_bytes_per_capture_high_water: screening
            .iter()
            .map(|record| record.survivor_cmp_vec_requested_capacity_bytes)
            .max()
            .unwrap_or(0),
        survivor_ids_digest_vec_requested_capacity_bytes_per_capture_high_water: screening
            .iter()
            .map(|record| record.survivor_ids_digest_vec_requested_capacity_bytes)
            .max()
            .unwrap_or(0),
        hidden_norm_stack_working_set_bytes_per_capture: BLOCKS * 4,
        bound_terms_stack_working_set_bytes_per_row: BLOCKS * 8,
        norm_terms_stack_working_set_bytes_per_block: BLOCK * 8,
        screening_vec_requested_capacity_bytes,
        exact_records_vec_requested_capacity_bytes,
        ledgers_vec_requested_capacity_bytes,
        retained_survivor_ids_vec_requested_capacity_bytes: retained_survivor_ids_bytes,
        retained_survivor_cmp_vec_requested_capacity_bytes: retained_survivor_cmp_bytes,
        retained_ledger_stream_vec_requested_capacity_bytes: retained_ledger_stream_bytes,
        persistent_before_capture_requested_capacity_high_water_bytes: vec_phase
            .persistent_before_high_water_bytes,
        current_capture_transient_requested_capacity_high_water_bytes: vec_phase
            .transient_high_water_bytes,
        persistent_after_capture_requested_capacity_high_water_bytes: vec_phase
            .persistent_after_high_water_bytes,
        owned_vec_requested_capacity_high_water_bytes: vec_phase.phase_high_water_bytes,
        largest_observed_bigint_coefficient_payload_bits: bigint_bits,
        largest_observed_bigint_coefficient_payload_rounded_limb_bytes: bigint_bits.div_ceil(64) * 8,
        note: "phase-wide requested Vec payload capacity high-water and largest observed BigInt coefficient payload; excludes Strings, allocator metadata, spare BigInt capacity, and allocator overhead".to_string(),
    };
    Ok(AnalyzerResult {
        block_norms: analyzer.block_norms,
        norm_exact_high_water_bits,
        norm_population_wall_nanoseconds,
        norm_exact_validation_wall_nanoseconds,
        screening_and_exact_survivor_wall_nanoseconds: u64::try_from(
            screening_start.elapsed().as_nanos(),
        )
        .unwrap_or(u64::MAX),
        screening,
        exact_records,
        ledgers,
        exact_dot_calls,
        exact_dot_term_operations,
        allocation_telemetry,
    })
}

fn join_analyzer_records(
    result: &AnalyzerResult,
    labels: &[CaptureLabel],
) -> Result<JoinedAnalyzerRecords> {
    let screening_ids = result
        .screening
        .iter()
        .map(|record| record.capture_identity.as_str())
        .collect::<Vec<_>>();
    let exact_ids = result
        .exact_records
        .iter()
        .map(|record| record.capture_identity.as_str())
        .collect::<Vec<_>>();
    let ledger_ids = result
        .ledgers
        .iter()
        .map(|record| record.capture_identity.as_str())
        .collect::<Vec<_>>();
    validate_analyzer_identity_join(labels, &screening_ids, &exact_ids, &ledger_ids)?;

    Ok(JoinedAnalyzerRecords {
        screening: labels
            .iter()
            .zip(&result.screening)
            .map(|(label, analysis)| PersistedCaptureScreening {
                call: label.call,
                sequence_position: label.sequence_position,
                analysis: analysis.clone(),
            })
            .collect(),
        exact_records: labels
            .iter()
            .zip(&result.exact_records)
            .map(|(label, analysis)| PersistedExactCaptureRecord {
                call: label.call,
                sequence_position: label.sequence_position,
                analysis: analysis.clone(),
            })
            .collect(),
        ledgers: labels
            .iter()
            .zip(&result.ledgers)
            .map(|(label, analysis)| PersistedCaptureLedgerRecord {
                call: label.call,
                sequence_position: label.sequence_position,
                analysis: analysis.clone(),
            })
            .collect(),
    })
}

fn validate_analyzer_identity_join(
    labels: &[CaptureLabel],
    screening_ids: &[&str],
    exact_ids: &[&str],
    ledger_ids: &[&str],
) -> Result<()> {
    ensure!(
        labels.len() == CALLS.len(),
        "missing controller capture label"
    );
    ensure!(
        labels
            .iter()
            .map(|label| label.capture_identity.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            == labels.len(),
        "duplicate controller capture identity"
    );
    ensure!(
        labels
            .iter()
            .zip(CALLS)
            .all(|(label, call)| label.call == call
                && label.sequence_position == PROMPT_TOKENS + call),
        "controller labels are not in frozen call order"
    );
    let expected = labels
        .iter()
        .map(|label| label.capture_identity.as_str())
        .collect::<Vec<_>>();
    ensure!(
        screening_ids == expected.as_slice(),
        "screening identities missing, unknown, or reordered"
    );
    ensure!(
        exact_ids == expected.as_slice(),
        "exact identities missing, unknown, or reordered"
    );
    ensure!(
        ledger_ids == expected.as_slice(),
        "ledger identities missing, unknown, or reordered"
    );
    Ok(())
}

fn persist_analyzer_result(
    packet: &mut Packet,
    result: &AnalyzerResult,
    joined: &JoinedAnalyzerRecords,
    analysis_retained_output_reauthentication_wall_nanoseconds: u64,
    analysis_retained_stamp: &FileStamp,
) -> Result<()> {
    let norm_bytes = f32_le_bytes(&result.block_norms);
    packet.write("metadata/block-norm.f32le", norm_bytes)?;
    packet.json_strict(
        "metadata-manifest.json",
        &MetadataManifestArtifact {
            schema: SCHEMA.to_string(),
            status: ArtifactStatus::Complete,
            name: "block_norm".to_string(),
            dtype: "F32".to_string(),
            shape: [VOCAB, BLOCKS],
            bytes: norm_bytes.len(),
            sha256: sha256(norm_bytes),
            source_complete_head_read_bytes: OUTPUT_BYTES,
            source_complete_head_sha256: OUTPUT_SHA256.to_string(),
            population_wall_nanoseconds: result.norm_population_wall_nanoseconds,
            exhaustive_exact_containment: true,
            exact_validation_logical_read_bytes: OUTPUT_BYTES,
            exact_validation_wall_nanoseconds: result.norm_exact_validation_wall_nanoseconds,
            largest_observed_bigint_coefficient_payload_bits: result.norm_exact_high_water_bits,
            bigint_payload_note: "largest observed BigInt coefficient payload during exact block-square validation; not allocator overhead".to_string(),
            analysis_retained_stamp: analysis_retained_stamp.clone(),
            analysis_retained_output_reauthentication_wall_nanoseconds,
        },
    )?;
    packet.json_strict(
        "exact-validation.json",
        &ExactValidationArtifact {
            schema: SCHEMA.to_string(),
            status: ArtifactStatus::Complete,
            dot_unit_exponent: DOT_EXP,
            dot_resource_bits: 512,
            required_high_water_bits_lt: 384,
            norm_square_floor_exponent: -298,
            fixed_dot_resource_bytes: 64,
            bigint_exact_dot_calls: result.exact_dot_calls,
            bigint_term_operation_proxy: result.exact_dot_term_operations,
            operation_proxy_note: "logical exact terms, not hardware operations".to_string(),
            norm_validation_wall_nanoseconds: result.norm_exact_validation_wall_nanoseconds,
            screening_and_exact_survivor_wall_nanoseconds: result
                .screening_and_exact_survivor_wall_nanoseconds,
            allocation_telemetry: result.allocation_telemetry.clone(),
            records: joined.exact_records.clone(),
        },
    )?;
    packet.json_strict(
        "screening-result.json",
        &ScreeningResultArtifact {
            schema: SCHEMA.to_string(),
            status: ArtifactStatus::Complete,
            competitors: VOCAB - 1,
            pruning_floor: PRUNING_FLOOR,
            captures: joined.screening.clone(),
        },
    )?;
    packet.json_strict(
        "byte-ledger.json",
        &ByteLedgerArtifact {
            schema: SCHEMA.to_string(),
            status: ArtifactStatus::Complete,
            competitors: VOCAB - 1,
            pruning_floor: PRUNING_FLOOR,
            full_head_denominator_bytes: OUTPUT_BYTES,
            charged_128_limit_bytes: BYTE_LIMIT,
            gate_alignment: 128,
            ledgers: joined.ledgers.clone(),
        },
    )?;
    Ok(())
}

fn write_skipped_analysis_artifacts(
    packet: &mut Packet,
    reason: SkippedAnalysisReason,
) -> Result<()> {
    for name in [
        "metadata-manifest.json",
        "exact-validation.json",
        "screening-result.json",
        "byte-ledger.json",
    ] {
        packet.json_strict(
            name,
            &SkippedAnalysisArtifact {
                schema: SCHEMA.to_string(),
                status: ArtifactStatus::NotRunCaptureCoverage,
                reason,
            },
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SkippedAnalysisReason {
    ProducerEosBeforeRequiredCaptureCoverage,
    ProducerEosInsteadOfTokenLimit,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SkippedAnalysisArtifact {
    schema: String,
    status: ArtifactStatus,
    reason: SkippedAnalysisReason,
}

pub fn run(args: LmHeadScreeningOracleArgs) -> Result<()> {
    ensure!(
        args.packet_dir == Path::new(PACKET_PATH) && normalized_relative(&args.packet_dir),
        "refusing to reserve a noncanonical packet root"
    );
    let mut packet = Packet::reserve(&args.packet_dir)?;
    let acquired = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let build_identity = super::recorded_build_identity();
        let qwen_env = super::capture_qwen_env();
        run_reserved(&mut packet, args, build_identity, qwen_env)
    }));
    let outcome = match acquired {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(error)) if packet.readiness_published() => invalid_outcome(format!("{error:#}")),
        Ok(Err(error)) => {
            return Err(anyhow!(
                "packet root was consumed before authenticated readiness: {error:#}"
            ));
        }
        Err(payload) if packet.readiness_published() => {
            invalid_outcome(format!("panic: {}", panic_message(payload)))
        }
        Err(payload) => {
            return Err(anyhow!(
                "packet root was consumed before authenticated readiness: panic: {}",
                panic_message(payload)
            ));
        }
    };
    seal(&mut packet, &outcome)?;
    println!("{}", outcome.disposition.label());
    Ok(())
}

fn invalid_outcome(failure: String) -> TerminalOutcome {
    TerminalOutcome {
        disposition: Disposition::Invalid,
        any_pruning_failure: false,
        any_byte_failure: false,
        any_ideal_mismatch: false,
        coverage_eos_call: None,
        failure: Some(failure),
        require_complete_inventory: false,
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|value| (*value).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

fn run_reserved(
    packet: &mut Packet,
    args: LmHeadScreeningOracleArgs,
    build_identity: super::BuildIdentity,
    qwen_env: BTreeMap<String, String>,
) -> Result<TerminalOutcome> {
    validate_args(&args)?;
    validate_build(&build_identity)?;
    ensure!(
        qwen_env.is_empty(),
        "v0.664 forbids every inherited QWEN_* control"
    );
    let executable = executable_identity()?;
    packet.executable_commitment = Some(executable.clone());
    let readiness = readiness_payload(&args, &build_identity, &executable)?;
    validate_readiness_constants(&readiness)?;
    packet.json_strict("readiness.json", &readiness)?;
    for directory in ["captures", "winner-evidence", "analyzer-input", "metadata"] {
        packet.mkdir(Path::new(directory))?;
    }
    let tests = self_tests()?;
    packet.json_strict("self-tests.json", &tests)?;

    let host_before_raw = HostSnapshot::capture("v0664_before_model")?;
    host_before_raw.validate()?;
    let host_before = host_evidence("v0664_before_model", &host_before_raw)?;
    let AuthenticatedModel {
        file: authenticated_file,
        gguf: authenticated_gguf,
        analysis_gguf,
        stamp: authenticated_stamp,
        identity: authenticated_identity,
        authentication_telemetry,
    } = authenticate_raw_model(&args.model)?;
    let runtime = Runtime::metal()?;
    let loaded = runtime.load_opened_gguf(authenticated_gguf, args.model.clone())?;
    let runtime_stamp = stamp_retained_gguf(loaded.gguf())?;
    ensure!(runtime_stamp == authenticated_stamp);
    ensure!(validate_raw_gguf(loaded.gguf(), &authenticated_stamp)? == authenticated_identity);
    let post_runtime_load_stamp = FileStamp::from_metadata(&authenticated_file.metadata()?);
    ensure!(post_runtime_load_stamp == authenticated_stamp);
    let tokenizer = loaded.tokenizer()?;
    ensure!(tokenizer.n_vocab() as usize == VOCAB);
    let (prompt_text, prompt_ids) = prompt_identity(&tokenizer, &args.prompt_file)?;
    let generation = capture_request(&loaded, &prompt_ids, &authenticated_identity.stop_token_ids)?;
    let post_capture_stamp = FileStamp::from_metadata(&authenticated_file.metadata()?);
    ensure!(post_capture_stamp == authenticated_stamp);
    ensure!(stamp_retained_gguf(loaded.gguf())? == authenticated_stamp);
    drop(loaded);
    drop(runtime);
    let host_after_raw = HostSnapshot::capture("v0664_after_capture")?;
    host_after_raw.validate()?;
    let host_after = host_evidence("v0664_after_capture", &host_after_raw)?;

    for capture in &generation.captures {
        persist_capture(packet, capture, &prompt_ids)?;
    }
    let runtime_identity = OpenedGgufRuntimeIdentity {
        runtime_load_api: "Runtime::load_opened_gguf".to_string(),
        exact_opened_gguf_consumed: true,
        pre_runtime_stamp: authenticated_stamp.clone(),
        runtime_retained_stamp: runtime_stamp,
        post_runtime_load_stamp,
        post_capture_stamp: Some(post_capture_stamp),
    };
    let capture_manifest = capture_manifest_payload(
        &build_identity,
        &executable,
        &authenticated_identity,
        &runtime_identity,
        &prompt_text,
        &generation,
        &host_before,
        &host_after,
        &authentication_telemetry,
    );
    ensure!(
        serde_json::to_vec(&capture_manifest.build_identity)?
            == serde_json::to_vec(&readiness.build_identity)?,
        "capture and readiness build identities differ before publication"
    );
    ensure!(
        capture_manifest.executable_identity == readiness.executable_identity,
        "capture and readiness executable identities differ before publication"
    );
    packet.json_strict("capture-manifest.json", &capture_manifest)?;

    if let Some(call) = generation.coverage_eos_call {
        let reason = if CALLS.iter().all(|required| *required <= call) {
            SkippedAnalysisReason::ProducerEosInsteadOfTokenLimit
        } else {
            SkippedAnalysisReason::ProducerEosBeforeRequiredCaptureCoverage
        };
        write_skipped_analysis_artifacts(packet, reason)?;
        return Ok(TerminalOutcome {
            disposition: Disposition::KillCaptureCoverage,
            any_pruning_failure: false,
            any_byte_failure: false,
            any_ideal_mismatch: false,
            coverage_eos_call: Some(call),
            failure: None,
            require_complete_inventory: false,
        });
    }
    if !generation
        .captures
        .iter()
        .map(|capture| capture.call)
        .eq(CALLS)
    {
        return Err(anyhow!(
            "reachable non-EOS request is missing a frozen capture call"
        ));
    }
    ensure!(generation.generated.len() == TOKENS && generation.transitions == TOKENS - 1);

    let analysis_reauthentication_start = Instant::now();
    let analysis_stamp = stamp_retained_gguf(&analysis_gguf)?;
    ensure!(analysis_stamp == authenticated_stamp);
    ensure!(FileStamp::from_metadata(&authenticated_file.metadata()?) == authenticated_stamp);
    let output_weight = validate_analysis_output(&analysis_gguf, &authenticated_stamp)?;
    let analysis_retained_output_reauthentication_wall_nanoseconds =
        u64::try_from(analysis_reauthentication_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let capture_labels = generation
        .captures
        .iter()
        .map(|capture| CaptureLabel {
            capture_identity: sha256(f32_le_bytes(&capture.hidden)),
            call: capture.call,
            sequence_position: capture.sequence_position,
        })
        .collect::<Vec<_>>();
    let analyzer_captures = generation
        .captures
        .into_iter()
        .zip(&capture_labels)
        .map(|(capture, label)| AnalyzerCapture {
            capture_identity: label.capture_identity.clone(),
            winner: capture.winner,
            hidden: capture.hidden,
        })
        .collect::<Vec<_>>();
    let analyzer_result = run_analyzer(output_weight, analyzer_captures)?;
    let joined = join_analyzer_records(&analyzer_result, &capture_labels)?;
    persist_analyzer_result(
        packet,
        &analyzer_result,
        &joined,
        analysis_retained_output_reauthentication_wall_nanoseconds,
        &analysis_stamp,
    )?;
    ensure!(stamp_retained_gguf(&analysis_gguf)? == authenticated_stamp);
    ensure!(FileStamp::from_metadata(&authenticated_file.metadata()?) == authenticated_stamp);

    let any_mismatch = analyzer_result
        .screening
        .iter()
        .any(|result| !result.unique_ideal_winner);
    let any_pruning_failure = analyzer_result
        .screening
        .iter()
        .any(|result| !result.pruning_pass);
    let any_byte_failure = analyzer_result
        .screening
        .iter()
        .any(|result| !result.bytes_pass);
    let disposition = mechanism_disposition(any_mismatch, any_pruning_failure, any_byte_failure);
    Ok(TerminalOutcome {
        disposition,
        any_pruning_failure,
        any_byte_failure,
        any_ideal_mismatch: any_mismatch,
        coverage_eos_call: None,
        failure: None,
        require_complete_inventory: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_q6() -> [u8; BLOCK_BYTES] {
        let mut bytes = [0u8; BLOCK_BYTES];
        bytes[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
        bytes
    }

    #[test]
    fn q6_unpack_matches_independent_hand_constructed_paths() {
        // (logical index, ql byte, qh byte, qh shift, scale byte)
        let paths = [
            (0, 0, 128, 0, 192),
            (32, 32, 128, 2, 194),
            (64, 0, 128, 4, 196),
            (96, 32, 128, 6, 198),
            (128, 64, 160, 0, 200),
            (160, 96, 160, 2, 202),
            (192, 64, 160, 4, 204),
            (224, 96, 160, 6, 206),
        ];
        for (logical, ql, qh, shift, scale) in paths {
            let mut bytes = empty_q6();
            bytes[ql] = if logical % 128 < 64 { 0x0f } else { 0xf0 };
            bytes[qh] = 3 << shift;
            bytes[scale] = 7;
            assert_eq!(
                Q6Block::parse(&bytes)
                    .unwrap()
                    .quant_scale(logical)
                    .unwrap(),
                (31, 7)
            );
        }
        let mut extrema = empty_q6();
        extrema[..128].fill(0xff);
        extrema[128..192].fill(0xff);
        extrema[192..208].fill(0x80);
        let block = Q6Block::parse(&extrema).unwrap();
        assert_eq!(block.quant_scale(0).unwrap(), (31, -128));
        extrema[..128].fill(0);
        extrema[128..192].fill(0);
        extrema[192..208].fill(0x7f);
        let block = Q6Block::parse(&extrema).unwrap();
        assert_eq!(block.quant_scale(255).unwrap(), (-32, 127));
    }

    #[test]
    fn dyadic_f16_f32_subnormal_and_signed_zero() {
        assert_eq!(f16_dyadic(1).unwrap(), Dyadic::new(BigInt::one(), -24));
        assert_eq!(f16_dyadic(0x3c00).unwrap(), Dyadic::new(BigInt::one(), 0));
        assert_eq!(
            f32_dyadic(f32::from_bits(1)).unwrap(),
            Dyadic::new(BigInt::one(), -149)
        );
        assert_eq!(f32_dyadic(1.0).unwrap(), Dyadic::new(BigInt::one(), 0));
        assert_eq!(f32_dyadic(0.0).unwrap(), f32_dyadic(-0.0).unwrap());
        assert_eq!(
            f16_dyadic(0x7bff).unwrap(),
            Dyadic::new(BigInt::from(2047), 5)
        );
        assert_eq!(
            f16_dyadic(0xfbff).unwrap(),
            Dyadic::new(BigInt::from(-2047), 5)
        );
        assert_eq!(
            f32_dyadic(f32::MIN_POSITIVE).unwrap(),
            Dyadic::new(BigInt::one(), -126)
        );
        assert!(f32_dyadic(f32::MAX).unwrap().cmp(&Dyadic::zero()).is_gt());
        assert!(f16_dyadic(0x7c00).is_err());
        assert!(f32_dyadic(f32::NAN).is_err());
    }

    #[test]
    fn fixed_dot_512_bit_high_water_contract() {
        let valid = Dyadic::new((BigInt::one() << 382usize) - 1, DOT_EXP);
        assert!(valid.signed_bits() < 384);
        let resource_max = Dyadic::new((BigInt::one() << 510usize) - 1, DOT_EXP);
        assert_eq!(resource_max.signed_bits(), 511);
        assert!(resource_max.signed_bits() <= 512);
        let mut row = vec![0u8; 2 * BLOCK_BYTES];
        for block in row.chunks_exact_mut(BLOCK_BYTES) {
            block[192..208].fill(1);
            block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
        }
        assert_eq!(
            exact_dot(&row, &vec![1.0; 512])
                .unwrap()
                .cmp(&Dyadic::new(BigInt::from(-16_384), 0)),
            Ordering::Equal
        );
    }

    #[test]
    fn exact_dot_payload_tracker_observes_shifted_partial_before_cancellation() {
        let mut row = empty_q6();
        row[192..208].fill(1);
        let mut hidden = vec![0.0f32; BLOCK];
        hidden[0] = f32::MAX;
        hidden[1] = -f32::MAX;
        let mut tracker = BigIntPayloadTracker::default();
        let score = exact_dot_profiled(&row, &hidden, &mut tracker).unwrap();
        assert!(score.coefficient.is_zero());
        assert_eq!(score.signed_bits(), 1);
        assert!(tracker.largest_observed_coefficient_payload_bits > 250);
        assert!(tracker.largest_observed_coefficient_payload_bits > score.signed_bits());
    }

    #[test]
    fn fixed_i512_encoding_avoids_untracked_bigint_temporaries() {
        assert_eq!(
            bigint_twos_hex(&BigInt::zero(), 8).unwrap(),
            "0".repeat(128)
        );
        assert_eq!(
            bigint_twos_hex(&BigInt::from(-1), 8).unwrap(),
            "f".repeat(128)
        );
        let maximum = (BigInt::one() << 511usize) - 1;
        let minimum = -(BigInt::one() << 511usize);
        assert!(bigint_twos_hex(&maximum, 8).unwrap().starts_with("7fff"));
        assert_eq!(
            bigint_twos_hex(&minimum, 8).unwrap(),
            format!("8{}", "0".repeat(127))
        );
        assert!(bigint_twos_hex(&(BigInt::one() << 511usize), 8).is_err());
        assert!(bigint_twos_hex(&(minimum - 1), 8).is_err());
    }

    #[test]
    fn owned_vec_phase_high_water_includes_prior_retained_and_current_transient() {
        let mut tracker = OwnedVecPhaseTracker::new(100);
        tracker.observe_capture(50, 30).unwrap();
        tracker.observe_capture(80, 20).unwrap();
        assert_eq!(tracker.persistent_before_high_water_bytes, 130);
        assert_eq!(tracker.transient_high_water_bytes, 80);
        assert_eq!(tracker.persistent_after_high_water_bytes, 150);
        assert_eq!(tracker.phase_high_water_bytes, 210);
        assert!(tracker.phase_high_water_bytes > 100 + 80);
    }

    #[test]
    fn authenticated_stop_vector_requires_exact_frozen_order_and_membership() {
        assert!(validate_authenticated_stop_ids(&STOP_IDS).is_ok());
        assert!(validate_authenticated_stop_ids(&[]).is_err());
        assert!(validate_authenticated_stop_ids(&[STOP_IDS[0], 1]).is_err());
        assert!(validate_authenticated_stop_ids(&[1, STOP_IDS[0]]).is_err());
    }

    #[test]
    fn shared_model_name_extractor_distinguishes_both_identity_keys() {
        let observation = runtime_opened_gguf_rewind_self_test().unwrap();
        assert_eq!(
            observation.model_names,
            ModelNameIdentity {
                base_model_name: BASE_MODEL_NAME.to_string(),
                basename: BASENAME.to_string(),
            }
        );
        assert!(observation.diagnostic_path_absent_before_parse);
        assert_eq!(
            observation.shared_cursor_position_before_parse,
            observation.fixture_bytes as u64
        );
    }

    #[test]
    fn prompt_and_generated_token_digest_contracts_remain_distinct() {
        let tokens = [1, -2, 248_319];
        let prompt = token_ids_sha256_i32le(&tokens);
        let generated = generated_token_digest(&tokens);
        assert_eq!(
            prompt,
            "3f37364bc87f9ff835c64d4bdb3d993fe35e530097697da8cbacb5e6f92119d5"
        );
        assert_eq!(
            generated,
            "4b93ca7810c97ba477771dba407b0e8b9dd27e743f21d3a1fa9242d01fe06a9d"
        );
        assert_ne!(prompt, generated);
    }

    #[test]
    fn sum_norm_upper_and_next_up_cast_contain_exact() {
        let exact = Dyadic::new(BigInt::from(5), -2);
        let upper = norm_upper(&exact, &[1.0, 0.25]).unwrap();
        assert_ne!(
            f32_dyadic(upper).unwrap().square().cmp(&exact),
            Ordering::Less
        );
        let source = 1.0 + 2.0f64.powi(-30);
        assert_eq!(f32_upper(source).unwrap(), next_up_f32(1.0));
        assert!(gamma_sum_upper(&[f64::NAN]).is_err());
        assert!(norm_upper(&Dyadic::new(BigInt::from(-1), 0), &[1.0]).is_err());
        let square = Dyadic::new(BigInt::from(4), 0);
        let exact_boundary = norm_upper(&square, &[4.0]).unwrap();
        assert!(exact_boundary >= 2.0);
        let above = Dyadic::new((BigInt::one() << 48usize) + 1, -46);
        let above_upper = norm_upper(&above, &[above.to_f64_rn().unwrap()]).unwrap();
        assert!(f32_dyadic(above_upper).unwrap().square().cmp(&above) != Ordering::Less);
    }

    #[test]
    fn strict_total_cmp_ties_and_signed_zero() {
        let mut logits = vec![-1.0; VOCAB];
        logits[3] = 7.0;
        logits[9] = 7.0;
        assert_eq!(total_cmp_winner(&logits).unwrap(), 9);
        logits.fill(-1.0);
        logits[2] = -0.0;
        logits[4] = 0.0;
        assert_eq!(total_cmp_winner(&logits).unwrap(), 4);
        logits[7] = f32::NAN;
        assert!(total_cmp_winner(&logits).is_err());
    }

    #[test]
    fn persisted_analysis_types_round_trip_with_explicit_nesting() {
        fn round_trip<T: Serialize + DeserializeOwned>(value: &T) {
            let encoded = serde_json::to_vec(value).unwrap();
            let tree: Value = serde_json::from_slice(&encoded).unwrap();
            assert!(tree.get("analysis").is_some());
            let parsed: T = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(serde_json::to_vec(&parsed).unwrap(), encoded);
        }

        let identity = "0".repeat(64);
        let screening = PersistedCaptureScreening {
            call: 0,
            sequence_position: PROMPT_TOKENS,
            analysis: CaptureScreening {
                capture_identity: identity.clone(),
                winner_id: 0,
                winner_lower_bits: "0000000000000000".to_string(),
                winner_upper_bits: "0000000000000000".to_string(),
                winner_exact_coefficient_hex: "0".repeat(128),
                winner_exact_exponent: DOT_EXP,
                competitor_count: VOCAB - 1,
                bound_pruned_count: VOCAB - 1,
                survivor_count: 0,
                less_than_winner_count: 0,
                tie_count: 0,
                greater_than_winner_count: 0,
                unique_ideal_winner: true,
                pruning_pass: true,
                bytes_pass: true,
                charged_64_bytes: 0,
                charged_128_bytes: 0,
                charged_256_bytes: 0,
                unique_q6_footprint_bytes: ROW_BYTES as u64,
                survivor_ids_sha256: sha256(&[]),
                survivor_cmp_sha256: sha256(&[]),
                upper_vec_requested_capacity_bytes: 0,
                active_vec_requested_capacity_bytes: 0,
                survivor_ids_vec_requested_capacity_bytes: 0,
                survivor_cmp_vec_requested_capacity_bytes: 0,
                survivor_ids_digest_vec_requested_capacity_bytes: 0,
                largest_observed_bigint_coefficient_payload_bits: 1,
            },
        };
        let exact = PersistedExactCaptureRecord {
            call: 0,
            sequence_position: PROMPT_TOKENS,
            analysis: ExactCaptureRecord {
                capture_identity: identity.clone(),
                winner_id: 0,
                winner: ExactWinnerRecord {
                    coefficient_twos_complement_u64x8: "0".repeat(128),
                    exponent: DOT_EXP,
                    signed_bits: 1,
                    lower_f64_bits: "0000000000000000".to_string(),
                    upper_f64_bits: "0000000000000000".to_string(),
                },
                survivors: 0,
                less: 0,
                ties: 0,
                greater: 0,
                survivor_ids: vec![],
                survivor_cmp: vec![],
                comparison_less: 0,
                comparison_equal: 1,
                comparison_greater: 2,
            },
        };
        let ledger = PersistedCaptureLedgerRecord {
            call: 0,
            sequence_position: PROMPT_TOKENS,
            analysis: CaptureLedgerRecord {
                capture_identity: identity,
                full_head_denominator_bytes: OUTPUT_BYTES as u64,
                gate_limit_bytes: BYTE_LIMIT,
                winner_excluded: true,
                logical_survivors: 0,
                alignments: [64, 128, 256].map(|alignment| ledger_for(&[], 0, alignment).unwrap()),
                reduction: reduce_survivors(0, &[], &[], (0.0, 0.0)).unwrap(),
            },
        };
        round_trip(&screening);
        round_trip(&exact);
        round_trip(&ledger);

        let mut tree = serde_json::to_value(&screening).unwrap();
        tree["analysis"]["unknown"] = Value::Bool(true);
        assert!(serde_json::from_value::<PersistedCaptureScreening>(tree).is_err());
    }

    #[test]
    fn synthetic_prune_survivor_and_reduction_invariants() {
        let lower = 2.0;
        let bounds = [1.0, 2.0, 3.0, 0.0];
        let survivors: Vec<u32> = bounds
            .iter()
            .enumerate()
            .filter(|&(id, bound)| id != 3 && *bound >= lower)
            .map(|(id, _)| id as u32)
            .collect();
        assert_eq!(survivors, [1, 2]);
        validate_survivors(&survivors, 4, 3).unwrap();
        assert!(validate_survivors(&[1, 1], 4, 3).is_err());
        assert!(validate_survivors(&[1, 3], 4, 3).is_err());
        assert!(reduce_survivors(3, &[1, 1], &[0, 1], (1.0, 1.0)).is_err());
        assert!(reduce_survivors(3, &[1, 3], &[0, 1], (1.0, 1.0)).is_err());
        assert!(reduce_survivors(3, &[1, 2], &[0, 3], (1.0, 1.0)).is_err());
        assert!(reduce_survivors(3, &[1], &[0, 1], (1.0, 1.0)).is_err());
        assert!(reduce_survivors(3, &[1], &[0], (2.0, 1.0)).is_err());
        assert_eq!(1 + survivors.len(), 3);
        let winner = Dyadic::new(BigInt::from(10), 0);
        assert_eq!(Dyadic::new(BigInt::from(9), 0).cmp(&winner), Ordering::Less);
        assert_eq!(
            Dyadic::new(BigInt::from(10), 0).cmp(&winner),
            Ordering::Equal
        );
        assert_eq!(
            Dyadic::new(BigInt::from(11), 0).cmp(&winner),
            Ordering::Greater
        );
        assert_eq!(
            mechanism_disposition(true, true, true),
            Disposition::KillIdealMismatch
        );
        assert_eq!(
            mechanism_disposition(false, true, true),
            Disposition::KillPruningAndBytes
        );
        assert_eq!(
            mechanism_disposition(false, true, false),
            Disposition::KillPruning
        );
        assert_eq!(
            mechanism_disposition(false, false, true),
            Disposition::KillBytes
        );
    }

    #[test]
    fn byte_span_epoch_ledgers_are_exact_at_all_alignments() {
        let survivors = [0, 2, 4, 7];
        let ledgers = [64, 128, 256].map(|alignment| ledger_for(&survivors, 3, alignment).unwrap());
        assert_eq!(
            ledgers.each_ref().map(|ledger| ledger.alignment),
            [64, 128, 256]
        );
        for ledger in ledgers {
            assert_eq!(ledger.streams.len(), 22);
            assert_eq!(
                ledger.total_charged_bytes,
                ledger
                    .streams
                    .iter()
                    .map(|stream| stream.charged_bytes)
                    .sum::<u64>()
            );
        }
        let ledger = ledger_for(&[0, 2, 4], 3, 128).unwrap();
        let survivor_rows = ledger
            .streams
            .iter()
            .find(|stream| stream.epoch == "survivor" && stream.resource == "output_weight")
            .unwrap();
        assert_eq!(survivor_rows.logical_bytes, 3 * ROW_BYTES as u64);
        assert_eq!(survivor_rows.charged_bytes, 5_376);
        assert_eq!(rounded_union(&[(1, 65), (63, 129)], 64).unwrap(), 192);
        assert_eq!(rounded_union(&[(1, 65), (63, 129)], 128).unwrap(), 256);
        assert!(rounded_union(&[(2, 1)], 128).is_err());
    }

    #[test]
    fn malformed_q6_and_dyadic_inputs_reject() {
        let bytes = empty_q6();
        assert!(Q6Block::parse(&bytes[..209]).is_err());
        assert!(Q6Block::parse(&[0; 211]).is_err());
        assert!(f64_dyadic(f64::INFINITY).is_err());
        assert!(gamma_sum_upper(&[]).is_err());
        assert!(decode_f32_raw(&[0; 7], 2).is_err());
        assert!(decode_f32_raw(&f32::NAN.to_bits().to_le_bytes(), 1).is_err());
        let duplicate = br#"{"schema":"x","schema":"y","status":"x","reason":"producer_eos_before_required_capture_coverage"}"#;
        assert!(serde_json::from_slice::<SkippedAnalysisArtifact>(duplicate).is_err());
        assert!(self_tests().is_ok());
    }

    #[test]
    fn analyzer_positive_capability_set_is_exact() {
        let input = AnalyzerInput {
            geometry: Geometry {
                hidden: 1,
                vocab: 2,
                blocks: 1,
                row_bytes: 1,
            },
            captures: vec![],
            block_norms: vec![],
            output_weight: &[],
            ledger: LedgerConstants {
                full_head: 2,
                byte_limit: 1,
                pruning_floor: 1,
            },
        };
        assert_eq!(
            analyzer_capabilities(&input),
            [
                "profile_geometry",
                "capture_identity",
                "winner_id",
                "owned_hidden_f32_bits",
                "owned_authenticated_block_norm_f32_metadata",
                "readonly_authenticated_output_weight_bytes",
                "frozen_ledger_constants",
            ]
        );
    }

    #[test]
    fn analyzer_identity_join_rejects_duplicate_missing_unknown_and_reordered() {
        let labels = CALLS
            .iter()
            .map(|&call| CaptureLabel {
                capture_identity: format!("identity-{call}"),
                call,
                sequence_position: PROMPT_TOKENS + call,
            })
            .collect::<Vec<_>>();
        let ids = labels
            .iter()
            .map(|label| label.capture_identity.as_str())
            .collect::<Vec<_>>();
        validate_analyzer_identity_join(&labels, &ids, &ids, &ids).unwrap();

        let mut duplicate_labels = labels.clone();
        let duplicate_identity = duplicate_labels[0].capture_identity.clone();
        duplicate_labels[1].capture_identity = duplicate_identity;
        assert!(validate_analyzer_identity_join(&duplicate_labels, &ids, &ids, &ids).is_err());
        assert!(
            validate_analyzer_identity_join(&labels, &ids[..ids.len() - 1], &ids, &ids).is_err()
        );
        let mut unknown = ids.clone();
        unknown[0] = "unknown";
        assert!(validate_analyzer_identity_join(&labels, &unknown, &ids, &ids).is_err());
        let mut reordered = ids.clone();
        reordered.swap(0, 1);
        assert!(validate_analyzer_identity_join(&labels, &ids, &reordered, &ids).is_err());
    }

    #[test]
    fn artifact_inventory_rejects_symlink_hardlink_and_foreign_file() {
        let root = std::env::temp_dir().join(format!(
            "qwen-v0664-inventory-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("a"), b"a").unwrap();
        let mut files = Vec::new();
        walk_packet(&root, Path::new(""), &mut files).unwrap();
        assert_eq!(files, [PathBuf::from("a")]);
        let expected = BTreeSet::from([PathBuf::from("a")]);
        std::fs::write(root.join("foreign"), b"foreign").unwrap();
        let mut with_foreign = Vec::new();
        walk_packet(&root, Path::new(""), &mut with_foreign).unwrap();
        assert!(validate_inventory_set(&with_foreign.into_iter().collect(), &expected).is_err());
        std::fs::remove_file(root.join("foreign")).unwrap();
        std::os::unix::fs::symlink("a", root.join("link")).unwrap();
        assert!(walk_packet(&root, Path::new(""), &mut Vec::new()).is_err());
        std::fs::remove_file(root.join("link")).unwrap();
        std::fs::hard_link(root.join("a"), root.join("alias")).unwrap();
        assert!(walk_packet(&root, Path::new(""), &mut Vec::new()).is_err());
        std::fs::remove_file(root.join("alias")).unwrap();
        assert!(!normalized_relative(Path::new("../escape")));
        assert!(!normalized_relative(Path::new("a/../escape")));
        std::fs::write(
            root.join("bad.json"),
            b"{\"schema\":\"wrong\",\"status\":\"x\",\"payload\":null}\n",
        )
        .unwrap();
        assert!(strict_reparse_json(&root.join("bad.json"), Path::new("bad.json")).is_err());
        std::fs::remove_file(root.join("bad.json")).unwrap();
        assert!(
            validate_raw_artifact_size(Path::new("captures/call-000/hidden.f32le"), 7).is_err()
        );
        std::fs::create_dir(root.join("foreign")).unwrap();
        assert!(walk_packet(&root, Path::new(""), &mut Vec::new()).is_err());
        std::fs::remove_dir(root.join("foreign")).unwrap();
        std::fs::remove_file(root.join("a")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn invalid_terminalization_is_exclusive_and_atomic_visible() {
        let root = std::env::temp_dir().join(format!(
            "qwen-v0664-terminal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let metadata = std::fs::symlink_metadata(&root).unwrap();
        let mut packet = Packet {
            root: root.clone(),
            written: BTreeSet::new(),
            commitments: BTreeMap::new(),
            root_stamp: FileStamp::from_metadata(&metadata),
            terminalized: false,
            executable_commitment: None,
        };
        let outcome = invalid_outcome("synthetic".to_string());
        assert!(seal(&mut packet, &outcome).is_err());
        let source_state = format!(
            "{}{}",
            super::super::source_identity::SOURCE_STATE_PREFIX,
            "0".repeat(64)
        );
        let build_identity = super::super::BuildIdentity {
            schema_version: 2,
            build_commit: "0".repeat(40),
            build_commit_short: "0".repeat(9),
            build_dirty: Some(false),
            build_source_state: Some(source_state.clone()),
            stamp_source: "git".to_string(),
            stamp_error: None,
            runtime_commit: Some("0".repeat(40)),
            runtime_dirty: Some(false),
            runtime_source_state: Some(source_state),
            status: "match".to_string(),
            problems: vec![],
            overrides: vec![],
        };
        let executable_identity = ExecutableIdentity {
            path: "/synthetic/target/release/qwen-bench".to_string(),
            sha256: "0".repeat(64),
            stamp: FileStamp::from_metadata(&metadata),
            debug_assertions: false,
        };
        let readiness = ReadinessArtifact {
            schema: SCHEMA.to_string(),
            build_identity,
            executable_identity,
            operator_attestation: true,
            command_arguments: ReadinessCommandArguments {
                model: MODEL_PATH.to_string(),
                prompt_file: PROMPT_PATH.to_string(),
                tokens: TOKENS,
                capture_calls: CALLS.to_vec(),
                packet_dir: PACKET_PATH.to_string(),
                attest_no_other_user_gpu_workload: true,
            },
            predecessor_decision_sha256: PREDECESSOR_DECISION_SHA256.to_string(),
        };
        packet.json_strict("readiness.json", &readiness).unwrap();
        seal(&mut packet, &outcome).unwrap();
        assert!(root.join("decision.json").is_file());
        assert!(!root.join(".decision.json.tmp").exists());
        assert!(publish_decision_exclusive(&packet, b"{}\n").is_err());
        std::fs::remove_file(root.join("decision.json")).unwrap();
        std::fs::remove_file(root.join("artifact-manifest.json")).unwrap();
        std::fs::remove_file(root.join("readiness.json")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}
