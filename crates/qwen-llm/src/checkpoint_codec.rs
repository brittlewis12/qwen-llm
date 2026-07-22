//! Bounded whole-checkpoint wire format for durable prefix-cache storage.
//!
//! The raw Metal arenas are a versioned little-endian runtime ABI, not a
//! hardware-independent tensor encoding. Publication, locking, and catalog
//! policy deliberately live above this module.

use crate::checkpoint_identity::STATE_ENCODING_ABI_VERSION;
use crate::metal_forward::{
    SessionSnapshot, SnapshotIdentity, SnapshotKvStorageKind, SnapshotValidationError,
};
use std::io::{self, Read, Write};

const MAGIC: &[u8; 8] = b"QWENCKP\0";
const CODEC_VERSION: u32 = 1;
const HEADER_BYTES: usize = 256;
const PAYLOAD_OFFSET: usize = 16 * 1024;
const DIGEST_BYTES: usize = 32;
pub const SNAPSHOT_RECORD_FIXED_BYTES: u64 = (PAYLOAD_OFFSET + DIGEST_BYTES) as u64;
const FLAG_PENDING_TOKEN: u64 = 1 << 0;
const FLAG_FINAL_LOGITS: u64 = 1 << 1;
const KNOWN_FLAGS: u64 = FLAG_PENDING_TOKEN | FLAG_FINAL_LOGITS;

const OFF_MAGIC: usize = 0x00;
const OFF_VERSION: usize = 0x08;
const OFF_HEADER_BYTES: usize = 0x0c;
const OFF_FLAGS: usize = 0x10;
const OFF_RECORD_BYTES: usize = 0x18;
const OFF_PAYLOAD_OFFSET: usize = 0x20;
const OFF_PAYLOAD_BYTES: usize = 0x28;
const OFF_PREFIX_COUNT: usize = 0x30;
const OFF_PENDING_TOKEN: usize = 0x38;
const OFF_STATE_ENCODING: usize = 0x3c;
const OFF_MODEL_ID: usize = 0x40;
const OFF_TOKENIZER_ID: usize = 0x48;
const OFF_LAYOUT_VERSION: usize = 0x50;
const OFF_ATTN_LAYERS: usize = 0x54;
const OFF_GDN_LAYERS: usize = 0x58;
const OFF_KV_DIM: usize = 0x5c;
const OFF_KV_BYTES_PER_TOKEN: usize = 0x60;
const OFF_GDN_STATE_ELEMENTS: usize = 0x64;
const OFF_GDN_CONV_ELEMENTS: usize = 0x68;
const OFF_KV_STORAGE_KIND: usize = 0x6c;
const OFF_PREFIX_BYTES: usize = 0x70;
const OFF_KV_POSITION_BYTES: usize = 0x78;
const OFF_KV_K_BYTES: usize = 0x80;
const OFF_KV_V_BYTES: usize = 0x88;
const OFF_GDN_CONV_BYTES: usize = 0x90;
const OFF_GDN_STATE_BYTES: usize = 0x98;
const OFF_LOGITS_BYTES: usize = 0xa0;
const OFF_COMPATIBILITY_ID: usize = 0xa8;
const OFF_RESERVED: usize = 0xc8;

#[derive(Clone, Copy, Debug)]
pub struct SnapshotCodecConstraints<'a> {
    pub expected_identity: &'a SnapshotIdentity,
    pub expected_compatibility_id: &'a [u8; 32],
    pub expected_vocab_size: usize,
    pub max_context_tokens: usize,
    pub max_record_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodedSnapshot {
    pub record_bytes: u64,
    pub digest: [u8; DIGEST_BYTES],
}

#[derive(Debug, thiserror::Error)]
pub enum SnapshotCodecError {
    #[error("checkpoint I/O: {0}")]
    Io(#[from] io::Error),
    #[error("unsupported checkpoint host byte order")]
    UnsupportedByteOrder,
    #[error("invalid checkpoint header: {0}")]
    InvalidHeader(&'static str),
    #[error("checkpoint compatibility identity mismatch")]
    CompatibilityMismatch,
    #[error("checkpoint snapshot identity mismatch")]
    IdentityMismatch,
    #[error("checkpoint {field} arithmetic overflow")]
    ArithmeticOverflow { field: &'static str },
    #[error("checkpoint record size {record_bytes} exceeds budget {max_record_bytes}")]
    RecordBudgetExceeded {
        record_bytes: u64,
        max_record_bytes: u64,
    },
    #[error("checkpoint consumed prefix length {prefix_len} exceeds context {max_context_tokens}")]
    ContextCapacityExceeded {
        prefix_len: u64,
        max_context_tokens: usize,
    },
    #[error("checkpoint {section} length {actual} != expected {expected}")]
    SectionLength {
        section: &'static str,
        actual: u64,
        expected: u64,
    },
    #[error("checkpoint {section} allocation of {bytes} bytes failed")]
    AllocationFailed { section: &'static str, bytes: usize },
    #[error("checkpoint payload digest mismatch")]
    DigestMismatch,
    #[error("checkpoint has trailing bytes")]
    TrailingBytes,
    #[error("checkpoint snapshot validation: {0}")]
    Snapshot(#[from] SnapshotValidationError),
}

#[derive(Clone, Copy, Debug)]
struct WireLayout {
    flags: u64,
    prefix_count: u64,
    pending_token: i32,
    prefix_bytes: u64,
    kv_position_bytes: u64,
    kv_k_bytes: u64,
    kv_v_bytes: u64,
    gdn_conv_bytes: u64,
    gdn_state_bytes: u64,
    logits_bytes: u64,
    payload_bytes: u64,
    record_bytes: u64,
}

impl WireLayout {
    fn derive(
        prefix_count: u64,
        pending_token: Option<i32>,
        has_logits: bool,
        constraints: SnapshotCodecConstraints<'_>,
    ) -> Result<Self, SnapshotCodecError> {
        if prefix_count > constraints.max_context_tokens as u64 {
            return Err(SnapshotCodecError::ContextCapacityExceeded {
                prefix_len: prefix_count,
                max_context_tokens: constraints.max_context_tokens,
            });
        }
        let identity = constraints.expected_identity;
        let prefix_bytes = checked_mul("prefix_bytes", prefix_count, 4)?;
        let kv_position_bytes = checked_mul("kv_position_bytes", identity.n_attn_layers as u64, 8)?;
        let kv_per_layer = checked_mul(
            "kv_per_layer",
            prefix_count,
            identity.kv_bytes_per_token as u64,
        )?;
        let kv_bytes = checked_mul("kv_bytes", identity.n_attn_layers as u64, kv_per_layer)?;
        let gdn_conv_bytes = checked_mul3(
            "gdn_conv_bytes",
            identity.n_gdn_layers as u64,
            identity.gdn_conv_elements_per_layer as u64,
            4,
        )?;
        let gdn_state_bytes = checked_mul3(
            "gdn_state_bytes",
            identity.n_gdn_layers as u64,
            identity.gdn_state_elements_per_layer as u64,
            4,
        )?;
        let logits_bytes = if has_logits {
            checked_mul("logits_bytes", constraints.expected_vocab_size as u64, 4)?
        } else {
            0
        };
        let payload_bytes = checked_sum(
            "payload_bytes",
            &[
                kv_bytes,
                kv_bytes,
                gdn_conv_bytes,
                gdn_state_bytes,
                prefix_bytes,
                kv_position_bytes,
                logits_bytes,
            ],
        )?;
        let record_bytes = checked_sum(
            "record_bytes",
            &[PAYLOAD_OFFSET as u64, payload_bytes, DIGEST_BYTES as u64],
        )?;
        if record_bytes > constraints.max_record_bytes {
            return Err(SnapshotCodecError::RecordBudgetExceeded {
                record_bytes,
                max_record_bytes: constraints.max_record_bytes,
            });
        }
        Ok(Self {
            flags: u64::from(pending_token.is_some()) * FLAG_PENDING_TOKEN
                | u64::from(has_logits) * FLAG_FINAL_LOGITS,
            prefix_count,
            pending_token: pending_token.unwrap_or(0),
            prefix_bytes,
            kv_position_bytes,
            kv_k_bytes: kv_bytes,
            kv_v_bytes: kv_bytes,
            gdn_conv_bytes,
            gdn_state_bytes,
            logits_bytes,
            payload_bytes,
            record_bytes,
        })
    }
}

pub fn encode_snapshot<W: Write>(
    dst: &mut W,
    snapshot: &SessionSnapshot,
    constraints: SnapshotCodecConstraints<'_>,
) -> Result<EncodedSnapshot, SnapshotCodecError> {
    require_little_endian()?;
    if &snapshot.identity != constraints.expected_identity {
        return Err(SnapshotCodecError::IdentityMismatch);
    }
    snapshot.validate_for_restore(
        constraints.expected_identity,
        constraints.max_context_tokens,
        Some(constraints.expected_vocab_size),
    )?;
    let layout = WireLayout::derive(
        snapshot.prefix_len() as u64,
        snapshot.pending_token,
        snapshot.final_logits.is_some(),
        constraints,
    )?;
    require_len(
        "prefix",
        snapshot.prefix_tokens.len(),
        layout.prefix_bytes / 4,
    )?;
    require_len(
        "kv_positions",
        snapshot.kv_n_pos.len(),
        layout.kv_position_bytes / 8,
    )?;
    require_len("kv_k", snapshot.kv_k_arena.len(), layout.kv_k_bytes)?;
    require_len("kv_v", snapshot.kv_v_arena.len(), layout.kv_v_bytes)?;
    require_len(
        "gdn_conv",
        snapshot.gdn_conv_arena.len(),
        layout.gdn_conv_bytes,
    )?;
    require_len(
        "gdn_state",
        snapshot.gdn_state_arena.len(),
        layout.gdn_state_bytes,
    )?;
    if let Some(logits) = snapshot.final_logits.as_ref() {
        require_len("logits", logits.len(), layout.logits_bytes / 4)?;
    }

    let header = build_header(snapshot, layout, constraints.expected_compatibility_id);
    let mut hasher = blake3::Hasher::new();
    write_hashed(dst, &mut hasher, &header)?;
    write_zero_padding(dst, &mut hasher, PAYLOAD_OFFSET - HEADER_BYTES)?;
    write_hashed(dst, &mut hasher, &snapshot.kv_k_arena)?;
    write_hashed(dst, &mut hasher, &snapshot.kv_v_arena)?;
    write_hashed(dst, &mut hasher, &snapshot.gdn_conv_arena)?;
    write_hashed(dst, &mut hasher, &snapshot.gdn_state_arena)?;
    write_hashed(
        dst,
        &mut hasher,
        bytemuck::cast_slice(&snapshot.prefix_tokens),
    )?;
    for &position in &snapshot.kv_n_pos {
        write_hashed(dst, &mut hasher, &(position as u64).to_le_bytes())?;
    }
    if let Some(logits) = snapshot.final_logits.as_ref() {
        write_hashed(dst, &mut hasher, bytemuck::cast_slice(logits))?;
    }
    let digest = *hasher.finalize().as_bytes();
    dst.write_all(&digest)?;
    Ok(EncodedSnapshot {
        record_bytes: layout.record_bytes,
        digest,
    })
}

pub fn decode_snapshot<R: Read>(
    src: &mut R,
    constraints: SnapshotCodecConstraints<'_>,
) -> Result<SessionSnapshot, SnapshotCodecError> {
    require_little_endian()?;
    let mut header = [0u8; HEADER_BYTES];
    src.read_exact(&mut header)?;
    let layout = parse_and_validate_header(&header, constraints)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&header);
    read_zero_padding(src, &mut hasher, PAYLOAD_OFFSET - HEADER_BYTES)?;

    let kv_k_arena = read_hashed_vec(src, &mut hasher, "kv_k", layout.kv_k_bytes)?;
    let kv_v_arena = read_hashed_vec(src, &mut hasher, "kv_v", layout.kv_v_bytes)?;
    let gdn_conv_arena = read_hashed_vec(src, &mut hasher, "gdn_conv", layout.gdn_conv_bytes)?;
    let gdn_state_arena = read_hashed_vec(src, &mut hasher, "gdn_state", layout.gdn_state_bytes)?;
    let prefix_raw = read_hashed_vec(src, &mut hasher, "prefix", layout.prefix_bytes)?;
    let position_raw = read_hashed_vec(src, &mut hasher, "kv_positions", layout.kv_position_bytes)?;
    let logits_raw = read_hashed_vec(src, &mut hasher, "logits", layout.logits_bytes)?;

    let mut expected_digest = [0u8; DIGEST_BYTES];
    src.read_exact(&mut expected_digest)?;
    let actual_digest = *hasher.finalize().as_bytes();
    if actual_digest != expected_digest {
        return Err(SnapshotCodecError::DigestMismatch);
    }
    require_eof(src)?;

    let prefix_tokens = parse_i32_le("prefix", &prefix_raw)?;
    let mut kv_n_pos = Vec::new();
    let position_count = to_usize("kv_positions", layout.kv_position_bytes / 8)?;
    kv_n_pos.try_reserve_exact(position_count).map_err(|_| {
        SnapshotCodecError::AllocationFailed {
            section: "kv_positions",
            bytes: position_count.saturating_mul(std::mem::size_of::<usize>()),
        }
    })?;
    for bytes in position_raw.chunks_exact(8) {
        let position = u64::from_le_bytes(bytes.try_into().expect("eight-byte chunk"));
        kv_n_pos
            .push(usize::try_from(position).map_err(|_| {
                SnapshotCodecError::InvalidHeader("KV position does not fit usize")
            })?);
    }
    let final_logits = if layout.flags & FLAG_FINAL_LOGITS != 0 {
        Some(parse_f32_le("logits", &logits_raw)?)
    } else {
        None
    };
    let snapshot = SessionSnapshot {
        identity: constraints.expected_identity.clone(),
        prefix_tokens,
        pending_token: (layout.flags & FLAG_PENDING_TOKEN != 0).then_some(layout.pending_token),
        kv_n_pos,
        kv_k_arena,
        kv_v_arena,
        gdn_conv_arena,
        gdn_state_arena,
        final_logits,
    };
    snapshot.validate_for_restore(
        constraints.expected_identity,
        constraints.max_context_tokens,
        Some(constraints.expected_vocab_size),
    )?;
    Ok(snapshot)
}

fn build_header(
    snapshot: &SessionSnapshot,
    layout: WireLayout,
    compatibility_id: &[u8; 32],
) -> [u8; HEADER_BYTES] {
    let mut header = [0u8; HEADER_BYTES];
    header[OFF_MAGIC..OFF_MAGIC + MAGIC.len()].copy_from_slice(MAGIC);
    put_u32(&mut header, OFF_VERSION, CODEC_VERSION);
    put_u32(&mut header, OFF_HEADER_BYTES, HEADER_BYTES as u32);
    put_u64(&mut header, OFF_FLAGS, layout.flags);
    put_u64(&mut header, OFF_RECORD_BYTES, layout.record_bytes);
    put_u64(&mut header, OFF_PAYLOAD_OFFSET, PAYLOAD_OFFSET as u64);
    put_u64(&mut header, OFF_PAYLOAD_BYTES, layout.payload_bytes);
    put_u64(&mut header, OFF_PREFIX_COUNT, layout.prefix_count);
    put_i32(&mut header, OFF_PENDING_TOKEN, layout.pending_token);
    put_u32(&mut header, OFF_STATE_ENCODING, STATE_ENCODING_ABI_VERSION);
    put_u64(&mut header, OFF_MODEL_ID, snapshot.identity.model_id);
    put_u64(
        &mut header,
        OFF_TOKENIZER_ID,
        snapshot.identity.tokenizer_id,
    );
    put_u32(
        &mut header,
        OFF_LAYOUT_VERSION,
        snapshot.identity.layout_version,
    );
    put_u32(
        &mut header,
        OFF_ATTN_LAYERS,
        snapshot.identity.n_attn_layers,
    );
    put_u32(&mut header, OFF_GDN_LAYERS, snapshot.identity.n_gdn_layers);
    put_u32(&mut header, OFF_KV_DIM, snapshot.identity.kv_dim_elements);
    put_u32(
        &mut header,
        OFF_KV_BYTES_PER_TOKEN,
        snapshot.identity.kv_bytes_per_token,
    );
    put_u32(
        &mut header,
        OFF_GDN_STATE_ELEMENTS,
        snapshot.identity.gdn_state_elements_per_layer,
    );
    put_u32(
        &mut header,
        OFF_GDN_CONV_ELEMENTS,
        snapshot.identity.gdn_conv_elements_per_layer,
    );
    put_u32(
        &mut header,
        OFF_KV_STORAGE_KIND,
        snapshot.identity.kv_storage_kind as u32,
    );
    put_u64(&mut header, OFF_PREFIX_BYTES, layout.prefix_bytes);
    put_u64(&mut header, OFF_KV_POSITION_BYTES, layout.kv_position_bytes);
    put_u64(&mut header, OFF_KV_K_BYTES, layout.kv_k_bytes);
    put_u64(&mut header, OFF_KV_V_BYTES, layout.kv_v_bytes);
    put_u64(&mut header, OFF_GDN_CONV_BYTES, layout.gdn_conv_bytes);
    put_u64(&mut header, OFF_GDN_STATE_BYTES, layout.gdn_state_bytes);
    put_u64(&mut header, OFF_LOGITS_BYTES, layout.logits_bytes);
    header[OFF_COMPATIBILITY_ID..OFF_COMPATIBILITY_ID + 32].copy_from_slice(compatibility_id);
    header
}

fn parse_and_validate_header(
    header: &[u8; HEADER_BYTES],
    constraints: SnapshotCodecConstraints<'_>,
) -> Result<WireLayout, SnapshotCodecError> {
    if &header[OFF_MAGIC..OFF_MAGIC + MAGIC.len()] != MAGIC {
        return Err(SnapshotCodecError::InvalidHeader("magic"));
    }
    if get_u32(header, OFF_VERSION) != CODEC_VERSION {
        return Err(SnapshotCodecError::InvalidHeader("codec version"));
    }
    if get_u32(header, OFF_HEADER_BYTES) != HEADER_BYTES as u32 {
        return Err(SnapshotCodecError::InvalidHeader("header size"));
    }
    let flags = get_u64(header, OFF_FLAGS);
    if flags & !KNOWN_FLAGS != 0 {
        return Err(SnapshotCodecError::InvalidHeader("unknown flags"));
    }
    if get_u64(header, OFF_PAYLOAD_OFFSET) != PAYLOAD_OFFSET as u64 {
        return Err(SnapshotCodecError::InvalidHeader("payload offset"));
    }
    if get_u32(header, OFF_STATE_ENCODING) != STATE_ENCODING_ABI_VERSION {
        return Err(SnapshotCodecError::InvalidHeader("state encoding"));
    }
    if header[OFF_RESERVED..].iter().any(|&byte| byte != 0) {
        return Err(SnapshotCodecError::InvalidHeader("reserved bytes"));
    }
    if &header[OFF_COMPATIBILITY_ID..OFF_COMPATIBILITY_ID + 32]
        != constraints.expected_compatibility_id
    {
        return Err(SnapshotCodecError::CompatibilityMismatch);
    }
    let declared_record_bytes = get_u64(header, OFF_RECORD_BYTES);
    if declared_record_bytes > constraints.max_record_bytes {
        return Err(SnapshotCodecError::RecordBudgetExceeded {
            record_bytes: declared_record_bytes,
            max_record_bytes: constraints.max_record_bytes,
        });
    }
    let actual_identity = SnapshotIdentity {
        model_id: get_u64(header, OFF_MODEL_ID),
        tokenizer_id: get_u64(header, OFF_TOKENIZER_ID),
        layout_version: get_u32(header, OFF_LAYOUT_VERSION),
        n_attn_layers: get_u32(header, OFF_ATTN_LAYERS),
        n_gdn_layers: get_u32(header, OFF_GDN_LAYERS),
        kv_dim_elements: get_u32(header, OFF_KV_DIM),
        kv_bytes_per_token: get_u32(header, OFF_KV_BYTES_PER_TOKEN),
        kv_storage_kind: parse_kv_storage_kind(get_u32(header, OFF_KV_STORAGE_KIND))?,
        gdn_state_elements_per_layer: get_u32(header, OFF_GDN_STATE_ELEMENTS),
        gdn_conv_elements_per_layer: get_u32(header, OFF_GDN_CONV_ELEMENTS),
    };
    if actual_identity.abi() != constraints.expected_identity.abi() {
        return Err(SnapshotCodecError::IdentityMismatch);
    }
    let pending_token = get_i32(header, OFF_PENDING_TOKEN);
    if flags & FLAG_PENDING_TOKEN == 0 && pending_token != 0 {
        return Err(SnapshotCodecError::InvalidHeader(
            "pending token without flag",
        ));
    }
    if flags & FLAG_PENDING_TOKEN != 0
        && (pending_token < 0 || pending_token as usize >= constraints.expected_vocab_size)
    {
        return Err(SnapshotCodecError::InvalidHeader("pending token range"));
    }
    let derived = WireLayout::derive(
        get_u64(header, OFF_PREFIX_COUNT),
        (flags & FLAG_PENDING_TOKEN != 0).then_some(pending_token),
        flags & FLAG_FINAL_LOGITS != 0,
        constraints,
    )?;
    require_header_u64(header, OFF_RECORD_BYTES, "record", derived.record_bytes)?;
    require_header_u64(header, OFF_PAYLOAD_BYTES, "payload", derived.payload_bytes)?;
    require_header_u64(header, OFF_PREFIX_BYTES, "prefix", derived.prefix_bytes)?;
    require_header_u64(
        header,
        OFF_KV_POSITION_BYTES,
        "kv_positions",
        derived.kv_position_bytes,
    )?;
    require_header_u64(header, OFF_KV_K_BYTES, "kv_k", derived.kv_k_bytes)?;
    require_header_u64(header, OFF_KV_V_BYTES, "kv_v", derived.kv_v_bytes)?;
    require_header_u64(
        header,
        OFF_GDN_CONV_BYTES,
        "gdn_conv",
        derived.gdn_conv_bytes,
    )?;
    require_header_u64(
        header,
        OFF_GDN_STATE_BYTES,
        "gdn_state",
        derived.gdn_state_bytes,
    )?;
    require_header_u64(header, OFF_LOGITS_BYTES, "logits", derived.logits_bytes)?;
    Ok(derived)
}

fn require_header_u64(
    header: &[u8; HEADER_BYTES],
    offset: usize,
    section: &'static str,
    expected: u64,
) -> Result<(), SnapshotCodecError> {
    let actual = get_u64(header, offset);
    if actual != expected {
        return Err(SnapshotCodecError::SectionLength {
            section,
            actual,
            expected,
        });
    }
    Ok(())
}

fn require_len(
    section: &'static str,
    actual: usize,
    expected: u64,
) -> Result<(), SnapshotCodecError> {
    if actual as u64 != expected {
        return Err(SnapshotCodecError::SectionLength {
            section,
            actual: actual as u64,
            expected,
        });
    }
    Ok(())
}

fn checked_mul(field: &'static str, a: u64, b: u64) -> Result<u64, SnapshotCodecError> {
    a.checked_mul(b)
        .ok_or(SnapshotCodecError::ArithmeticOverflow { field })
}

fn checked_mul3(field: &'static str, a: u64, b: u64, c: u64) -> Result<u64, SnapshotCodecError> {
    checked_mul(field, checked_mul(field, a, b)?, c)
}

fn checked_sum(field: &'static str, values: &[u64]) -> Result<u64, SnapshotCodecError> {
    values.iter().try_fold(0u64, |sum, &value| {
        sum.checked_add(value)
            .ok_or(SnapshotCodecError::ArithmeticOverflow { field })
    })
}

fn to_usize(section: &'static str, bytes: u64) -> Result<usize, SnapshotCodecError> {
    usize::try_from(bytes).map_err(|_| SnapshotCodecError::AllocationFailed {
        section,
        bytes: usize::MAX,
    })
}

fn read_hashed_vec<R: Read>(
    src: &mut R,
    hasher: &mut blake3::Hasher,
    section: &'static str,
    bytes: u64,
) -> Result<Vec<u8>, SnapshotCodecError> {
    let bytes = to_usize(section, bytes)?;
    let mut out = Vec::new();
    out.try_reserve_exact(bytes)
        .map_err(|_| SnapshotCodecError::AllocationFailed { section, bytes })?;
    let mut limited = src.take(bytes as u64);
    limited.read_to_end(&mut out)?;
    if out.len() != bytes {
        return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
    }
    hasher.update(&out);
    Ok(out)
}

fn parse_i32_le(section: &'static str, bytes: &[u8]) -> Result<Vec<i32>, SnapshotCodecError> {
    if bytes.len() % 4 != 0 {
        return Err(SnapshotCodecError::InvalidHeader("i32 section alignment"));
    }
    let count = bytes.len() / 4;
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| SnapshotCodecError::AllocationFailed {
            section,
            bytes: bytes.len(),
        })?;
    for chunk in bytes.chunks_exact(4) {
        out.push(i32::from_le_bytes(
            chunk.try_into().expect("four-byte chunk"),
        ));
    }
    Ok(out)
}

fn parse_f32_le(section: &'static str, bytes: &[u8]) -> Result<Vec<f32>, SnapshotCodecError> {
    if bytes.len() % 4 != 0 {
        return Err(SnapshotCodecError::InvalidHeader("f32 section alignment"));
    }
    let count = bytes.len() / 4;
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| SnapshotCodecError::AllocationFailed {
            section,
            bytes: bytes.len(),
        })?;
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_bits(u32::from_le_bytes(
            chunk.try_into().expect("four-byte chunk"),
        )));
    }
    Ok(out)
}

fn write_hashed<W: Write>(
    dst: &mut W,
    hasher: &mut blake3::Hasher,
    bytes: &[u8],
) -> Result<(), SnapshotCodecError> {
    dst.write_all(bytes)?;
    hasher.update(bytes);
    Ok(())
}

fn write_zero_padding<W: Write>(
    dst: &mut W,
    hasher: &mut blake3::Hasher,
    mut bytes: usize,
) -> Result<(), SnapshotCodecError> {
    let zeros = [0u8; 4096];
    while bytes != 0 {
        let count = bytes.min(zeros.len());
        write_hashed(dst, hasher, &zeros[..count])?;
        bytes -= count;
    }
    Ok(())
}

fn read_zero_padding<R: Read>(
    src: &mut R,
    hasher: &mut blake3::Hasher,
    mut bytes: usize,
) -> Result<(), SnapshotCodecError> {
    let mut buf = [0u8; 4096];
    while bytes != 0 {
        let count = bytes.min(buf.len());
        src.read_exact(&mut buf[..count])?;
        if buf[..count].iter().any(|&byte| byte != 0) {
            return Err(SnapshotCodecError::InvalidHeader("alignment padding"));
        }
        hasher.update(&buf[..count]);
        bytes -= count;
    }
    Ok(())
}

fn require_eof<R: Read>(src: &mut R) -> Result<(), SnapshotCodecError> {
    let mut trailing = [0u8; 1];
    loop {
        match src.read(&mut trailing) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err(SnapshotCodecError::TrailingBytes),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
}

fn parse_kv_storage_kind(value: u32) -> Result<SnapshotKvStorageKind, SnapshotCodecError> {
    match value {
        0 => Ok(SnapshotKvStorageKind::None),
        1 => Ok(SnapshotKvStorageKind::F16),
        2 => Ok(SnapshotKvStorageKind::Q8_0),
        _ => Err(SnapshotCodecError::InvalidHeader("KV storage kind")),
    }
}

fn require_little_endian() -> Result<(), SnapshotCodecError> {
    if cfg!(target_endian = "little") && std::mem::size_of::<usize>() == 8 {
        Ok(())
    } else {
        Err(SnapshotCodecError::UnsupportedByteOrder)
    }
}

fn put_u32(header: &mut [u8], offset: usize, value: u32) {
    header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_i32(header: &mut [u8], offset: usize, value: i32) {
    header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(header: &mut [u8], offset: usize, value: u64) {
    header[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u32(header: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(header[offset..offset + 4].try_into().expect("header field"))
}

fn get_i32(header: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(header[offset..offset + 4].try_into().expect("header field"))
}

fn get_u64(header: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(header[offset..offset + 8].try_into().expect("header field"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal_forward::SNAPSHOT_LAYOUT_VERSION;
    use std::io::Cursor;

    const COMPATIBILITY_ID: [u8; 32] = [0x5a; 32];

    fn identity() -> SnapshotIdentity {
        SnapshotIdentity {
            model_id: 11,
            tokenizer_id: 12,
            layout_version: SNAPSHOT_LAYOUT_VERSION,
            n_attn_layers: 2,
            n_gdn_layers: 1,
            kv_dim_elements: 2,
            kv_bytes_per_token: 4,
            kv_storage_kind: SnapshotKvStorageKind::F16,
            gdn_state_elements_per_layer: 4,
            gdn_conv_elements_per_layer: 3,
        }
    }

    fn snapshot(identity: SnapshotIdentity) -> SessionSnapshot {
        SessionSnapshot {
            identity,
            prefix_tokens: vec![1, 2],
            pending_token: Some(3),
            kv_n_pos: vec![2, 2],
            kv_k_arena: (0..16).collect(),
            kv_v_arena: (16..32).collect(),
            gdn_conv_arena: (32..44).collect(),
            gdn_state_arena: (44..60).collect(),
            final_logits: Some(vec![
                -0.0,
                f32::INFINITY,
                f32::from_bits(0x7fc0_1234),
                -3.5,
                9.0,
            ]),
        }
    }

    fn constraints(identity: &SnapshotIdentity) -> SnapshotCodecConstraints<'_> {
        SnapshotCodecConstraints {
            expected_identity: identity,
            expected_compatibility_id: &COMPATIBILITY_ID,
            expected_vocab_size: 5,
            max_context_tokens: 8,
            max_record_bytes: 1 << 20,
        }
    }

    fn encode(snapshot: &SessionSnapshot) -> Vec<u8> {
        let mut out = Vec::new();
        let encoded = encode_snapshot(&mut out, snapshot, constraints(&snapshot.identity))
            .expect("encode fixture");
        assert_eq!(encoded.record_bytes as usize, out.len());
        assert_eq!(encoded.digest, out[out.len() - DIGEST_BYTES..]);
        out
    }

    fn decode(bytes: &[u8], identity: &SnapshotIdentity) -> SessionSnapshot {
        decode_snapshot(&mut Cursor::new(bytes), constraints(identity)).expect("decode fixture")
    }

    fn assert_snapshot_bits_eq(a: &SessionSnapshot, b: &SessionSnapshot) {
        assert_eq!(a.identity, b.identity);
        assert_eq!(a.prefix_tokens, b.prefix_tokens);
        assert_eq!(a.pending_token, b.pending_token);
        assert_eq!(a.kv_n_pos, b.kv_n_pos);
        assert_eq!(a.kv_k_arena, b.kv_k_arena);
        assert_eq!(a.kv_v_arena, b.kv_v_arena);
        assert_eq!(a.gdn_conv_arena, b.gdn_conv_arena);
        assert_eq!(a.gdn_state_arena, b.gdn_state_arena);
        let a_logits = a.final_logits.as_ref().map(|values| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        });
        let b_logits = b.final_logits.as_ref().map(|values| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        });
        assert_eq!(a_logits, b_logits);
    }

    fn refresh_digest(record: &mut [u8]) {
        let digest_offset = record.len() - DIGEST_BYTES;
        let digest = blake3::hash(&record[..digest_offset]);
        record[digest_offset..].copy_from_slice(digest.as_bytes());
    }

    #[test]
    fn checkpoint_codec_round_trips_bits_and_is_canonical() {
        let snapshot = snapshot(identity());
        let first = encode(&snapshot);
        let second = encode(&snapshot);
        assert_eq!(first, second);
        assert_eq!(&first[OFF_MAGIC..OFF_MAGIC + 8], MAGIC);
        assert_eq!(get_u32(&first, OFF_VERSION), CODEC_VERSION);
        assert_eq!(get_u32(&first, OFF_HEADER_BYTES), HEADER_BYTES as u32);
        assert_eq!(get_u64(&first, OFF_FLAGS), 3);
        assert_eq!(get_u64(&first, OFF_RECORD_BYTES), first.len() as u64);
        assert_eq!(
            first.len() as u64,
            snapshot.n_bytes() - 4 + SNAPSHOT_RECORD_FIXED_BYTES
        );
        assert_eq!(get_u64(&first, OFF_PAYLOAD_OFFSET), PAYLOAD_OFFSET as u64);
        assert_eq!(get_u64(&first, OFF_PREFIX_COUNT), 2);
        assert_eq!(get_i32(&first, OFF_PENDING_TOKEN), 3);
        assert_eq!(get_u32(&first, OFF_KV_STORAGE_KIND), 1);
        assert_eq!(get_u64(&first, OFF_PREFIX_BYTES), 8);
        assert_eq!(get_u64(&first, OFF_KV_POSITION_BYTES), 16);
        assert_eq!(get_u64(&first, OFF_KV_K_BYTES), 16);
        assert_eq!(
            &first[OFF_COMPATIBILITY_ID..OFF_RESERVED],
            &COMPATIBILITY_ID
        );
        assert!(
            first[OFF_RESERVED..HEADER_BYTES]
                .iter()
                .all(|&byte| byte == 0)
        );
        assert_snapshot_bits_eq(&snapshot, &decode(&first, &snapshot.identity));
    }

    #[test]
    fn checkpoint_codec_round_trips_state_families_and_capabilities() {
        let base = snapshot(identity());

        let mut attention = base.clone();
        attention.identity.n_gdn_layers = 0;
        attention.gdn_conv_arena.clear();
        attention.gdn_state_arena.clear();
        assert_snapshot_bits_eq(
            &attention,
            &decode(&encode(&attention), &attention.identity),
        );

        let mut gdn = base;
        gdn.identity.n_attn_layers = 0;
        gdn.identity.kv_dim_elements = 0;
        gdn.identity.kv_bytes_per_token = 0;
        gdn.identity.kv_storage_kind = SnapshotKvStorageKind::None;
        gdn.kv_n_pos.clear();
        gdn.kv_k_arena.clear();
        gdn.kv_v_arena.clear();
        gdn.pending_token = None;
        gdn.final_logits = None;
        assert_snapshot_bits_eq(&gdn, &decode(&encode(&gdn), &gdn.identity));

        let mut q8 = snapshot(identity());
        q8.identity.kv_storage_kind = SnapshotKvStorageKind::Q8_0;
        assert_snapshot_bits_eq(&q8, &decode(&encode(&q8), &q8.identity));
    }

    #[test]
    fn checkpoint_codec_rejects_noncanonical_headers_before_payload() {
        let snapshot = snapshot(identity());
        let encoded = encode(&snapshot);
        for (offset, value) in [
            (OFF_MAGIC, 0xff),
            (OFF_VERSION, 0xff),
            (OFF_FLAGS, 0x80),
            (OFF_RESERVED, 1),
            (OFF_COMPATIBILITY_ID, 0),
            (OFF_KV_STORAGE_KIND, 0xff),
        ] {
            let mut corrupt = encoded.clone();
            corrupt[offset] = value;
            assert!(
                decode_snapshot(&mut Cursor::new(&corrupt), constraints(&snapshot.identity))
                    .is_err(),
                "offset {offset:#x}"
            );
        }

        let mut wrong_length = encoded;
        let value = get_u64(&wrong_length, OFF_KV_K_BYTES) + 1;
        put_u64(&mut wrong_length, OFF_KV_K_BYTES, value);
        assert!(matches!(
            decode_snapshot(
                &mut Cursor::new(&wrong_length),
                constraints(&snapshot.identity)
            ),
            Err(SnapshotCodecError::SectionLength {
                section: "kv_k",
                ..
            })
        ));

        let mut absent_pending = snapshot.clone();
        absent_pending.pending_token = None;
        absent_pending.final_logits = None;
        let mut noncanonical_pending = encode(&absent_pending);
        put_i32(&mut noncanonical_pending, OFF_PENDING_TOKEN, 1);
        assert!(matches!(
            decode_snapshot(
                &mut Cursor::new(&noncanonical_pending),
                constraints(&snapshot.identity)
            ),
            Err(SnapshotCodecError::InvalidHeader(
                "pending token without flag"
            ))
        ));

        let mut logits_flag_mismatch = encode(&snapshot);
        put_u64(&mut logits_flag_mismatch, OFF_FLAGS, FLAG_PENDING_TOKEN);
        assert!(matches!(
            decode_snapshot(
                &mut Cursor::new(&logits_flag_mismatch),
                constraints(&snapshot.identity)
            ),
            Err(SnapshotCodecError::SectionLength {
                section: "record",
                ..
            })
        ));
    }

    #[test]
    fn checkpoint_codec_enforces_budget_integrity_and_record_boundary() {
        let snapshot = snapshot(identity());
        let encoded = encode(&snapshot);
        let mut exact = constraints(&snapshot.identity);
        exact.max_record_bytes = encoded.len() as u64;
        decode_snapshot(&mut Cursor::new(&encoded), exact).expect("exact budget");

        let mut too_small = exact;
        too_small.max_record_bytes -= 1;
        assert!(matches!(
            decode_snapshot(&mut Cursor::new(&encoded), too_small),
            Err(SnapshotCodecError::RecordBudgetExceeded { .. })
        ));

        let mut short_context = exact;
        short_context.max_context_tokens = 1;
        assert!(matches!(
            decode_snapshot(&mut Cursor::new(&encoded), short_context),
            Err(SnapshotCodecError::ContextCapacityExceeded { .. })
        ));

        let mut corrupt = encoded.clone();
        corrupt[PAYLOAD_OFFSET] ^= 1;
        assert!(matches!(
            decode_snapshot(&mut Cursor::new(&corrupt), constraints(&snapshot.identity)),
            Err(SnapshotCodecError::DigestMismatch)
        ));

        let mut corrupt_padding = encoded.clone();
        corrupt_padding[PAYLOAD_OFFSET - 1] = 1;
        assert!(matches!(
            decode_snapshot(
                &mut Cursor::new(&corrupt_padding),
                constraints(&snapshot.identity)
            ),
            Err(SnapshotCodecError::InvalidHeader("alignment padding"))
        ));

        let mut corrupt_digest = encoded.clone();
        let last = corrupt_digest.len() - 1;
        corrupt_digest[last] ^= 1;
        assert!(matches!(
            decode_snapshot(
                &mut Cursor::new(&corrupt_digest),
                constraints(&snapshot.identity)
            ),
            Err(SnapshotCodecError::DigestMismatch)
        ));

        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(matches!(
            decode_snapshot(&mut Cursor::new(&trailing), constraints(&snapshot.identity)),
            Err(SnapshotCodecError::TrailingBytes)
        ));

        for end in [0, HEADER_BYTES - 1, PAYLOAD_OFFSET, encoded.len() - 1] {
            assert!(
                decode_snapshot(
                    &mut Cursor::new(&encoded[..end]),
                    constraints(&snapshot.identity)
                )
                .is_err(),
                "truncation at {end}"
            );
        }
    }

    #[test]
    fn checkpoint_codec_validation_backstop_survives_recomputed_digest() {
        let snapshot = snapshot(identity());
        let mut encoded = encode(&snapshot);
        let prefix_offset = PAYLOAD_OFFSET
            + snapshot.kv_k_arena.len()
            + snapshot.kv_v_arena.len()
            + snapshot.gdn_conv_arena.len()
            + snapshot.gdn_state_arena.len();
        encoded[prefix_offset..prefix_offset + 4].copy_from_slice(&(-1i32).to_le_bytes());
        refresh_digest(&mut encoded);
        assert!(matches!(
            decode_snapshot(&mut Cursor::new(&encoded), constraints(&snapshot.identity)),
            Err(SnapshotCodecError::Snapshot(
                SnapshotValidationError::TokenOutOfRange { .. }
            ))
        ));

        let mut encoded = encode(&snapshot);
        put_i32(&mut encoded, OFF_PENDING_TOKEN, 5);
        refresh_digest(&mut encoded);
        assert!(matches!(
            decode_snapshot(&mut Cursor::new(&encoded), constraints(&snapshot.identity)),
            Err(SnapshotCodecError::InvalidHeader("pending token range"))
        ));
    }

    #[test]
    fn checkpoint_codec_uses_strong_identity_plus_structural_abi() {
        let snapshot = snapshot(identity());
        let mut renamed = encode(&snapshot);
        put_u64(&mut renamed, OFF_MODEL_ID, 99);
        put_u64(&mut renamed, OFF_TOKENIZER_ID, 100);
        refresh_digest(&mut renamed);
        let restored = decode(&renamed, &snapshot.identity);
        assert_snapshot_bits_eq(&snapshot, &restored);

        let mut wrong_abi = encode(&snapshot);
        put_u32(
            &mut wrong_abi,
            OFF_LAYOUT_VERSION,
            snapshot.identity.layout_version + 1,
        );
        refresh_digest(&mut wrong_abi);
        assert!(matches!(
            decode_snapshot(
                &mut Cursor::new(&wrong_abi),
                constraints(&snapshot.identity)
            ),
            Err(SnapshotCodecError::IdentityMismatch)
        ));
    }

    struct OneByteReader<R>(R);

    impl<R: Read> Read for OneByteReader<R> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let count = buf.len().min(1);
            self.0.read(&mut buf[..count])
        }
    }

    #[test]
    fn checkpoint_codec_accepts_arbitrary_short_reads() {
        let snapshot = snapshot(identity());
        let encoded = encode(&snapshot);
        let mut reader = OneByteReader(Cursor::new(encoded));
        let restored = decode_snapshot(&mut reader, constraints(&snapshot.identity)).unwrap();
        assert_snapshot_bits_eq(&snapshot, &restored);
    }

    struct InterruptOnceAtEof {
        inner: Cursor<Vec<u8>>,
        interrupted: bool,
    }

    impl Read for InterruptOnceAtEof {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.inner.position() == self.inner.get_ref().len() as u64 && !self.interrupted {
                self.interrupted = true;
                return Err(io::Error::from(io::ErrorKind::Interrupted));
            }
            self.inner.read(buf)
        }
    }

    #[test]
    fn checkpoint_codec_retries_interrupted_eof_probe() {
        let snapshot = snapshot(identity());
        let mut reader = InterruptOnceAtEof {
            inner: Cursor::new(encode(&snapshot)),
            interrupted: false,
        };
        let restored = decode_snapshot(&mut reader, constraints(&snapshot.identity)).unwrap();
        assert!(reader.interrupted);
        assert_snapshot_bits_eq(&snapshot, &restored);
    }

    #[test]
    fn checkpoint_codec_preserves_pending_token_at_consumed_capacity() {
        let snapshot = snapshot(identity());
        let mut exact_capacity = constraints(&snapshot.identity);
        exact_capacity.max_context_tokens = snapshot.prefix_len();
        let mut encoded = Vec::new();
        encode_snapshot(&mut encoded, &snapshot, exact_capacity).expect("encode at capacity");
        let restored = decode_snapshot(&mut Cursor::new(encoded), exact_capacity).unwrap();
        assert_snapshot_bits_eq(&snapshot, &restored);
    }

    struct FailingWriter {
        remaining: usize,
    }

    impl Write for FailingWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                return Err(io::Error::other("injected writer failure"));
            }
            let count = self.remaining.min(buf.len());
            self.remaining -= count;
            Ok(count)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn checkpoint_codec_propagates_writer_failures() {
        let snapshot = snapshot(identity());
        let record_bytes = encode(&snapshot).len();
        for remaining in [
            0,
            HEADER_BYTES / 2,
            PAYLOAD_OFFSET,
            PAYLOAD_OFFSET + 10,
            record_bytes - DIGEST_BYTES / 2,
        ] {
            let mut writer = FailingWriter { remaining };
            assert!(
                encode_snapshot(&mut writer, &snapshot, constraints(&snapshot.identity)).is_err(),
                "writer limit {remaining}"
            );
        }
    }

    #[test]
    fn checkpoint_codec_arithmetic_is_checked() {
        assert!(matches!(
            checked_mul("test", u64::MAX, 2),
            Err(SnapshotCodecError::ArithmeticOverflow { field: "test" })
        ));
        assert!(matches!(
            checked_sum("test", &[u64::MAX, 1]),
            Err(SnapshotCodecError::ArithmeticOverflow { field: "test" })
        ));
    }
}
