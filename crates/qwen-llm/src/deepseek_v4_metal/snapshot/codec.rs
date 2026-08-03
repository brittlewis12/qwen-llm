//! Bounded little-endian wire format for typed DeepSeek V4 causal state.
//!
//! Decode validates all record geometry before allocation. It then retains the
//! bounded byte sections while constructing typed bit arenas, so peak host
//! allocation is approximately twice the payload rather than one payload.

use super::*;
use std::io::{self, Read, Write};

const MAGIC: &[u8; 8] = b"DS4CKP\0\0";
const CODEC_VERSION: u32 = 1;
const HEADER_BYTES: usize = 256;
pub(super) const PAYLOAD_OFFSET: usize = 16 * 1024;
const DIGEST_BYTES: usize = 32;
const STATE_WORD_ENCODING: u32 = 1;
const FLAG_SOURCE_OBSERVATION: u64 = 1 << 0;
const KNOWN_FLAGS: u64 = FLAG_SOURCE_OBSERVATION;

const OFF_MAGIC: usize = 0x00;
const OFF_VERSION: usize = 0x08;
const OFF_HEADER_BYTES: usize = 0x0c;
const OFF_FLAGS: usize = 0x10;
const OFF_RECORD_BYTES: usize = 0x18;
const OFF_PAYLOAD_OFFSET: usize = 0x20;
const OFF_PAYLOAD_BYTES: usize = 0x28;
const OFF_NEXT_POSITION: usize = 0x30;
const OFF_STATE_WORD_ENCODING: usize = 0x34;
const OFF_PREFIX_COUNT: usize = 0x38;
const OFF_PREFIX_BYTES: usize = 0x40;
const OFF_RAW_BYTES: usize = 0x48;
const OFF_COMPRESSOR_BYTES: usize = 0x50;
const OFF_PUBLISHED_BYTES: usize = 0x58;
const OFF_MODEL_CONTENT_ID: usize = 0x60;
const OFF_COMPATIBILITY_DIGEST: usize = 0x80;
const OFF_PREFIX_DIGEST: usize = 0xa0;
const OFF_CAUSAL_DIGEST: usize = 0xc0;
const OFF_RESERVED: usize = 0xe0;

#[derive(Clone, Copy, Debug)]
pub struct DeepSeekV4SnapshotCodecConstraints<'a> {
    pub config: &'a DeepSeekV4Config,
    pub session_capacity: DeepSeekV4SessionCapacity,
    pub expected_model_content_id: DeepSeekV4ModelContentId,
    pub max_record_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4EncodedSnapshot {
    pub record_bytes: u64,
    pub digest: [u8; DIGEST_BYTES],
}

#[derive(Debug, thiserror::Error)]
pub enum DeepSeekV4SnapshotCodecError {
    #[error("DeepSeek V4 checkpoint I/O: {0}")]
    Io(#[from] io::Error),
    #[error("unsupported DeepSeek V4 checkpoint host byte order")]
    UnsupportedByteOrder,
    #[error("invalid DeepSeek V4 checkpoint header: {0}")]
    InvalidHeader(&'static str),
    #[error("DeepSeek V4 checkpoint model-content identity mismatch")]
    ModelIdentityMismatch,
    #[error("DeepSeek V4 checkpoint compatibility digest mismatch")]
    CompatibilityMismatch,
    #[error("DeepSeek V4 checkpoint {field} arithmetic overflow")]
    ArithmeticOverflow { field: &'static str },
    #[error("DeepSeek V4 checkpoint record size {record_bytes} exceeds budget {max_record_bytes}")]
    RecordBudgetExceeded {
        record_bytes: u64,
        max_record_bytes: u64,
    },
    #[error("DeepSeek V4 checkpoint {section} length {actual} != expected {expected}")]
    SectionLength {
        section: &'static str,
        actual: u64,
        expected: u64,
    },
    #[error("DeepSeek V4 checkpoint {section} allocation of {bytes} bytes failed")]
    AllocationFailed { section: &'static str, bytes: usize },
    #[error("DeepSeek V4 checkpoint record digest mismatch")]
    DigestMismatch,
    #[error("DeepSeek V4 checkpoint has trailing bytes")]
    TrailingBytes,
    #[error("invalid DeepSeek V4 causal snapshot: {0}")]
    Snapshot(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WireLayout {
    flags: u64,
    next_position: u32,
    prefix_bytes: u64,
    raw_bytes: u64,
    compressor_bytes: u64,
    published_bytes: u64,
    payload_bytes: u64,
    record_bytes: u64,
}

impl WireLayout {
    fn derive(
        next_position: u32,
        source_observation: DeepSeekV4SnapshotObservation,
        constraints: DeepSeekV4SnapshotCodecConstraints<'_>,
    ) -> Result<Self, DeepSeekV4SnapshotCodecError> {
        let geometry = snapshot_geometry(
            constraints.config,
            constraints.session_capacity,
            next_position,
        )
        .map_err(snapshot_codec_error)?;
        let prefix_bytes = checked_mul("prefix bytes", u64::from(next_position), 4)?;
        let raw_bytes = checked_mul("raw bytes", geometry.raw_elements as u64, 2)?;
        let compressor_bytes =
            checked_mul("compressor bytes", geometry.compressor_elements as u64, 4)?;
        let published_bytes =
            checked_mul("published bytes", geometry.published_elements as u64, 2)?;
        let payload_bytes = checked_sum(
            "payload bytes",
            &[prefix_bytes, raw_bytes, compressor_bytes, published_bytes],
        )?;
        let record_bytes = checked_sum(
            "record bytes",
            &[PAYLOAD_OFFSET as u64, payload_bytes, DIGEST_BYTES as u64],
        )?;
        if record_bytes > constraints.max_record_bytes {
            return Err(DeepSeekV4SnapshotCodecError::RecordBudgetExceeded {
                record_bytes,
                max_record_bytes: constraints.max_record_bytes,
            });
        }
        Ok(Self {
            flags: u64::from(source_observation == DeepSeekV4SnapshotObservation::Available)
                * FLAG_SOURCE_OBSERVATION,
            next_position,
            prefix_bytes,
            raw_bytes,
            compressor_bytes,
            published_bytes,
            payload_bytes,
            record_bytes,
        })
    }
}

pub fn encode_causal_snapshot<W: Write>(
    dst: &mut W,
    snapshot: &DeepSeekV4CausalSnapshot,
    constraints: DeepSeekV4SnapshotCodecConstraints<'_>,
) -> Result<DeepSeekV4EncodedSnapshot, DeepSeekV4SnapshotCodecError> {
    require_little_endian()?;
    validate_snapshot(
        snapshot,
        constraints.config,
        constraints.session_capacity,
        constraints.expected_model_content_id,
    )
    .map_err(snapshot_codec_error)?;
    let layout = WireLayout::derive(
        snapshot.next_position,
        snapshot.source_observation,
        constraints,
    )?;
    require_len(
        "prefix",
        snapshot.prefix_tokens.len() as u64 * 4,
        layout.prefix_bytes,
    )?;
    require_len(
        "raw",
        snapshot.raw_f16_bits.len() as u64 * 2,
        layout.raw_bytes,
    )?;
    require_len(
        "compressor",
        snapshot.compressor_f32_bits.len() as u64 * 4,
        layout.compressor_bytes,
    )?;
    require_len(
        "published",
        snapshot.published_f16_bits.len() as u64 * 2,
        layout.published_bytes,
    )?;

    let header = build_header(snapshot, layout);
    let mut hasher = blake3::Hasher::new();
    write_hashed(dst, &mut hasher, &header)?;
    write_zero_padding(dst, &mut hasher, PAYLOAD_OFFSET - HEADER_BYTES)?;
    write_hashed(
        dst,
        &mut hasher,
        bytemuck::cast_slice(&snapshot.prefix_tokens),
    )?;
    write_hashed(
        dst,
        &mut hasher,
        bytemuck::cast_slice(&snapshot.raw_f16_bits),
    )?;
    write_hashed(
        dst,
        &mut hasher,
        bytemuck::cast_slice(&snapshot.compressor_f32_bits),
    )?;
    write_hashed(
        dst,
        &mut hasher,
        bytemuck::cast_slice(&snapshot.published_f16_bits),
    )?;
    let digest = *hasher.finalize().as_bytes();
    dst.write_all(&digest)?;
    Ok(DeepSeekV4EncodedSnapshot {
        record_bytes: layout.record_bytes,
        digest,
    })
}

pub fn decode_causal_snapshot<R: Read>(
    src: &mut R,
    constraints: DeepSeekV4SnapshotCodecConstraints<'_>,
) -> Result<DeepSeekV4CausalSnapshot, DeepSeekV4SnapshotCodecError> {
    require_little_endian()?;
    let mut header = [0u8; HEADER_BYTES];
    src.read_exact(&mut header)?;
    let (layout, header_identity) = parse_and_validate_header(&header, constraints)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&header);
    read_zero_padding(src, &mut hasher, PAYLOAD_OFFSET - HEADER_BYTES)?;
    let prefix_raw = read_hashed_vec(src, &mut hasher, "prefix", layout.prefix_bytes)?;
    let raw_raw = read_hashed_vec(src, &mut hasher, "raw", layout.raw_bytes)?;
    let compressor_raw = read_hashed_vec(src, &mut hasher, "compressor", layout.compressor_bytes)?;
    let published_raw = read_hashed_vec(src, &mut hasher, "published", layout.published_bytes)?;
    let mut expected_record_digest = [0u8; DIGEST_BYTES];
    src.read_exact(&mut expected_record_digest)?;
    if *hasher.finalize().as_bytes() != expected_record_digest {
        return Err(DeepSeekV4SnapshotCodecError::DigestMismatch);
    }
    require_eof(src)?;

    let snapshot = DeepSeekV4CausalSnapshot {
        model_content_id: constraints.expected_model_content_id,
        compatibility_digest: snapshot_compatibility_digest(
            constraints.expected_model_content_id,
            constraints.config,
        ),
        next_position: layout.next_position,
        prefix_tokens: parse_u32_le("prefix", &prefix_raw)?.into_boxed_slice(),
        prefix_digest: header_identity.prefix_digest,
        source_observation: if layout.flags & FLAG_SOURCE_OBSERVATION != 0 {
            DeepSeekV4SnapshotObservation::Available
        } else {
            DeepSeekV4SnapshotObservation::Unavailable
        },
        raw_f16_bits: parse_u16_le("raw", &raw_raw)?.into_boxed_slice(),
        compressor_f32_bits: parse_u32_le("compressor", &compressor_raw)?.into_boxed_slice(),
        published_f16_bits: parse_u16_le("published", &published_raw)?.into_boxed_slice(),
        causal_digest: header_identity.causal_digest,
    };
    validate_snapshot(
        &snapshot,
        constraints.config,
        constraints.session_capacity,
        constraints.expected_model_content_id,
    )
    .map_err(snapshot_codec_error)?;
    Ok(snapshot)
}

#[derive(Clone, Copy)]
struct HeaderIdentity {
    prefix_digest: [u8; 32],
    causal_digest: [u8; 32],
}

fn build_header(snapshot: &DeepSeekV4CausalSnapshot, layout: WireLayout) -> [u8; HEADER_BYTES] {
    let mut header = [0u8; HEADER_BYTES];
    header[OFF_MAGIC..OFF_MAGIC + MAGIC.len()].copy_from_slice(MAGIC);
    put_u32(&mut header, OFF_VERSION, CODEC_VERSION);
    put_u32(&mut header, OFF_HEADER_BYTES, HEADER_BYTES as u32);
    put_u64(&mut header, OFF_FLAGS, layout.flags);
    put_u64(&mut header, OFF_RECORD_BYTES, layout.record_bytes);
    put_u64(&mut header, OFF_PAYLOAD_OFFSET, PAYLOAD_OFFSET as u64);
    put_u64(&mut header, OFF_PAYLOAD_BYTES, layout.payload_bytes);
    put_u32(&mut header, OFF_NEXT_POSITION, layout.next_position);
    put_u32(&mut header, OFF_STATE_WORD_ENCODING, STATE_WORD_ENCODING);
    put_u64(
        &mut header,
        OFF_PREFIX_COUNT,
        u64::from(layout.next_position),
    );
    put_u64(&mut header, OFF_PREFIX_BYTES, layout.prefix_bytes);
    put_u64(&mut header, OFF_RAW_BYTES, layout.raw_bytes);
    put_u64(&mut header, OFF_COMPRESSOR_BYTES, layout.compressor_bytes);
    put_u64(&mut header, OFF_PUBLISHED_BYTES, layout.published_bytes);
    header[OFF_MODEL_CONTENT_ID..OFF_MODEL_CONTENT_ID + 32]
        .copy_from_slice(snapshot.model_content_id.as_bytes());
    header[OFF_COMPATIBILITY_DIGEST..OFF_COMPATIBILITY_DIGEST + 32]
        .copy_from_slice(snapshot.compatibility_digest.as_bytes());
    header[OFF_PREFIX_DIGEST..OFF_PREFIX_DIGEST + 32].copy_from_slice(&snapshot.prefix_digest);
    header[OFF_CAUSAL_DIGEST..OFF_CAUSAL_DIGEST + 32].copy_from_slice(&snapshot.causal_digest);
    header
}

fn parse_and_validate_header(
    header: &[u8; HEADER_BYTES],
    constraints: DeepSeekV4SnapshotCodecConstraints<'_>,
) -> Result<(WireLayout, HeaderIdentity), DeepSeekV4SnapshotCodecError> {
    if &header[OFF_MAGIC..OFF_MAGIC + MAGIC.len()] != MAGIC {
        return Err(DeepSeekV4SnapshotCodecError::InvalidHeader("magic"));
    }
    if get_u32(header, OFF_VERSION) != CODEC_VERSION {
        return Err(DeepSeekV4SnapshotCodecError::InvalidHeader("codec version"));
    }
    if get_u32(header, OFF_HEADER_BYTES) != HEADER_BYTES as u32 {
        return Err(DeepSeekV4SnapshotCodecError::InvalidHeader("header size"));
    }
    let flags = get_u64(header, OFF_FLAGS);
    if flags & !KNOWN_FLAGS != 0 {
        return Err(DeepSeekV4SnapshotCodecError::InvalidHeader("unknown flags"));
    }
    if get_u64(header, OFF_PAYLOAD_OFFSET) != PAYLOAD_OFFSET as u64 {
        return Err(DeepSeekV4SnapshotCodecError::InvalidHeader(
            "payload offset",
        ));
    }
    if get_u32(header, OFF_STATE_WORD_ENCODING) != STATE_WORD_ENCODING {
        return Err(DeepSeekV4SnapshotCodecError::InvalidHeader(
            "state word encoding",
        ));
    }
    if header[OFF_RESERVED..].iter().any(|&byte| byte != 0) {
        return Err(DeepSeekV4SnapshotCodecError::InvalidHeader(
            "reserved bytes",
        ));
    }
    if &header[OFF_MODEL_CONTENT_ID..OFF_MODEL_CONTENT_ID + 32]
        != constraints.expected_model_content_id.as_bytes()
    {
        return Err(DeepSeekV4SnapshotCodecError::ModelIdentityMismatch);
    }
    let expected_compatibility =
        snapshot_compatibility_digest(constraints.expected_model_content_id, constraints.config);
    if &header[OFF_COMPATIBILITY_DIGEST..OFF_COMPATIBILITY_DIGEST + 32]
        != expected_compatibility.as_bytes()
    {
        return Err(DeepSeekV4SnapshotCodecError::CompatibilityMismatch);
    }
    let declared_record_bytes = get_u64(header, OFF_RECORD_BYTES);
    if declared_record_bytes > constraints.max_record_bytes {
        return Err(DeepSeekV4SnapshotCodecError::RecordBudgetExceeded {
            record_bytes: declared_record_bytes,
            max_record_bytes: constraints.max_record_bytes,
        });
    }
    let next_position = get_u32(header, OFF_NEXT_POSITION);
    if get_u64(header, OFF_PREFIX_COUNT) != u64::from(next_position) {
        return Err(DeepSeekV4SnapshotCodecError::InvalidHeader("prefix count"));
    }
    let source_observation = if flags & FLAG_SOURCE_OBSERVATION != 0 {
        DeepSeekV4SnapshotObservation::Available
    } else {
        DeepSeekV4SnapshotObservation::Unavailable
    };
    let derived = WireLayout::derive(next_position, source_observation, constraints)?;
    require_header_u64(header, OFF_RECORD_BYTES, "record", derived.record_bytes)?;
    require_header_u64(header, OFF_PAYLOAD_BYTES, "payload", derived.payload_bytes)?;
    require_header_u64(header, OFF_PREFIX_BYTES, "prefix", derived.prefix_bytes)?;
    require_header_u64(header, OFF_RAW_BYTES, "raw", derived.raw_bytes)?;
    require_header_u64(
        header,
        OFF_COMPRESSOR_BYTES,
        "compressor",
        derived.compressor_bytes,
    )?;
    require_header_u64(
        header,
        OFF_PUBLISHED_BYTES,
        "published",
        derived.published_bytes,
    )?;
    let mut prefix_digest = [0u8; 32];
    prefix_digest.copy_from_slice(&header[OFF_PREFIX_DIGEST..OFF_PREFIX_DIGEST + 32]);
    let mut causal_digest = [0u8; 32];
    causal_digest.copy_from_slice(&header[OFF_CAUSAL_DIGEST..OFF_CAUSAL_DIGEST + 32]);
    Ok((
        derived,
        HeaderIdentity {
            prefix_digest,
            causal_digest,
        },
    ))
}

fn snapshot_codec_error(error: DeepSeekV4MetalError) -> DeepSeekV4SnapshotCodecError {
    DeepSeekV4SnapshotCodecError::Snapshot(error.to_string())
}

fn checked_mul(
    field: &'static str,
    left: u64,
    right: u64,
) -> Result<u64, DeepSeekV4SnapshotCodecError> {
    left.checked_mul(right)
        .ok_or(DeepSeekV4SnapshotCodecError::ArithmeticOverflow { field })
}

fn checked_sum(field: &'static str, values: &[u64]) -> Result<u64, DeepSeekV4SnapshotCodecError> {
    values.iter().try_fold(0u64, |sum, value| {
        sum.checked_add(*value)
            .ok_or(DeepSeekV4SnapshotCodecError::ArithmeticOverflow { field })
    })
}

fn require_len(
    section: &'static str,
    actual: u64,
    expected: u64,
) -> Result<(), DeepSeekV4SnapshotCodecError> {
    if actual != expected {
        return Err(DeepSeekV4SnapshotCodecError::SectionLength {
            section,
            actual,
            expected,
        });
    }
    Ok(())
}

fn require_header_u64(
    header: &[u8; HEADER_BYTES],
    offset: usize,
    section: &'static str,
    expected: u64,
) -> Result<(), DeepSeekV4SnapshotCodecError> {
    require_len(section, get_u64(header, offset), expected)
}

fn read_hashed_vec<R: Read>(
    src: &mut R,
    hasher: &mut blake3::Hasher,
    section: &'static str,
    bytes: u64,
) -> Result<Vec<u8>, DeepSeekV4SnapshotCodecError> {
    let bytes =
        usize::try_from(bytes).map_err(|_| DeepSeekV4SnapshotCodecError::AllocationFailed {
            section,
            bytes: usize::MAX,
        })?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes)
        .map_err(|_| DeepSeekV4SnapshotCodecError::AllocationFailed { section, bytes })?;
    let mut limited = src.take(bytes as u64);
    limited.read_to_end(&mut output)?;
    if output.len() != bytes {
        return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
    }
    hasher.update(&output);
    Ok(output)
}

fn parse_u16_le(
    section: &'static str,
    bytes: &[u8],
) -> Result<Vec<u16>, DeepSeekV4SnapshotCodecError> {
    parse_words(section, bytes, 2, |chunk| {
        u16::from_le_bytes(chunk.try_into().expect("two-byte chunk"))
    })
}

fn parse_u32_le(
    section: &'static str,
    bytes: &[u8],
) -> Result<Vec<u32>, DeepSeekV4SnapshotCodecError> {
    parse_words(section, bytes, 4, |chunk| {
        u32::from_le_bytes(chunk.try_into().expect("four-byte chunk"))
    })
}

fn parse_words<T>(
    section: &'static str,
    bytes: &[u8],
    width: usize,
    parse: impl Fn(&[u8]) -> T,
) -> Result<Vec<T>, DeepSeekV4SnapshotCodecError> {
    if !bytes.len().is_multiple_of(width) {
        return Err(DeepSeekV4SnapshotCodecError::InvalidHeader(
            "section word alignment",
        ));
    }
    let count = bytes.len() / width;
    let mut output = Vec::new();
    output.try_reserve_exact(count).map_err(|_| {
        DeepSeekV4SnapshotCodecError::AllocationFailed {
            section,
            bytes: bytes.len(),
        }
    })?;
    output.extend(bytes.chunks_exact(width).map(parse));
    Ok(output)
}

fn write_hashed<W: Write>(
    dst: &mut W,
    hasher: &mut blake3::Hasher,
    bytes: &[u8],
) -> Result<(), io::Error> {
    dst.write_all(bytes)?;
    hasher.update(bytes);
    Ok(())
}

fn write_zero_padding<W: Write>(
    dst: &mut W,
    hasher: &mut blake3::Hasher,
    mut bytes: usize,
) -> Result<(), io::Error> {
    const ZEROS: [u8; 4096] = [0; 4096];
    while bytes > 0 {
        let count = bytes.min(ZEROS.len());
        write_hashed(dst, hasher, &ZEROS[..count])?;
        bytes -= count;
    }
    Ok(())
}

fn read_zero_padding<R: Read>(
    src: &mut R,
    hasher: &mut blake3::Hasher,
    mut bytes: usize,
) -> Result<(), DeepSeekV4SnapshotCodecError> {
    let mut buffer = [0u8; 4096];
    while bytes > 0 {
        let count = bytes.min(buffer.len());
        src.read_exact(&mut buffer[..count])?;
        if buffer[..count].iter().any(|&byte| byte != 0) {
            return Err(DeepSeekV4SnapshotCodecError::InvalidHeader(
                "payload padding",
            ));
        }
        hasher.update(&buffer[..count]);
        bytes -= count;
    }
    Ok(())
}

fn require_eof<R: Read>(src: &mut R) -> Result<(), DeepSeekV4SnapshotCodecError> {
    let mut byte = [0u8; 1];
    match src.read(&mut byte)? {
        0 => Ok(()),
        _ => Err(DeepSeekV4SnapshotCodecError::TrailingBytes),
    }
}

fn require_little_endian() -> Result<(), DeepSeekV4SnapshotCodecError> {
    if cfg!(target_endian = "little") {
        Ok(())
    } else {
        Err(DeepSeekV4SnapshotCodecError::UnsupportedByteOrder)
    }
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 field"))
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("u64 field"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn tiny_config() -> DeepSeekV4Config {
        let mut config = crate::deepseek_v4::flash_0731_config_fixture();
        config.key_length = 4;
        config.value_length = 4;
        config.indexer_key_length = 2;
        config.attention_kinds = vec![AttentionKind::SlidingWindow; DEEPSEEK_V4_LAYER_COUNT];
        config
    }

    fn capacity(config: &DeepSeekV4Config) -> DeepSeekV4SessionCapacity {
        DeepSeekV4SessionCapacity::for_forward_limit(3_073, config.context_length).unwrap()
    }

    fn test_snapshot(
        config: &DeepSeekV4Config,
        position: u32,
        observation: DeepSeekV4SnapshotObservation,
    ) -> DeepSeekV4CausalSnapshot {
        let model_content_id = DeepSeekV4ModelContentId::new([0x5a; 32]);
        let session_capacity = capacity(config);
        let geometry = snapshot_geometry(config, session_capacity, position).unwrap();
        let prefix_tokens = (0..position)
            .map(|token| token % config.vocab_size)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let raw_f16_bits = (0..geometry.raw_elements)
            .map(|index| index as u16 ^ 0x2d5a)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let compressor_f32_bits = vec![0u32; geometry.compressor_elements].into_boxed_slice();
        let published_f16_bits = (0..geometry.published_elements)
            .map(|index| index as u16 ^ 0x1357)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut snapshot = DeepSeekV4CausalSnapshot {
            model_content_id,
            compatibility_digest: snapshot_compatibility_digest(model_content_id, config),
            next_position: position,
            prefix_digest: prefix_digest(&prefix_tokens),
            prefix_tokens,
            source_observation: observation,
            raw_f16_bits,
            compressor_f32_bits,
            published_f16_bits,
            causal_digest: [0; 32],
        };
        snapshot.causal_digest = causal_digest(&snapshot);
        validate_snapshot(&snapshot, config, session_capacity, model_content_id).unwrap();
        snapshot
    }

    fn constraints(config: &DeepSeekV4Config) -> DeepSeekV4SnapshotCodecConstraints<'_> {
        DeepSeekV4SnapshotCodecConstraints {
            config,
            session_capacity: capacity(config),
            expected_model_content_id: DeepSeekV4ModelContentId::new([0x5a; 32]),
            max_record_bytes: 64 * 1024 * 1024,
        }
    }

    fn encode(snapshot: &DeepSeekV4CausalSnapshot, config: &DeepSeekV4Config) -> Vec<u8> {
        let mut bytes = Vec::new();
        let encoded = encode_causal_snapshot(&mut bytes, snapshot, constraints(config)).unwrap();
        assert_eq!(encoded.record_bytes, bytes.len() as u64);
        bytes
    }

    fn reseal(bytes: &mut [u8]) {
        let digest_offset = bytes.len() - DIGEST_BYTES;
        let digest = blake3::hash(&bytes[..digest_offset]);
        bytes[digest_offset..].copy_from_slice(digest.as_bytes());
    }

    #[test]
    fn codec_roundtrip_is_canonical_and_observation_explicit() {
        let config = tiny_config();
        for observation in [
            DeepSeekV4SnapshotObservation::Unavailable,
            DeepSeekV4SnapshotObservation::Available,
        ] {
            let snapshot = test_snapshot(&config, 3, observation);
            let bytes = encode(&snapshot, &config);
            let decoded =
                decode_causal_snapshot(&mut Cursor::new(&bytes), constraints(&config)).unwrap();
            assert_eq!(decoded, snapshot);
            assert_eq!(encode(&decoded, &config), bytes);
        }
    }

    #[test]
    fn codec_rejects_header_drift_before_payload_allocation() {
        let config = tiny_config();
        let snapshot = test_snapshot(&config, 3, DeepSeekV4SnapshotObservation::Available);
        let original = encode(&snapshot, &config);
        for (offset, value, expected) in [
            (OFF_VERSION, CODEC_VERSION + 1, "codec version"),
            (
                OFF_STATE_WORD_ENCODING,
                STATE_WORD_ENCODING + 1,
                "state word encoding",
            ),
        ] {
            let mut bytes = original.clone();
            put_u32(&mut bytes, offset, value);
            reseal(&mut bytes);
            let error =
                decode_causal_snapshot(&mut Cursor::new(bytes), constraints(&config)).unwrap_err();
            assert!(error.to_string().contains(expected));
        }

        let mut unknown_flags = original.clone();
        put_u64(&mut unknown_flags, OFF_FLAGS, 1 << 63);
        reseal(&mut unknown_flags);
        assert!(
            decode_causal_snapshot(&mut Cursor::new(unknown_flags), constraints(&config))
                .unwrap_err()
                .to_string()
                .contains("unknown flags")
        );

        let mut reserved = original;
        reserved[OFF_RESERVED] = 1;
        reseal(&mut reserved);
        assert!(
            decode_causal_snapshot(&mut Cursor::new(reserved), constraints(&config))
                .unwrap_err()
                .to_string()
                .contains("reserved bytes")
        );
    }

    #[test]
    fn codec_rejects_budget_corruption_truncation_and_trailing_bytes() {
        let config = tiny_config();
        let snapshot = test_snapshot(&config, 3, DeepSeekV4SnapshotObservation::Available);
        let bytes = encode(&snapshot, &config);

        let mut too_small = constraints(&config);
        too_small.max_record_bytes = bytes.len() as u64 - 1;
        assert!(matches!(
            decode_causal_snapshot(&mut Cursor::new(&bytes), too_small),
            Err(DeepSeekV4SnapshotCodecError::RecordBudgetExceeded { .. })
        ));

        let mut corrupt = bytes.clone();
        corrupt[PAYLOAD_OFFSET] ^= 1;
        assert!(matches!(
            decode_causal_snapshot(&mut Cursor::new(corrupt), constraints(&config)),
            Err(DeepSeekV4SnapshotCodecError::DigestMismatch)
        ));

        let truncated = &bytes[..bytes.len() - 1];
        assert!(matches!(
            decode_causal_snapshot(&mut Cursor::new(truncated), constraints(&config)),
            Err(DeepSeekV4SnapshotCodecError::Io(_))
        ));

        let mut trailing = bytes;
        trailing.push(0);
        assert!(matches!(
            decode_causal_snapshot(&mut Cursor::new(trailing), constraints(&config)),
            Err(DeepSeekV4SnapshotCodecError::TrailingBytes)
        ));
    }

    #[test]
    fn production_and_legacy_layouts_stay_bounded_and_exact() {
        let config = crate::deepseek_v4::flash_0731_config_fixture();
        let promoted_capacity = DeepSeekV4SessionCapacity::for_forward_limit(
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
            config.context_length,
        )
        .unwrap();
        let constraints = DeepSeekV4SnapshotCodecConstraints {
            config: &config,
            session_capacity: promoted_capacity,
            expected_model_content_id: DeepSeekV4ModelContentId::new([0x5a; 32]),
            max_record_bytes: 1024 * 1024 * 1024,
        };
        let legacy = WireLayout::derive(
            1_025,
            DeepSeekV4SnapshotObservation::Unavailable,
            constraints,
        )
        .unwrap();
        assert_eq!(legacy.prefix_bytes, 4_100);
        assert_eq!(legacy.raw_bytes, 5_636_096);
        assert_eq!(legacy.compressor_bytes, 12_206_080);
        assert_eq!(legacy.published_bytes, 7_045_120);
        assert_eq!(legacy.payload_bytes, 24_891_396);
        assert_eq!(legacy.record_bytes, 24_907_812);

        let terminal = WireLayout::derive(
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY as u32,
            DeepSeekV4SnapshotObservation::Unavailable,
            constraints,
        )
        .unwrap();
        assert_eq!(terminal.prefix_bytes, 262_144);
        assert_eq!(terminal.raw_bytes, 5_636_096);
        assert_eq!(terminal.compressor_bytes, 12_206_080);
        assert_eq!(terminal.published_bytes, 450_887_680);
        assert_eq!(terminal.payload_bytes, 468_992_000);
        assert_eq!(terminal.record_bytes, 469_008_416);
    }
}
