//! Binary array payloads and JSON descriptors share one durable snapshot commit.

use super::*;
use sha2::{Digest, Sha256};

pub(crate) const MAX_ARCHIVE_BYTES: u64 = 32 * 1024 * 1024;
pub(crate) const MAX_ARRAY_BYTES: usize = 4 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    dtype: String,
    length: usize,
    offset: u64,
    byte_length: usize,
    sha256: String,
    url: String,
}

fn descriptor(record: &Value, snapshot: &Snapshot, record_offset: u64) -> Result<Descriptor> {
    if record.get("kind").and_then(Value::as_str) != Some("retained_array") {
        return Err(invalid("record is not a retained array"));
    }
    let value: Descriptor = serde_json::from_value(
        record
            .get("array")
            .cloned()
            .ok_or_else(|| corrupt("missing array descriptor"))?,
    )?;
    if value.dtype != "f32le"
        || value.length == 0
        || value.length.checked_mul(4) != Some(value.byte_length)
        || value.byte_length > MAX_ARRAY_BYTES
        || value
            .offset
            .checked_add(value.byte_length as u64)
            .is_none_or(|end| end > snapshot.archive_committed_bytes)
        || value.sha256.len() != 64
        || !value
            .sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        || value.url
            != format!(
                "/v1/lens/jobs/{}/arrays/{record_offset}",
                snapshot.status.id
            )
    {
        return Err(corrupt("invalid committed array descriptor"));
    }
    Ok(value)
}

fn payload(file: &mut File, descriptor: &Descriptor) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(descriptor.offset))?;
    let mut bytes = vec![0; descriptor.byte_length];
    file.read_exact(&mut bytes)?;
    if format!("{:x}", Sha256::digest(&bytes)) != descriptor.sha256 {
        return Err(corrupt("retained array digest mismatch"));
    }
    if !bytes
        .chunks_exact(4)
        .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
    {
        return Err(corrupt("nonfinite retained array"));
    }
    Ok(bytes)
}

pub(super) fn recover(directory: &Path, snapshot: &Snapshot, limits: Limits) -> Result<()> {
    if snapshot.archive_committed_bytes > snapshot.archive_reserved_bytes
        || snapshot.archive_reserved_bytes > MAX_ARCHIVE_BYTES
        || snapshot
            .archive_reserved_bytes
            .checked_add(snapshot.committed_bytes)
            .is_none_or(|n| n > limits.max_job_bytes)
    {
        return Err(corrupt("invalid archive watermark/reservation"));
    }
    let mut arrays = (snapshot.archive_reserved_bytes != 0)
        .then(|| open_regular(&directory.join("arrays.bin"), true))
        .transpose()?;
    let records = open_regular(&directory.join("records.jsonl"), false)?;
    let mut reader = BufReader::new(records.take(snapshot.committed_bytes));
    let mut record_offset = 0;
    let mut array_offset = 0;
    let mut seq = 0;
    while record_offset < snapshot.committed_bytes {
        let mut line = Vec::new();
        (&mut reader)
            .take(limits.max_record_bytes as u64 + 1)
            .read_until(b'\n', &mut line)?;
        if line.last() != Some(&b'\n') || line.len() > limits.max_record_bytes {
            return Err(corrupt("invalid committed archive record"));
        }
        let record: Value = serde_json::from_slice(&line)?;
        if record.get("seq").and_then(Value::as_u64) != Some(seq) {
            return Err(corrupt("invalid archive record sequence"));
        }
        if record.get("kind").and_then(Value::as_str).is_none() {
            return Err(corrupt("committed record kind is missing"));
        }
        if record.get("kind").and_then(Value::as_str) == Some("retained_array") {
            let desc = descriptor(&record, snapshot, record_offset)?;
            if desc.offset != array_offset {
                return Err(corrupt("noncontiguous archive payloads"));
            }
            payload(
                arrays
                    .as_mut()
                    .ok_or_else(|| corrupt("array record without an archive"))?,
                &desc,
            )?;
            array_offset += desc.byte_length as u64;
        }
        record_offset += line.len() as u64;
        seq += 1;
    }
    if array_offset != snapshot.archive_committed_bytes || seq != snapshot.next_seq {
        return Err(corrupt("archive descriptor coverage mismatch"));
    }
    if let Some(arrays) = arrays {
        arrays.set_len(array_offset)?;
        arrays.sync_all()?;
    }
    Ok(())
}

impl JobStore {
    pub(crate) fn append_array(
        &self,
        id: &str,
        mut record: Value,
        bytes: &[u8],
    ) -> Result<JobStatus> {
        if bytes.is_empty()
            || bytes.len() % 4 != 0
            || bytes.len() > MAX_ARRAY_BYTES
            || !bytes
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
        {
            return Err(invalid("array must be bounded finite F32LE"));
        }
        let object = record
            .as_object_mut()
            .ok_or_else(|| invalid("array metadata must be an object"))?;
        if object.contains_key("array") || object.contains_key("seq") {
            return Err(invalid("array descriptor fields are store-owned"));
        }
        object.insert("kind".into(), "retained_array".into());
        let entry = self.entry(id)?;
        let _writer = entry.writer.lock().unwrap();
        entry.writable()?;
        let mut next = entry.snapshot();
        if !matches!(next.status.state, JobState::Running | JobState::Finalizing) {
            return Err(invalid("job does not accept arrays"));
        }
        let offset = next.archive_committed_bytes;
        next.archive_committed_bytes = offset
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| invalid("archive byte overflow"))?;
        if next.archive_committed_bytes > next.archive_reserved_bytes {
            return Err(StoreError::Full);
        }
        record["array"] = serde_json::to_value(Descriptor {
            dtype: "f32le".into(),
            length: bytes.len() / 4,
            offset,
            byte_length: bytes.len(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            url: format!("/v1/lens/jobs/{id}/arrays/{}", next.committed_bytes),
        })?;
        self.append_locked(&entry, next, &[record], Some(bytes))
    }

    pub(crate) fn array(&self, id: &str, record_offset: u64) -> Result<Vec<u8>> {
        let entry = self.entry(id)?;
        let _payload = entry.payload.read().unwrap();
        let snapshot = entry.snapshot();
        if snapshot.status.deleted {
            return Err(StoreError::Deleted);
        }
        if record_offset >= snapshot.committed_bytes {
            return Err(StoreError::NotFound);
        }
        let mut file = open_regular(&entry.directory.join("records.jsonl"), false)?;
        if record_offset != 0 {
            file.seek(SeekFrom::Start(record_offset - 1))?;
            let mut byte = [0];
            file.read_exact(&mut byte)?;
            if byte[0] != b'\n' {
                return Err(invalid("array locator is not a record boundary"));
            }
        }
        let mut line = Vec::new();
        BufReader::new(file.take(
            (snapshot.committed_bytes - record_offset).min(self.limits.max_record_bytes as u64 + 1),
        ))
        .read_until(b'\n', &mut line)?;
        if line.last() != Some(&b'\n') || line.len() > self.limits.max_record_bytes {
            return Err(corrupt("invalid retained array record"));
        }
        let desc = descriptor(&serde_json::from_slice(&line)?, &snapshot, record_offset)?;
        payload(
            &mut open_regular(&entry.directory.join("arrays.bin"), false)?,
            &desc,
        )
    }
}
