//! Row-addressed PLE embedding access for Qwen3.8-Flash-Next.
//!
//! The released table is too large to make random GPU page faults attractive.
//! This view touches only selected mmap rows on the CPU and copies them into a
//! bounded staging buffer for native IQ4_NL dequantization.

use crate::tensor::{GgmlType, TensorDesc};

const IQ4_NL_BLOCK_ELEMENTS: usize = 32;
const IQ4_NL_BLOCK_BYTES: usize = 18;
const IQ4_NL_VALUES: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PleGatherError {
    #[error("PLE embedding must use IQ4_NL storage, got {0:?}")]
    UnsupportedDtype(GgmlType),
    #[error("PLE embedding must be a nonempty row matrix, got shape {0:?}")]
    InvalidShape(Vec<u64>),
    #[error("PLE embedding row width {0} is not divisible by 32")]
    InvalidRowWidth(u64),
    #[error("PLE logical row count {logical} is outside physical row count {physical}")]
    InvalidLogicalRows { logical: u64, physical: u64 },
    #[error("PLE embedding geometry overflows host addressing")]
    SizeOverflow,
    #[error(
        "PLE embedding byte length mismatch: descriptor={descriptor}, source={source_bytes}, expected={expected}"
    )]
    ByteLength {
        descriptor: u64,
        source_bytes: usize,
        expected: usize,
    },
    #[error("PLE gather output length mismatch: got {got}, expected {expected}")]
    OutputLength { got: usize, expected: usize },
    #[error("PLE row id {row} at lookup {lookup} is outside {row_count} rows")]
    RowOutOfRange {
        lookup: usize,
        row: u32,
        row_count: usize,
    },
}

#[derive(Clone, Copy)]
pub struct PleIq4NlTable<'a> {
    bytes: &'a [u8],
    row_width: usize,
    row_count: usize,
    physical_row_count: usize,
    row_bytes: usize,
}

impl<'a> PleIq4NlTable<'a> {
    pub fn new(
        desc: &TensorDesc,
        bytes: &'a [u8],
        logical_row_count: u64,
    ) -> Result<Self, PleGatherError> {
        if desc.dtype != GgmlType::IQ4_NL {
            return Err(PleGatherError::UnsupportedDtype(desc.dtype));
        }
        if desc.shape.len() != 2 || desc.shape[0] == 0 || desc.shape[1] == 0 {
            return Err(PleGatherError::InvalidShape(desc.shape.clone()));
        }
        if !desc.shape[0].is_multiple_of(IQ4_NL_BLOCK_ELEMENTS as u64) {
            return Err(PleGatherError::InvalidRowWidth(desc.shape[0]));
        }

        let row_width = usize::try_from(desc.shape[0]).map_err(|_| PleGatherError::SizeOverflow)?;
        if logical_row_count == 0 || logical_row_count > desc.shape[1] {
            return Err(PleGatherError::InvalidLogicalRows {
                logical: logical_row_count,
                physical: desc.shape[1],
            });
        }
        let row_count =
            usize::try_from(logical_row_count).map_err(|_| PleGatherError::SizeOverflow)?;
        let physical_row_count =
            usize::try_from(desc.shape[1]).map_err(|_| PleGatherError::SizeOverflow)?;
        let row_bytes = row_width
            .checked_div(IQ4_NL_BLOCK_ELEMENTS)
            .and_then(|blocks| blocks.checked_mul(IQ4_NL_BLOCK_BYTES))
            .ok_or(PleGatherError::SizeOverflow)?;
        let expected = row_bytes
            .checked_mul(physical_row_count)
            .ok_or(PleGatherError::SizeOverflow)?;
        if desc.n_bytes != expected as u64 || bytes.len() != expected {
            return Err(PleGatherError::ByteLength {
                descriptor: desc.n_bytes,
                source_bytes: bytes.len(),
                expected,
            });
        }

        Ok(Self {
            bytes,
            row_width,
            row_count,
            physical_row_count,
            row_bytes,
        })
    }

    pub fn row_width(&self) -> usize {
        self.row_width
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn physical_row_count(&self) -> usize {
        self.physical_row_count
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub fn packed_staging_bytes(&self, lookup_count: usize) -> Result<usize, PleGatherError> {
        lookup_count
            .checked_mul(self.row_bytes)
            .ok_or(PleGatherError::SizeOverflow)
    }

    pub fn f32_staging_elements(&self, lookup_count: usize) -> Result<usize, PleGatherError> {
        lookup_count
            .checked_mul(self.row_width)
            .ok_or(PleGatherError::SizeOverflow)
    }

    pub fn gather_packed_into(
        &self,
        row_ids: &[u32],
        output: &mut [u8],
    ) -> Result<(), PleGatherError> {
        let expected = self.packed_staging_bytes(row_ids.len())?;
        validate_gather(row_ids, output.len(), expected, self.row_count)?;

        for (lookup, &row) in row_ids.iter().enumerate() {
            let source_start = row as usize * self.row_bytes;
            let output_start = lookup * self.row_bytes;
            output[output_start..output_start + self.row_bytes]
                .copy_from_slice(&self.bytes[source_start..source_start + self.row_bytes]);
        }
        Ok(())
    }

    pub fn gather_f32_into(
        &self,
        row_ids: &[u32],
        output: &mut [f32],
    ) -> Result<(), PleGatherError> {
        let expected = self.f32_staging_elements(row_ids.len())?;
        validate_gather(row_ids, output.len(), expected, self.row_count)?;

        let blocks_per_row = self.row_width / IQ4_NL_BLOCK_ELEMENTS;
        for (lookup, &row) in row_ids.iter().enumerate() {
            let source_row = row as usize * self.row_bytes;
            let output_row = lookup * self.row_width;
            for block_index in 0..blocks_per_row {
                let source_start = source_row + block_index * IQ4_NL_BLOCK_BYTES;
                let block = &self.bytes[source_start..source_start + IQ4_NL_BLOCK_BYTES];
                let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                let output_start = output_row + block_index * IQ4_NL_BLOCK_ELEMENTS;
                for lane in 0..16 {
                    let packed = block[2 + lane];
                    output[output_start + lane] = d * IQ4_NL_VALUES[(packed & 0x0f) as usize];
                    output[output_start + 16 + lane] = d * IQ4_NL_VALUES[(packed >> 4) as usize];
                }
            }
        }
        Ok(())
    }
}

fn validate_gather(
    row_ids: &[u32],
    output_len: usize,
    expected_output_len: usize,
    row_count: usize,
) -> Result<(), PleGatherError> {
    if output_len != expected_output_len {
        return Err(PleGatherError::OutputLength {
            got: output_len,
            expected: expected_output_len,
        });
    }
    if let Some((lookup, &row)) = row_ids
        .iter()
        .enumerate()
        .find(|&(_, &row)| row as usize >= row_count)
    {
        return Err(PleGatherError::RowOutOfRange {
            lookup,
            row,
            row_count,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(rows: usize, row_width: usize) -> (TensorDesc, Vec<u8>) {
        let blocks_per_row = row_width / IQ4_NL_BLOCK_ELEMENTS;
        let mut bytes = Vec::with_capacity(rows * blocks_per_row * IQ4_NL_BLOCK_BYTES);
        for row in 0..rows {
            for block in 0..blocks_per_row {
                let ordinal = row * blocks_per_row + block;
                let sign = if ordinal.is_multiple_of(2) { 1.0 } else { -1.0 };
                let d = sign * (ordinal % 7 + 1) as f32 / 1024.0;
                bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
                for lane in 0..16 {
                    let low = (ordinal + lane * 3) % 16;
                    let high = (ordinal * 5 + lane * 7 + 1) % 16;
                    bytes.push(low as u8 | ((high as u8) << 4));
                }
            }
        }
        let desc = TensorDesc {
            name: "per_layer_token_embd.weight".into(),
            shape: vec![row_width as u64, rows as u64],
            dtype: GgmlType::IQ4_NL,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: bytes.len() as u64,
        };
        (desc, bytes)
    }

    #[test]
    fn packed_gather_copies_only_selected_rows() {
        let (desc, bytes) = fixture(4, 160);
        let table = PleIq4NlTable::new(&desc, &bytes, 4).unwrap();
        assert_eq!(table.row_width(), 160);
        assert_eq!(table.row_count(), 4);
        assert_eq!(table.row_bytes(), 90);

        let row_ids = [3, 0, 3];
        let mut gathered = vec![0xa5; table.packed_staging_bytes(row_ids.len()).unwrap()];
        table.gather_packed_into(&row_ids, &mut gathered).unwrap();
        for (lookup, &row) in row_ids.iter().enumerate() {
            let expected = &bytes[row as usize * 90..(row as usize + 1) * 90];
            assert_eq!(&gathered[lookup * 90..(lookup + 1) * 90], expected);
        }
    }

    #[test]
    fn f32_gather_matches_llama_codec() {
        let (desc, bytes) = fixture(4, 160);
        let table = PleIq4NlTable::new(&desc, &bytes, 4).unwrap();
        let row_ids = [2, 0, 2];
        let mut packed = vec![0; table.packed_staging_bytes(row_ids.len()).unwrap()];
        table.gather_packed_into(&row_ids, &mut packed).unwrap();
        let packed_desc = TensorDesc {
            name: "selected_ple_rows".into(),
            shape: vec![160, row_ids.len() as u64],
            dtype: GgmlType::IQ4_NL,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: packed.len() as u64,
        };
        let expected = crate::codec::dequant_to_f32(&packed_desc, &packed).unwrap();
        let mut actual = vec![f32::NAN; table.f32_staging_elements(row_ids.len()).unwrap()];
        table.gather_f32_into(&row_ids, &mut actual).unwrap();
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(candidate, reference)| candidate.to_bits() == reference.to_bits())
        );
    }

    #[test]
    fn gather_errors_do_not_modify_output() {
        let (desc, bytes) = fixture(3, 160);
        let table = PleIq4NlTable::new(&desc, &bytes, 3).unwrap();
        let mut output = vec![0xa5; table.packed_staging_bytes(2).unwrap()];
        let original = output.clone();
        assert_eq!(
            table.gather_packed_into(&[0, 3], &mut output),
            Err(PleGatherError::RowOutOfRange {
                lookup: 1,
                row: 3,
                row_count: 3,
            })
        );
        assert_eq!(output, original);

        let mut f32_output = vec![7.0; 319];
        assert_eq!(
            table.gather_f32_into(&[0, 1], &mut f32_output),
            Err(PleGatherError::OutputLength {
                got: 319,
                expected: 320,
            })
        );
        assert!(f32_output.iter().all(|&value| value == 7.0));
    }

    #[test]
    fn constructor_rejects_malformed_storage() {
        let (desc, bytes) = fixture(3, 160);

        let mut wrong_dtype = desc.clone();
        wrong_dtype.dtype = GgmlType::Q8_0;
        assert_eq!(
            PleIq4NlTable::new(&wrong_dtype, &bytes, 3).err(),
            Some(PleGatherError::UnsupportedDtype(GgmlType::Q8_0))
        );

        let mut bad_width = desc.clone();
        bad_width.shape[0] = 159;
        assert_eq!(
            PleIq4NlTable::new(&bad_width, &bytes, 3).err(),
            Some(PleGatherError::InvalidRowWidth(159))
        );

        let mut bad_descriptor_bytes = desc.clone();
        bad_descriptor_bytes.n_bytes -= 1;
        assert!(matches!(
            PleIq4NlTable::new(&bad_descriptor_bytes, &bytes, 3),
            Err(PleGatherError::ByteLength { .. })
        ));
        assert!(matches!(
            PleIq4NlTable::new(&desc, &bytes[..bytes.len() - 1], 3),
            Err(PleGatherError::ByteLength { .. })
        ));
    }

    #[test]
    fn padded_physical_rows_remain_outside_logical_addressing() {
        let (desc, bytes) = fixture(4, 160);
        let table = PleIq4NlTable::new(&desc, &bytes, 3).unwrap();
        assert_eq!(table.row_count(), 3);
        assert_eq!(table.physical_row_count(), 4);
        let mut output = vec![0xa5; table.row_bytes()];
        assert_eq!(
            table.gather_packed_into(&[3], &mut output),
            Err(PleGatherError::RowOutOfRange {
                lookup: 0,
                row: 3,
                row_count: 3,
            })
        );
        assert!(output.iter().all(|&value| value == 0xa5));

        assert_eq!(
            PleIq4NlTable::new(&desc, &bytes, 5).err(),
            Some(PleGatherError::InvalidLogicalRows {
                logical: 5,
                physical: 4,
            })
        );
    }
}
