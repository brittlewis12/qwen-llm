//! CPU integer addresses for the released Engram profile, not model execution.
//!
//! Source: `inference/engram.py`, revision
//! `1bae0e1088993b9b82de08f59de5be5a34e42170`, SHA-256
//! `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897`
//! (pinned by `sources.json` and `engram-addresses.json`). Parameters below are
//! the verified upstream-AST fixture values, not a reimplementation of NumPy RNG.
//! Inputs are ALREADY compressed IDs. There is no tokenizer mapping, disk I/O,
//! framework dependency, or floating-point Engram operator here.

pub const COMPRESSED_VOCAB_SIZE: u32 = 99_092;
pub const MAX_POSITION: u64 = 1_048_576;
pub const LAYER_IDS: [u32; 2] = [1, 14];
pub const TABLE_TOTALS: [u64; 2] = [384_006_168, 384_016_682];

const PRIMES: [[u64; 24]; 2] = [
    [
        16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183, 16000189,
        16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321, 16000339, 16000381,
        16000393, 16000399, 16000403, 16000409, 16000447, 16000463,
    ],
    [
        16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627, 16000667,
        16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769, 16000781, 16000799,
        16000813, 16000819, 16000841, 16000877, 16000879, 16000889,
    ],
];
const MULTIPLIERS: [[u64; 4]; 2] = [
    [
        76632096046245,
        4839876093313,
        35959672319349,
        73987337458391,
    ],
    [
        67716810739261,
        51510806800915,
        30921347202721,
        82619226485591,
    ],
];

/// Rows by layer [1, 14], then n-gram length [2, 3, 4], eight heads each.
pub type Rows = [[u64; 24]; 2];
pub type LayerRows = std::collections::BTreeMap<usize, [u64; 24]>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerHash {
    pub layer: usize,
    pub primes: [u64; 24],
    pub offsets: [u64; 24],
    pub multipliers: [u64; 4],
    pub rows: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashConfig {
    pub layers: Vec<LayerHash>,
    pub compressed_vocab_size: u32,
    pub max_position: u64,
}
impl Default for HashConfig {
    fn default() -> Self {
        Self {
            layers: (0..2)
                .map(|i| {
                    let mut offset = 0;
                    let offsets = PRIMES[i].map(|prime| {
                        let start = offset;
                        offset += prime;
                        start
                    });
                    LayerHash {
                        layer: LAYER_IDS[i] as usize,
                        primes: PRIMES[i],
                        offsets,
                        multipliers: MULTIPLIERS[i],
                        rows: TABLE_TOTALS[i],
                    }
                })
                .collect(),
            compressed_vocab_size: COMPRESSED_VOCAB_SIZE,
            max_position: MAX_POSITION,
        }
    }
}
impl HashConfig {
    pub fn validate(&self) -> Result<(), AddressError> {
        if self.compressed_vocab_size == 0 || self.max_position == 0 {
            return Err(AddressError::InvalidConfig("empty vocabulary/context"));
        }
        if self.layers.windows(2).any(|w| w[0].layer >= w[1].layer) {
            return Err(AddressError::InvalidConfig(
                "unordered or duplicate hash layers",
            ));
        }
        for layer in &self.layers {
            let mut end = 0u64;
            for (&prime, &offset) in layer.primes.iter().zip(&layer.offsets) {
                if prime < 2 || offset != end {
                    return Err(AddressError::InvalidConfig("bucket geometry"));
                }
                end = end
                    .checked_add(prime)
                    .ok_or(AddressError::InvalidConfig("bucket overflow"))?;
            }
            if end != layer.rows {
                return Err(AddressError::InvalidConfig("bucket/table rows"));
            }
            for &m in &layer.multipliers {
                if m % 2 == 0
                    || u64::from(self.compressed_vocab_size - 1)
                        .checked_mul(m)
                        .is_none_or(|n| n > i64::MAX as u64)
                {
                    return Err(AddressError::InvalidConfig(
                        "hash product outside signed int64",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// `Dead` is a boundary, never an integer ID or an alias for padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Token {
    Compressed(u32),
    Dead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressError {
    InvalidConfig(&'static str),
    InvalidCompressedId(u32),
    InvalidPadId(u32),
    InvalidPosition(u64),
    HistoryLength { expected: usize, actual: usize },
    ContextFull,
}

/// Only the last three tokens (oldest first), compressed pad, and absolute
/// committed position. A staged result becomes committed only when the caller
/// replaces its state, after all other work (including GPU work) succeeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddressState {
    history: [Token; 3],
    pad: u32,
    position: u64,
    vocab_size: u32,
    max_position: u64,
}

impl AddressState {
    /// `pad` must be the caller's compressed pad ID, NOT raw tokenizer token 2.
    pub fn new(pad: u32) -> Result<Self, AddressError> {
        Self::restore(pad, 0, &[])
    }

    pub fn new_with_config(pad: u32, config: &HashConfig) -> Result<Self, AddressError> {
        Self::restore_with_config(pad, 0, &[], config)
    }

    /// Restore bounded address history, not a whole-session snapshot. This
    /// checks shape and ranges only; the engine owns session/content identity.
    /// A position equal to MAX_POSITION is valid but cannot accept another step.
    pub fn restore(pad: u32, position: u64, history: &[Token]) -> Result<Self, AddressError> {
        Self::restore_with_config(pad, position, history, &HashConfig::default())
    }

    pub fn restore_with_config(
        pad: u32,
        position: u64,
        history: &[Token],
        config: &HashConfig,
    ) -> Result<Self, AddressError> {
        config.validate()?;
        if pad >= config.compressed_vocab_size {
            return Err(AddressError::InvalidPadId(pad));
        }
        if position > config.max_position {
            return Err(AddressError::InvalidPosition(position));
        }
        let expected = position.min(3) as usize;
        if history.len() != expected {
            return Err(AddressError::HistoryLength {
                expected,
                actual: history.len(),
            });
        }
        for &token in history {
            validate_token_with_limit(token, config.compressed_vocab_size)?;
        }
        let mut state = Self {
            history: [Token::Dead; 3],
            pad,
            position,
            vocab_size: config.compressed_vocab_size,
            max_position: config.max_position,
        };
        state.history[..expected].copy_from_slice(history);
        Ok(state)
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    pub fn pad(&self) -> u32 {
        self.pad
    }

    pub fn history(&self) -> &[Token] {
        &self.history[..self.position.min(3) as usize]
    }

    /// Pure staged step using the legacy family-default hash configuration.
    /// before hashing (state invariants are established by new/restore).
    /// Valid products are nonnegative and <= i64::MAX, matching upstream int64.
    pub fn step(&self, token: Token) -> Result<(Rows, Self), AddressError> {
        let (rows, next) = self.step_with_config(token, &HashConfig::default())?;
        Ok(([rows[&1], rows[&14]], next))
    }

    pub fn step_with_config(
        &self,
        token: Token,
        config: &HashConfig,
    ) -> Result<(LayerRows, Self), AddressError> {
        config.validate()?;
        if self.vocab_size != config.compressed_vocab_size
            || self.max_position != config.max_position
        {
            return Err(AddressError::InvalidConfig("state/config bounds differ"));
        }
        validate_token_with_limit(token, self.vocab_size)?;
        if self.position == self.max_position {
            return Err(AddressError::ContextFull);
        }
        let len = self.history().len();
        let mut tokens = [u64::from(self.pad); 4];
        let mut blocked = false;
        for (shift, value) in tokens.iter_mut().enumerate() {
            let source = if shift == 0 {
                token
            } else if shift <= len {
                self.history[len - shift]
            } else {
                Token::Dead
            };
            match source {
                Token::Compressed(id) if !blocked => *value = u64::from(id),
                _ => blocked = true,
            }
        }
        let mut rows = LayerRows::new();
        for layer in &config.layers {
            let mut row = [0; 24];
            let mut rolling = tokens[0] * layer.multipliers[0];
            for shift in 1..4 {
                rolling ^= tokens[shift] * layer.multipliers[shift];
                for head in 0..8 {
                    let column = (shift - 1) * 8 + head;
                    row[column] = layer.offsets[column] + rolling % layer.primes[column];
                }
            }
            rows.insert(layer.layer, row);
        }
        let mut next = *self;
        if len < 3 {
            next.history[len] = token;
        } else {
            next.history = [self.history[1], self.history[2], token];
        }
        next.position += 1;
        Ok((rows, next))
    }
}

fn validate_token_with_limit(token: Token, limit: u32) -> Result<(), AddressError> {
    if let Token::Compressed(id) = token {
        if id >= limit {
            return Err(AddressError::InvalidCompressedId(id));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn metadata_layers_buckets_and_multipliers_change_addresses() {
        let config = HashConfig {
            compressed_vocab_size: 10,
            max_position: 4,
            layers: vec![LayerHash {
                layer: 5,
                primes: [5; 24],
                offsets: std::array::from_fn(|i| i as u64 * 5),
                multipliers: [3, 5, 7, 9],
                rows: 120,
            }],
        };
        let initial = AddressState::new_with_config(0, &config).unwrap();
        let (rows, next) = initial
            .step_with_config(Token::Compressed(3), &config)
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[&5], std::array::from_fn(|i| i as u64 * 5 + 4));
        let restored =
            AddressState::restore_with_config(0, next.position(), next.history(), &config).unwrap();
        assert_eq!(next, restored);
        // 2*3 XOR 3*5 = 9; the other history terms are zero padding.
        assert_eq!(
            next.step_with_config(Token::Compressed(2), &config)
                .unwrap()
                .0[&5],
            std::array::from_fn(|i| i as u64 * 5 + 4)
        );
        assert!(
            initial
                .step_with_config(Token::Compressed(10), &config)
                .is_err()
        );
        let mut bad = config.clone();
        bad.layers[0].multipliers[0] = u64::MAX;
        assert!(bad.validate().is_err());
        bad = config.clone();
        bad.layers[0].offsets[1] += 1;
        assert!(bad.validate().is_err());
    }

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../scripts/reference/deepseek_v41/fixtures/engram-addresses.json"
        ))
        .unwrap()
    }

    fn tokens(value: &Value) -> Vec<Token> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| match v.as_i64().unwrap() {
                -1 => Token::Dead,
                id => Token::Compressed(u32::try_from(id).unwrap()),
            })
            .collect()
    }

    fn check_ranges(rows: &Rows) {
        for layer in 0..2 {
            let mut offset = 0;
            for column in 0..24 {
                assert!((offset..offset + PRIMES[layer][column]).contains(&rows[layer][column]));
                assert!(rows[layer][column] < TABLE_TOTALS[layer]);
                offset += PRIMES[layer][column];
            }
        }
    }

    #[test]
    fn pinned_parameters_and_product_bounds() {
        let fixture = fixture();
        let sources: Value = serde_json::from_str(include_str!(
            "../../../../scripts/reference/deepseek_v41/sources.json"
        ))
        .unwrap();
        assert_eq!(fixture["source_revision"], sources["revision"]);
        assert_eq!(
            fixture["source_sha256"],
            sources["sha256"]["inference/engram.py"]
        );
        assert_eq!(fixture["compressed_vocab_size"], COMPRESSED_VOCAB_SIZE);
        let params = &fixture["parameters"];
        let bound = (i64::MAX as u64 / u64::from(COMPRESSED_VOCAB_SIZE)) / 2;
        for layer in 0..2 {
            assert_eq!(params["layer_ids"][layer], LAYER_IDS[layer]);
            assert_eq!(params["table_totals"][layer], TABLE_TOTALS[layer]);
            assert_eq!(PRIMES[layer].iter().sum::<u64>(), TABLE_TOTALS[layer]);
            let mut offset = 0;
            for column in 0..24 {
                assert_eq!(
                    params["primes"][layer][column / 8][column % 8],
                    PRIMES[layer][column]
                );
                assert_eq!(params["offsets"][layer][column], offset);
                offset += PRIMES[layer][column];
            }
            for (shift, &m) in MULTIPLIERS[layer].iter().enumerate() {
                assert_eq!(params["multipliers"][layer][shift], m);
                assert_eq!(m % 2, 1);
                assert!((m - 1) / 2 < bound);
                assert!(
                    u64::from(COMPRESSED_VOCAB_SIZE - 1).checked_mul(m).unwrap() <= i64::MAX as u64
                );
            }
        }
    }

    #[test]
    fn every_source_row_chunking_and_restoration() {
        let fixture = fixture();
        let pad = fixture["synthetic_pad_compressed_id"].as_u64().unwrap() as u32;
        let mut positions = 0;
        for case in fixture["cases"].as_array().unwrap() {
            let history = tokens(&case["history"]);
            let inputs = tokens(&case["compressed_ids"]);
            let initial = AddressState::restore(pad, history.len() as u64, &history).unwrap();
            let mut singleton = initial;
            let mut expected = Vec::new();
            for (i, &token) in inputs.iter().enumerate() {
                let before = singleton;
                let (rows, next) = singleton.step(token).unwrap();
                assert_eq!(singleton, before);
                for layer in 0..2 {
                    for column in 0..24 {
                        assert_eq!(
                            rows[layer][column],
                            case["expected_rows"][i][layer][column].as_u64().unwrap(),
                            "case {} position {i} layer {layer} column {column}",
                            case["name"]
                        );
                    }
                }
                check_ranges(&rows);
                let restored = AddressState::restore(
                    singleton.pad(),
                    singleton.position(),
                    singleton.history(),
                )
                .unwrap();
                assert_eq!(restored.step(token), Ok((rows, next)));
                singleton = next;
                expected.push(rows);
                positions += 1;
            }
            assert_eq!(singleton.history(), tokens(&case["final_history"]));
            let mut chunked = initial;
            let mut index = 0;
            for size in case["chunks"].as_array().unwrap() {
                for &token in &inputs[index..index + size.as_u64().unwrap() as usize] {
                    let (rows, next) = chunked.step(token).unwrap();
                    assert_eq!(rows, expected[index]);
                    chunked = next;
                    index += 1;
                }
            }
            assert_eq!(index, inputs.len());
            assert_eq!(chunked, singleton);
        }
        assert_eq!(positions, 21);
    }

    #[test]
    fn dead_blocks_current_and_every_older_phase() {
        let live = Token::Compressed;
        let pad = 17;
        for dead_shift in 0..4 {
            let mut window = [live(11), live(12), live(13), live(14)];
            window[3 - dead_shift] = Token::Dead;
            let actual = AddressState::restore(pad, 50, &window[..3]).unwrap();
            let mut padded = window;
            padded[..4 - dead_shift].fill(live(pad));
            let equivalent = AddressState::restore(pad, 50, &padded[..3]).unwrap();
            assert_eq!(
                actual.step(window[3]).unwrap().0,
                equivalent.step(padded[3]).unwrap().0
            );
        }
        let mut state = AddressState::new(pad).unwrap();
        let all_pad = AddressState::restore(pad, 3, &[live(pad); 3])
            .unwrap()
            .step(live(pad))
            .unwrap()
            .0;
        for _ in 0..5 {
            let (rows, next) = state.step(Token::Dead).unwrap();
            assert_eq!(rows, all_pad);
            state = next;
        }
    }

    #[test]
    fn checked_boundaries_and_discarded_staging() {
        let live = Token::Compressed;
        for bad in [COMPRESSED_VOCAB_SIZE, u32::MAX] {
            assert_eq!(AddressState::new(bad), Err(AddressError::InvalidPadId(bad)));
            assert_eq!(
                AddressState::restore(17, 1, &[live(bad)]),
                Err(AddressError::InvalidCompressedId(bad))
            );
            let state = AddressState::new(17).unwrap();
            let before = state;
            assert_eq!(
                state.step(live(bad)),
                Err(AddressError::InvalidCompressedId(bad))
            );
            assert_eq!(state, before);
        }
        for position in [MAX_POSITION + 1, u64::MAX] {
            assert_eq!(
                AddressState::restore(17, position, &[Token::Dead; 3]),
                Err(AddressError::InvalidPosition(position))
            );
        }
        for position in [0, 1, 2, 3, 4, MAX_POSITION] {
            for len in 0..=4 {
                let result = AddressState::restore(17, position, &[Token::Dead; 4][..len]);
                assert_eq!(result.is_ok(), len == position.min(3) as usize);
            }
        }
        for pad in [0, 2, 17, COMPRESSED_VOCAB_SIZE - 1] {
            let state = AddressState::restore(pad, MAX_POSITION - 1, &[live(99091); 3]).unwrap();
            let before = state;
            let (rows, next) = state.step(live(99091)).unwrap();
            check_ranges(&rows);
            // Simulated late downstream failure: discard next, then retry.
            assert_eq!(state, before);
            assert_eq!(state.step(live(99091)), Ok((rows, next)));
            let full = next;
            assert_eq!(full.step(Token::Dead), Err(AddressError::ContextFull));
            assert_eq!(full, next);
        }
    }
}
