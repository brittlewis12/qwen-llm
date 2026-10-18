#[cfg(test)]
pub(super) use super::super::operators::Session;
pub(super) type AttentionWeights<'a> = super::super::operators::AttentionWeights<super::Linear<'a>>;
pub(super) type CompressorWeights<'a> =
    super::super::operators::CompressorWeights<super::Linear<'a>>;
pub(super) type IndexWeights<'a> = super::super::operators::IndexWeights<super::Linear<'a>>;
