use super::*;
use crate::gguf::GgufShardStamp;
use std::path::{Component, Path};

/// Separate logical source groups; auxiliary tables are never backbone splits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceGroups {
    pub backbone: Vec<GgufShardStamp>,
    pub auxiliary: Vec<GgufShardStamp>,
}

pub(super) struct TableContext {
    pub(super) auxiliary: Option<Arc<GgufFile>>,
    pub(super) tables: Vec<engram::TableView>,
    pub(super) stamps: SourceGroups,
}

impl std::fmt::Debug for TableContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableContext")
            .field("tables", &self.tables)
            .field("stamps", &self.stamps)
            .finish()
    }
}

impl TableContext {
    pub(super) fn bookkeeping_bytes(&self) -> Result<u64, LoadError> {
        let mut bytes = std::mem::size_of::<Self>() as u64;
        for stamp in self.stamps.backbone.iter().chain(&self.stamps.auxiliary) {
            let one = add(
                std::mem::size_of_val(stamp) as u64,
                add(stamp.path.as_os_str().len() as u64, 128)?,
            )?;
            bytes = add(bytes, add(one, one)?)?;
        }
        for table in &self.tables {
            let shape = (table.tensor.shape.len() as u64)
                .checked_mul(8)
                .ok_or_else(|| invalid("table descriptor bytes overflow"))?;
            let one = add(
                std::mem::size_of_val(table) as u64,
                add(table.tensor.name.len() as u64, add(shape, 128)?)?,
            )?;
            bytes = add(bytes, add(one, one)?)?;
        }
        Ok(bytes)
    }

    pub(super) fn revalidate(&self, backbone: &GgufFile) -> Result<(), LoadError> {
        let backbone = backbone.revalidate_retained_shard_stamps()?;
        let auxiliary = self
            .auxiliary
            .as_ref()
            .map(|g| g.revalidate_retained_shard_stamps())
            .transpose()?
            .unwrap_or_default();
        if backbone != self.stamps.backbone || auxiliary != self.stamps.auxiliary {
            return Err(invalid("Engram source groups changed after planning"));
        }
        Ok(())
    }
}

pub(super) fn resolve(
    g: &GgufFile,
    hash: &HashConfig,
    layers: usize,
) -> Result<TableContext, LoadError> {
    let key = "deepseek41.engram.file";
    let auxiliary = if g.model.metadata().get(key).is_some() {
        let name = g
            .get_str(key)
            .ok_or_else(|| invalid("Engram file must be a relative path"))?;
        let path = Path::new(name);
        if name.is_empty()
            || name.contains('\\')
            || !path.components().all(|c| matches!(c, Component::Normal(_)))
        {
            return Err(invalid("unsafe relative Engram companion path"));
        }
        if hash.layers.is_empty() || g.tensors.iter().any(|t| engram::is_table(&t.name)) {
            return Err(invalid(
                "ambiguous inline/auxiliary Engram tables or empty Engram layers",
            ));
        }
        let source = g
            .get_str("deepseek41.engram.source")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| invalid("missing declared Engram source identity"))?;
        // Use the logical snapshot parent, NOT canonicalize(): HF shard symlinks
        // point into blobs while the companion lives beside the snapshot link.
        let parent = g
            .primary_shard()
            .path
            .parent()
            .ok_or_else(|| invalid("missing backbone parent"))?;
        let logical_path = parent.join(path);
        let aux = Arc::new(GgufFile::from_opened_file(
            std::fs::File::open(&logical_path)?,
            logical_path,
        )?);
        if aux.architecture().as_deref() != Some("deepseek41-engram") || aux.shard_count() != 1 {
            return Err(invalid(
                "companion must be a standalone deepseek41-engram container",
            ));
        }
        if aux.get_str("deepseek41.engram.source") != Some(source) {
            return Err(invalid("Engram declared source identity mismatch"));
        }
        if aux.tensors.len() != hash.layers.len()
            || aux
                .tensors
                .iter()
                .any(|t| !engram::is_table(&t.name) || t.name == "per_layer_token_embd.weight")
        {
            return Err(invalid(
                "companion must contain only the declared per-layer lookup tables",
            ));
        }
        Some(aux)
    } else {
        None
    };
    let table_source = auxiliary.as_deref().unwrap_or(g);
    let tables = engram::resolve_tables(table_source, hash, layers)?;
    if auxiliary.is_some() {
        for table in &tables {
            engram::validate(g, &table.tensor)?;
        }
    }
    let stamps = SourceGroups {
        backbone: g.revalidate_retained_shard_stamps()?,
        auxiliary: auxiliary
            .as_ref()
            .map(|g| g.revalidate_retained_shard_stamps())
            .transpose()?
            .unwrap_or_default(),
    };
    Ok(TableContext {
        auxiliary,
        tables,
        stamps,
    })
}
