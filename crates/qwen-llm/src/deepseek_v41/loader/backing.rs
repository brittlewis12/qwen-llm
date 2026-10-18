use super::*;
use crate::{
    metal::{MetalContext, MetalTensor},
    tensor::{GgmlType, TensorDesc},
};
use memmap2::Mmap;
use std::{
    ffi::CString,
    io::Write,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::ffi::OsStrExt,
    },
};

pub(super) fn padded_bytes(bytes: u64) -> Result<u64, String> {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 || bytes == 0 {
        return Err("invalid backing extent/page size".into());
    }
    let page = page as u64;
    bytes
        .checked_add(page - 1)
        .map(|n| n / page * page)
        .ok_or("backing extent overflow".into())
}

pub(super) struct BackingWriter {
    file: File,
    bytes: u64,
    written: u64,
    path: Option<std::path::PathBuf>,
}

impl Drop for BackingWriter {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl BackingWriter {
    pub(super) fn new(dir: &Path, bytes: u64) -> Result<Self, String> {
        Self::create(dir, bytes, false)
    }

    pub(super) fn persistent(dir: &Path, bytes: u64) -> Result<Self, String> {
        Self::create(dir, bytes, true)
    }

    fn create(dir: &Path, bytes: u64, persistent: bool) -> Result<Self, String> {
        let padded = padded_bytes(bytes)?;
        let mut template = CString::new(dir.join("qwen-weights-XXXXXX").as_os_str().as_bytes())
            .map_err(|e| e.to_string())?
            .into_bytes_with_nul();
        let fd = unsafe { libc::mkstemp(template.as_mut_ptr().cast()) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        let file = unsafe { File::from_raw_fd(fd) };
        // Ephemeral files are unlinked immediately; persistent staging is RAII-owned.
        if !persistent && unsafe { libc::unlink(template.as_ptr().cast()) } != 0 {
            let error = std::io::Error::last_os_error();
            let path = std::ffi::CStr::from_bytes_with_nul(&template).unwrap();
            let _ = std::fs::remove_file(std::ffi::OsStr::from_bytes(path.to_bytes()));
            return Err(error.to_string());
        }
        let writer = Self {
            file,
            bytes,
            written: 0,
            path: persistent.then(|| {
                Path::new(std::ffi::OsStr::from_bytes(&template[..template.len() - 1])).to_owned()
            }),
        };
        let file = &writer.file;
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } == -1 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        #[cfg(target_os = "macos")]
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        file.set_len(padded).map_err(|e| e.to_string())?;
        Ok(writer)
    }

    pub(super) fn write(&mut self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() > MAX_READ_BYTES || bytes.len() as u64 > self.bytes - self.written {
            return Err("backing write exceeds extent/staging bound".into());
        }
        self.file.write_all(bytes).map_err(|e| e.to_string())?;
        self.written += bytes.len() as u64;
        Ok(())
    }

    fn finish(self) -> Result<Mmap, String> {
        if self.written != self.bytes {
            return Err("short backing write".into());
        }
        self.file.sync_all().map_err(|e| e.to_string())?;
        // No other owner can mutate this unlinked file after mapping.
        unsafe { Mmap::map(&self.file) }.map_err(|e| e.to_string())
    }

    pub(super) fn publish(mut self, path: &Path) -> Result<(), String> {
        if self.written != self.bytes {
            return Err("short persistent backing write".into());
        }
        self.file.sync_all().map_err(|e| e.to_string())?;
        let temporary = self.path.as_ref().ok_or("not a persistent writer")?;
        // Never replace an inode that a concurrent loader may have mapped.
        std::fs::hard_link(temporary, path).map_err(|e| e.to_string())?;
        std::fs::remove_file(temporary).map_err(|e| e.to_string())?;
        self.path = None;
        Ok(())
    }

    pub(super) fn tensor(
        self,
        ctx: &MetalContext,
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> Result<MetalTensor, String> {
        let bytes = self.bytes;
        mapped_tensor(ctx, self.finish()?, shape, dtype, bytes)
    }
}

pub(super) fn mapped_tensor(
    ctx: &MetalContext,
    mmap: Mmap,
    shape: Vec<u64>,
    dtype: GgmlType,
    bytes: u64,
) -> Result<MetalTensor, String> {
    let desc = TensorDesc {
        name: "converted weight".into(),
        shape,
        dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: bytes,
    };
    let mmap = Arc::new(mmap);
    let backing = ctx
        .gguf_no_copy_backing(mmap, 0, 32)
        .map_err(|e| e.to_string())?;
    let (eligibility, tensor) = backing.tensor(&desc).map_err(|e| e.to_string())?;
    tensor.ok_or_else(|| format!("weight no-copy backing ineligible: {eligibility:?}"))
}
