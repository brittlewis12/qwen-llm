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
}

impl BackingWriter {
    pub(super) fn new(dir: &Path, bytes: u64) -> Result<Self, String> {
        let padded = padded_bytes(bytes)?;
        let mut template = CString::new(dir.join("qwen-weights-XXXXXX").as_os_str().as_bytes())
            .map_err(|e| e.to_string())?
            .into_bytes_with_nul();
        let fd = unsafe { libc::mkstemp(template.as_mut_ptr().cast()) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        let file = unsafe { File::from_raw_fd(fd) };
        // Unlink before any setup or payload writes; the descriptor owns cleanup.
        if unsafe { libc::unlink(template.as_ptr().cast()) } != 0 {
            let error = std::io::Error::last_os_error();
            let path = std::ffi::CStr::from_bytes_with_nul(&template).unwrap();
            let _ = std::fs::remove_file(std::ffi::OsStr::from_bytes(path.to_bytes()));
            return Err(error.to_string());
        }
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } == -1 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        #[cfg(target_os = "macos")]
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        file.set_len(padded).map_err(|e| e.to_string())?;
        Ok(Self {
            file,
            bytes,
            written: 0,
        })
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

    pub(super) fn tensor(
        self,
        ctx: &MetalContext,
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> Result<MetalTensor, String> {
        let desc = TensorDesc {
            name: "ephemeral weight".into(),
            shape,
            dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: self.bytes,
        };
        let mmap = Arc::new(self.finish()?);
        let backing = ctx
            .gguf_no_copy_backing(mmap, 0, 32)
            .map_err(|e| e.to_string())?;
        let (eligibility, tensor) = backing.tensor(&desc).map_err(|e| e.to_string())?;
        tensor.ok_or_else(|| format!("weight no-copy backing ineligible: {eligibility:?}"))
    }
}
