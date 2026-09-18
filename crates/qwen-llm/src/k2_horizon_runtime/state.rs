use super::{K2RuntimeError, Result, invalid};

#[derive(Default, Debug)]
pub(super) struct Ledger {
    prefix: u32,
    poisoned: bool,
}

impl Ledger {
    pub fn prefix(&self) -> u32 {
        self.prefix
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    pub fn poison(&mut self) {
        self.poisoned = true;
    }

    pub fn begin(&mut self, tokens: &[u32], vocab: u32, capacity: u32) -> Result<Append<'_>> {
        if self.poisoned {
            return Err(K2RuntimeError::Poisoned);
        }
        let count =
            u32::try_from(tokens.len()).map_err(|_| invalid("append length exceeds u32"))?;
        if count == 0
            || self
                .prefix
                .checked_add(count)
                .is_none_or(|end| end > capacity)
        {
            return Err(invalid("empty append or capacity exceeded"));
        }
        if tokens.iter().any(|&id| id >= vocab || id > i32::MAX as u32) {
            return Err(invalid("token ID outside vocabulary/I32"));
        }
        Ok(Append {
            ledger: self,
            count,
            checked: 0,
            inflight: false,
            submitted: false,
            committed: false,
        })
    }
}

/// Drop is conservative: any abandoned transaction that submitted work poisons
/// the session, even if earlier token commands completed successfully.
pub(super) struct Append<'a> {
    ledger: &'a mut Ledger,
    count: u32,
    checked: u32,
    inflight: bool,
    submitted: bool,
    committed: bool,
}

impl Append<'_> {
    pub fn old_prefix(&self) -> u32 {
        self.ledger.prefix
    }

    pub fn submitting(&mut self) -> Result<()> {
        if self.inflight || self.checked == self.count {
            return Err(invalid("invalid command submission transition"));
        }
        self.submitted = true;
        self.inflight = true;
        Ok(())
    }

    pub fn checked(&mut self) -> Result<()> {
        if !self.inflight {
            return Err(invalid("completion without submitted command"));
        }
        self.inflight = false;
        self.checked += 1;
        Ok(())
    }

    pub fn commit(mut self) -> Result<()> {
        if self.inflight || self.checked != self.count {
            return Err(invalid("incomplete append cannot commit"));
        }
        self.ledger.prefix += self.count;
        self.committed = true;
        Ok(())
    }
}

impl Drop for Append<'_> {
    fn drop(&mut self) {
        if self.submitted && !self.committed {
            self.ledger.poison();
        }
    }
}
