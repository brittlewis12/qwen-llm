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

    pub fn begin(&mut self, tokens: &[u32], vocab: u32, capacity: u32) -> Result<Transaction<'_>> {
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
        Ok(self.transaction(Operation::Append(count)))
    }

    pub fn begin_readout(&mut self) -> Result<Transaction<'_>> {
        if self.poisoned {
            return Err(K2RuntimeError::Poisoned);
        }
        Ok(self.transaction(Operation::Readout))
    }

    fn transaction(&mut self, operation: Operation) -> Transaction<'_> {
        Transaction {
            ledger: self,
            operation,
            checked: 0,
            inflight: false,
            submitted: false,
            committed: false,
        }
    }
}

enum Operation {
    Append(u32),
    Readout,
}

impl Operation {
    fn commands(&self) -> u32 {
        match self {
            Self::Append(count) => *count,
            Self::Readout => 1,
        }
    }

    fn advance(&self) -> u32 {
        match self {
            Self::Append(count) => *count,
            Self::Readout => 0,
        }
    }
}

/// Drop is conservative: any abandoned transaction that submitted work poisons
/// the session, even if earlier token commands completed successfully.
pub(super) struct Transaction<'a> {
    ledger: &'a mut Ledger,
    operation: Operation,
    checked: u32,
    inflight: bool,
    submitted: bool,
    committed: bool,
}

impl Transaction<'_> {
    pub fn old_prefix(&self) -> u32 {
        self.ledger.prefix
    }

    pub fn submitting(&mut self) -> Result<()> {
        if self.inflight || self.checked == self.operation.commands() {
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
        if self.inflight || self.checked != self.operation.commands() {
            return Err(invalid("incomplete transaction cannot commit"));
        }
        self.ledger.prefix += self.operation.advance();
        self.committed = true;
        Ok(())
    }
}

impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        if self.submitted && !self.committed {
            self.ledger.poison();
        }
    }
}
