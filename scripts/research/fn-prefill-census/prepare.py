#!/usr/bin/env python3
"""Install the scratch-only Flash-Next prefill census seams, once.

Adds:
  * `append_research` (route-count capture, shared with append-roofline);
  * a route-copy call at the end of the packed MoE bucket stage;
  * a `prefill_continuation_profiled` method so appends after a restored
    snapshot can use the in-tree staged profiler.

Never run in the maintained checkout; use an isolated worktree.
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ROOFLINE = HERE.parent / "append-roofline"


def replace(path, old, new):
    p = ROOT / path
    s = p.read_text()
    assert s.count(old) == 1, (path, s.count(old), old[:80])
    p.write_text(s.replace(old, new))


def main():
    shutil.copyfile(
        ROOFLINE / "instrument.rs", ROOT / "crates/qwen-llm/src/append_research.rs"
    )
    shutil.copyfile(
        HERE / "probe.rs", ROOT / "crates/qwen-llm/examples/fn_prefill_census.rs"
    )
    p = ROOT / "crates/qwen-llm/src/lib.rs"
    p.write_text(
        p.read_text()
        + "\n// Scratch research seam, not a runtime policy.\npub mod append_research;\n"
    )
    replace(
        Path("crates/qwen-llm/src/qwen4exp_moe.rs"),
        "        end_optional(&mut profile, enc, marker)?;\n        Ok(())\n    }\n\n    fn encode_route(",
        """        crate::append_research::routes_copy(ctx,enc,layer,&self.views.route_counts,
        [self.weights.routed_gate,self.weights.routed_up,self.weights.routed_down])?;
        end_optional(&mut profile, enc, marker)?;
        Ok(())
    }

    fn encode_route(""",
    )
    p = ROOT / "crates/qwen-llm/src/qwen4exp_runtime.rs"
    p.write_text(
        p.read_text()
        + """
impl Qwen4ExpLoadedModel<'_> {
    pub fn append_research_inventory(&self) -> serde_json::Value {
        serde_json::Value::Array(self.weights.tensors().map(|(name, t)| {
            serde_json::json!({
                "name": name,
                "shape": t.shape,
                "dtype": format!("{:?}", t.dtype),
                "bytes": t.n_bytes(),
            })
        }).collect())
    }
}
"""
        + """
impl Qwen4ExpTextRunner<'_, '_, '_> {
    /// Scratch research seam: profile one packed continuation command from the
    /// committed position. Same encoder and arithmetic as `prefill_packed_profiled`.
    pub fn prefill_continuation_profiled(
        &mut self,
        token_ids: &[u32],
    ) -> Result<Qwen4ExpPackedProfileOutcome, Qwen4ExpRuntimeError> {
        self.last_prefill_timing = None;
        self.validate_prefill_request(token_ids)?;
        let packed_capacity = self.workspace.packed_prefill_capacity().ok_or_else(|| {
            Qwen4ExpRuntimeError::Invalid("packed prefill was not admitted for this runner".into())
        })?;
        if token_ids.len() < 2 || token_ids.len() > packed_capacity {
            return invalid(format!(
                "profiled packed continuation length {} is outside 2..={packed_capacity}",
                token_ids.len()
            ));
        }
        let start = self.next_position();
        let outcome = match execute_qwen4exp_text_packed_profiled_sync(
            self.ctx,
            token_ids,
            self.ple_table,
            &self.weights,
            &mut self.workspace,
        ) {
            Ok(outcome) => outcome,
            Err(source) => {
                return Err(Qwen4ExpRuntimeError::Prefill {
                    committed: self.next_position().saturating_sub(start),
                    requested: token_ids.len(),
                    source: Box::new(source),
                });
            }
        };
        self.last_token_timing = None;
        let mut timing = Qwen4ExpPrefillTiming::new(token_ids.len(), token_ids.len(), false);
        timing.record(outcome.token);
        self.last_prefill_timing = Some(timing);
        Ok(outcome)
    }
}
"""
    )
    print("Installed scratch-only prefill census seams")


if __name__ == "__main__":
    main()
