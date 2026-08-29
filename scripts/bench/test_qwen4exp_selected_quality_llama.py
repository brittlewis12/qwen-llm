from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qwen4exp_selected_quality_analyze as analyzer
import qwen4exp_selected_quality_llama as llama


def valid_context() -> dict[str, object]:
    return {
        "requested_n_ctx": 4_224,
        "effective_n_ctx": 4_352,
        "effective_n_ctx_seq": 4_352,
        "required_decoded_tokens": 4_195,
        "context_padding_multiple": 256,
        "n_batch": 512,
        "n_ubatch": 512,
        "n_seq_max": 1,
        "kv_unified": False,
        "kv_type_k": "f16",
        "kv_type_v": "f16",
        "flash_attention": "enabled",
        "gpu_layers": "all",
        "memory_cleared_with_data_before_each_operation": True,
    }


class SelectedQualityLlamaContextTests(unittest.TestCase):
    def test_accepts_exact_requested_and_effective_context(self) -> None:
        self.assertEqual(llama.validate_core_context(valid_context()), valid_context())

    def test_rejects_context_realization_drift(self) -> None:
        cases = {
            "unrounded effective capacity": {"effective_n_ctx": 4_224},
            "arbitrary larger capacity": {"effective_n_ctx": 4_608},
            "per-sequence mismatch": {"effective_n_ctx_seq": 4_224},
            "required-token drift": {"required_decoded_tokens": 4_196},
            "padding drift": {"context_padding_multiple": 128},
            "boolean integer alias": {"requested_n_ctx": True},
            "unified KV": {"kv_unified": True},
            "quantized K cache": {"kv_type_k": "q8_0"},
            "quantized V cache": {"kv_type_v": "q8_0"},
        }
        for label, updates in cases.items():
            with self.subTest(label=label):
                context = valid_context()
                context.update(updates)
                with self.assertRaises(RuntimeError):
                    llama.validate_core_context(context)

    def test_rejects_missing_and_legacy_context_fields(self) -> None:
        missing = valid_context()
        del missing["effective_n_ctx_seq"]
        with self.assertRaises(RuntimeError):
            llama.validate_core_context(missing)

        legacy = valid_context()
        del legacy["requested_n_ctx"]
        del legacy["effective_n_ctx"]
        del legacy["effective_n_ctx_seq"]
        legacy["n_ctx"] = 4_224
        with self.assertRaises(RuntimeError):
            llama.validate_core_context(legacy)

    def test_padding_contract_covers_longest_operation(self) -> None:
        rounded = (
            (llama.REQUESTED_CONTEXT_TOKENS + llama.CONTEXT_PADDING_MULTIPLE - 1)
            // llama.CONTEXT_PADDING_MULTIPLE
            * llama.CONTEXT_PADDING_MULTIPLE
        )
        self.assertEqual(rounded, llama.EFFECTIVE_CONTEXT_TOKENS)
        self.assertLessEqual(
            llama.REQUIRED_DECODED_TOKENS, llama.REQUESTED_CONTEXT_TOKENS
        )


class SelectedQualityPacketTests(unittest.TestCase):
    def test_v3_manifest_preserves_frozen_v2_payload(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        manifest_path = (
            repository
            / "docs/bench/2026-08-28-qwen4exp-selected-quality-v3-prereg"
            / "fixtures.json"
        )
        manifest, tokens, fixtures, operations = llama.validate_fixtures(manifest_path)
        self.assertEqual(manifest["packet_id"], llama.PACKET_ID)
        self.assertFalse(
            manifest["lineage"]["semantic_outputs_inspected_before_freeze"]
        )
        self.assertEqual(len(tokens), 21)
        self.assertEqual(len(fixtures), 21)
        self.assertEqual(len(operations), 20)

        changed_payload = copy.deepcopy(manifest)
        changed_payload["execution"]["operation_plan"][0]["arm"] = "D"
        with self.assertRaises(RuntimeError):
            llama.validate_packet_lineage(manifest_path, changed_payload)

        changed_preregistration = copy.deepcopy(manifest)
        changed_preregistration["preregistration"]["bytes"] += 1
        with self.assertRaises(RuntimeError):
            llama.validate_packet_lineage(manifest_path, changed_preregistration)

    def test_analysis_rejects_cross_commit_evidence(self) -> None:
        commit = "1" * 40
        self.assertEqual(analyzer.require_common_source_commit(commit, commit), commit)
        with self.assertRaises(RuntimeError):
            analyzer.require_common_source_commit(commit, "2" * 40)
        with self.assertRaises(RuntimeError):
            analyzer.require_common_source_commit("z" * 40, "z" * 40)


if __name__ == "__main__":
    unittest.main()
