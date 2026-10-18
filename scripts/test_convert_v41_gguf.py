#!/usr/bin/env -S uv run --no-project python
"""Tiny real-safetensors parity tests; never reads an installed model.

Run with --canonical PATH --patched PATH using the CPU oracle Python environment.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import convert_v41_gguf as driver


def make_fixture(root, split_engram=False, bad_scale=False):
    import numpy as np
    import torch
    from conversion.deepseek import _v41_build_compressed_token_map
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    root.mkdir(parents=True)
    vocab = {"<bos>": 0, "<eos>": 1, "<pad>": 2}
    vocab.update(
        {c: i + 3 for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    )
    backend = Tokenizer(models.BPE(vocab, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel()
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    tok.save_pretrained(root)
    _, compressed = _v41_build_compressed_token_map(tok)
    config = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": len(tok),
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "qk_rope_head_dim": 16,
            "q_lora_rank": 32,
            "o_lora_rank": 32,
            "o_groups": 2,
            "rms_norm_eps": 1e-20,
            "max_position_embeddings": 128,
            "rope_theta": 10000,
            "moe_intermediate_size": 64,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.5,
            "sliding_window": 16,
            "swiglu_limit": 10.0,
            "compress_ratios": [0, 0],
            "compress_rope_theta": 160000,
            "index_n_heads": 2,
            "index_head_dim": 32,
            "index_topk": 4,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "engram_layer_ids": [1],
            "engram_num_embeddings": [36],
            "engram_n_heads": 2,
            "engram_head_dim": 64,
            "engram_max_ngram_size": 3,
            "engram_vocab_size": 5,
            "engram_compressed_vocab_size": compressed,
        },
    }
    (root / "config.json").write_text(json.dumps(config))
    weights, scales = {}, {}

    def scale(shape):
        # Exercise true E8M0 values, not a float scale approximation.
        values = (torch.arange(int(np.prod(shape))) % 7 + 123).to(torch.uint8)
        return values.reshape(shape).view(torch.float8_e8m0fnu)

    def fp8(shape):
        values = (torch.arange(int(np.prod(shape))) % 41 - 20).float() / 8
        return values.reshape(shape).to(torch.float8_e4m3fn)

    weights["embed.weight"] = (
        torch.arange(len(tok) * 64).reshape(len(tok), 64).to(torch.bfloat16) / 128
    )
    weights["head.weight"] = weights["embed.weight"].clone()
    weights["norm.weight"] = torch.ones(64, dtype=torch.bfloat16)
    for bid in range(2):
        prefix = f"layers.{bid}"
        weights[f"{prefix}.attn.attn_sink"] = torch.tensor([0.25, -0.5])
        weights[f"{prefix}.attn.wkv.weight"] = fp8((64, 64))
        scales[f"{prefix}.attn.wkv.scale"] = scale((2, 2))
        weights[f"{prefix}.ffn.gate.weight"] = torch.ones((2, 64), dtype=torch.bfloat16)
        for eid in range(2):
            for proj in ("w1", "w2", "w3"):
                name = f"{prefix}.ffn.experts.{eid}.{proj}"
                weights[name + ".weight"] = (
                    torch.arange(64 * 32).reshape(64, 32).to(torch.int8)
                )
                scales[name + ".scale"] = scale((64, 2))
    weights["layers.1.engram.embed.weight"] = fp8((36, 64))
    engram_scale = scale((35 if bad_scale else 36, 2))
    (scales if split_engram else weights)["layers.1.engram.embed.scale"] = engram_scale
    weights["layers.1.engram.k_weight"] = torch.ones((2, 64, 64), dtype=torch.bfloat16)
    weights["layers.1.engram.q_weight"] = torch.ones((2, 64, 64), dtype=torch.bfloat16)
    weights["vision.unused.weight"] = torch.zeros((2, 2), dtype=torch.bfloat16)
    weights["mtp.0.unused.weight"] = torch.zeros((2, 2), dtype=torch.bfloat16)
    index = {}
    for filename, tensors in (
        ("model-00001-of-00002.safetensors", weights),
        ("model-00002-of-00002.safetensors", scales),
    ):
        save_file(tensors, root / filename)
        index.update({name: filename for name in tensors})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )


def fixture_vocab(model_class):
    # A synthetic byte-level tokenizer has no entry in the canonical hash registry.
    # Only its pre-tokenizer label is supplied; vocabulary and Engram hashes remain canonical.
    return patch.object(
        model_class, "get_vocab_base_pre", lambda self, tokenizer: "gpt-2"
    )


def child(args):
    gguf, torch, cls = driver.import_converter(Path(args.source))
    torch.set_num_threads(1)
    root, out = Path(args.model), Path(args.output)
    with fixture_vocab(cls), torch.inference_mode():
        if args.operation == "fixture":
            make_fixture(root, args.split_engram, args.bad_scale)
        elif args.operation == "canonical":
            model = cls(
                root,
                gguf.LlamaFileType.MOSTLY_BF16,
                out,
                eager=False,
                use_temp_file=False,
            )
            model._V41_ENGRAM_CHUNK_ROWS = args.chunk_rows
            model.write()
        elif args.operation in ("patched", "preflight"):
            with patch(
                "tempfile.SpooledTemporaryFile",
                side_effect=AssertionError("global spool"),
            ):
                model, report = driver.prepare(
                    Path(args.source), root, out, args.chunk_rows
                )
                if args.operation == "patched":
                    # Input mappings are allowed; writable staging mappings are not.
                    import numpy as np

                    original = np.memmap

                    def input_only(*a, **kw):
                        if kw.get("mode", "r+") not in ("r", "c"):
                            raise AssertionError("writable staging memmap")
                        return original(*a, **kw)

                    with patch.object(np, "memmap", input_only):
                        driver.write_output(model, out, report["output_bytes"])
            print(json.dumps(report))


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="v41-tiny-")
        cls.root = Path(cls.temp.name)
        cls.model = cls.root / "same" / "DeepSeek-V4.1-Tiny"
        cls.split = cls.root / "split" / "DeepSeek-V4.1-Tiny"
        cls.run_child("fixture", cls.model, canonical=True)
        cls.run_child("fixture", cls.split, canonical=True, extras=["--split-engram"])
        cls.reference = cls.root / "canonical.gguf"
        cls.run_child("canonical", cls.model, cls.reference, canonical=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def run_child(
        cls,
        operation,
        model,
        output=None,
        canonical=False,
        rows=7,
        extras=(),
        check=True,
    ):
        env = dict(
            os.environ, V41_ENGRAM_TMPDIR=str(cls.root), PYTHONDONTWRITEBYTECODE="1"
        )
        command = [
            sys.executable,
            "-B",
            __file__,
            "--operation",
            operation,
            "--source",
            str(CANONICAL if canonical else PATCHED),
            "--model",
            str(model),
            "--output",
            str(output or cls.root / "not-written.gguf"),
            "--chunk-rows",
            str(rows),
            *extras,
        ]
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, check=False
        )
        if check and result.returncode:
            raise AssertionError(
                f"{operation} failed:\n{result.stdout}\n{result.stderr}"
            )
        return result

    def test_source_patch_audit(self):
        self.assertEqual(driver.check_source(CANONICAL), "canonical")
        self.assertEqual(driver.check_source(PATCHED), "patched")

    def test_canonical_whole_file_bytes_and_headers(self):
        gguf, _, _ = driver.import_converter(PATCHED)
        for i, (model, rows) in enumerate(
            ((self.model, 1), (self.model, 7), (self.model, 36), (self.split, 11))
        ):
            with self.subTest(model=model, rows=rows):
                output = self.root / f"patched-{i}.gguf"
                before = set(self.root.rglob("*"))
                result = self.run_child("patched", model, output, rows=rows)
                report = json.loads(result.stdout)
                self.assertEqual(set(self.root.rglob("*")) - before, {output})
                self.assertEqual(output.read_bytes(), self.reference.read_bytes())
                self.assertEqual(output.stat().st_size, report["output_bytes"])
                reader = gguf.GGUFReader(output)
                expected = gguf.GGUFReader(self.reference)
                self.assertEqual(
                    [(t.name, t.tensor_type, t.shape.tolist()) for t in reader.tensors],
                    [
                        (t.name, t.tensor_type, t.shape.tolist())
                        for t in expected.tensors
                    ],
                )
                kinds = {t.tensor_type for t in reader.tensors}
                self.assertTrue(
                    {
                        gguf.GGMLQuantizationType.MXFP4,
                        gguf.GGMLQuantizationType.Q8_0,
                        gguf.GGMLQuantizationType.BF16,
                        gguf.GGMLQuantizationType.F32,
                    }
                    <= kinds
                )

    def test_preflight_has_no_payload_reads_or_writes(self):
        before = set(self.root.rglob("*"))
        result = self.run_child("preflight", self.split)
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assertEqual(json.loads(result.stdout)["temporary_payload_bytes"], 0)

    def test_bad_scale_fails_before_output(self):
        model = self.root / "bad" / "DeepSeek-V4.1-Tiny"
        self.run_child("fixture", model, canonical=True, extras=["--bad-scale"])
        output = self.root / "bad.gguf"
        result = self.run_child("patched", model, output, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Scale layout mismatch", result.stderr)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name + ".incomplete").exists())

    def test_output_never_overwrites(self):
        output = self.root / "existing.gguf"
        output.write_bytes(b"preserve me")
        result = self.run_child("patched", self.model, output, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), b"preserve me")

    def test_metadata_mismatch_fails_before_output(self):
        model = self.root / "bad-hashes" / "DeepSeek-V4.1-Tiny"
        self.run_child("fixture", model, canonical=True)
        path = model / "config.json"
        config = json.loads(path.read_text())
        config["text_config"]["engram_compressed_vocab_size"] += 1
        path.write_text(json.dumps(config))
        output = self.root / "bad-hashes.gguf"
        result = self.run_child("patched", model, output, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compressed vocab size", result.stderr)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name + ".incomplete").exists())

    def test_failure_keeps_only_incomplete_output_and_refuses_resume(self):
        gguf, _, cls = driver.import_converter(PATCHED)
        output = self.root / "interrupted.gguf"
        with fixture_vocab(cls):
            model, report = driver.prepare(PATCHED, self.model, output, 7)
        stream = next(
            ti.tensor
            for shard in model.gguf_writer.tensors
            for ti in shard.values()
            if isinstance(ti.tensor, gguf.LazyChunkedTensor)
        )

        def fail():
            raise OSError("simulated source read failure")

        stream._chunks[1] = fail
        with self.assertRaisesRegex(OSError, "simulated source read failure"):
            driver.write_output(model, output, report["output_bytes"])
        partial_dir = output.with_name(output.name + ".incomplete")
        partial = partial_dir / output.name
        self.assertFalse(output.exists())
        self.assertEqual(list(partial_dir.iterdir()), [partial])
        self.assertGreater(partial.stat().st_size, report["header_bytes_with_padding"])
        self.assertLess(partial.stat().st_size, report["output_bytes"])
        contents = partial.read_bytes()
        with self.assertRaises(FileExistsError):
            driver.write_output(model, output, report["output_bytes"])
        self.assertEqual(partial.read_bytes(), contents)

    def test_chunks_bounded_and_released(self):
        gguf, torch, cls = driver.import_converter(PATCHED)
        import numpy as np

        with fixture_vocab(cls):
            model, _ = driver.prepare(PATCHED, self.split, self.root / "unused.gguf", 7)
        streams = [
            ti.tensor
            for shard in model.gguf_writer.tensors
            for ti in shard.values()
            if isinstance(ti.tensor, gguf.LazyChunkedTensor)
        ]
        self.assertEqual(len(streams), 1)
        stream = streams[0]
        self.assertEqual(stream.nbytes, 36 * 68)
        previous = None
        shapes = []
        original = gguf.quants.quantize
        import safetensors

        original_open = safetensors.safe_open
        slices = []

        class Slice:
            def __init__(self, source, name):
                self.source, self.name = source, name

            def __getitem__(self, key):
                self_test.assertLessEqual(key[0].stop - key[0].start, 7)
                slices.append(self.name)
                return self.source[key]

        class Open:
            def __init__(self, *a, **kw):
                self.source = original_open(*a, **kw)

            def __enter__(self):
                self.source.__enter__()
                return self

            def __exit__(self, *a):
                return self.source.__exit__(*a)

            def get_slice(self, name):
                return Slice(self.source.get_slice(name), name)

            def get_tensor(self, name):
                raise AssertionError("whole-table materialization")

        self_test = self

        def checked(data, qtype):
            nonlocal previous
            gc.collect()
            self.assertIsNone(previous() if previous else None)
            self.assertLessEqual(data.shape[0], 7)
            self.assertEqual(data.dtype, np.float32)
            shapes.append(data.shape)
            previous = weakref.ref(data)
            return original(data, qtype)

        with (
            tempfile.TemporaryFile() as fp,
            patch.object(gguf.quants, "quantize", checked),
            patch.object(safetensors, "safe_open", Open),
            torch.inference_mode(),
        ):
            stream.tofile(fp)
            self.assertEqual(fp.tell(), 36 * 68)
        self.assertEqual([s[0] for s in shapes], [7, 7, 7, 7, 7, 1])
        self.assertEqual(slices.count("layers.1.engram.embed.weight"), 6)
        self.assertEqual(slices.count("layers.1.engram.embed.scale"), 6)
        experts = [
            ti.tensor
            for shard in model.gguf_writer.tensors
            for ti in shard.values()
            if ti.dtype == gguf.GGMLQuantizationType.MXFP4
        ]
        self.assertEqual(len(experts), 6)
        self.assertTrue(
            all(
                isinstance(t, gguf.LazyNumpyTensor) and t._data is None for t in experts
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path)
    parser.add_argument("--patched", type=Path)
    parser.add_argument("--operation")
    parser.add_argument("--source")
    parser.add_argument("--model")
    parser.add_argument("--output")
    parser.add_argument("--chunk-rows", type=int, default=7)
    parser.add_argument("--split-engram", action="store_true")
    parser.add_argument("--bad-scale", action="store_true")
    args, extra = parser.parse_known_args()
    if args.operation:
        child(args)
    else:
        CANONICAL, PATCHED = args.canonical, args.patched
        if CANONICAL is None or PATCHED is None:
            parser.error("--canonical and --patched are required")
        unittest.main(argv=[sys.argv[0], *extra])
