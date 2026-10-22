# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf>=0.19.0"]
# ///
"""Tiny generated GGUF only. No real model, inference or GPU imports."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import gguf
import numpy as np

import extract


def fixture(
    root, *, split=False, legacy=False, omit=None, extra=None, wrong_shape=False
):
    arch = "glm5next" if legacy else extract.ARCH
    values = {
        "block_count": 46,
        "nextn_predict_layers": 1,
        "embedding_length": 256,
        "vocab_size": 4,
        "attention.q_lora_rank": 32,
        "attention.kv_lora_rank": 32,
        "attention.head_count": 1,
        "attention.key_length_mla": 32,
        "attention.value_length_mla": 32,
        "attention.indexer.head_count": 1,
        "attention.indexer.key_length": 32,
        "attention.indexer.kpool": 4,
        "expert_count": 2,
        "expert_feed_forward_length": 256,
        "expert_shared_count": 1,
    }
    path = root / "tiny.gguf"
    w = gguf.GGUFWriter(
        path, arch, split_max_tensors=16 if split else 0, small_first_shard=split
    )
    for key, val in values.items():
        w.add_uint32(f"{arch}.{key}", val)
    w.add_array(f"{arch}.attention.head_count_kv", [0] * 45 + [1])
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "c", "d"])
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "glm4")
    w.add_string("tokenizer.chat_template", "{{ messages }}")
    w.add_bool("fixture.bool", True)
    w.add_array("fixture.float_array", [0.25, -0.5])
    meta = {f"{extract.ARCH}.{k}": (v,) for k, v in values.items()}
    meta[f"{extract.ARCH}.attention.head_count_kv"] = ([0] * 45 + [1],)
    meta["tokenizer.ggml.tokens"] = (["a", "b", "c", "d"],)
    shapes = extract.tensor_shapes(meta)
    w.add_tensor("blk.0.attn_norm.weight", np.arange(32, dtype=np.float32))
    for name, dims in shapes.items():
        if name == omit:
            continue
        dims = list(reversed(dims))
        if wrong_shape and name == "blk.45.nextn.hnorm.weight":
            dims = [64]
        dtype = {
            "blk.45.nextn.eh_proj.weight": gguf.GGMLQuantizationType.Q8_0,
            "blk.45.ffn_gate_exps.weight": gguf.GGMLQuantizationType.Q2_K,
            "blk.45.ffn_up_exps.weight": gguf.GGMLQuantizationType.Q2_K,
            "blk.45.ffn_down_exps.weight": gguf.GGMLQuantizationType.Q3_K,
            "output.weight": gguf.GGMLQuantizationType.BF16,
        }.get(name)
        if dtype is not None:
            block, size = gguf.GGML_QUANT_SIZES[dtype]
            byte_shape = (*dims[:-1], dims[-1] // block * size)
            data = np.arange(np.prod(byte_shape), dtype=np.uint8).reshape(byte_shape)
            w.add_tensor(name, data, raw_dtype=dtype)
        else:
            w.add_tensor(name, np.arange(np.prod(dims), dtype=np.float32).reshape(dims))
    if extra:
        w.add_tensor(extra, np.ones(32, dtype=np.float32))
    paths = w.format_shard_names(path)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return paths


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "head.gguf"

    def tearDown(self):
        self.tmp.cleanup()

    def test_copy_inventory_metadata_and_quantized_bytes(self):
        paths = fixture(self.root, split=True)
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        report = extract.extract(paths[0], self.output)
        out = gguf.GGUFReader(self.output)
        self.assertEqual(len(out.tensors), 32)
        self.assertFalse(any(k.startswith("split.") for k in out.fields))
        self.assertFalse(any(t.name.startswith("blk.0.") for t in out.tensors))
        source = {t.name: t for p in paths for t in gguf.GGUFReader(p).tensors}
        for tensor in out.tensors:
            original = source[tensor.name]
            self.assertEqual(tensor.tensor_type, original.tensor_type)
            np.testing.assert_array_equal(tensor.shape, original.shape)
            self.assertEqual(tensor.data.tobytes(), original.data.tobytes())
        self.assertEqual(report["payload_bytes"], sum(t.n_bytes for t in out.tensors))
        self.assertEqual(
            extract.value(out, "tokenizer.chat_template"), "{{ messages }}"
        )
        self.assertEqual(extract.value(out, "glm5-next.block_count"), 46)
        self.assertEqual(extract.value(out, "fixture.bool"), True)
        self.assertEqual(extract.value(out, "fixture.float_array"), [0.25, -0.5])
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        )

    def test_symlink_entrypoint_and_legacy_namespace(self):
        paths = fixture(self.root, split=True, legacy=True)
        backing = self.root / "publisher.gguf_file"
        paths[0].rename(backing)
        paths[0].symlink_to(backing)
        extract.extract(paths[0], self.output)
        reader = gguf.GGUFReader(self.output)
        self.assertEqual(extract.value(reader, "general.architecture"), "glm5-next")
        self.assertFalse(any(k.startswith("glm5next.") for k in reader.fields))

    def test_missing_tensor_and_unexpected_optional_head_refused(self):
        for kwargs in [
            {"omit": "blk.45.nextn.hnorm.weight"},
            {"omit": "output.weight"},
            {"extra": "blk.45.nextn.shared_head_head.weight"},
            {"wrong_shape": True},
        ]:
            with (
                self.subTest(kwargs=kwargs),
                tempfile.TemporaryDirectory(dir=self.root) as d,
            ):
                source = fixture(Path(d), **kwargs)[0]
                with self.assertRaises(ValueError):
                    extract.extract(source, self.output)
                self.assertFalse(os.path.lexists(self.output))

    def test_missing_sibling_refused(self):
        paths = fixture(self.root, split=True)
        paths[-1].unlink()
        with self.assertRaises(FileNotFoundError):
            extract.extract(paths[0], self.output)
        self.assertFalse(self.output.exists())

    def test_source_or_output_collision_preserves_original(self):
        source = fixture(self.root)[0]
        original = source.read_bytes()
        with self.assertRaises(ValueError):
            extract.extract(source, source)
        self.output.write_bytes(b"owned by another caller")
        with self.assertRaises(ValueError):
            extract.extract(source, self.output)
        self.assertEqual(self.output.read_bytes(), b"owned by another caller")
        self.assertEqual(source.read_bytes(), original)

    def test_dangling_output_symlink_refused(self):
        source = fixture(self.root)[0]
        self.output.symlink_to(self.root / "absent")
        with self.assertRaises(ValueError):
            extract.extract(source, self.output)
        self.assertTrue(self.output.is_symlink())
        self.assertFalse((self.root / "absent").exists())

    def test_metadata_conflict_refused(self):
        source = fixture(self.root)[0]
        reader = gguf.GGUFReader(source)

        class ConflictingReader:
            fields = {
                "general.architecture": gguf.ReaderField(
                    0,
                    "general.architecture",
                    [np.frombuffer(b"other", dtype=np.uint8)],
                    [-1],
                    [gguf.GGUFValueType.STRING],
                )
            }

        with self.assertRaisesRegex(ValueError, "conflicting architecture"):
            extract.metadata([reader, ConflictingReader()])

    def test_duplicate_tensor_refused(self):
        source = fixture(self.root)[0]
        reader = gguf.GGUFReader(source)
        shapes = extract.tensor_shapes(extract.metadata([reader]))
        with self.assertRaisesRegex(ValueError, "duplicate tensor"):
            extract.inventory([reader, reader], shapes)


if __name__ == "__main__":
    unittest.main()
