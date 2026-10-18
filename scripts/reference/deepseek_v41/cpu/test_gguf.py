import json
from pathlib import Path
import re
import struct
import tempfile
import unittest

import torch

from gguf_weights import ALIASES, GGUFWeights, decode_bytes


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def value(v):
    if isinstance(v, str):
        return struct.pack("<I", 8) + string(v)
    if isinstance(v, list):
        return struct.pack("<IIQ", 9, 11, len(v)) + b"".join(
            struct.pack("<q", x) for x in v
        )
    return struct.pack("<Iq", 11, v)


def write_gguf(path, tensors, metadata=None):
    metadata = {"general.architecture": "deepseek41", **(metadata or {})}
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    header += b"".join(string(k) + value(v) for k, v in metadata.items())
    payload = bytearray()
    for name, shape, kind, raw in tensors:
        payload.extend(b"\0" * (-len(payload) % 32))
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        header += struct.pack("<IQ", kind, len(payload))
        payload.extend(raw)
    path.write_bytes(header + b"\0" * (-len(header) % 32) + payload)


class GGUFTest(unittest.TestCase):
    def test_decoders(self):
        levels = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
        packed = bytes(i | ((15 - i) << 4) for i in range(16))
        out = decode_bytes(bytes([128]) + packed, 39, (1, 32))
        torch.testing.assert_close(out[0], 2 * torch.tensor(levels + levels[::-1]))
        for exponent in (0, 1, 254, 255):
            out = decode_bytes(bytes([exponent]) + b"\x11" * 16, 39, (32,))
            self.assertEqual(out[0].item(), 2.0 ** (exponent - 128))
        codes = list(range(-16, 16))
        out = decode_bytes(struct.pack("<e32b", 0.25, *codes), 8, (1, 32))
        torch.testing.assert_close(out[0], torch.tensor(codes) * 0.25)
        torch.testing.assert_close(
            decode_bytes(struct.pack("<HH", 0x3F80, 0xC020), 30, (2,)),
            torch.tensor([1.0, -2.5]),
        )
        torch.testing.assert_close(
            decode_bytes(struct.pack("<ff", 1.25, -3.0), 0, (2,)),
            torch.tensor([1.25, -3.0]),
        )
        with self.assertRaises(ValueError):
            decode_bytes(b"\0" * 16, 39, (32,))

    def test_experts_engram_bounds_and_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.gguf"
            blocks = b"".join(
                bytes([127]) + bytes([i | (i << 4)]) * 16 for i in range(6)
            )
            qrow = struct.pack("<e32b", 0.5, *range(32))
            write_gguf(
                path,
                [
                    ("blk.0.ffn_gate_exps.weight", (3, 2, 32), 39, blocks),
                    ("blk.0.engram_embd.weight", (70, 32), 8, qrow * 70),
                ],
            )
            store = GGUFWeights(path, rows=32)
            try:
                self.assertEqual(
                    store.shape("layers.0.ffn.experts.2.w1.weight"), (2, 32)
                )
                torch.testing.assert_close(
                    store.decode("layers.0.ffn.experts.2.w1.weight", 0, 2),
                    torch.tensor([[2.0] * 32, [3.0] * 32]),
                )
                name = "layers.0.engram.embed.weight"
                out = store.embedding(name, torch.tensor([[69, 0, 69]]), True)
                torch.testing.assert_close(
                    out, (torch.arange(32) * 0.5).expand(1, 3, 32)
                )
                for lo, hi in [(-1, 1), (69, 71), (0, 33), (2, 1)]:
                    with self.assertRaises(ValueError):
                        store.decode(name, lo, hi)
                with self.assertRaises(ValueError):
                    store.tensor(name)
                with self.assertRaises(TypeError):
                    store.data[0] = 0
            finally:
                store.close()
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                GGUFWeights(path)

    def test_metadata_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.gguf"
            write_gguf(path, [], {"deepseek41.expert_count": 3})
            store = GGUFWeights(path)
            try:
                store.validate_config({"n_routed_experts": 3})
                with self.assertRaises(ValueError):
                    store.validate_config({"n_routed_experts": 2})
            finally:
                store.close()

    def test_operative_aliases(self):
        cases = [
            ("attention.kv_source_layers", "kv_source_layer_ids", [2, 8], [2, 9]),
            (
                "attention.indexer.source_layers",
                "index_source_layer_ids",
                [2, 8],
                [2, 9],
            ),
            (
                "attention.indexer.candidate_source_layer",
                "candidate_source_layer_id",
                20,
                21,
            ),
            ("attention.indexer.candidate_block_size", "candidate_block_size", 8, 16),
            (
                "attention.indexer.candidate_top_k_blocks",
                "candidate_topk_blocks",
                2048,
                1024,
            ),
            ("attention.o_lora_rank", "o_lora_rank", 1024, 512),
            ("attention.o_groups", "o_groups", 8, 4),
            ("engram.layers", "engram_layer_ids", [1, 14], [1, 15]),
            ("engram.n_heads", "engram_n_heads", 8, 4),
            ("engram.head_dim", "engram_head_dim", 256, 128),
            (
                "engram.compressed_vocab_size",
                "engram_compressed_vocab_size",
                99092,
                99093,
            ),
            ("engram.pad_token_id", "engram_pad_token_id", 2, 3),
            ("engram.num_embeddings", "engram_num_embeddings", [24, 36], [24, 37]),
            ("engram.rows", "engram_num_embeddings", [24, 36], [24, 37]),
            ("hidden_activation", "hidden_act", "silu", "gelu"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.gguf"
            for alias, field, good, bad in cases:
                with self.subTest(alias=alias):
                    key = "deepseek41." + alias
                    direct = "deepseek41." + field
                    write_gguf(
                        path,
                        [],
                        {
                            key: good,
                            direct: good,
                            "deepseek41.unrelated_annotation": "allowed",
                        },
                    )
                    store = GGUFWeights(path)
                    try:
                        config = {"text_config": {field: good}}
                        store.validate_config(config)
                        store.metadata[key] = bad
                        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                            store.validate_config(config)
                        store.metadata[key] = good
                        store.metadata[direct] = bad
                        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                            store.validate_config(config)
                        with self.assertRaisesRegex(
                            ValueError, "missing independent HF"
                        ):
                            store.validate_config({})
                    finally:
                        store.close()

    def test_fixed_gate_temperature_and_engram_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.gguf"
            write_gguf(
                path,
                [("blk.0.engram_embd.weight", (1, 32), 8, b"\0" * 34)],
                {
                    "deepseek41.engram.encoding": "Q8_0",
                    "deepseek41.gate_temperature": 1,
                },
            )
            store = GGUFWeights(path)
            try:
                store.validate_config({})
                store.metadata["deepseek41.gate_temperature"] = 2
                with self.assertRaisesRegex(ValueError, "gate_temperature"):
                    store.validate_config({})
                store.metadata["deepseek41.gate_temperature"] = 1
                store.metadata["deepseek41.engram.encoding"] = "BF16"
                with self.assertRaisesRegex(ValueError, "encoding"):
                    store.validate_config({})
            finally:
                store.close()

    def test_tiny_forward_matches_hf(self):
        from oracle import Oracle
        from test_cpu import synthetic_checkpoint

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_checkpoint(root, ratio_one=True)
            hf = Oracle(root, root / "hf", max_seq_len=8)
            reverse = {v: k for k, v in ALIASES.items()}
            tensors, banks = [], {}
            for name in hf.store.names:
                if name.endswith(".scale"):
                    continue
                shape = hf.store.shape(name)
                decoded = hf.store.decode(name, 0, shape[0], ".engram.embed." in name)
                match = re.fullmatch(
                    r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight", name
                )
                if match:
                    layer, expert, part = map(int, match.groups())
                    banks.setdefault((layer, part), {})[expert] = decoded
                    continue
                match = re.fullmatch(r"layers\.(\d+)\.(.+)", name)
                disk = f"blk.{match[1]}.{reverse[match[2]]}" if match else reverse[name]
                tensors.append(
                    (disk, tuple(decoded.shape), 0, decoded.numpy().tobytes())
                )
            for (layer, part), experts in banks.items():
                decoded = torch.stack([experts[i] for i in range(len(experts))])
                disk = f"blk.{layer}.ffn_{ {1: 'gate', 2: 'down', 3: 'up'}[part] }_exps.weight"
                tensors.append(
                    (disk, tuple(decoded.shape), 0, decoded.numpy().tobytes())
                )
            metadata = {}
            for key in ("primes", "offsets", "multipliers", "token_map", "pad_id"):
                item = getattr(hf.model.engram_hash, key)
                metadata["deepseek41.engram." + key] = (
                    item.flatten().tolist() if isinstance(item, torch.Tensor) else item
                )
            path = root / "tiny.gguf"
            write_gguf(path, tensors, metadata)
            gg = Oracle(root, root / "gg", max_seq_len=8, gguf=path)
            try:
                torch.testing.assert_close(
                    gg.feed([1, 2, 3, 4, 5, 6, 7]),
                    hf.feed([1, 2, 3, 4, 5, 6, 7]),
                    rtol=0,
                    atol=0,
                )
                for dump in (root / "hf").glob("*.pt"):
                    torch.testing.assert_close(
                        torch.load(root / "gg" / dump.name),
                        torch.load(dump),
                        rtol=0,
                        atol=0,
                    )
                self.assertEqual(
                    json.loads((root / "gg/run.json").read_text())["router_tie_policy"],
                    "lower_expert_id_then_id_sorted",
                )
                for field in (
                    "pad_id",
                    "token_map",
                    "primes",
                    "offsets",
                    "multipliers",
                ):
                    key = "deepseek41.engram." + field
                    original = gg.store.metadata[key]
                    bad = (
                        [original[0] + 1, *original[1:]]
                        if isinstance(original, list)
                        else original + 1
                    )
                    gg.store.metadata[key] = bad
                    with (
                        self.subTest(hash_field=field),
                        self.assertRaisesRegex(ValueError, "Engram mismatch"),
                    ):
                        gg.store.validate_model(gg.model, gg.module)
                    gg.store.metadata[key] = original
                name = "layers.0.ffn.experts.0.w1.weight"
                shape, kind, offset = gg.store.names[name]
                gg.store.names[name] = ((shape[0] + 1, shape[1]), kind, offset)
                with self.assertRaises(ValueError):
                    gg.store.validate_model(gg.model, gg.module)
            finally:
                gg.store.close()


if __name__ == "__main__":
    unittest.main()
