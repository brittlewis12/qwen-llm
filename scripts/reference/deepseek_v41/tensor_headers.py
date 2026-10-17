"""Pinned HF1017 metadata-only fixture gate. Run with --capture to refresh.

Headers describe storage, not tensor values, nibble order, or numeric correctness.
No safetensors payload is requested or retained.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
REPOSITORY = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
BASE = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/"
MAX_HEADER = 1024 * 1024
MAX_INDEX = 8 * 1024 * 1024
MAX_TOTAL = 16 * 1024 * 1024
SOURCE_HASHES = {
    "config.json": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
    "inference/model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
}
INDEX_HASH = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
WIDTHS = {
    "BF16": 2,
    "F32": 4,
    "F16": 2,
    "I64": 8,
    "I32": 4,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class Reader:
    def __init__(self, opener=None):
        self.opener = opener or urllib.request.build_opener(BoundedRedirects()).open
        self.used = 0

    def fetch(self, name, start=None, length=None, total=None):
        ranged = start is not None
        limit = length if ranged else MAX_INDEX
        require(
            type(limit) is int and 0 < limit <= (MAX_HEADER if ranged else MAX_INDEX),
            "request exceeds metadata bound",
        )
        require(self.used + limit <= MAX_TOTAL, "total read limit exceeded")
        headers = {"Accept-Encoding": "identity"}
        if ranged:
            require(
                type(start) is int and 0 <= start <= MAX_HEADER + 8,
                "invalid range start",
            )
            headers["Range"] = f"bytes={start}-{start + length - 1}"
        # Distinct URLs prevent intermediary caches from reusing the length response.
        url = (
            BASE
            + name
            + (f"?header_range={start}-{start + length - 1}" if ranged else "")
        )
        with self.opener(
            urllib.request.Request(url, headers=headers), timeout=60
        ) as response:
            require(
                response.headers.get("Content-Encoding", "identity") == "identity",
                "encoded response",
            )
            content_length = response.headers.get("Content-Length", "")
            require(
                re.fullmatch(r"[0-9]+", content_length) is not None,
                "missing content length",
            )
            count = int(content_length)
            if ranged:
                require(response.status == 206, "server ignored Range")
                match = re.fullmatch(
                    r"bytes ([0-9]+)-([0-9]+)/([0-9]+)",
                    response.headers.get("Content-Range", ""),
                )
                require(match is not None, "malformed Content-Range")
                first, last, size = map(int, match.groups())
                require(
                    (first, last) == (start, start + length - 1) and last < size,
                    "mismatched Content-Range",
                )
                require(total is None or total == size, "source size changed")
                require(count == length, "mismatched content length")
            else:
                require(response.status == 200 and count <= MAX_INDEX, "oversize index")
                size = count
            # Never probe past the requested header into the first tensor byte.
            raw = response.read(count)
            self.used += count
            require(len(raw) == count, "truncated response")
            return raw, size


class BoundedRedirects(urllib.request.HTTPRedirectHandler):
    """Never drain a redirect body: urllib's default handler reads it unbounded."""

    MAX_HOPS = 5

    def http_error_302(self, request, response, code, message, headers):
        response.close()
        hops = getattr(request, "metadata_redirect_hops", 0)
        require(hops < self.MAX_HOPS, "too many metadata redirects")
        location = headers.get("Location")
        require(isinstance(location, str) and location, "redirect missing Location")
        target = urllib.parse.urljoin(request.full_url, location)
        require(urllib.parse.urlsplit(target).scheme == "https", "non-HTTPS redirect")
        # Do not copy Host or other unredirected headers installed by urllib.
        redirected = urllib.request.Request(target, headers=dict(request.headers))
        redirected.metadata_redirect_hops = hops + 1
        return self.parent.open(redirected, timeout=request.timeout)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def header_length(prefix):
    require(len(prefix) == 8, "invalid length prefix")
    length = struct.unpack("<Q", prefix)[0]
    require(0 < length <= MAX_HEADER and length % 8 == 0, "oversize or invalid header")
    return length


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def descriptors(raw, size):
    require(0 < len(raw) <= MAX_HEADER and len(raw) % 8 == 0, "invalid header size")
    obj = json.loads(raw, object_pairs_hook=unique_pairs)
    require(isinstance(obj, dict), "invalid header object")
    intervals = []
    for key, desc in obj.items():
        if key == "__metadata__":
            continue
        require(
            isinstance(desc, dict) and set(desc) == {"dtype", "shape", "data_offsets"},
            "invalid descriptor",
        )
        shape, offsets = desc["shape"], desc["data_offsets"]
        require(
            isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape),
            "invalid shape",
        )
        require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(type(n) is int for n in offsets),
            "invalid offsets",
        )
        a, b = offsets
        require(0 <= a <= b <= size - 8 - len(raw), "offset outside payload")
        require(desc["dtype"] in WIDTHS, f"unknown dtype {desc['dtype']}")
        require(
            b - a == math.ceil(math.prod(shape) * WIDTHS[desc["dtype"]]),
            "descriptor byte size mismatch",
        )
        intervals.append((a, b))
    end = 0
    for a, b in sorted(intervals):
        require(a == end, "offset overlap or gap")
        end = b
    require(end == size - 8 - len(raw), "payload extent mismatch")
    return obj


def selections():
    result = {}
    for prefix in ("layers.0.ffn.experts.0", "mtp.0.ffn.experts.0"):
        for w, role in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
            result[f"{prefix}.{w}.weight"] = role
    result["layers.0.attn.wq_a.weight"] = "dense_fp8"
    for layer in (1, 14):
        result[f"layers.{layer}.engram.embed.weight"] = "engram_table"
    return result


def geometry(key, weight, scale):
    if ".experts." in key:
        n, k = (5120, 2304) if ".w2." in key else (2304, 5120)
        require(
            weight["shape"] == [n, k // 2] and weight["dtype"] == "I8",
            "expert storage geometry/type",
        )
        require(
            scale["shape"] == [n, k // 32] and scale["dtype"] == "F8_E8M0",
            "expert scale geometry/type",
        )
        return [n, k]
    if ".engram." in key:
        n = 384006168 if key.startswith("layers.1.") else 384016682
        require(
            weight["shape"] == [n, 256] and weight["dtype"] == "F8_E4M3",
            "Engram storage geometry/type",
        )
        require(
            scale["shape"] == [n, 8] and scale["dtype"] == "F8_E8M0",
            "Engram scale geometry/type",
        )
        return [n, 256]
    require(
        weight["shape"] == [1280, 5120] and weight["dtype"] == "F8_E4M3",
        "dense storage geometry/type",
    )
    require(
        scale["shape"] == [40, 160] and scale["dtype"] == "F8_E8M0",
        "dense scale geometry/type",
    )
    return [1280, 5120]


def records(headers, routes, shards):
    result = {}
    for key, role in selections().items():
        scale_key = key.removesuffix("weight") + "scale"
        shard, scale_shard = routes[key], routes[scale_key]
        weight, scale = headers[shard][key], headers[scale_shard][scale_key]
        logical = geometry(key, weight, scale)
        for name, desc, file in ((key, weight, shard), (scale_key, scale, scale_shard)):
            base = shards[file]["header_length"] + 8
            result[name] = {
                "role": role if name == key else role + "_scale",
                "shard": file,
                "descriptor": desc,
                "source_byte_offsets": [base + x for x in desc["data_offsets"]],
            }
        result[key].update(scale_key=scale_key, source_logical_shape=logical)
    return result


def capture():
    verify_sources(ROOT)
    reader = Reader()
    index, _ = reader.fetch("model.safetensors.index.json")
    require(digest(index) == INDEX_HASH, "pinned index digest mismatch")
    weights = json.loads(index)["weight_map"]
    keys = list(selections())
    keys += [k.removesuffix("weight") + "scale" for k in keys]
    routes = {key: weights[key] for key in keys}
    shards, headers, blobs = {}, {}, {}
    for name in sorted(set(routes.values())):
        prefix, size = reader.fetch(name, 0, 8)
        length = header_length(prefix)
        require(8 + length <= size, "header exceeds file")
        raw, _ = reader.fetch(name, 8, length, size)
        headers[name] = descriptors(raw, size)
        fixture = f"fixtures/headers/{name}.header"
        blobs[fixture] = prefix + raw
        shards[name] = {
            "fixture": fixture,
            "sha256": digest(prefix + raw),
            "header_sha256": digest(raw),
            "header_length": length,
            "source_content_length": size,
            "ranges": [
                {"offset": 0, "content_length": 8},
                {"offset": 8, "content_length": length},
            ],
        }
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "evidence": "Header metadata only; no tensor values, nibble ordering, or matrix numeric verification.",
        "source_geometry": {
            "inference/model.py": "Linear (214-232), Expert (832-853), ParallelEngramEmbedding (296-310); w1=gate, w3=up, w2=down",
            "config.json": "hidden_size, moe_intermediate_size, q_lora_rank, engram_num_embeddings, engram_head_dim",
        },
        "source_sha256": SOURCE_HASHES,
        "index": {"sha256": digest(index), "content_length": len(index)},
        "read_bytes": reader.used,
        "shards": shards,
        "tensors": records(headers, routes, shards),
    }
    manifest_path = ROOT / "tensor-headers.json"
    check_existing_manifest(manifest_path, manifest)
    for path, blob in blobs.items():
        target = ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def check_existing_manifest(path, manifest):
    if path.exists():
        require(
            manifest == json.loads(path.read_text()),
            "capture differs from the pinned header manifest; review the source change",
        )


def verify_sources(root):
    for name, expected in SOURCE_HASHES.items():
        require(
            digest((root / "fixtures" / name).read_bytes()) == expected,
            "geometry source digest mismatch",
        )


def verify(root=ROOT):
    verify_sources(root)
    manifest = json.loads((root / "tensor-headers.json").read_text())
    require(manifest["source_sha256"] == SOURCE_HASHES, "wrong geometry source pins")
    require(
        manifest["index"] == {"sha256": INDEX_HASH, "content_length": 7470294},
        "wrong index pin",
    )
    require(
        (manifest["repository"], manifest["revision"]) == (REPOSITORY, REVISION),
        "wrong source pin",
    )
    headers = {}
    for name, entry in manifest["shards"].items():
        require(
            entry["fixture"] == f"fixtures/headers/{name}.header"
            and re.fullmatch(r"model-\d{5}-of-00048.safetensors", name),
            "invalid fixture path",
        )
        path = root / entry["fixture"]
        require(path.stat().st_size <= MAX_HEADER + 8, "oversize fixture")
        blob = path.read_bytes()
        length = header_length(blob[:8])
        require(
            len(blob) == 8 + length and length == entry["header_length"],
            "fixture contains payload or truncated header",
        )
        require(
            digest(blob) == entry["sha256"]
            and digest(blob[8:]) == entry["header_sha256"],
            "header digest mismatch",
        )
        require(
            entry["ranges"]
            == [
                {"offset": 0, "content_length": 8},
                {"offset": 8, "content_length": length},
            ],
            "invalid captured ranges",
        )
        headers[name] = descriptors(blob[8:], entry["source_content_length"])
    routes = {k: v["shard"] for k, v in manifest["tensors"].items()}
    require(
        records(headers, routes, manifest["shards"]) == manifest["tensors"],
        "mismatched selected descriptors",
    )
    expected_bytes = manifest["index"]["content_length"] + sum(
        8 + x["header_length"] for x in manifest["shards"].values()
    )
    require(
        expected_bytes == manifest["read_bytes"] <= MAX_TOTAL,
        "invalid total read bytes",
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    if parser.parse_args().capture:
        capture()
    manifest = verify()
    print(
        f"Verified {len(manifest['tensors'])} descriptors in {len(manifest['shards'])} metadata-only headers"
    )
