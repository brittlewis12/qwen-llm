#!/usr/bin/env python3
"""Read GGUF metadata and tensor tables without loading tensor data.

Usage:
  gguf_header.py [--all-tensors] [--json] FILE [FILE ...]

The default prints metadata keys that bear on MTP/NextN, sampling, and the
architecture, plus every tensor whose name matches an MTP/NextN signature.
`--all-tensors` prints the full tensor table; `--json` emits one object per
file. Pure stdlib; only the header is read.
"""

import json
import struct
import sys

MTP_WORDS = (
    "nextn",
    "mtp",
    "eh_proj",
    "enorm",
    "hnorm",
    "shared_head",
)

INTEREST_KEYS = (
    "general.architecture",
    "general.name",
    "general.basename",
    "general.sampling.temp",
    "general.sampling.top_k",
    "general.sampling.top_p",
    "general.sampling.min_p",
)


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.path = path

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"short read {len(b)}/{n} in {self.path}")
        return b

    def u8(self):
        return struct.unpack("<B", self.read(1))[0]

    def i8(self):
        return struct.unpack("<b", self.read(1))[0]

    def u16(self):
        return struct.unpack("<H", self.read(2))[0]

    def i16(self):
        return struct.unpack("<h", self.read(2))[0]

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def i32(self):
        return struct.unpack("<i", self.read(4))[0]

    def f32(self):
        return struct.unpack("<f", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def i64(self):
        return struct.unpack("<q", self.read(8))[0]

    def f64(self):
        return struct.unpack("<d", self.read(8))[0]

    def string(self):
        return self.read(self.u64()).decode("utf-8", "replace")


SCALARS = {
    0: ("u8", Reader.u8, 1),
    1: ("i8", Reader.i8, 1),
    2: ("u16", Reader.u16, 2),
    3: ("i16", Reader.i16, 2),
    4: ("u32", Reader.u32, 4),
    5: ("i32", Reader.i32, 4),
    6: ("f32", Reader.f32, 4),
    7: ("bool", Reader.u8, 1),
    10: ("u64", Reader.u64, 8),
    11: ("i64", Reader.i64, 8),
    12: ("f64", Reader.f64, 8),
}


def read_value(r, vtype):
    if vtype == 8:
        return r.string()
    if vtype == 9:
        etype = r.u32()
        n = r.u64()
        if etype in SCALARS and n <= 4096:
            return [SCALARS[etype][1](r) for _ in range(n)]
        if etype == 8 and n <= 4096:
            return [r.string() for _ in range(n)]
        if etype == 8:
            for _ in range(n):
                r.f.seek(r.u64(), 1)
            return f"<string[{n}]>"
        if etype in SCALARS:
            r.f.seek(SCALARS[etype][2] * n, 1)
            return f"<{SCALARS[etype][0]}[{n}]>"
        raise ValueError(f"array element type {etype}")
    if vtype in SCALARS:
        return SCALARS[vtype][1](r)
    raise ValueError(f"value type {vtype}")


def parse(path, all_tensors=False):
    r = Reader(path)
    magic = r.read(4)
    if magic != b"GGUF":
        raise ValueError(f"bad magic {magic!r}")
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        meta[key] = read_value(r, r.u32())
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.u64() for _ in range(r.u32())]
        ttype = r.u32()
        off = r.u64()
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})

    interested = {
        k: v
        for k, v in meta.items()
        if k in INTEREST_KEYS
        or "nextn" in k.lower()
        or k.lower().startswith("mtp")
        or ".mtp." in k.lower()
        or "sampling" in k.lower()
    }
    mtp = [t for t in tensors if any(w in t["name"].lower() for w in MTP_WORDS)]
    params = sum(prod(t["dims"]) for t in tensors)
    return {
        "path": path,
        "version": version,
        "tensor_count": n_tensors,
        "metadata_kv_count": n_kv,
        "metadata": interested,
        "nextn_predict_layers": next(
            (v for k, v in meta.items() if "nextn_predict" in k.lower()), None
        ),
        "matched_tensors": mtp,
        "tensor_params": params,
        "tensors": tensors if all_tensors else None,
    }


def prod(values):
    out = 1
    for value in values:
        out *= value
    return out


def main(argv):
    all_tensors = "--all-tensors" in argv
    as_json = "--json" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    reports = []
    for path in paths:
        try:
            reports.append(parse(path, all_tensors))
        except Exception as exc:  # keep going across shards/files
            reports.append({"path": path, "error": str(exc)})
    if as_json:
        print(json.dumps(reports, indent=2))
        return 0
    for rep in reports:
        print(f"# {rep['path']}")
        if "error" in rep:
            print(f"error: {rep['error']}")
            continue
        print(
            f"version={rep['version']} tensor_count={rep['tensor_count']} "
            f"metadata_kv_count={rep['metadata_kv_count']} "
            f"tensor_params={rep['tensor_params']:,}"
        )
        for key, value in rep["metadata"].items():
            text = repr(value)
            print(f"meta {key} = {text[:300]}{'...' if len(text) > 300 else ''}")
        print(f"nextn_predict_layers={rep['nextn_predict_layers']!r}")
        print(f"matched_tensors={len(rep['matched_tensors'])}")
        for t in rep["matched_tensors"]:
            print(
                f"tensor {t['name']} dims={t['dims']} "
                f"type={t['type']} offset={t['offset']}"
            )
        if all_tensors:
            for t in rep["tensors"]:
                print(
                    f"tensor {t['name']} dims={t['dims']} "
                    f"type={t['type']} offset={t['offset']}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
