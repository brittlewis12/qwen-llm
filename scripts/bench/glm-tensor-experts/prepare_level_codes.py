"""Losslessly re-encode selected real GLM IQ expert blocks as scalar level codes."""
from __future__ import annotations
import csv, hashlib, json, re, struct, sys
from pathlib import Path

if len(sys.argv) != 5:
    raise SystemExit("usage: prepare_level_codes.py RAW_DIR OUT_DIR quant_tiles.h stage-census.jsonl")
raw_dir, out_dir, header, census = map(Path, sys.argv[1:])
out_dir.mkdir(parents=True, exist_ok=True)
source = header.read_text()

def grid(name: str, count: int, fmt: str):
    m = re.search(rf"constant\s+\w+\s+{name}\[{count}\]\s*=\s*\{{(.*?)\}};", source, re.S)
    if not m:
        raise ValueError(f"cannot parse {name} from {header}")
    vals = re.findall(r"0x[0-9a-fA-F]+|[0-9]+", m.group(1))
    if len(vals) != count:
        raise ValueError(f"{name}: parsed {len(vals)} entries, expected {count}")
    return [int(v, 0) for v in vals]

iq2_grid = grid("qt_iq2s_grid", 1024, "Q")
iq3_grid = grid("qt_iq3s_grid", 512, "I")

def set_bits(dst: bytearray, bit: int, width: int, value: int):
    byte, shift = divmod(bit, 8)
    v = value << shift
    for j in range((shift + width + 7) // 8):
        dst[byte + j] |= (v >> (8*j)) & 0xff

def repack_iq2(block: bytes) -> bytes:
    if len(block) != 82: raise ValueError("IQ2_S block must be 82 bytes")
    out = bytearray(106)
    out[0:2] = block[0:2]
    out[2:10] = block[74:82]       # d + the exact original per-16 scales
    levels, signs = memoryview(out)[10:74], memoryview(out)[74:106]
    for il in range(16):
        ib, ih = il >> 1, il & 1
        qbase = 2 + 4*ib + 2*ih
        qh_lane = block[66 + ib] >> (4*ih)
        signbase = qbase + 32
        for group in range(2):
            grid_id = block[qbase + group] | ((qh_lane << (8 - 2*group)) & 0x300)
            word = iq2_grid[grid_id]
            for lane in range(8):
                mag = (word >> (8*lane)) & 0xff
                try: code = {8: 0, 25: 1, 43: 2}[mag]
                except KeyError: raise ValueError(f"unexpected IQ2_S grid level {mag}")
                idx = il*16 + group*8 + lane
                set_bits(levels, 2*idx, 2, code)
                set_bits(signs, idx, 1, (block[signbase + group] >> lane) & 1)
    return bytes(out)

def repack_iq3(block: bytes) -> bytes:
    if len(block) != 110: raise ValueError("IQ3_S block must be 110 bytes")
    out = bytearray(134)
    out[0:2] = block[0:2]
    out[2:6] = block[106:110]       # d + the exact original per-32 scales
    levels, signs = memoryview(out)[6:102], memoryview(out)[102:134]
    for il in range(16):
        ib, ih = il >> 1, il & 1
        qbase = 2 + 8*ib
        signbase = 74 + 4*ib + 2*ih
        qh_lane = block[66 + ib] >> (4*ih)
        for group in range(4):
            shift = 8 - group
            grid_id = block[qbase + 4*ih + group] | ((qh_lane << shift) & 0x100)
            word = iq3_grid[grid_id]
            signbyte = block[signbase + group//2]
            signshift = 4*(group & 1)
            for lane in range(4):
                mag = (word >> (8*lane)) & 0xff
                if not 1 <= mag <= 15 or mag % 2 == 0:
                    raise ValueError(f"unexpected IQ3_S grid level {mag}")
                code = (mag - 1)//2
                idx = il*16 + group*4 + lane
                set_bits(levels, 3*idx, 3, code)
                set_bits(signs, idx, 1, (signbyte >> (signshift + lane)) & 1)
    return bytes(out)

raw_rows = list(csv.DictReader((raw_dir/"manifest.csv").open()))
by_key = {(int(r["layer"]), int(r["expert"]), r["role"]): r for r in raw_rows}
cells = {}
for d in map(json.loads, census.open()):
    if d.get("kind") == "occupancy": cells[(d["depth"], d["width"])] = d
specs = [
    (7, 81920, 256, 3),
    (28, 81920, 1024, 3),
    (57, 49152, 2048, 3),
    (161, 49152, 2048, 8),
    (512, 49152, 2048, 39),
]
cases = []
for rows, depth, width, layer in specs:
    cell = cells[(depth, width)]
    lr = next(x for x in cell["per_layer"] if x["block"] == layer)
    expert = min(range(len(lr["counts"])), key=lambda e: (abs(lr["counts"][e]-rows), e))
    observed = int(lr["counts"][expert])
    for role in ("gate", "up", "down"):
        r = by_key[(layer, expert, role)]
        native_path = raw_dir/r["path"]
        native = native_path.read_bytes()
        code_bytes = 106 if r["dtype"] == "IQ2_S" else 134
        native_block = 82 if r["dtype"] == "IQ2_S" else 110
        if len(native) != int(r["bytes"]) or len(native) % native_block:
            raise ValueError(f"bad raw tensor slice {native_path}")
        fn = repack_iq2 if r["dtype"] == "IQ2_S" else repack_iq3
        encoded = b"".join(fn(native[i:i+native_block]) for i in range(0, len(native), native_block))
        level_path = out_dir/f"layer{layer}-expert{expert}-{role}.level"
        level_path.write_bytes(encoded)
        k, n = int(r["k"]), int(r["n"])
        cases.append(dict(rows=rows, observed_rows=observed, depth=depth, census_width=width,
                          layer=layer, expert=expert, role=role, dtype=r["dtype"].lower(),
                          k=k, m=n, native_block_bytes=native_block, level_block_bytes=code_bytes,
                          native_bytes=len(native), level_bytes=len(encoded),
                          native_path=str(native_path), level_path=str(level_path),
                          native_sha256=hashlib.sha256(native).hexdigest(),
                          level_sha256=hashlib.sha256(encoded).hexdigest()))
with (out_dir/"cases.jsonl").open("w") as f:
    for row in cases: f.write(json.dumps(row, sort_keys=True)+"\n")
print(json.dumps({"cases":len(cases),"experts":sorted({(c['layer'],c['expert']) for c in cases}),
                  "bytes_per_weight":{d:{"native":b*8/256,"level":v*8/256}
                     for d,b,v in (("IQ2_S",82,106),("IQ3_S",110,134))}}, sort_keys=True))
