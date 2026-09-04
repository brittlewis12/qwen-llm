# /// script
# requires-python = ">=3.12"
# ///

import json
from pathlib import Path


TRACE_ROOTS = [
    Path("target/qwen36-q8-error-tag-factorial-passive-closing-a-layer46-v1"),
    Path("target/qwen36-q8-error-tag-factorial-passive-closing-b-layer46-v1"),
]
RUN_ROOT = Path("target/qwen36-q8-error-tag-factorial-v1")
ERROR_TOKEN_ID = 815


def load(path: Path) -> object:
    return json.loads(path.read_text())


def closing_stem_completion(tokens: list[dict]) -> tuple[int, str]:
    close_start = max(
        index
        for index, token in enumerate(tokens)
        if bytes.fromhex(token["token_piece_hex"]) == b"</"
    )
    close_end = next(
        index
        for index in range(close_start + 1, len(tokens))
        if bytes.fromhex(tokens[index]["token_piece_hex"]) == b">"
    )
    completion = close_end - 1
    return tokens[completion]["position"], tokens[completion]["token_display_lossy"]


def main() -> None:
    root = Path(__file__).resolve().parent
    coding = load(root / "CODING.json")
    coding_by_id = {row["id"]: row for row in coding["rows"]}
    run_manifest = load(RUN_ROOT / "manifest.json")
    run_by_id = {child["id"]: child for child in run_manifest["sweeps"]}
    rows = []
    for trace_root in TRACE_ROOTS:
        manifest = load(trace_root / "manifest.json")
        for artifact in manifest["artifacts"]:
            item_id = artifact["request_id"]
            trace = load(trace_root / artifact["path"])
            position, piece = closing_stem_completion(trace["input_tokens"])
            cell = next(
                cell
                for cell in trace["cells"]
                if cell["source_layer"] == 46 and cell["source_position"] == position
            )
            error_entry = next(
                (
                    entry
                    for entry in cell["top_k"]
                    if entry["token_id"] == ERROR_TOKEN_ID
                ),
                None,
            )
            child = run_by_id[item_id]
            baseline = load(RUN_ROOT / child["path"] / "arms/000000/run.json")
            active = load(RUN_ROOT / child["path"] / "arms/000001/run.json")
            baseline_readout = next(
                readout
                for readout in baseline["live_readouts"]
                if readout["source_layer"] == 46
                and readout["phase"] == "prefill"
                and readout["index"] == position
            )
            active_readout = next(
                readout
                for readout in active["live_readouts"]
                if readout["source_layer"] == 46
                and readout["phase"] == "prefill"
                and readout["index"] == position
            )
            baseline_score = baseline_readout["scores"][0]["score"]
            active_score = active_readout["scores"][0]["score"]
            code = coding_by_id[item_id]
            rows.append(
                {
                    "id": item_id,
                    "stem": code["stem"],
                    "position": position,
                    "position_piece": piece,
                    "error_top25_rank": None
                    if error_entry is None
                    else error_entry["rank"],
                    "error_transported_logit": (
                        None if error_entry is None else error_entry["logit"]
                    ),
                    "top1_token": cell["top_k"][0]["token_display_lossy"],
                    "top1_logit": cell["top_k"][0]["logit"],
                    "baseline_error_numerator": baseline_score,
                    "active_error_numerator": active_score,
                    "error_numerator_delta": active_score - baseline_score,
                    "active_normalizes_corrupted_stem": code[
                        "active_normalizes_corrupted_stem"
                    ],
                    "active_explicit_pair_repair": code["active_explicit_pair_repair"],
                }
            )
    rows.sort(key=lambda row: row["baseline_error_numerator"], reverse=True)
    output = {
        "schema": "qwen.lens.error_tag_factorial.passive_closing_loci",
        "schema_version": 1,
        "locus_rule": "final token of the closing tag stem before the greater-than token",
        "rows": rows,
    }
    (root / "PASSIVE-LOCI.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
