# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow==25.0.1"]
# ///

"""Prepare the frozen WikiText source plan for Flash-Next quality fixtures."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq


SCHEMA = "qwen4exp-selected-quality-source-plan"
SCHEMA_VERSION = 1
SOURCE_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
    "6231e49f19a707241d6b84d9cff60a3a86b85a85/"
    "wikitext-2-raw-v1/test-00000-of-00001.parquet"
)
SOURCE_REVISION = "6231e49f19a707241d6b84d9cff60a3a86b85a85"
SOURCE_BYTES = 731_216
SOURCE_SHA256 = "3ee89cd6a2ab912afd5d01e98867b46a67d9ec7a9eca0910e5e4c3cdd4cc1925"
SOURCE_ROWS = 4_358
SOURCE_CONFIG = "wikitext-2-raw-v1"
MIN_DOCUMENT_BYTES = 18_000
TITLE_RE = re.compile(r"^ = ([^=].*?) = \n$")
EXCLUSION_REPOSITORY_COMMIT = "f9bbbdc0228e1029225fbdb6d4e83a14b846415e"
MAX_EXCLUSION_FILE_BYTES = 2 * 1024 * 1024
EXCLUSION_WINDOW_WORDS = 256
EXCLUSION_WINDOW_STRIDE = 128
EXCLUSION_MIN_WINDOW_WORDS = 128


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode()


def acquire_source(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
        data = response.read()
    path.write_bytes(data)


def validate_source(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) != SOURCE_BYTES:
        raise RuntimeError(f"source byte count {len(data)} != {SOURCE_BYTES}")
    observed = sha256(data)
    if observed != SOURCE_SHA256:
        raise RuntimeError(f"source SHA-256 {observed} != {SOURCE_SHA256}")
    return data


def parse_documents(rows: list[str]) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    title: str | None = None
    start = 0
    parts: list[str] = []

    def finish(end: int) -> None:
        if title is None:
            return
        text = "".join(parts)
        encoded = text.encode("utf-8")
        documents.append(
            {
                "source_ordinal": len(documents),
                "title": title,
                "row_start": start,
                "row_end_exclusive": end,
                "utf8_bytes": len(encoded),
                "utf8_sha256": sha256(encoded),
                "text": text,
            }
        )

    for index, row in enumerate(rows):
        match = TITLE_RE.match(row)
        if match:
            finish(index)
            title = match.group(1)
            start = index
            parts = [row]
        elif title is not None:
            parts.append(row)
    finish(len(rows))
    return documents


def word_five_grams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[index : index + 5]) for index in range(max(0, len(words) - 4))}


def word_five_gram_windows(
    text: str,
) -> list[tuple[int, int, set[tuple[str, ...]]]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < EXCLUSION_MIN_WINDOW_WORDS:
        return []
    starts = list(
        range(
            0, max(1, len(words) - EXCLUSION_WINDOW_WORDS + 1), EXCLUSION_WINDOW_STRIDE
        )
    )
    final_start = max(0, len(words) - EXCLUSION_WINDOW_WORDS)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    windows = []
    for start in starts:
        window = words[start : start + EXCLUSION_WINDOW_WORDS]
        if len(window) < EXCLUSION_MIN_WINDOW_WORDS:
            continue
        grams = {
            tuple(window[index : index + 5]) for index in range(max(0, len(window) - 4))
        }
        if grams:
            windows.append((start, start + len(window), grams))
    return windows


def near_duplicate_audit(documents: list[dict[str, object]]) -> dict[str, object]:
    grams = [word_five_grams(str(document["text"])) for document in documents]
    maximum = 0.0
    maximum_pair: tuple[str, str] | None = None
    for left in range(len(documents)):
        for right in range(left + 1, len(documents)):
            union = grams[left] | grams[right]
            score = len(grams[left] & grams[right]) / len(union) if union else 0.0
            if score > maximum:
                maximum = score
                maximum_pair = (
                    str(documents[left]["title"]),
                    str(documents[right]["title"]),
                )
    return {
        "algorithm": "lowercase_ascii_alnum_word_5gram_jaccard_v1",
        "maximum_jaccard": maximum,
        "maximum_pair": list(maximum_pair) if maximum_pair else None,
        "required_maximum_jaccard_exclusive": 0.05,
        "passed": maximum < 0.05,
    }


def repository_text_inventory(
    repository: Path,
) -> tuple[list[tuple[str, bytes]], list[dict[str, object]], str]:
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "archive",
            "--format=tar",
            EXCLUSION_REPOSITORY_COMMIT,
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    files: list[tuple[str, bytes]] = []
    token_fixtures: list[dict[str, object]] = []
    bound_files: list[tuple[str, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tree:
        for member in tree.getmembers():
            if not member.isfile() or member.size > MAX_EXCLUSION_FILE_BYTES:
                continue
            extracted = tree.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read()
            token_dtype: str | None = None
            if member.name.endswith(".tokens.u32le"):
                token_dtype = "u32"
            elif member.name.endswith(".tokens.i32le") or (
                "/tokens/" in member.name and member.name.endswith(".i32le")
            ):
                token_dtype = "i32"
            if token_dtype is not None and len(data) % 4 == 0:
                signed = token_dtype == "i32"
                token_ids = [
                    int.from_bytes(data[index : index + 4], "little", signed=signed)
                    for index in range(0, len(data), 4)
                ]
                if all(token >= 0 for token in token_ids):
                    token_fixtures.append(
                        {
                            "path": member.name,
                            "dtype": token_dtype,
                            "byte_order": "little",
                            "bytes": len(data),
                            "sha256": sha256(data),
                            "token_count": len(token_ids),
                            "token_ids_u32": token_ids,
                        }
                    )
                    bound_files.append((member.name, data))
                    continue
            if b"\0" in data:
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if word_five_grams(text):
                files.append((member.name, data))
                bound_files.append((member.name, data))

    digest = hashlib.sha256()
    digest.update(b"qwen4exp-selected-quality-exclusion-inventory-v2\0")
    digest.update(f"repository_commit={EXCLUSION_REPOSITORY_COMMIT}\n".encode())
    for path, data in bound_files:
        path_bytes = path.encode()
        digest.update(len(path_bytes).to_bytes(8, "little"))
        digest.update(path_bytes)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return files, token_fixtures, digest.hexdigest()


def repository_exclusion_audit(
    documents: list[dict[str, object]], repository: Path
) -> dict[str, object]:
    inventory, token_fixtures, inventory_sha256 = repository_text_inventory(repository)
    repository_grams = [
        (path, word_five_grams(data.decode("utf-8"))) for path, data in inventory
    ]
    maximum = 0.0
    maximum_pair: tuple[str, str] | None = None
    maximum_window_jaccard = 0.0
    maximum_window_containment = 0.0
    maximum_window_jaccard_pair: tuple[str, str, int, int] | None = None
    maximum_window_containment_pair: tuple[str, str, int, int] | None = None
    repository_windows = [
        (path, start, end, grams)
        for path, data in inventory
        for start, end, grams in word_five_gram_windows(data.decode("utf-8"))
    ]
    for document in documents:
        document_grams = word_five_grams(str(document["text"]))
        for path, candidate_grams in repository_grams:
            union = document_grams | candidate_grams
            score = len(document_grams & candidate_grams) / len(union) if union else 0.0
            if score > maximum:
                maximum = score
                maximum_pair = (str(document["title"]), path)
        for path, start, end, candidate_grams in repository_windows:
            intersection = len(document_grams & candidate_grams)
            union = len(document_grams | candidate_grams)
            jaccard = intersection / union if union else 0.0
            containment = (
                intersection / len(candidate_grams) if candidate_grams else 0.0
            )
            if jaccard > maximum_window_jaccard:
                maximum_window_jaccard = jaccard
                maximum_window_jaccard_pair = (
                    str(document["title"]),
                    path,
                    start,
                    end,
                )
            if containment > maximum_window_containment:
                maximum_window_containment = containment
                maximum_window_containment_pair = (
                    str(document["title"]),
                    path,
                    start,
                    end,
                )
    passed = (
        maximum < 0.05
        and maximum_window_jaccard < 0.05
        and maximum_window_containment < 0.5
    )
    return {
        "algorithm": "lowercase_ascii_alnum_word_5gram_repository_exclusion_v2",
        "repository_commit": EXCLUSION_REPOSITORY_COMMIT,
        "inventory_schema": "all_git_archive_utf8_word_files_and_token_fixtures_v2",
        "maximum_file_bytes_inclusive": MAX_EXCLUSION_FILE_BYTES,
        "inventory_files": len(inventory),
        "inventory_utf8_bytes": sum(len(data) for _, data in inventory),
        "token_fixture_inventory": token_fixtures,
        "token_fixture_count": len(token_fixtures),
        "inventory_sha256": inventory_sha256,
        "maximum_jaccard": maximum,
        "maximum_pair": list(maximum_pair) if maximum_pair else None,
        "required_maximum_jaccard_exclusive": 0.05,
        "window_audit": {
            "window_words": EXCLUSION_WINDOW_WORDS,
            "stride_words": EXCLUSION_WINDOW_STRIDE,
            "minimum_window_words": EXCLUSION_MIN_WINDOW_WORDS,
            "window_count": len(repository_windows),
            "maximum_jaccard": maximum_window_jaccard,
            "required_maximum_jaccard_exclusive": 0.05,
            "maximum_jaccard_pair": list(maximum_window_jaccard_pair)
            if maximum_window_jaccard_pair
            else None,
            "maximum_containment": maximum_window_containment,
            "required_maximum_containment_exclusive": 0.5,
            "maximum_containment_pair": list(maximum_window_containment_pair)
            if maximum_window_containment_pair
            else None,
        },
        "passed": passed,
    }


def build_plan(source: Path, repository: Path) -> dict[str, object]:
    validate_source(source)
    table = pq.read_table(source, columns=["text"])
    rows = table.column("text").to_pylist()
    if len(rows) != SOURCE_ROWS or not all(isinstance(row, str) for row in rows):
        raise RuntimeError("unexpected WikiText test schema or row count")
    all_documents = parse_documents(rows)
    documents = [
        document
        for document in all_documents
        if int(document["utf8_bytes"]) >= MIN_DOCUMENT_BYTES
    ]
    if len(all_documents) != 62 or len(documents) != 26:
        raise RuntimeError(
            f"unexpected document counts: all={len(all_documents)} qualifying={len(documents)}"
        )
    audit = near_duplicate_audit(documents)
    if not audit["passed"]:
        raise RuntimeError(f"near-duplicate audit failed: {audit}")
    exclusion_audit = repository_exclusion_audit(documents, repository)
    if not exclusion_audit["passed"]:
        raise RuntimeError(f"repository exclusion audit failed: {exclusion_audit}")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "dataset": "Salesforce/wikitext",
            "config": SOURCE_CONFIG,
            "split": "test",
            "shared_storage_path": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
            "revision": SOURCE_REVISION,
            "url": SOURCE_URL,
            "acquired_date": "2026-08-28",
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
            "spdx_licenses": ["CC-BY-SA-3.0", "GFDL-1.3-or-later"],
            "redistribution": "derived token IDs, titles, row metadata, and hashes only",
        },
        "selection": {
            "title_regex": TITLE_RE.pattern,
            "source_document_count": len(all_documents),
            "minimum_document_utf8_bytes": MIN_DOCUMENT_BYTES,
            "qualifying_document_count": len(documents),
            "order": "source row order",
            "near_duplicate_audit": audit,
            "repository_exclusion_audit": exclusion_audit,
            "forbidden_source_classes": [
                "repository text",
                "roadmap text",
                "benchmark or tuning prompts",
                "prior diagnostic prompts",
            ],
        },
        "documents": documents,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        acquire_source(args.source)
    content = canonical_json(build_plan(args.source, args.repository))
    if args.check:
        if not args.output.exists() or args.output.read_bytes() != content:
            raise RuntimeError(f"source plan drift: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(content)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": len(content),
                "sha256": sha256(content),
                "mode": "check" if args.check else "write",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
