#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run ignored local-artifact witnesses by target and complete test path."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

WITNESS_REASON = re.compile(
    r'^\s*#\[ignore\s*=\s*"(?P<reason>model witness: [^"]+|needs reference logits from the external llm tool)"\]\s*$'
)
FUNCTION = re.compile(r"(?:pub(?:\([^)]*\))?\s+)?fn\s+(?P<name>\w+)")

PACKAGES = ("qwen-llm", "qwen-cli")


def rust_code_only(source: str) -> str:
    """Mask strings and comments so braces in them cannot affect module scope."""
    output = list(source)
    i = 0
    while i < len(source):
        raw_start = i + 2 if source.startswith("br", i) else i + 1 if source.startswith("r", i) else i
        raw_end = raw_start
        if raw_start != i:
            while raw_end < len(source) and source[raw_end] == "#":
                raw_end += 1
        if raw_start != i and raw_end < len(source) and source[raw_end] == '"':
            hashes = source[raw_start:raw_end]
            start = i
            terminator = '"' + hashes
            end = source.find(terminator, raw_end + 1)
            i = len(source) if end < 0 else end + len(terminator)
            for j in range(start, i):
                if output[j] != "\n":
                    output[j] = " "
        elif source[i] == "'":
            character = re.match(r"'(?:\\.|[^'\\])'", source[i:])
            if character:
                start = i
                i += character.end()
                for j in range(start, i):
                    output[j] = " "
            else:
                i += 1
        elif source.startswith("//", i):
            end = source.find("\n", i)
            end = len(source) if end < 0 else end
            for j in range(i, end):
                output[j] = " "
            i = end
        elif source.startswith("/*", i):
            start = i
            i += 2
            depth = 1
            while i < len(source) and depth:
                if source.startswith("/*", i):
                    depth += 1
                    i += 2
                elif source.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            for j in range(start, i):
                if output[j] != "\n":
                    output[j] = " "
        elif source[i] == '"':
            start = i
            i += 1
            while i < len(source):
                if source[i] == "\\":
                    i += 2
                elif source[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            for j in range(start, min(i, len(source))):
                if output[j] != "\n":
                    output[j] = " "
        else:
            i += 1
    return "".join(output)


def module_ranges(code: str) -> list[tuple[int, int, str]]:
    declarations = {
        code.find("{", match.start(), match.end()): (match.start(), match.group(1))
        for match in re.finditer(r"\bmod\s+(\w+)\s*\{", code)
    }
    active: list[tuple[int, int, int, str]] = []
    modules: list[tuple[int, int, str]] = []
    depth = 0
    for offset, char in enumerate(code):
        if char == "{":
            depth += 1
            if offset in declarations:
                start, name = declarations[offset]
                active.append((start, depth, offset, name))
        elif char == "}":
            depth -= 1
            while active and active[-1][1] > depth:
                start, _, _, name = active.pop()
                modules.append((start, offset + 1, name))
    return modules


def source_base_path(path: pathlib.Path, source_root: pathlib.Path) -> list[str]:
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        return []
    parts = list(relative.with_suffix("").parts)
    if len(parts) == 1 and parts[0] in {"lib", "main"}:
        return []
    if parts[0] == "qwen":
        return parts[1:]
    return parts


def witness_annotations(source: str, path: pathlib.Path, package: str):
    lines = source.splitlines(keepends=True)
    offsets = []
    for line_index, line in enumerate(lines):
        annotation = WITNESS_REASON.match(line.rstrip("\r\n"))
        if not annotation:
            continue
        index = line_index + 1
        in_block_comment = False
        function = None
        while index < len(lines):
            candidate = lines[index].strip()
            if in_block_comment:
                if "*/" in candidate:
                    in_block_comment = False
                index += 1
                continue
            if not candidate:
                index += 1
                continue
            if candidate.startswith("//"):
                index += 1
                continue
            if candidate.startswith("/*"):
                in_block_comment = "*/" not in candidate[2:]
                index += 1
                continue
            if candidate.startswith("#["):
                index += 1
                continue
            function = FUNCTION.search(candidate)
            break
        if function is None:
            print(
                f"{package}: witness annotation in {path}:{line_index + 1} is not followed by a function",
                file=sys.stderr,
            )
            return None
        offset = sum(map(len, lines[:index])) + function.start()
        offsets.append((annotation.group("reason"), function.group("name"), offset))
    return offsets


def cargo_metadata() -> dict:
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version=1"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def target_args(target: dict) -> list[str]:
    kinds = set(target["kind"])
    if "lib" in kinds or "rlib" in kinds:
        return ["--lib"]
    if "test" in kinds:
        return ["--test", target["name"]]
    if "bin" in kinds:
        return ["--bin", target["name"]]
    if "example" in kinds:
        return ["--example", target["name"]]
    return []


def ignored_listing(package: str, selector: list[str]) -> list[str]:
    command = [
        "cargo", "test", "--release", "-p", package, *selector,
        "--", "--ignored", "--list", "--format", "terse",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"could not list ignored tests for {package} {selector}")
    return [line.removesuffix(": test") for line in result.stdout.splitlines() if line.endswith(": test")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filter", nargs="?", default="", help="function-name substring")
    parser.add_argument("--list-only", action="store_true", help="validate and print target-qualified selections")
    args = parser.parse_args()
    repo_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    )
    repo = pathlib.Path(repo_result.stdout.strip())
    fixture_source = (repo / "crates/qwen-llm/src/test_fixtures.rs").read_text()
    fixture_ids = set(re.findall(r'id: "([^"]+)"', fixture_source))
    metadata = cargo_metadata()
    packages = {package["name"]: package for package in metadata["packages"]}
    witnesses: list[tuple[str, str, pathlib.Path, str, bool]] = []

    for package_name in PACKAGES:
        package = packages[package_name]
        manifest = pathlib.Path(package["manifest_path"])
        src_root = manifest.parent / "src"
        roots = [src_root, manifest.parent / "tests"]
        if package_name == "qwen-llm":
            roots.append(manifest.parent / "examples")
        if package_name == "qwen-llm":
            roots.append(manifest.parent / "benches")
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*.rs"):
                if path.name.startswith("qwen4exp_"):
                    continue
                source = path.read_text()
                annotations = witness_annotations(source, path, package_name)
                if annotations is None:
                    return 1
                for reason, _, _ in annotations:
                    fixture_id = reason.removeprefix("model witness: ")
                    if reason.startswith("model witness: ") and fixture_id not in fixture_ids:
                        print(
                            f"{package_name}: witness annotation in {path} names unknown fixture {fixture_id!r}",
                            file=sys.stderr,
                        )
                        return 1
                code = rust_code_only(source)
                ranges = module_ranges(code)
                base = source_base_path(path, src_root)
                for _, name, offset in annotations:
                    if args.filter and args.filter not in name:
                        continue
                    module = base + [
                        module_name
                        for start, end, module_name in ranges
                        if start < offset < end
                    ]
                    full_name = "::".join([*module, name])
                    witnesses.append((package_name, full_name, path, name, path.is_relative_to(manifest.parent / "tests")))

    env = os.environ.copy()
    env["QWEN_REQUIRE_METAL_TESTS"] = "1"
    selected = 0
    selections: list[tuple[str, list[str], str]] = []
    passed_by_package = {name: 0 for name in PACKAGES}
    selected_by_package = {name: 0 for name in PACKAGES}
    failed_by_package = {name: 0 for name in PACKAGES}
    for package_name in PACKAGES:
        package_targets = packages[package_name]["targets"]
        package_witnesses = [witness for witness in witnesses if witness[0] == package_name]
        source_targets = {}
        for _, _, source_path, _, integration_source in package_witnesses:
            if integration_source:
                candidates = [
                    target
                    for target in package_targets
                    if "test" in target["kind"]
                    and pathlib.Path(target["src_path"]).resolve() == source_path.resolve()
                ]
            elif package_name == "qwen-cli":
                relative = source_path.relative_to(manifest.parent / "src")
                if relative.parts[0] in {"qwen", "serve"}:
                    target_name = "qwen"
                elif relative.parts[0] == "bench":
                    target_name = "qwen-bench"
                else:
                    print(
                        f"qwen-cli: no target mapping for witness source {source_path}",
                        file=sys.stderr,
                    )
                    return 1
                candidates = [
                    target
                    for target in package_targets
                    if target["name"] == target_name and target.get("test")
                ]
            else:
                candidates = [
                    target
                    for target in package_targets
                    if "lib" in target["kind"] and target.get("test")
                ]
            if len(candidates) != 1:
                print(
                    f"{package_name}: source {source_path} maps to {len(candidates)} test targets",
                    file=sys.stderr,
                )
                return 1
            source_targets[source_path] = candidates[0]
        listed: list[tuple[dict, str]] = []
        for target in {id(target): target for target in source_targets.values()}.values():
            selector = target_args(target)
            listed.extend((target, name) for name in ignored_listing(package_name, selector))

        for _, full_name, source_path, leaf_name, integration_source in package_witnesses:
            matches = [
                (target, test_name)
                for target, test_name in listed
                if test_name == full_name
                and (
                    not integration_source
                    or (
                        "test" in target["kind"]
                        and pathlib.Path(target["src_path"]).resolve() == source_path.resolve()
                    )
                )
            ]
            if len(matches) != 1:
                print(
                    f"{package_name}: witness {source_path}:{leaf_name} maps to {len(matches)} listed tests ({[name for _, name in matches]})",
                    file=sys.stderr,
                )
                return 1
            target, exact_name = matches[0]
            selector = target_args(target)
            selected += 1
            selected_by_package[package_name] += 1
            selections.append((package_name, selector, exact_name))
        if args.list_only:
            print(f"{package_name}: selected={selected_by_package[package_name]} passed=not-run")
            for selected_package, selector, exact_name in selections:
                if selected_package == package_name:
                    print(f"  {' '.join(selector)} :: {exact_name}")
            continue
        for selected_package, selector, exact_name in selections:
            if selected_package != package_name:
                continue
            result = subprocess.run(
                [
                    "cargo", "test", "--release", "-p", package_name, *selector,
                    exact_name, "--", "--ignored", "--exact", "--test-threads=1",
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            expected = re.search(r"test result: ok\. 1 passed; 0 failed", result.stdout)
            if result.returncode != 0 or expected is None:
                print(f"{package_name}: witness failed: {exact_name}", file=sys.stderr)
                failed_by_package[package_name] += 1
            else:
                passed_by_package[package_name] += 1
        print(
            f"{package_name}: selected={selected_by_package[package_name]} passed={passed_by_package[package_name]} failed={failed_by_package[package_name]}",
            flush=True,
        )
    if selected == 0:
        print("no witnesses matched", file=sys.stderr)
        return 1
    return int(any(failed_by_package.values()))


if __name__ == "__main__":
    raise SystemExit(main())
