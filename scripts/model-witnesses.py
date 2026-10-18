#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run ignored acceptance witnesses or opt-in characterization by exact test path."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

WITNESS_REASON = re.compile(
    r'ignore\s*=\s*"(?P<reason>model (?:witness|characterization): [^"]+|needs reference logits from the external llm tool)"',
    re.DOTALL,
)
FUNCTION = re.compile(r"(?:pub(?:\([^)]*\))?\s+)?fn\s+(?P<name>\w+)")

PACKAGES = ("qwen-llm", "qwen-cli")


def annotation_kind(reason: str) -> str:
    return (
        "characterization"
        if reason.startswith("model characterization: ")
        else "acceptance"
    )


def annotation_fixture(reason: str) -> str | None:
    for prefix in ("model witness: ", "model characterization: "):
        if reason.startswith(prefix):
            return reason.removeprefix(prefix)
    return None


def local_fixture_ids(source: str) -> set[str]:
    """Focused witnesses may declare the shared resolver's Fixture locally."""
    code = rust_code_only(source)
    return {
        match.group(1)
        for match in re.finditer(r'\bFixture\s*\{\s*id:\s*"([^"]+)"', source)
        if code[match.start() :].startswith("Fixture")
    }


def rust_code_only(source: str) -> str:
    """Mask strings and comments so braces in them cannot affect module scope."""
    output = list(source)
    i = 0
    while i < len(source):
        raw_start = (
            i + 2
            if source.startswith("br", i)
            else i + 1
            if source.startswith("r", i)
            else i
        )
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


def inline_module_path(ranges: list[tuple[int, int, str]], offset: int) -> list[str]:
    return [
        name
        for _, name in sorted(
            (start, name) for start, end, name in ranges if start < offset < end
        )
    ]


def witness_annotations(source: str, path: pathlib.Path, package: str):
    code = rust_code_only(source)
    offsets = []
    position = 0
    while True:
        start = code.find("#[", position)
        if start < 0:
            break
        depth = 0
        end = None
        for index in range(start + 1, len(code)):
            if code[index] == "[":
                depth += 1
            elif code[index] == "]":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            print(
                f"{package}: unterminated attribute in {path}:{source.count(chr(10), 0, start) + 1}",
                file=sys.stderr,
            )
            return None
        position = end
        annotation = WITNESS_REASON.search(source[start:end])
        if not annotation:
            continue
        function = FUNCTION.search(code, end)
        if function is None:
            line = source.count("\n", 0, start) + 1
            print(
                f"{package}: witness annotation in {path}:{line} is not followed by a function",
                file=sys.stderr,
            )
            return None
        offsets.append(
            (annotation.group("reason"), function.group("name"), function.start())
        )
    return offsets


def select_annotations(
    annotations: list[tuple[str, str]],
    prefixes: list[str],
    name_filter: str,
    kind: str = "acceptance",
) -> list[int]:
    if kind not in {"acceptance", "characterization"}:
        raise ValueError(f"unknown witness kind: {kind}")
    if "" in prefixes:
        raise ValueError("--fixture-prefix must not be empty")
    fixture_ids = [
        annotation_fixture(reason) if annotation_kind(reason) == kind else None
        for reason, _ in annotations
    ]
    unmatched = [
        prefix
        for prefix in prefixes
        if not any(
            fixture_id is not None and fixture_id.startswith(prefix)
            for fixture_id in fixture_ids
        )
    ]
    if unmatched:
        raise ValueError(
            f"fixture prefixes have no annotated model witnesses for kind={kind}: {', '.join(repr(prefix) for prefix in unmatched)}"
        )
    return [
        index
        for index, ((reason, name), fixture_id) in enumerate(
            zip(annotations, fixture_ids)
        )
        if annotation_kind(reason) == kind
        and name_filter in name
        and (
            not prefixes
            or (
                fixture_id is not None
                and any(fixture_id.startswith(prefix) for prefix in prefixes)
            )
        )
    ]


def self_test() -> None:
    assert local_fixture_ids(
        'const F: Fixture = Fixture { id: "local", env: &[], filename: "local.json" };'
    ) == {"local"}
    assert local_fixture_ids("""
const MODEL: qwen_llm::test_fixtures::Fixture = qwen_llm::test_fixtures::Fixture {
    id: "deepseek-v4.1-reap-272e-mxfp4-q8engram", env: &[],
    filename: "DeepSeek-V4.1-Flash-REAP-272E-MXFP4-Q8Engram.gguf",
};
const BASE: qwen_llm::test_fixtures::Fixture = qwen_llm::test_fixtures::Fixture {
    id: "deepseek-v4.1-reap-272e-cpu-source-config", env: &[],
    filename: "DeepSeek-V4.1-Flash-REAP-272E-config.json",
};
""") == {
        "deepseek-v4.1-reap-272e-mxfp4-q8engram",
        "deepseek-v4.1-reap-272e-cpu-source-config",
    }
    assert (
        local_fixture_ids(
            '// Fixture { id: "comment" }\nlet x = r#"Fixture { id: "string" }"#;'
        )
        == set()
    )
    assert source_base_path(
        pathlib.Path("src/deepseek_v41_tokenizer.rs"), pathlib.Path("src")
    ) == ["deepseek_v41_tokenizer"]
    source = """
mod outer {
    mod inner {
        #[test]
        #[ignore =
            "model witness: sample-fixture"
        ] // reason annotation may have a trailing comment
        fn nested_witness() {}
    }
}
"""
    path = pathlib.Path("fixture.rs")
    characterized = witness_annotations(
        """
#[ignore = "model witness: execution-fixture"]
fn acceptance() {}
#[ignore = "model characterization: converted-fixture"]
fn conversion_error() {}
""",
        path,
        "fixture",
    )
    assert characterized is not None
    classified = [(reason, name) for reason, name, _ in characterized]
    assert select_annotations(classified, [], "") == [0]
    assert select_annotations(classified, [], "", "characterization") == [1]
    assert select_annotations(
        classified, ["converted"], "error", "characterization"
    ) == [1]
    assert annotation_fixture(classified[1][0]) == "converted-fixture"
    try:
        select_annotations(classified, ["converted"], "")
    except ValueError as error:
        assert "kind=acceptance" in str(error)
    else:
        raise AssertionError("characterization selected as acceptance")
    oracle_source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "crates/qwen-cli/tests/deepseek_v41_oracle.rs"
    ).read_text()
    oracle_annotations = witness_annotations(oracle_source, path, "qwen-cli")
    assert oracle_annotations is not None
    oracle_identities = [(reason, name) for reason, name, _ in oracle_annotations]
    acceptance_indices = select_annotations(oracle_identities, [], "")
    assert {oracle_identities[index][1] for index in acceptance_indices} == {
        "independent_gguf_seven_token_execution_acceptance",
    }
    conversion_indices = select_annotations(
        oracle_identities,
        ["deepseek-v4.1-reap-272e-mxfp4-q8engram"],
        "",
        "characterization",
    )
    assert {oracle_identities[index][1] for index in conversion_indices} == {
        "hf_conversion_seven_token_characterization",
        "hf_conversion_ten_thousand_token_attention_characterization",
    }
    assert all(
        annotation_fixture(reason) in local_fixture_ids(oracle_source)
        for reason, _ in oracle_identities
    )
    assert "DeepSeek-V4.1-Flash-REAP-272E-MXFP4-Q8Engram.gguf" in oracle_source
    assert "NativeF32-oracle.gguf" not in oracle_source
    annotations = witness_annotations(source, path, "fixture")
    assert annotations == [
        (
            "model witness: sample-fixture",
            "nested_witness",
            source.index("fn nested_witness"),
        )
    ]
    code = rust_code_only(source)
    start = annotations[0][2]
    modules = inline_module_path(module_ranges(code), start)
    assert modules == ["outer", "inner"], modules
    trailing = '#[ignore = "needs reference logits from the external llm tool"] // comment\nfn oracle() {}'
    assert witness_annotations(trailing, path, "fixture")[0][:2] == (
        "needs reference logits from the external llm tool",
        "oracle",
    )

    selection_source = """
#[ignore = "model witness: deepseek-v4-flash-0731-small"]
fn alpha_keep() {}
#[ignore = "model witness: glm-5.3-flash-small"]
fn beta_drop() {}
#[ignore = "model witness: qwen-small"]
fn gamma_keep() {}
#[ignore = "model witness: dflash-small"]
fn delta_drop() {}
#[ignore = "needs reference logits from the external llm tool"]
fn qwen_oracle_keep() {}
#[ignore = "other reason"]
fn qwen_unannotated_keep() {}
"""
    parsed = witness_annotations(selection_source, path, "fixture")
    assert parsed is not None
    identities = [(reason, name) for reason, name, _ in parsed]
    prefixes = ["deepseek-v4-flash-0731", "glm-5.3-flash", "qwen", "dflash"]
    assert select_annotations(identities, [], "") == [0, 1, 2, 3, 4]
    assert select_annotations(identities, [], "qwen") == [4]
    assert select_annotations(identities, prefixes, "") == [0, 1, 2, 3]
    assert select_annotations(identities, prefixes, "keep") == [0, 2]
    assert select_annotations(identities, ["qwen", "deepseek", "qwen-small"], "") == [
        0,
        2,
    ]
    assert select_annotations(identities, ["qwen"], "alpha") == []
    renamed_source = selection_source
    for _, name in identities:
        renamed_source = renamed_source.replace(
            f"fn {name}()", f"fn renamed_{len(name)}_{name[::-1]}()"
        )
    renamed = witness_annotations(renamed_source, path, "fixture")
    assert renamed is not None
    assert select_annotations(
        [(reason, name) for reason, name, _ in renamed], prefixes, ""
    ) == [0, 1, 2, 3]
    for requested in (["missing"], ["qwen", "missing"], ["needs reference"]):
        try:
            select_annotations(identities, requested, "no_matching_name")
        except ValueError as error:
            assert repr(requested[-1]) in str(error)
            assert "no annotated model witnesses" in str(error)
        else:
            raise AssertionError(f"unmatched prefix accepted: {requested}")
    for requested in ([""], ["qwen", ""]):
        try:
            select_annotations(identities, requested, "")
        except ValueError as error:
            assert "--fixture-prefix must not be empty" in str(error)
        else:
            raise AssertionError(f"empty prefix accepted: {requested}")


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
        "cargo",
        "test",
        "--release",
        "-p",
        package,
        *selector,
        "--",
        "--ignored",
        "--list",
        "--format",
        "terse",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"could not list ignored tests for {package} {selector}")
    return [
        line.removesuffix(": test")
        for line in result.stdout.splitlines()
        if line.endswith(": test")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filter", nargs="?", default="", help="function-name substring")
    parser.add_argument(
        "--kind",
        choices=["acceptance", "characterization"],
        default="acceptance",
        help="acceptance is the default; characterization collects complete/finite metrics, not numerical parity acceptance",
    )
    parser.add_argument(
        "--fixture-prefix",
        action="append",
        default=[],
        metavar="PREFIX",
        help="annotated fixture ID prefix (repeatable OR; AND with name filter); each must match an annotation before name filtering",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="validate and print target-qualified selections",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="exercise source annotation parsing and selection",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("model-witnesses: self-test passed")
        return 0
    repo_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    repo = pathlib.Path(repo_result.stdout.strip())
    fixture_source = (repo / "crates/qwen-llm/src/test_fixtures.rs").read_text()
    fixture_ids = set(re.findall(r'id: "([^"]+)"', fixture_source))
    metadata = cargo_metadata()
    packages = {package["name"]: package for package in metadata["packages"]}
    witnesses: list[tuple[str, str, pathlib.Path, str, bool]] = []
    identities: list[tuple[str, str]] = []

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
                    fixture_id = annotation_fixture(reason)
                    if (
                        fixture_id is not None
                        and fixture_id not in fixture_ids | local_fixture_ids(source)
                    ):
                        print(
                            f"{package_name}: witness annotation in {path} names unknown fixture {fixture_id!r}",
                            file=sys.stderr,
                        )
                        return 1
                code = rust_code_only(source)
                ranges = module_ranges(code)
                base = source_base_path(path, src_root)
                for reason, name, offset in annotations:
                    module = base + inline_module_path(ranges, offset)
                    full_name = "::".join([*module, name])
                    witnesses.append(
                        (
                            package_name,
                            full_name,
                            path,
                            name,
                            path.is_relative_to(manifest.parent / "tests"),
                        )
                    )
                    identities.append((reason, name))

    try:
        indices = select_annotations(
            identities, args.fixture_prefix, args.filter, args.kind
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    witnesses = [witnesses[index] for index in indices]

    env = os.environ.copy()
    completion_label = "completed" if args.kind == "characterization" else "passed"
    env["QWEN_REQUIRE_METAL_TESTS"] = "1"
    selected = 0
    selections: list[tuple[str, list[str], str]] = []
    passed_by_package = {name: 0 for name in PACKAGES}
    selected_by_package = {name: 0 for name in PACKAGES}
    failed_by_package = {name: 0 for name in PACKAGES}
    for package_name in PACKAGES:
        package_targets = packages[package_name]["targets"]
        package_witnesses = [
            witness for witness in witnesses if witness[0] == package_name
        ]
        source_targets = {}
        for _, _, source_path, _, integration_source in package_witnesses:
            if integration_source:
                candidates = [
                    target
                    for target in package_targets
                    if "test" in target["kind"]
                    and pathlib.Path(target["src_path"]).resolve()
                    == source_path.resolve()
                ]
            elif package_name == "qwen-cli":
                relative = source_path.relative_to(
                    pathlib.Path(packages[package_name]["manifest_path"]).parent / "src"
                )
                if relative.as_posix() in {
                    "deepseek_v41_tokenizer.rs",
                }:
                    target_name = "qwen_cli"
                elif relative.parts[0] in {"qwen", "serve"}:
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
        for target in {
            id(target): target for target in source_targets.values()
        }.values():
            selector = target_args(target)
            listed.extend(
                (target, name) for name in ignored_listing(package_name, selector)
            )

        for (
            _,
            full_name,
            source_path,
            leaf_name,
            integration_source,
        ) in package_witnesses:
            matches = [
                (target, test_name)
                for target, test_name in listed
                if test_name == full_name
                and (
                    not integration_source
                    or (
                        "test" in target["kind"]
                        and pathlib.Path(target["src_path"]).resolve()
                        == source_path.resolve()
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
            print(
                f"{package_name}: kind={args.kind} selected={selected_by_package[package_name]} {completion_label}=not-run"
            )
            for selected_package, selector, exact_name in selections:
                if selected_package == package_name:
                    print(f"  {' '.join(selector)} :: {exact_name}")
            continue
        for selected_package, selector, exact_name in selections:
            if selected_package != package_name:
                continue
            result = subprocess.run(
                [
                    "cargo",
                    "test",
                    "--release",
                    "-p",
                    package_name,
                    *selector,
                    exact_name,
                    "--",
                    "--ignored",
                    "--exact",
                    "--test-threads=1",
                    *(["--nocapture"] if args.kind == "characterization" else []),
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            expected = re.search(r"test result: ok\. 1 passed; 0 failed", result.stdout)
            if result.returncode != 0 or expected is None:
                print(
                    f"{package_name}: {args.kind} failed: {exact_name}", file=sys.stderr
                )
                failed_by_package[package_name] += 1
            else:
                passed_by_package[package_name] += 1
        print(
            f"{package_name}: kind={args.kind} selected={selected_by_package[package_name]} {completion_label}={passed_by_package[package_name]} failed={failed_by_package[package_name]}",
            flush=True,
        )
    if selected == 0:
        print("no witnesses matched", file=sys.stderr)
        return 1
    return int(any(failed_by_package.values()))


if __name__ == "__main__":
    raise SystemExit(main())
