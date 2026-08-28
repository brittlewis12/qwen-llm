# /// script
# requires-python = ">=3.12"
# ///

"""Acquire same-token Qwen3.8 Flash-Next evidence from pinned llama.cpp."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PACKET_ID = "2026-08-28-qwen4exp-selected-quality-v2"
MANIFEST_SHA256 = "689e94bf135eac09f50cbf88de046004301f7937d4ded2d5cd4434ef7e45ced1"
LLAMA_CPP_COMMIT = "6c84c7d5d8833c6e0df69628f75a0f599797934e"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
LOCAL_OPERATION_COUNT = 78
TOTAL_OPERATION_COUNT = 98
VOCAB_SIZE = 248_320
PRODUCER_STOP_TOKEN_ID = 248_046
CORE_OPERATION_BINDING_SCHEMA = "qwen4exp-selected-quality-llama-operation-binding-v1"
CORE_OPERATION_BINDING_ENCODING = "nlohmann::json compact UTF-8 before binding"
CORE_REPORT_BINDING_SCHEMA = "qwen4exp-selected-quality-llama-core-binding-v1"
BUILD_POLICY_SCHEMA = "qwen4exp-selected-quality-llama-isolated-build-v1"
FORBIDDEN_BUILD_ENVIRONMENT = {
    "AR",
    "ASM",
    "CC",
    "CFLAGS",
    "CMAKE_GENERATOR",
    "CMAKE_PREFIX_PATH",
    "CPATH",
    "CPPFLAGS",
    "CPLUS_INCLUDE_PATH",
    "CXX",
    "CXXFLAGS",
    "C_INCLUDE_PATH",
    "DEVELOPER_DIR",
    "LD",
    "LDFLAGS",
    "LIBRARY_PATH",
    "MACOSX_DEPLOYMENT_TARGET",
    "NINJAFLAGS",
    "OBJC",
    "SDKROOT",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode()


def is_lower_hex(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(byte in "0123456789abcdef" for byte in value)
    )


def run_git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def parse_json_strict(data: bytes) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON number {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        data,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def validate_environment() -> dict[str, object]:
    forbidden_prefixes = (
        "CMAKE_",
        "DYLD_",
        "GGML_",
        "LLAMA_",
        "METAL_",
        "MTL_",
        "QWEN",
    )
    forbidden = sorted(
        name
        for name in os.environ
        if name.startswith(forbidden_prefixes) or name in FORBIDDEN_BUILD_ENVIRONMENT
    )
    if forbidden:
        raise RuntimeError(f"undeclared runtime environment overrides: {forbidden}")
    return {
        "forbidden_prefixes": list(forbidden_prefixes),
        "forbidden_build_variables": sorted(FORBIDDEN_BUILD_ENVIRONMENT),
        "undeclared_overrides_rejected": True,
    }


def validate_source(
    repository: Path,
    llama_source: Path,
) -> dict[str, object]:
    source_commit = run_git(repository, "rev-parse", "HEAD").decode().strip()
    tracked_diff = run_git(
        repository, "diff", "--binary", "--no-ext-diff", "HEAD", "--"
    )
    if sha256(tracked_diff) != EMPTY_SHA256:
        raise RuntimeError("D acquisition requires a clean tracked qwen tree")
    required_paths = [
        "scripts/bench/qwen4exp_selected_quality_llama.py",
        "scripts/bench/qwen4exp_selected_quality_llama/CMakeLists.txt",
        "scripts/bench/qwen4exp_selected_quality_llama/main.cpp",
        "docs/bench/2026-08-28-qwen4exp-selected-quality-prereg/README.md",
        "docs/bench/2026-08-28-qwen4exp-selected-quality-prereg/fixtures.json",
    ]
    for path in required_paths:
        subprocess.run(
            [
                "/usr/bin/git",
                "-C",
                str(repository),
                "ls-files",
                "--error-unmatch",
                "--",
                path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    scoped_status = run_git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "scripts/bench/qwen4exp_selected_quality_llama.py",
        "scripts/bench/qwen4exp_selected_quality_llama",
        "docs/bench/2026-08-28-qwen4exp-selected-quality-prereg",
    )
    if scoped_status:
        raise RuntimeError(f"D source scope is dirty:\n{scoped_status.decode()}")

    llama_commit = run_git(llama_source, "rev-parse", "HEAD").decode().strip()
    if llama_commit != LLAMA_CPP_COMMIT:
        raise RuntimeError(f"llama.cpp commit {llama_commit} != {LLAMA_CPP_COMMIT}")
    llama_status = run_git(
        llama_source, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if llama_status:
        raise RuntimeError(f"llama.cpp checkout is dirty:\n{llama_status.decode()}")
    llama_tree = run_git(llama_source, "rev-parse", "HEAD^{tree}").decode().strip()
    object_format = (
        run_git(llama_source, "rev-parse", "--show-object-format").decode().strip()
    )
    if object_format not in {"sha1", "sha256"}:
        raise RuntimeError(f"unsupported llama.cpp Git object format {object_format}")

    source_rows = []
    for relative in required_paths[:3]:
        path = repository / relative
        before = file_stamp(path)
        observed_sha256 = sha256_file(path)
        after = file_stamp(path)
        if before != after:
            raise RuntimeError(f"runner source changed while hashing: {relative}")
        source_rows.append(
            {
                "path": relative,
                "bytes": after["bytes"],
                "sha256": observed_sha256,
                "stamp": after,
            }
        )
    source_domain = b"qwen4exp-selected-quality-llama-runner-sources-v1\0"
    for row in source_rows:
        source_domain += f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n".encode()
    source_by_name = {Path(str(row["path"])).name: row for row in source_rows}
    main_sha256 = str(source_by_name["main.cpp"]["sha256"])
    cmake_sha256 = str(source_by_name["CMakeLists.txt"]["sha256"])
    core_source_domain = (
        "qwen4exp-selected-quality-llama-core-sources-v1\n"
        f"main.cpp={main_sha256}\n"
        f"CMakeLists.txt={cmake_sha256}\n"
    )
    return {
        "qwen_source_commit": source_commit,
        "qwen_tracked_diff_sha256": sha256(tracked_diff),
        "required_head_paths": required_paths,
        "runner_sources": source_rows,
        "runner_source_manifest_sha256": sha256(source_domain),
        "core_build_expected": {
            "main_cpp_sha256": main_sha256,
            "cmake_lists_sha256": cmake_sha256,
            "source_manifest_schema": (
                "qwen4exp-selected-quality-llama-core-sources-v1"
            ),
            "source_manifest_domain_utf8": core_source_domain,
            "source_manifest_sha256": sha256(core_source_domain.encode()),
            "build_type": "Release",
        },
        "llama_cpp": {
            "repository": "ggml-org/llama.cpp",
            "commit": llama_commit,
            "tree": llama_tree,
            "object_format": object_format,
            "support_pull_request": 27742,
            "scoped_status_sha256": sha256(llama_status),
        },
    }


def parse_git_tree(repository: Path) -> dict[str, dict[str, str]]:
    raw = run_git(
        repository,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        LLAMA_CPP_COMMIT,
    )
    entries: dict[str, dict[str, str]] = {}
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        metadata, raw_path = encoded.split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii").split(" ")
        path = raw_path.decode("utf-8")
        if path in entries or any(byte in path for byte in "\t\r\n"):
            raise RuntimeError(f"unsupported llama.cpp tree path {path!r}")
        entries[path] = {"mode": mode, "kind": kind, "object_id": object_id}
    return entries


def git_blob_and_sha256(path: Path, object_format: str) -> tuple[int, str, str]:
    size = path.stat().st_size
    content = hashlib.sha256()
    blob = hashlib.new(object_format)
    blob.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            content.update(chunk)
            blob.update(chunk)
    return size, content.hexdigest(), blob.hexdigest()


def exported_tree_manifest(
    root: Path,
    git_entries: dict[str, dict[str, str]],
    object_format: str,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        base = Path(directory)
        symlink_directories = [
            name for name in directory_names if (base / name).is_symlink()
        ]
        directory_names[:] = [
            name for name in directory_names if name not in symlink_directories
        ]
        for name in [*symlink_directories, *file_names]:
            path = base / name
            relative = path.relative_to(root).as_posix()
            expected = git_entries.get(relative)
            if expected is None or expected["kind"] != "blob":
                raise RuntimeError(f"exported path is not a pinned blob: {relative}")
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raw = os.readlink(path).encode()
                blob = hashlib.new(object_format)
                blob.update(f"blob {len(raw)}\0".encode())
                blob.update(raw)
                row = {
                    "path": relative,
                    "git_mode": "120000",
                    "kind": "symlink",
                    "bytes": len(raw),
                    "sha256": sha256(raw),
                    "git_blob_id": blob.hexdigest(),
                }
            elif stat.S_ISREG(metadata.st_mode):
                size, content_sha256, blob_id = git_blob_and_sha256(path, object_format)
                git_mode = "100755" if metadata.st_mode & stat.S_IXUSR else "100644"
                row = {
                    "path": relative,
                    "git_mode": git_mode,
                    "kind": "regular",
                    "bytes": size,
                    "sha256": content_sha256,
                    "git_blob_id": blob_id,
                }
            else:
                raise RuntimeError(f"unsupported exported file type: {relative}")
            if (
                row["git_mode"] != expected["mode"]
                or row["git_blob_id"] != expected["object_id"]
            ):
                raise RuntimeError(
                    f"exported blob differs from pinned tree: {relative}"
                )
            rows.append(row)
    rows.sort(key=lambda row: str(row["path"]))
    critical = {
        "include/llama.h",
        "src/models/qwen4exp.cpp",
        "src/llama-memory-hybrid-idx.cpp",
        "ggml/src/ggml-metal/ggml-metal.cpp",
        "ggml/src/ggml-metal/kernels/gated_delta_net.metal",
        "vendor/hash/sha256/sha256.c",
    }
    observed = {str(row["path"]) for row in rows}
    if not critical <= observed:
        raise RuntimeError(
            f"llama.cpp export omits critical inputs: {sorted(critical - observed)}"
        )
    domain = bytearray(b"qwen4exp-selected-quality-llama-export-v1\0")
    for row in rows:
        domain.extend(
            (
                f"{row['path']}\t{row['git_mode']}\t{row['kind']}\t{row['bytes']}\t"
                f"{row['sha256']}\t{row['git_blob_id']}\n"
            ).encode()
        )
    return {
        "schema": "qwen4exp-selected-quality-llama-export-v1",
        "files": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "manifest_sha256": sha256(bytes(domain)),
        "rows": rows,
    }


def make_tree_read_only(root: Path) -> None:
    directories: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        directories.append(base)
        for name in [*directory_names, *file_names]:
            path = base / name
            if path.is_symlink():
                continue
            mode = path.stat().st_mode
            if path.is_dir():
                continue
            path.chmod(0o500 if mode & stat.S_IXUSR else 0o400)
    for directory in reversed(directories):
        directory.chmod(0o500)


def make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        base.chmod(0o700)
        for name in [*directory_names, *file_names]:
            path = base / name
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() else 0o600)


def run_checked(command: list[str], environment: dict[str, str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"{completed.stdout.decode(errors='replace')}"
        )
    return {
        "command": command,
        "stdout_bytes": len(completed.stdout),
        "stdout_sha256": sha256(completed.stdout),
        "returncode": completed.returncode,
    }


def parse_cmake_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        typed_key, value = line.split("=", 1)
        key = typed_key.split(":", 1)[0]
        if key in values:
            raise RuntimeError(f"duplicate CMake cache key {key}")
        values[key] = value
    return values


def validate_compile_commands(
    path: Path, allowed_roots: list[Path]
) -> dict[str, object]:
    raw = path.read_bytes()
    commands = parse_json_strict(raw)
    if not isinstance(commands, list) or not commands:
        raise RuntimeError("empty compile_commands.json")
    forbidden = {
        "-Ofast",
        "-fassociative-math",
        "-ffast-math",
        "-ffinite-math-only",
        "-fno-signed-zeros",
        "-fno-trapping-math",
        "-freciprocal-math",
        "-funsafe-math-optimizations",
        "-march=native",
        "-mtune=native",
    }
    required_safe = {
        "-ffp-contract=off",
        "-fno-fast-math",
        "-fno-unsafe-math-optimizations",
    }
    language_commands = 0
    for ordinal, value in enumerate(commands):
        if not isinstance(value, dict):
            raise RuntimeError(f"compile command {ordinal} is not an object")
        if "arguments" in value:
            arguments = value["arguments"]
            if not isinstance(arguments, list) or not all(
                isinstance(argument, str) for argument in arguments
            ):
                raise RuntimeError(f"compile command {ordinal} arguments")
        elif isinstance(value.get("command"), str):
            arguments = shlex.split(value["command"])
        else:
            raise RuntimeError(f"compile command {ordinal} encoding")
        if forbidden & set(arguments):
            raise RuntimeError(f"unsafe compile flags in command {ordinal}")
        if any(
            argument in {"-include", "-include-pch", "-imacros"}
            or argument.startswith(("-include=", "-imacros="))
            for argument in arguments
        ):
            raise RuntimeError(f"forced include in compile command {ordinal}")
        source = Path(str(value.get("file", ""))).resolve(strict=True)
        if not any(source.is_relative_to(root) for root in allowed_roots):
            raise RuntimeError(f"compile source outside isolated roots: {source}")
        if source.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".m", ".mm"}:
            language_commands += 1
            if not required_safe <= set(arguments):
                raise RuntimeError(
                    f"safe FP flags absent from compile command {ordinal}"
                )
    if language_commands < 100:
        raise RuntimeError(
            "unexpectedly small compiled C/C++/Objective-C source census"
        )
    return {
        "path": "compile_commands.json",
        "bytes": len(raw),
        "sha256": sha256(raw),
        "commands": len(commands),
        "language_commands_with_safe_fp_policy": language_commands,
        "forbidden_flags_absent": sorted(forbidden),
        "forced_includes_absent": True,
    }


def validate_dynamic_linkage(core: Path) -> dict[str, object]:
    linkage = subprocess.run(
        ["/usr/bin/otool", "-L", str(core)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    dependencies = []
    for line in linkage.splitlines()[1:]:
        dependency = line.strip().split(" (", 1)[0]
        if not dependency.startswith(("/usr/lib/", "/System/Library/Frameworks/")):
            raise RuntimeError(f"non-system dynamic dependency: {dependency}")
        dependencies.append(dependency)
    load_commands = subprocess.run(
        ["/usr/bin/otool", "-l", str(core)],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if b"LC_RPATH" in load_commands:
        raise RuntimeError("core executable contains LC_RPATH")
    return {
        "policy": "only /usr/lib and /System/Library/Frameworks dependencies",
        "dependencies": dependencies,
        "otool_L_sha256": sha256(linkage.encode()),
        "otool_l_bytes": len(load_commands),
        "otool_l_sha256": sha256(load_commands),
        "lc_rpath_absent": True,
    }


@dataclass
class IsolatedCoreBuild:
    workspace: Path
    source_root: Path
    harness_root: Path
    build_root: Path
    core: Path
    git_entries: dict[str, dict[str, str]]
    object_format: str
    source_manifest_sha256: str
    report: dict[str, object]


def resolve_executable(name: str) -> Path:
    candidates: list[Path] = []
    mise = shutil.which("mise")
    if mise is not None:
        resolved = subprocess.run(
            [mise, "which", name],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if resolved.returncode == 0 and resolved.stdout.strip():
            candidates.append(Path(resolved.stdout.strip()))
    candidates.extend(
        Path(prefix) / name
        for prefix in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")
    )
    discovered = shutil.which(name)
    if discovered is not None:
        candidates.append(Path(discovered))
    seen: set[str] = set()
    for candidate in candidates:
        path = candidate.absolute()
        if str(path) in seen:
            continue
        seen.add(str(path))
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        probe = subprocess.run(
            [str(path), "--version"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return path
    raise RuntimeError(f"required build tool is unavailable: {name}")


def build_isolated_core(
    repository: Path,
    llama_repository: Path,
    source: dict[str, object],
    output_directory: Path,
) -> IsolatedCoreBuild:
    workspace = Path(
        tempfile.mkdtemp(
            prefix=".qwen4exp-selected-quality-build.", dir=output_directory
        )
    )
    workspace.chmod(0o700)
    source_root = workspace / "llama-source"
    harness_root = workspace / "harness"
    build_root = workspace / "build"
    temporary_root = workspace / "tmp"
    home_root = workspace / "home"
    for path in (source_root, harness_root, build_root, temporary_root, home_root):
        path.mkdir(mode=0o700)

    git_entries = parse_git_tree(llama_repository)
    archive = workspace / "llama-source.tar"
    archive_descriptor = os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(archive_descriptor, "wb") as stream:
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(llama_repository),
                    "archive",
                    "--format=tar",
                    LLAMA_CPP_COMMIT,
                ],
                check=True,
                stdout=stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        subprocess.run(
            ["/usr/bin/tar", "-xf", str(archive), "-C", str(source_root)],
            check=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        export_manifest = exported_tree_manifest(
            source_root,
            git_entries,
            str(source["llama_cpp"]["object_format"]),
        )
        make_tree_read_only(source_root)

        source_rows = {str(row["path"]): row for row in source["runner_sources"]}
        for name, relative in (
            (
                "CMakeLists.txt",
                "scripts/bench/qwen4exp_selected_quality_llama/CMakeLists.txt",
            ),
            ("main.cpp", "scripts/bench/qwen4exp_selected_quality_llama/main.cpp"),
        ):
            data = (repository / relative).read_bytes()
            if sha256(data) != source_rows[relative]["sha256"]:
                raise RuntimeError(
                    f"runner source changed before isolated copy: {relative}"
                )
            destination = harness_root / name
            descriptor = os.open(
                destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        harness_root.chmod(0o500)

        cmake = resolve_executable("cmake")
        ninja = resolve_executable("ninja")
        clang = Path("/usr/bin/clang").resolve(strict=True)
        clangxx = Path("/usr/bin/clang++").resolve(strict=True)
        sdk = Path(
            subprocess.run(
                ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
        ).resolve(strict=True)
        tools = {
            "cmake": {
                "path": str(cmake),
                "version": subprocess.run(
                    [str(cmake), "--version"],
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                ).stdout.splitlines()[0],
            },
            "ninja": {
                "path": str(ninja),
                "version": subprocess.run(
                    [str(ninja), "--version"],
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                ).stdout.strip(),
            },
            "clang": {
                "path": str(clang),
                "version_sha256": sha256(
                    subprocess.run(
                        [str(clang), "--version"], check=True, stdout=subprocess.PIPE
                    ).stdout
                ),
            },
            "clangxx": {
                "path": str(clangxx),
                "version_sha256": sha256(
                    subprocess.run(
                        [str(clangxx), "--version"], check=True, stdout=subprocess.PIPE
                    ).stdout
                ),
            },
            "macos_sdk": str(sdk),
            "git": "/usr/bin/git",
            "tar": "/usr/bin/tar",
            "otool": "/usr/bin/otool",
        }
        safe_flags = "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off"
        definitions = {
            "CMAKE_ASM_COMPILER": str(clang),
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_C_COMPILER": str(clang),
            "CMAKE_CXX_COMPILER": str(clangxx),
            "CMAKE_C_FLAGS": safe_flags,
            "CMAKE_C_FLAGS_RELEASE": "-O3 -DNDEBUG",
            "CMAKE_CXX_FLAGS": safe_flags,
            "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG",
            "CMAKE_EXE_LINKER_FLAGS": "",
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "CMAKE_MAKE_PROGRAM": str(ninja),
            "CMAKE_OSX_ARCHITECTURES": "arm64",
            "CMAKE_OSX_SYSROOT": str(sdk),
            "GGML_ACCELERATE": "ON",
            "GGML_BACKEND_DL": "OFF",
            "GGML_BLAS": "OFF",
            "GGML_CCACHE": "OFF",
            "GGML_CPU": "ON",
            "GGML_CPU_ALL_VARIANTS": "OFF",
            "GGML_CPU_KLEIDIAI": "OFF",
            "GGML_LLAMAFILE": "OFF",
            "GGML_METAL": "ON",
            "GGML_METAL_EMBED_LIBRARY": "ON",
            "GGML_METAL_NDEBUG": "OFF",
            "GGML_METAL_SHADER_DEBUG": "OFF",
            "GGML_NATIVE": "OFF",
            "GGML_OPENMP": "OFF",
        }
        policy = {
            "schema": BUILD_POLICY_SCHEMA,
            "llama_cpp_commit": LLAMA_CPP_COMMIT,
            "llama_cpp_tree": source["llama_cpp"]["tree"],
            "llama_cpp_export_manifest_sha256": export_manifest["manifest_sha256"],
            "generator": "Ninja",
            "parallel_jobs": 8,
            "tools": tools,
            "definitions": definitions,
            "environment": {
                "HOME": "<private-build-home>",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "TMPDIR": "<private-build-tmp>",
                "ZERO_AR_DATE": "1",
            },
        }
        policy_bytes = canonical_json(policy)
        policy_sha256 = sha256(policy_bytes)
        configure_definitions = {
            **definitions,
            "LLAMA_CPP_SOURCE": str(source_root),
            "QWEN4EXP_BUILD_POLICY_SHA256": policy_sha256,
            "QWEN4EXP_LLAMA_EXPORT_MANIFEST_SHA256": str(
                export_manifest["manifest_sha256"]
            ),
            "QWEN4EXP_LLAMA_TREE": str(source["llama_cpp"]["tree"]),
        }
        environment = {
            "HOME": str(home_root),
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "TMPDIR": str(temporary_root),
            "ZERO_AR_DATE": "1",
        }
        configure_command = [
            str(cmake),
            "-S",
            str(harness_root),
            "-B",
            str(build_root),
            "-G",
            "Ninja",
            *[
                f"-D{key}={value}"
                for key, value in sorted(configure_definitions.items())
            ],
        ]
        configure = run_checked(configure_command, environment)
        build_command = [
            str(cmake),
            "--build",
            str(build_root),
            "--target",
            "qwen4exp-selected-quality-llama",
            "-j",
            "8",
        ]
        build = run_checked(build_command, environment)

        cache_path = build_root / "CMakeCache.txt"
        cache = parse_cmake_cache(cache_path)
        expected_cache = {
            **definitions,
            "BUILD_SHARED_LIBS": "OFF",
            "LLAMA_BUILD_EXAMPLES": "OFF",
            "LLAMA_BUILD_SERVER": "OFF",
            "LLAMA_BUILD_TESTS": "OFF",
            "LLAMA_BUILD_TOOLS": "OFF",
            "LLAMA_OPENSSL": "OFF",
            "LLAMA_SUBPROCESS": "OFF",
            "QWEN4EXP_BUILD_POLICY_SHA256": policy_sha256,
            "QWEN4EXP_LLAMA_EXPORT_MANIFEST_SHA256": str(
                export_manifest["manifest_sha256"]
            ),
            "QWEN4EXP_LLAMA_TREE": str(source["llama_cpp"]["tree"]),
        }
        for key, expected in expected_cache.items():
            if cache.get(key) != expected:
                raise RuntimeError(
                    f"CMake cache policy drift: {key}={cache.get(key)!r}"
                )

        compile_commands = validate_compile_commands(
            build_root / "compile_commands.json",
            [source_root, harness_root, build_root],
        )
        command_census_bytes = subprocess.run(
            [
                str(ninja),
                "-C",
                str(build_root),
                "-t",
                "commands",
                "qwen4exp-selected-quality-llama",
            ],
            check=True,
            env=environment,
            stdout=subprocess.PIPE,
        ).stdout
        command_census_text = command_census_bytes.decode()
        for unsafe in (
            " -Ofast ",
            " -fassociative-math ",
            " -ffast-math ",
            " -ffinite-math-only ",
            " -funsafe-math-optimizations ",
            " -march=native ",
            " -mtune=native ",
        ):
            if unsafe in f" {command_census_text} ":
                raise RuntimeError(f"unsafe build command option {unsafe.strip()}")

        core = build_root / "qwen4exp-selected-quality-llama"
        if not core.is_file() or not os.access(core, os.X_OK):
            raise RuntimeError("isolated core executable was not built")
        core_before = file_stamp(core)
        core_sha256 = sha256_file(core)
        core_after = file_stamp(core)
        if core_before != core_after:
            raise RuntimeError("isolated core changed while hashing")
        linkage = validate_dynamic_linkage(core)
        source_after = exported_tree_manifest(
            source_root,
            git_entries,
            str(source["llama_cpp"]["object_format"]),
        )
        if source_after["manifest_sha256"] != export_manifest["manifest_sha256"]:
            raise RuntimeError("read-only llama.cpp export changed during build")
        archive_before = file_stamp(archive)
        archive_sha256 = sha256_file(archive)
        archive_after = file_stamp(archive)
        if archive_before != archive_after:
            raise RuntimeError("llama.cpp Git archive changed while hashing")

        core_build_expected = {
            **source["core_build_expected"],
            "llama_cpp_tree": source["llama_cpp"]["tree"],
            "llama_cpp_export_manifest_sha256": export_manifest["manifest_sha256"],
            "build_policy_sha256": policy_sha256,
        }
        report = {
            "schema": BUILD_POLICY_SCHEMA,
            "policy": policy,
            "policy_bytes": len(policy_bytes),
            "policy_sha256": policy_sha256,
            "llama_cpp_export": {
                "archive_bytes": archive_after["bytes"],
                "archive_sha256": archive_sha256,
                "tree_manifest": export_manifest,
                "read_only_during_build": True,
                "manifest_revalidated_after_build": True,
            },
            "configure": configure,
            "build": build,
            "cmake_cache": {
                "bytes": cache_path.stat().st_size,
                "sha256": sha256_file(cache_path),
                "enforced_values": expected_cache,
            },
            "compile_commands": compile_commands,
            "ninja_commands": {
                "bytes": len(command_census_bytes),
                "sha256": sha256(command_census_bytes),
                "unsafe_options_absent": True,
            },
            "build_ninja": {
                "bytes": (build_root / "build.ninja").stat().st_size,
                "sha256": sha256_file(build_root / "build.ninja"),
            },
            "dynamic_linkage": linkage,
            "core_build_expected": core_build_expected,
            "core_executable": {
                "path": str(core),
                "bytes": core_after["bytes"],
                "sha256": core_sha256,
                "stamp": core_after,
                "linkage": "static llama.cpp/ggml; allowlisted system libraries only",
            },
        }
        return IsolatedCoreBuild(
            workspace=workspace,
            source_root=source_root,
            harness_root=harness_root,
            build_root=build_root,
            core=core,
            git_entries=git_entries,
            object_format=str(source["llama_cpp"]["object_format"]),
            source_manifest_sha256=str(export_manifest["manifest_sha256"]),
            report=report,
        )
    except BaseException:
        make_tree_writable(source_root)
        make_tree_writable(harness_root)
        shutil.rmtree(workspace, ignore_errors=True)
        raise


def revalidate_isolated_build(build: IsolatedCoreBuild) -> None:
    source_manifest = exported_tree_manifest(
        build.source_root,
        build.git_entries,
        build.object_format,
    )
    if source_manifest["manifest_sha256"] != build.source_manifest_sha256:
        raise RuntimeError("isolated llama.cpp export changed during acquisition")
    expected = build.report["core_executable"]
    before = file_stamp(build.core)
    observed_sha256 = sha256_file(build.core)
    after = file_stamp(build.core)
    if before != after or after != expected["stamp"]:
        raise RuntimeError("isolated core inode changed during acquisition")
    if observed_sha256 != expected["sha256"]:
        raise RuntimeError("isolated core bytes changed during acquisition")
    build.report["source_and_core_revalidated_after_acquisition"] = True


def cleanup_isolated_build(build: IsolatedCoreBuild) -> None:
    make_tree_writable(build.source_root)
    make_tree_writable(build.harness_root)
    shutil.rmtree(build.workspace)


def read_tokens(root: Path, fixture: dict[str, object]) -> list[int]:
    fixture_id = str(fixture["fixture_id"])
    record = fixture["tokens"]
    assert isinstance(record, dict)
    path = root / str(record["path"])
    data = path.read_bytes()
    if len(data) != int(record["bytes"]):
        raise RuntimeError(f"{fixture_id} byte count")
    if sha256(data) != record["sha256"] or sha256(data) != record["sha256_raw_i32le"]:
        raise RuntimeError(f"{fixture_id} token hash")
    tokens = [
        int.from_bytes(data[index : index + 4], "little", signed=True)
        for index in range(0, len(data), 4)
    ]
    if len(tokens) != int(record["token_count"]):
        raise RuntimeError(f"{fixture_id} token count")
    if not all(0 <= token < VOCAB_SIZE for token in tokens):
        raise RuntimeError(f"{fixture_id} token range")
    return tokens


def ordering_key(fixture: dict[str, object]) -> str:
    record = fixture["tokens"]
    assert isinstance(record, dict)
    domain = (
        "qwen4exp-selected-quality-llama-order-v1\0"
        f"fixture_id={fixture['fixture_id']}\n"
        f"tokens_sha256={record['sha256_raw_i32le']}\n"
    )
    return sha256(domain.encode())


def validate_fixtures(
    path: Path,
) -> tuple[
    dict[str, object],
    dict[str, list[int]],
    dict[str, dict[str, object]],
    list[tuple[str, str, str]],
]:
    data = path.read_bytes()
    if sha256(data) != MANIFEST_SHA256:
        raise RuntimeError("fixture manifest SHA-256")
    manifest = parse_json_strict(data)
    assert isinstance(manifest, dict)
    if manifest["packet_id"] != PACKET_ID or manifest["schema_version"] != 2:
        raise RuntimeError("fixture manifest contract")
    root = path.parent
    tokens: dict[str, list[int]] = {}
    fixtures: dict[str, dict[str, object]] = {}
    all_fixtures = [
        *manifest["natural_fixtures"],
        manifest["scope_control"],
        *manifest["retrieval_fixtures"],
    ]
    for fixture in all_fixtures:
        fixture_id = str(fixture["fixture_id"])
        if fixture_id in fixtures:
            raise RuntimeError(f"duplicate fixture ID {fixture_id}")
        fixtures[fixture_id] = fixture
        tokens[fixture_id] = read_tokens(root, fixture)

    for fixture in manifest["natural_fixtures"]:
        fixture_tokens = tokens[str(fixture["fixture_id"])]
        prompt_count = int(fixture["prompt_token_count"])
        continuation_count = int(fixture["continuation_token_count"])
        if continuation_count != 96 or len(fixture_tokens) != prompt_count + 96:
            raise RuntimeError(f"natural fixture shape {fixture['fixture_id']}")
        if fixture_tokens[-1] != int(fixture["terminal_feed_token_id"]):
            raise RuntimeError(f"natural terminal token {fixture['fixture_id']}")
    for fixture in manifest["retrieval_fixtures"]:
        fixture_id = str(fixture["fixture_id"])
        if len(tokens[fixture_id]) != 4099:
            raise RuntimeError(f"retrieval fixture shape {fixture_id}")
        answers = fixture["answer_token_ids"]
        if (
            len(answers) != int(fixture["answer_token_count"])
            or not 1 <= len(answers) <= 2
        ):
            raise RuntimeError(f"retrieval answer shape {fixture_id}")
        if fixture["producer_stop_token_ids"] != [PRODUCER_STOP_TOKEN_ID]:
            raise RuntimeError(f"retrieval stop token {fixture_id}")

    expected = []
    for fixture in manifest["natural_fixtures"]:
        expected.append(
            (ordering_key(fixture), fixture["fixture_id"], "teacher_forced_nll_96")
        )
    for fixture in manifest["retrieval_fixtures"]:
        expected.append(
            (
                ordering_key(fixture),
                fixture["fixture_id"],
                "answer_nll_and_exact_prefix",
            )
        )
    expected.sort()
    plan = manifest["execution"]["operation_plan"]
    if len(plan) != TOTAL_OPERATION_COUNT:
        raise RuntimeError("operation count")
    for offset, row in enumerate(plan[LOCAL_OPERATION_COUNT:]):
        key, fixture_id, mode = expected[offset]
        required = {
            "ordinal": LOCAL_OPERATION_COUNT + offset,
            "phase": "llama_cpp_triangulation",
            "fixture_id": fixture_id,
            "mode": mode,
            "arm": "D",
            "ordering_key_sha256": key,
        }
        if row != required:
            raise RuntimeError(f"D operation {offset} drift")
    return manifest, tokens, fixtures, expected


def file_stamp(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "mode": stat.st_mode,
    }


def validate_model(
    first_shard: Path, lock: dict[str, object]
) -> tuple[dict[str, object], list[dict[str, int | str]], list[Path]]:
    expected_shards = sorted(lock["shards"], key=lambda row: int(row["index"]))
    if [int(row["index"]) for row in expected_shards] != list(
        range(len(expected_shards))
    ):
        raise RuntimeError("model shard indices")
    if not expected_shards:
        raise RuntimeError("empty model shard lock")
    first_expected = expected_shards[0]
    if first_shard.name != first_expected["filename"]:
        raise RuntimeError(
            f"--model must name locked shard index 0: {first_expected['filename']}"
        )
    model_directory = first_shard.parent.resolve(strict=True)
    canonical_first = (model_directory / str(first_expected["filename"])).resolve(
        strict=True
    )
    if first_shard.resolve(strict=True) != canonical_first:
        raise RuntimeError("--model does not resolve to locked shard index 0")
    rows = []
    stamps = []
    canonical_paths = []
    for expected in expected_shards:
        path = (model_directory / str(expected["filename"])).resolve(strict=True)
        before = file_stamp(path)
        if before["bytes"] != expected["bytes"]:
            raise RuntimeError(f"model shard size: {path}")
        observed_hash = sha256_file(path)
        after = file_stamp(path)
        if before != after:
            raise RuntimeError(f"model shard changed while hashing: {path}")
        if observed_hash != expected["sha256"]:
            raise RuntimeError(f"model shard SHA-256: {path}")
        canonical_paths.append(path)
        stamps.append(after)
        rows.append(
            {
                "index": expected["index"],
                "file_name": expected["filename"],
                "bytes": expected["bytes"],
                "sha256": observed_hash,
                "stamp": after,
            }
        )
    return (
        {
            "repository": lock["repository"],
            "revision": lock["revision"],
            "quant": lock["quant"],
            "shard_manifest_sha256": lock["shard_manifest_sha256"],
            "qualification": "every local shard was hashed in full before llama.cpp execution",
            "locked_first_shard_path": str(canonical_first),
            "shards": rows,
        },
        stamps,
        canonical_paths,
    )


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def stage_core_executable(
    core: Path,
    output_directory: Path,
    expected: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    source_before = file_stamp(core)
    if source_before != expected["stamp"]:
        raise RuntimeError("validated core runner changed before staging")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".qwen4exp-selected-quality-llama-core.",
        dir=output_directory,
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    copied = 0
    try:
        with core.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            destination.flush()
            os.fsync(destination.fileno())
            os.fchmod(destination.fileno(), 0o500)
            os.fsync(destination.fileno())
        source_after = file_stamp(core)
        if source_before != source_after:
            raise RuntimeError("validated core runner changed while staging")
        if copied != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            raise RuntimeError("staged core bytes do not match validated executable")
        staged_before = file_stamp(temporary)
        staged_sha256 = sha256_file(temporary)
        staged_after = file_stamp(temporary)
        if staged_before != staged_after or staged_sha256 != expected["sha256"]:
            raise RuntimeError("staged core verification failed")
        if stat.S_IMODE(int(staged_after["mode"])) != 0o500:
            raise RuntimeError("staged core is not read-only executable")
        sync_directory(output_directory)
        return temporary, {
            "strategy": "exclusive byte copy, fsync, chmod 0500, execute copied inode",
            "source_path": str(core),
            "source_stamp": source_after,
            "bytes": copied,
            "sha256": staged_sha256,
            "stamp_before_execution": staged_after,
        }
    except BaseException:
        temporary.unlink(missing_ok=True)
        sync_directory(output_directory)
        raise


def revalidate_staged_core(
    core: Path,
    record: dict[str, object],
) -> None:
    before = file_stamp(core)
    observed_hash = sha256_file(core)
    after = file_stamp(core)
    if before != after or after != record["stamp_before_execution"]:
        raise RuntimeError("staged core inode changed during execution")
    if observed_hash != record["sha256"]:
        raise RuntimeError("staged core bytes changed during execution")
    record["stamp_after_execution"] = after
    record["revalidated_after_execution"] = True


class OutputReservation:
    def __init__(self, output: Path) -> None:
        if output.exists():
            raise RuntimeError(f"output already exists: {output}")
        self.output = output
        self.lock = (
            output.parent / f".{output.name}.qwen4exp-selected-quality-llama.lock"
        )
        descriptor = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"pid={os.getpid()}\noutput={output}\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._sync_directory()
        self.active = True

    def _sync_directory(self) -> None:
        descriptor = os.open(self.output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def release(self) -> None:
        if self.active:
            self.lock.unlink()
            self._sync_directory()
            self.active = False

    def __enter__(self) -> OutputReservation:
        return self

    def __exit__(self, *_: object) -> None:
        if self.active:
            try:
                self.release()
            except OSError as error:
                print(
                    f"warning: could not remove output reservation: {error}",
                    file=sys.stderr,
                )

    def publish(self, data: bytes) -> str:
        report_sha256 = sha256(data)
        temporary = self.output.parent / (
            f".{self.output.name}.tmp.{os.getpid()}.{report_sha256[:16]}"
        )
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, self.output)
            try:
                self._sync_directory()
            except OSError:
                self.output.unlink(missing_ok=True)
                self._sync_directory()
                raise
            try:
                temporary.unlink()
            except OSError as error:
                print(
                    f"warning: could not remove temporary report: {error}",
                    file=sys.stderr,
                )
            try:
                self.release()
            except OSError as error:
                print(
                    f"warning: could not remove output reservation: {error}",
                    file=sys.stderr,
                )
            return report_sha256
        finally:
            temporary.unlink(missing_ok=True)


def require_dict(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must be an object")
    return value


def require_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeError(f"{path} must be an array")
    return value


def require_exact_keys(value: dict[str, object], keys: set[str], path: str) -> None:
    observed = set(value)
    if observed != keys:
        missing = sorted(keys - observed)
        extra = sorted(observed - keys)
        raise RuntimeError(f"{path} keys: missing={missing} extra={extra}")


def require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{path} must be an integer")
    return value


def require_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{path} must be a number")
    observed = float(value)
    if not math.isfinite(observed):
        raise RuntimeError(f"{path} must be finite")
    return observed


def require_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{path} must be a boolean")
    return value


def require_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{path} must be a string")
    return value


def require_close(observed: float, expected: float, path: str) -> None:
    tolerance = 2e-12 * max(1.0, abs(observed), abs(expected))
    if abs(observed - expected) > tolerance:
        raise RuntimeError(f"{path}: {observed} != {expected}")


def validate_logit_identity(value: object, expected_rows: int, path: str) -> None:
    identity = require_dict(value, path)
    require_exact_keys(
        identity,
        {
            "endpoint_logits_sha256_f32le",
            "terminal_logits_sha256_f32le",
            "trace_sha256_f32le",
            "trace_rows",
        },
        path,
    )
    for key in (
        "endpoint_logits_sha256_f32le",
        "terminal_logits_sha256_f32le",
        "trace_sha256_f32le",
    ):
        if not is_lower_hex(identity[key]):
            raise RuntimeError(f"{path}.{key}")
    if require_int(identity["trace_rows"], f"{path}.trace_rows") != expected_rows:
        raise RuntimeError(f"{path}.trace_rows")


def validate_score_row(
    value: object,
    ordinal: int,
    target: int,
    path: str,
) -> tuple[float, bool, int]:
    row = require_dict(value, path)
    require_exact_keys(
        row,
        {
            "ordinal",
            "target_token_id",
            "target_logit_f32",
            "logsumexp_f64",
            "nll_f64",
            "argmax_token_id",
            "top1",
        },
        path,
    )
    if require_int(row["ordinal"], f"{path}.ordinal") != ordinal:
        raise RuntimeError(f"{path}.ordinal")
    if require_int(row["target_token_id"], f"{path}.target_token_id") != target:
        raise RuntimeError(f"{path}.target_token_id")
    target_logit = require_number(row["target_logit_f32"], f"{path}.target_logit_f32")
    f32_roundtrip = struct.unpack("<f", struct.pack("<f", target_logit))[0]
    if f32_roundtrip != target_logit:
        raise RuntimeError(f"{path}.target_logit_f32 is not an exact F32 value")
    logsumexp = require_number(row["logsumexp_f64"], f"{path}.logsumexp_f64")
    nll = require_number(row["nll_f64"], f"{path}.nll_f64")
    require_close(nll, logsumexp - target_logit, f"{path}.nll_f64")
    if nll < -2e-12:
        raise RuntimeError(f"{path}.nll_f64 is negative")
    prediction = require_int(row["argmax_token_id"], f"{path}.argmax_token_id")
    if not 0 <= prediction < VOCAB_SIZE:
        raise RuntimeError(f"{path}.argmax_token_id range")
    top1 = require_bool(row["top1"], f"{path}.top1")
    if top1 != (prediction == target):
        raise RuntimeError(f"{path}.top1")
    return nll, top1, prediction


def validate_operation_binding(operation: dict[str, object], path: str) -> None:
    binding = require_dict(operation["binding"], f"{path}.binding")
    require_exact_keys(
        binding,
        {
            "schema",
            "semantic_payload_encoding",
            "semantic_payload_bytes",
            "semantic_payload_sha256",
            "semantic_payload_json_compact",
        },
        f"{path}.binding",
    )
    if binding["schema"] != CORE_OPERATION_BINDING_SCHEMA:
        raise RuntimeError(f"{path}.binding.schema")
    if binding["semantic_payload_encoding"] != CORE_OPERATION_BINDING_ENCODING:
        raise RuntimeError(f"{path}.binding.semantic_payload_encoding")
    payload = dict(operation)
    del payload["binding"]
    compact = require_string(
        binding["semantic_payload_json_compact"],
        f"{path}.binding.semantic_payload_json_compact",
    )
    encoded = compact.encode()
    if parse_json_strict(encoded) != payload:
        raise RuntimeError(f"{path}.binding semantic payload content")
    if require_int(
        binding["semantic_payload_bytes"],
        f"{path}.binding.semantic_payload_bytes",
    ) != len(encoded):
        raise RuntimeError(f"{path}.binding.semantic_payload_bytes")
    if binding["semantic_payload_sha256"] != sha256(encoded):
        raise RuntimeError(f"{path}.binding.semantic_payload_sha256")


def validate_natural_operation(
    operation: dict[str, object],
    fixture: dict[str, object],
    token_ids: list[int],
    operation_ordinal: int,
    ordering_sha256: str,
    path: str,
) -> None:
    require_exact_keys(
        operation,
        {
            "schema_version",
            "operation_ordinal",
            "fixture_id",
            "mode",
            "arm",
            "document_source_ordinal",
            "prompt_token_count",
            "selected_suffix_tokens",
            "continuation",
            "logit_identity",
            "input_token_ids_sha256_raw_i32le",
            "ordering_key_sha256",
            "binding",
        },
        path,
    )
    prompt_count = int(fixture["prompt_token_count"])
    continuation_count = int(fixture["continuation_token_count"])
    expected_scalars = {
        "schema_version": 1,
        "operation_ordinal": operation_ordinal,
        "fixture_id": fixture["fixture_id"],
        "mode": "teacher_forced_nll_96",
        "arm": "D",
        "document_source_ordinal": fixture["document"]["source_ordinal"],
        "prompt_token_count": prompt_count,
        "selected_suffix_tokens": fixture["selected_suffix_tokens"],
        "input_token_ids_sha256_raw_i32le": fixture["tokens"]["sha256_raw_i32le"],
        "ordering_key_sha256": ordering_sha256,
    }
    for key, expected in expected_scalars.items():
        if operation[key] != expected:
            raise RuntimeError(f"{path}.{key}")
    continuation = require_dict(operation["continuation"], f"{path}.continuation")
    require_exact_keys(
        continuation,
        {"tokens", "nll_sum_f64", "mean_nll_f64", "top1_hits", "scored_rows"},
        f"{path}.continuation",
    )
    if require_int(continuation["tokens"], f"{path}.continuation.tokens") != 96:
        raise RuntimeError(f"{path}.continuation.tokens")
    rows = require_list(continuation["scored_rows"], f"{path}.continuation.scored_rows")
    if len(rows) != continuation_count:
        raise RuntimeError(f"{path}.continuation.scored_rows count")
    nll_values = []
    top1_hits = 0
    for ordinal, row in enumerate(rows):
        target = token_ids[prompt_count + ordinal]
        nll, top1, _ = validate_score_row(
            row,
            ordinal,
            target,
            f"{path}.continuation.scored_rows[{ordinal}]",
        )
        nll_values.append(nll)
        top1_hits += int(top1)
    nll_sum = sum(nll_values)
    require_close(
        require_number(continuation["nll_sum_f64"], f"{path}.continuation.nll_sum_f64"),
        nll_sum,
        f"{path}.continuation.nll_sum_f64",
    )
    require_close(
        require_number(
            continuation["mean_nll_f64"], f"{path}.continuation.mean_nll_f64"
        ),
        nll_sum / continuation_count,
        f"{path}.continuation.mean_nll_f64",
    )
    if (
        require_int(continuation["top1_hits"], f"{path}.continuation.top1_hits")
        != top1_hits
    ):
        raise RuntimeError(f"{path}.continuation.top1_hits")
    validate_logit_identity(operation["logit_identity"], 97, f"{path}.logit_identity")
    validate_operation_binding(operation, path)


def validate_retrieval_operation(
    operation: dict[str, object],
    fixture: dict[str, object],
    operation_ordinal: int,
    ordering_sha256: str,
    path: str,
) -> None:
    require_exact_keys(
        operation,
        {
            "schema_version",
            "operation_ordinal",
            "fixture_id",
            "mode",
            "arm",
            "kind",
            "document_source_ordinal",
            "prompt_token_count",
            "selected_suffix_tokens",
            "answer",
            "logit_identity",
            "input_token_ids_sha256_raw_i32le",
            "ordering_key_sha256",
            "binding",
        },
        path,
    )
    expected_scalars = {
        "schema_version": 1,
        "operation_ordinal": operation_ordinal,
        "fixture_id": fixture["fixture_id"],
        "mode": "answer_nll_and_exact_prefix",
        "arm": "D",
        "kind": fixture["kind"],
        "document_source_ordinal": fixture["document"]["source_ordinal"],
        "prompt_token_count": 4099,
        "selected_suffix_tokens": 2048,
        "input_token_ids_sha256_raw_i32le": fixture["tokens"]["sha256_raw_i32le"],
        "ordering_key_sha256": ordering_sha256,
    }
    for key, expected in expected_scalars.items():
        if operation[key] != expected:
            raise RuntimeError(f"{path}.{key}")
    answer = require_dict(operation["answer"], f"{path}.answer")
    require_exact_keys(
        answer,
        {
            "expected_token_ids",
            "tokens",
            "nll_sum_f64",
            "mean_nll_f64",
            "scored_rows",
            "greedy_prefix_through_first_mismatch_or_stop",
            "greedy_prefix_contract",
            "exact_pass",
        },
        f"{path}.answer",
    )
    expected_tokens = [int(value) for value in fixture["answer_token_ids"]]
    if answer["expected_token_ids"] != expected_tokens:
        raise RuntimeError(f"{path}.answer.expected_token_ids")
    if require_int(answer["tokens"], f"{path}.answer.tokens") != len(expected_tokens):
        raise RuntimeError(f"{path}.answer.tokens")
    rows = require_list(answer["scored_rows"], f"{path}.answer.scored_rows")
    if len(rows) != len(expected_tokens):
        raise RuntimeError(f"{path}.answer.scored_rows count")
    nll_values = []
    predictions = []
    for ordinal, (row, target) in enumerate(zip(rows, expected_tokens, strict=True)):
        nll, _, prediction = validate_score_row(
            row,
            ordinal,
            target,
            f"{path}.answer.scored_rows[{ordinal}]",
        )
        nll_values.append(nll)
        predictions.append(prediction)
    nll_sum = sum(nll_values)
    require_close(
        require_number(answer["nll_sum_f64"], f"{path}.answer.nll_sum_f64"),
        nll_sum,
        f"{path}.answer.nll_sum_f64",
    )
    require_close(
        require_number(answer["mean_nll_f64"], f"{path}.answer.mean_nll_f64"),
        nll_sum / len(expected_tokens),
        f"{path}.answer.mean_nll_f64",
    )
    contract = (
        "true free-running prefix; after a mismatch, remaining expected answer "
        "tokens are teacher-forced only"
    )
    if answer["greedy_prefix_contract"] != contract:
        raise RuntimeError(f"{path}.answer.greedy_prefix_contract")
    prefix = require_list(
        answer["greedy_prefix_through_first_mismatch_or_stop"],
        f"{path}.answer.greedy_prefix_through_first_mismatch_or_stop",
    )
    if not all(
        not isinstance(token, bool)
        and isinstance(token, int)
        and 0 <= token < VOCAB_SIZE
        for token in prefix
    ):
        raise RuntimeError(f"{path}.answer greedy prefix token range")
    expected_prefix: list[int] = []
    all_answers_match = True
    for prediction, target in zip(predictions, expected_tokens, strict=True):
        if all_answers_match:
            expected_prefix.append(prediction)
            all_answers_match = prediction == target
    exact_pass = require_bool(answer["exact_pass"], f"{path}.answer.exact_pass")
    if all_answers_match:
        if len(prefix) != len(expected_tokens) + 1 or prefix[:-1] != expected_prefix:
            raise RuntimeError(f"{path}.answer greedy stop row")
        expected_exact = prefix[-1] in fixture["producer_stop_token_ids"]
    else:
        if prefix != expected_prefix:
            raise RuntimeError(f"{path}.answer greedy mismatch prefix")
        expected_exact = False
    if len(prefix) > int(fixture["max_generated_tokens"]):
        raise RuntimeError(f"{path}.answer greedy prefix length")
    if exact_pass != expected_exact:
        raise RuntimeError(f"{path}.answer.exact_pass")
    validate_logit_identity(
        operation["logit_identity"],
        len(expected_tokens) + 1,
        f"{path}.logit_identity",
    )
    validate_operation_binding(operation, path)


def validate_backends(value: object) -> None:
    backends = require_dict(value, "core.backends")
    require_exact_keys(
        backends,
        {
            "registration",
            "dynamic_backend_discovery_invoked",
            "registries",
            "devices",
        },
        "core.backends",
    )
    if backends["registration"] != (
        "link-time CPU and Metal registry populated before llama_backend_init"
    ):
        raise RuntimeError("core.backends.registration")
    if require_bool(
        backends["dynamic_backend_discovery_invoked"],
        "core.backends.dynamic_backend_discovery_invoked",
    ):
        raise RuntimeError("core invoked dynamic backend discovery")
    registries = require_list(backends["registries"], "core.backends.registries")
    if len(registries) != 2:
        raise RuntimeError("core.backends.registries count")
    registry_counts: dict[str, int] = {}
    for ordinal, value in enumerate(registries):
        registry = require_dict(value, f"core.backends.registries[{ordinal}]")
        require_exact_keys(
            registry,
            {"ordinal", "name", "device_count"},
            f"core.backends.registries[{ordinal}]",
        )
        if require_int(registry["ordinal"], "backend registry ordinal") != ordinal:
            raise RuntimeError("backend registry ordinal")
        name = require_string(registry["name"], "backend registry name")
        count = require_int(registry["device_count"], "backend registry device count")
        if count <= 0:
            raise RuntimeError("backend registry has no devices")
        registry_counts[name] = count
    if list(registry_counts) != ["MTL", "CPU"]:
        raise RuntimeError("core backend registry set or order")
    devices = require_list(backends["devices"], "core.backends.devices")
    observed_counts = {"MTL": 0, "CPU": 0}
    has_cpu = False
    has_metal_accelerator = False
    for ordinal, value in enumerate(devices):
        device = require_dict(value, f"core.backends.devices[{ordinal}]")
        require_exact_keys(
            device,
            {
                "ordinal",
                "registry",
                "name",
                "description",
                "device_id",
                "type",
                "memory_free",
                "memory_total",
                "capabilities",
            },
            f"core.backends.devices[{ordinal}]",
        )
        if require_int(device["ordinal"], "backend device ordinal") != ordinal:
            raise RuntimeError("backend device ordinal")
        registry = require_string(device["registry"], "backend device registry")
        if registry not in observed_counts:
            raise RuntimeError("backend device registry")
        observed_counts[registry] += 1
        if not require_string(device["name"], "backend device name"):
            raise RuntimeError("empty backend device name")
        require_string(device["description"], "backend device description")
        if device["device_id"] is not None:
            require_string(device["device_id"], "backend device ID")
        device_type = require_string(device["type"], "backend device type")
        if device_type not in {"CPU", "GPU", "IGPU", "ACCEL", "META"}:
            raise RuntimeError("backend device type")
        memory_free = require_int(device["memory_free"], "backend memory free")
        memory_total = require_int(device["memory_total"], "backend memory total")
        if memory_free < 0 or memory_total <= 0 or memory_free > memory_total:
            raise RuntimeError("backend memory accounting")
        capabilities = require_dict(device["capabilities"], "backend capabilities")
        require_exact_keys(
            capabilities,
            {"async", "host_buffer", "buffer_from_host_ptr", "events", "mmap_support"},
            "backend capabilities",
        )
        for capability in capabilities.values():
            require_bool(capability, "backend capability")
        has_cpu = has_cpu or (registry == "CPU" and device_type == "CPU")
        has_metal_accelerator = has_metal_accelerator or (
            registry == "MTL" and device_type in {"GPU", "IGPU"}
        )
    if observed_counts != registry_counts or not has_cpu or not has_metal_accelerator:
        raise RuntimeError("static backend device inventory")


def validate_core_report(
    core: object,
    manifest: dict[str, object],
    tokens: dict[str, list[int]],
    fixtures: dict[str, dict[str, object]],
    expected_operations: list[tuple[str, str, str]],
    build_evidence: dict[str, object],
    canonical_model_shards: list[Path],
) -> dict[str, object]:
    report = require_dict(core, "core")
    require_exact_keys(
        report,
        {
            "schema",
            "schema_version",
            "packet_id",
            "fixture_manifest_sha256",
            "runner_build",
            "llama_cpp",
            "backends",
            "model",
            "context",
            "scoring",
            "operations",
            "binding",
        },
        "core",
    )
    if (
        report["schema"] != "qwen4exp-selected-quality-llama-core"
        or report["schema_version"] != 1
        or report["packet_id"] != PACKET_ID
        or report["fixture_manifest_sha256"] != MANIFEST_SHA256
    ):
        raise RuntimeError("llama.cpp core evidence identity")

    build = require_dict(report["runner_build"], "core.runner_build")
    require_exact_keys(
        build,
        {
            "main_cpp_sha256",
            "cmake_lists_sha256",
            "source_manifest_schema",
            "source_manifest_sha256",
            "llama_cpp_tree",
            "llama_cpp_export_manifest_sha256",
            "build_policy_sha256",
            "compiler_id",
            "compiler_version",
            "build_type",
            "cxx_flags_sha256",
        },
        "core.runner_build",
    )
    build_expected = build_evidence["core_build_expected"]
    for key in (
        "main_cpp_sha256",
        "cmake_lists_sha256",
        "source_manifest_schema",
        "source_manifest_sha256",
        "llama_cpp_tree",
        "llama_cpp_export_manifest_sha256",
        "build_policy_sha256",
        "build_type",
    ):
        if build[key] != build_expected[key]:
            raise RuntimeError(f"stale core build attestation: {key}")
    if not require_string(build["compiler_id"], "core compiler ID"):
        raise RuntimeError("empty core compiler ID")
    if not require_string(build["compiler_version"], "core compiler version"):
        raise RuntimeError("empty core compiler version")
    if not is_lower_hex(build["cxx_flags_sha256"]):
        raise RuntimeError("core C++ flags hash")

    llama_cpp = require_dict(report["llama_cpp"], "core.llama_cpp")
    require_exact_keys(llama_cpp, {"commit", "version"}, "core.llama_cpp")
    if llama_cpp["commit"] != LLAMA_CPP_COMMIT:
        raise RuntimeError("core llama.cpp commit")
    if not require_string(llama_cpp["version"], "core llama.cpp version"):
        raise RuntimeError("empty core llama.cpp version")
    validate_backends(report["backends"])

    model = require_dict(report["model"], "core.model")
    require_exact_keys(
        model,
        {
            "split_paths",
            "split_count",
            "description",
            "bytes",
            "parameters",
            "vocab_size",
            "architecture",
            "tokenizer_model",
            "tokenizer_pre",
            "producer_stop_token_id",
            "producer_stop_token_is_eog",
            "ftype",
        },
        "core.model",
    )
    expected_model = {
        "split_paths": [str(path) for path in canonical_model_shards],
        "split_count": len(canonical_model_shards),
        "vocab_size": VOCAB_SIZE,
        "architecture": "qwen4exp",
        "tokenizer_model": "gpt2",
        "tokenizer_pre": "qwen35",
        "producer_stop_token_id": PRODUCER_STOP_TOKEN_ID,
        "producer_stop_token_is_eog": True,
    }
    for key, expected in expected_model.items():
        if model[key] != expected:
            raise RuntimeError(f"core.model.{key}")
    if not require_string(model["description"], "core.model.description"):
        raise RuntimeError("empty model description")
    if require_int(model["bytes"], "core.model.bytes") <= 0:
        raise RuntimeError("core.model.bytes")
    if require_int(model["parameters"], "core.model.parameters") <= 0:
        raise RuntimeError("core.model.parameters")
    if require_int(model["ftype"], "core.model.ftype") < 0:
        raise RuntimeError("core.model.ftype")

    context = require_dict(report["context"], "core.context")
    require_exact_keys(
        context,
        {
            "n_ctx",
            "n_batch",
            "n_ubatch",
            "n_seq_max",
            "flash_attention",
            "gpu_layers",
            "memory_cleared_with_data_before_each_operation",
        },
        "core.context",
    )
    expected_context = {
        "n_ctx": 4224,
        "n_batch": 512,
        "n_ubatch": 512,
        "n_seq_max": 1,
        "flash_attention": "enabled",
        "gpu_layers": "all",
        "memory_cleared_with_data_before_each_operation": True,
    }
    for key, expected in expected_context.items():
        if context[key] != expected:
            raise RuntimeError(f"core.context.{key}")

    scoring = require_dict(report["scoring"], "core.scoring")
    require_exact_keys(
        scoring,
        {
            "vocab_size",
            "logsumexp",
            "argmax_tie_policy",
            "natural_forwards",
            "retrieval_exact",
        },
        "core.scoring",
    )
    expected_scoring = {
        "vocab_size": VOCAB_SIZE,
        "logsumexp": "max-subtracted F64 over every finite F32 logit",
        "argmax_tie_policy": "lowest token ID",
        "natural_forwards": (
            "score each of 96 current rows, feed target once, retain one unscored "
            "terminal row"
        ),
        "retrieval_exact": (
            "answer from generated token zero followed immediately by producer stop"
        ),
    }
    if scoring != expected_scoring:
        raise RuntimeError("core.scoring")

    operations = require_list(report["operations"], "core.operations")
    if len(operations) != 20 or len(expected_operations) != 20:
        raise RuntimeError("core operation count")
    for index, (ordering_sha256, fixture_id, mode) in enumerate(expected_operations):
        operation = require_dict(operations[index], f"core.operations[{index}]")
        fixture = fixtures[fixture_id]
        ordinal = LOCAL_OPERATION_COUNT + index
        if mode == "teacher_forced_nll_96":
            validate_natural_operation(
                operation,
                fixture,
                tokens[fixture_id],
                ordinal,
                ordering_sha256,
                f"core.operations[{index}]",
            )
        elif mode == "answer_nll_and_exact_prefix":
            validate_retrieval_operation(
                operation,
                fixture,
                ordinal,
                ordering_sha256,
                f"core.operations[{index}]",
            )
        else:
            raise RuntimeError(f"unexpected D mode {mode}")

    binding = require_dict(report["binding"], "core.binding")
    require_exact_keys(
        binding,
        {
            "schema",
            "semantic_payload_encoding",
            "semantic_payload_bytes",
            "semantic_payload_sha256",
            "semantic_payload_json_compact",
        },
        "core.binding",
    )
    if binding["schema"] != CORE_REPORT_BINDING_SCHEMA:
        raise RuntimeError("core.binding.schema")
    if binding["semantic_payload_encoding"] != CORE_OPERATION_BINDING_ENCODING:
        raise RuntimeError("core.binding.semantic_payload_encoding")
    semantic_sha256 = binding["semantic_payload_sha256"]
    if not is_lower_hex(semantic_sha256):
        raise RuntimeError("core.binding semantic payload hash syntax")
    payload = dict(report)
    del payload["binding"]
    compact = require_string(
        binding["semantic_payload_json_compact"],
        "core.binding.semantic_payload_json_compact",
    )
    encoded = compact.encode()
    if parse_json_strict(encoded) != payload:
        raise RuntimeError("core.binding semantic payload content")
    if require_int(
        binding["semantic_payload_bytes"], "core.binding.semantic_payload_bytes"
    ) != len(encoded):
        raise RuntimeError("core.binding semantic payload byte count")
    if semantic_sha256 != sha256(encoded):
        raise RuntimeError("core.binding semantic payload hash")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[2]
    output_parent = args.output.parent.resolve(strict=True)
    if not output_parent.is_dir():
        raise RuntimeError(f"output parent is not a directory: {output_parent}")
    args.output = output_parent / args.output.name
    fixtures_path = args.fixtures.resolve(strict=True)
    with OutputReservation(args.output) as reservation:
        environment = validate_environment()
        manifest, tokens, fixtures, expected_operations = validate_fixtures(
            fixtures_path
        )
        llama_source = args.llama_source.resolve(strict=True)
        source = validate_source(repository, llama_source)
        isolated_build: IsolatedCoreBuild | None = None
        staged_core: Path | None = None
        core_output: Path | None = None
        try:
            isolated_build = build_isolated_core(
                repository,
                llama_source,
                source,
                args.output.parent,
            )
            runtime_root = isolated_build.workspace / "runtime"
            runtime_root.mkdir(mode=0o700)
            runtime_home = runtime_root / "home"
            runtime_tmp = runtime_root / "tmp"
            runtime_home.mkdir(mode=0o700)
            runtime_tmp.mkdir(mode=0o700)
            core_output = runtime_root / "core-evidence.json"
            model, initial_stamps, canonical_model_shards = validate_model(
                args.model, manifest["acquisition_model_lock"]
            )
            staged_core, core_execution = stage_core_executable(
                isolated_build.core,
                runtime_root,
                isolated_build.report["core_executable"],
            )
            clean_environment = {
                "HOME": str(runtime_home),
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "TMPDIR": str(runtime_tmp),
            }
            try:
                subprocess.run(
                    [
                        str(staged_core),
                        "--model-shard",
                        str(canonical_model_shards[0]),
                        "--model-shard",
                        str(canonical_model_shards[1]),
                        "--model-shard",
                        str(canonical_model_shards[2]),
                        "--fixtures",
                        str(fixtures_path),
                        "--output",
                        str(core_output),
                    ],
                    check=True,
                    env=clean_environment,
                )
            finally:
                revalidate_staged_core(staged_core, core_execution)
            core_bytes = core_output.read_bytes()
            core = validate_core_report(
                parse_json_strict(core_bytes),
                manifest,
                tokens,
                fixtures,
                expected_operations,
                isolated_build.report,
                canonical_model_shards,
            )
            final_stamps = [
                file_stamp(Path(str(stamp["path"]))) for stamp in initial_stamps
            ]
            if final_stamps != initial_stamps:
                raise RuntimeError("model shards changed during llama.cpp acquisition")
            revalidate_isolated_build(isolated_build)
            final_source = validate_source(repository, llama_source)
            if final_source != source:
                raise RuntimeError(
                    "runner or source checkout changed during acquisition"
                )
            core_sha256 = sha256(core_bytes)
            evidence_domain = (
                "qwen4exp-selected-quality-llama-evidence-v1\0"
                f"packet_id={PACKET_ID}\n"
                f"fixture_manifest_sha256={MANIFEST_SHA256}\n"
                f"qwen_source_commit={source['qwen_source_commit']}\n"
                f"runner_source_manifest_sha256={source['runner_source_manifest_sha256']}\n"
                f"llama_cpp_tree={source['llama_cpp']['tree']}\n"
                f"llama_cpp_export_manifest_sha256={isolated_build.source_manifest_sha256}\n"
                f"build_policy_sha256={isolated_build.report['policy_sha256']}\n"
                f"core_executable_sha256={isolated_build.report['core_executable']['sha256']}\n"
                f"llama_cpp_commit={LLAMA_CPP_COMMIT}\n"
                f"model_shard_manifest_sha256={model['shard_manifest_sha256']}\n"
                f"core_output_sha256={core_sha256}\n"
            )
            report = {
                "schema": "qwen4exp-selected-quality-llama-evidence",
                "schema_version": 1,
                "packet_id": PACKET_ID,
                "status": "llama_cpp_d_acquired_unanalyzed",
                "disposition": None,
                "fixture_manifest": {
                    "path": str(fixtures_path),
                    "bytes": fixtures_path.stat().st_size,
                    "sha256": MANIFEST_SHA256,
                },
                "environment": environment,
                "source": source,
                "isolated_build": isolated_build.report,
                "core_execution": core_execution,
                "model": model,
                "host": {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                },
                "evidence_domain_utf8": evidence_domain,
                "evidence_binding_sha256": sha256(evidence_domain.encode()),
                "core_output": {
                    "bytes": len(core_bytes),
                    "sha256": core_sha256,
                    "report": core,
                },
            }
            report_bytes = canonical_json(report)
            report_sha256 = reservation.publish(report_bytes)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "bytes": len(report_bytes),
                        "sha256": report_sha256,
                    },
                    sort_keys=True,
                )
            )
        finally:
            if core_output is not None:
                core_output.unlink(missing_ok=True)
            if staged_core is not None:
                staged_core.unlink(missing_ok=True)
            if isolated_build is not None:
                cleanup_isolated_build(isolated_build)
            sync_directory(args.output.parent)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
