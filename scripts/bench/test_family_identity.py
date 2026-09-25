from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import family


COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
STATE_A = f"{family.SOURCE_STATE_PREFIX}{'a' * 64}"
STATE_B = f"{family.SOURCE_STATE_PREFIX}{'b' * 64}"


def identity(*, commit: str = COMMIT_A, dirty: bool = False) -> dict:
    return {
        "schema_version": 2,
        "build_commit": commit,
        "build_commit_short": commit[:9],
        "build_dirty": dirty,
        "build_source_state": STATE_A,
        "stamp_source": "git",
        "stamp_error": None,
        "runtime_commit": commit,
        "runtime_dirty": dirty,
        "runtime_source_state": STATE_A,
        "status": "dirty" if dirty else "match",
        "problems": ["dirty"] if dirty else [],
        "overrides": [],
    }


class FamilyIdentityTests(unittest.TestCase):
    def assert_rejected(self, call) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            call()

    def test_clean_matching_identity_passes(self) -> None:
        family.validate_qwen_identity(
            identity(), COMMIT_A, False, STATE_A, allow_dirty=False
        )

    def test_binary_source_mismatch_is_rejected(self) -> None:
        self.assert_rejected(
            lambda: family.validate_qwen_identity(
                identity(commit=COMMIT_B),
                COMMIT_A,
                False,
                STATE_A,
                allow_dirty=False,
            )
        )

    def test_dirty_identity_requires_override(self) -> None:
        packet = identity(dirty=True)
        self.assert_rejected(
            lambda: family.validate_qwen_identity(
                packet, COMMIT_A, True, STATE_A, allow_dirty=False
            )
        )
        family.validate_qwen_identity(packet, COMMIT_A, True, STATE_A, allow_dirty=True)

    def test_dirty_observations_must_agree(self) -> None:
        self.assert_rejected(
            lambda: family.validate_qwen_identity(
                identity(dirty=True), COMMIT_A, False, STATE_A, allow_dirty=True
            )
        )

        stale_build = identity(dirty=True)
        stale_build["build_dirty"] = False
        self.assert_rejected(
            lambda: family.validate_qwen_identity(
                stale_build, COMMIT_A, True, STATE_A, allow_dirty=True
            )
        )

    def test_unknown_dirty_observation_is_rejected(self) -> None:
        packet = identity()
        packet["runtime_dirty"] = None
        self.assert_rejected(
            lambda: family.validate_qwen_identity(
                packet, COMMIT_A, False, STATE_A, allow_dirty=True
            )
        )

    def test_source_state_mismatch_is_rejected(self) -> None:
        self.assert_rejected(
            lambda: family.validate_qwen_identity(
                identity(), COMMIT_A, False, STATE_B, allow_dirty=True
            )
        )

    def test_row_identity_drift_is_rejected(self) -> None:
        expected = identity()
        changed = identity(commit=COMMIT_B)
        self.assert_rejected(
            lambda: family.validate_qwen_rows(
                [{"test": "pp1", "build_identity": changed}],
                expected,
                allow_dirty=False,
                expected_tests={"pp1"},
            )
        )

    def test_row_requires_exact_override_and_aliases(self) -> None:
        expected = identity(dirty=True)
        row_identity = identity(dirty=True)
        row_identity["overrides"] = ["allow_dirty"]
        row = {
            "schema_version": 3,
            "engine": "qwen-llm",
            "build_commit": COMMIT_A[:9],
            "build_dirty": 1,
            "test": "pp1",
            "build_identity": row_identity,
        }
        family.validate_qwen_rows(
            [row], expected, allow_dirty=True, expected_tests={"pp1"}
        )

        row_identity["overrides"] = []
        self.assert_rejected(
            lambda: family.validate_qwen_rows(
                [row], expected, allow_dirty=True, expected_tests={"pp1"}
            )
        )

    def test_rows_require_complete_identity_and_shape_set(self) -> None:
        expected = identity()
        row_identity = identity()
        row = {
            "schema_version": 3,
            "engine": "qwen-llm",
            "build_commit": COMMIT_A[:9],
            "build_dirty": 0,
            "test": "pp1",
            "build_identity": row_identity,
        }
        del row_identity["stamp_error"]
        self.assert_rejected(
            lambda: family.validate_qwen_rows(
                [row], expected, allow_dirty=False, expected_tests={"pp1"}
            )
        )
        self.assert_rejected(
            lambda: family.validate_qwen_rows(
                [], expected, allow_dirty=False, expected_tests={"pp1"}
            )
        )

    def test_tracked_source_state_distinguishes_dirty_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            subprocess.run(["git", "init", "-q", repo], check=True)
            tracked = repo / "tracked.txt"
            tracked.write_text("clean\n")
            subprocess.run(["git", "-C", repo, "add", "tracked.txt"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    repo,
                    "-c",
                    "user.name=Qwen Test",
                    "-c",
                    "user.email=qwen-test@example.invalid",
                    "commit",
                    "-qm",
                    "initial",
                ],
                check=True,
            )
            clean = family.tracked_source_state(repo)
            tracked.write_text("dirty-a\n")
            dirty_a = family.tracked_source_state(repo)
            tracked.write_text("dirty-b\n")
            dirty_b = family.tracked_source_state(repo)
            subprocess.run(["git", "-C", repo, "add", "tracked.txt"], check=True)
            staged_b = family.tracked_source_state(repo)

            self.assertEqual(len({clean, dirty_a, dirty_b, staged_b}), 4)

            subprocess.run(
                [
                    "git",
                    "-C",
                    repo,
                    "-c",
                    "user.name=Qwen Test",
                    "-c",
                    "user.email=qwen-test@example.invalid",
                    "commit",
                    "-qm",
                    "tracked update",
                ],
                check=True,
            )
            committed = family.tracked_source_state(repo)
            self.assertFalse(family.source_identity(repo)[1])

            # Untracked files (bench output, work-in-progress docs) change
            # neither the source state nor the dirty flag, matching the
            # binary's tracked-only identity (32bacc9a, 4bf0482).
            untracked = repo / "untracked.rs"
            untracked.write_text("fn untracked() {}\n")
            self.assertEqual(family.tracked_source_state(repo), committed)
            self.assertFalse(family.source_identity(repo)[1])
            untracked.unlink()

            subprocess.run(
                [
                    "git",
                    "-C",
                    repo,
                    "update-index",
                    "--assume-unchanged",
                    "tracked.txt",
                ],
                check=True,
            )
            tracked.write_text("hidden-change\n")
            hidden = family.tracked_source_state(repo)
            self.assertNotEqual(hidden, committed)
            self.assertTrue(family.source_identity(repo)[1])


class MergeBlocksTests(unittest.TestCase):
    def row(self, samples: list[float], **extra) -> dict:
        return {
            "test": "tg128",
            "samples_ts": samples,
            "samples_ns": [int(1e9 / s) for s in samples],
            "avg_ts": sum(samples) / len(samples),
            "stddev_ns": 0,
            "avg_session_alloc_ns": extra.pop("alloc", 100),
            "prefill_mode": extra.pop("mode", "packed"),
            **extra,
        }

    def merge(self, *blocks: dict) -> dict:
        with contextlib.redirect_stderr(io.StringIO()):
            (merged,) = family.merge_blocks(
                [[b] for b in blocks], key=lambda r: r["test"]
            )
        return merged

    def test_statistics_cover_every_block(self) -> None:
        merged = self.merge(
            self.row([10.0, 12.0], alloc=100), self.row([14.0], alloc=400)
        )
        self.assertEqual(merged["samples_ts"], [10.0, 12.0, 14.0])
        self.assertAlmostEqual(merged["avg_ts"], 12.0)
        self.assertAlmostEqual(merged["stddev_ts"], 2.0)
        self.assertGreater(merged["stddev_ns"], 0)
        self.assertEqual(merged["n_repetitions"], 3)
        self.assertEqual(merged["n_blocks"], 2)
        self.assertEqual(merged["block_avg_ts"], [11.0, 14.0])
        # Repetition-weighted: (100*2 + 400*1) / 3.
        self.assertEqual(merged["avg_session_alloc_ns"], 200)
        self.assertNotIn("heterogeneous_fields", merged)

    def test_blocks_that_executed_differently_are_flagged(self) -> None:
        merged = self.merge(
            self.row([10.0], mode="packed"), self.row([9.0], mode="scalar")
        )
        self.assertEqual(
            merged["heterogeneous_fields"], {"prefill_mode": ['"packed"', '"scalar"']}
        )


class BuildScriptIntegrationTests(unittest.TestCase):
    def git(self, repo: Path, *args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", repo, *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()

    def commit(self, repo: Path, message: str, *, empty: bool = False) -> None:
        args = [
            "-c",
            "user.name=Qwen Test",
            "-c",
            "user.email=qwen-test@example.invalid",
            "commit",
            "-qm",
            message,
        ]
        if empty:
            args.append("--allow-empty")
        self.git(repo, *args)

    def run_fixture(
        self, repo: Path, target: Path, overrides: dict[str, str] | None = None
    ) -> tuple[str, bool, str, str, str]:
        env = os.environ.copy()
        for name in (
            "QWEN_BUILD_COMMIT",
            "QWEN_BUILD_DIRTY",
            "QWEN_BUILD_SOURCE_STATE",
        ):
            env.pop(name, None)
        env.update(overrides or {})
        env["CARGO_TARGET_DIR"] = str(target)
        proc = subprocess.run(
            ["cargo", "run", "--quiet"],
            cwd=repo,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        commit, dirty, state, source, error = proc.stdout.strip().splitlines()
        return commit, dirty == "1", state, source, error

    def test_build_script_tracks_worktree_and_git_topology(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            repo.mkdir()
            self.git(repo, "init", "-q")
            (repo / "src").mkdir()
            (repo / "Cargo.toml").write_text(
                "[package]\n"
                'name = "identity-fixture"\n'
                'version = "0.0.0"\n'
                'edition = "2024"\n\n'
                "[build-dependencies]\n"
                'sha2 = "0.10"\n'
            )
            shutil.copy2(family.ROOT / "crates/qwen-cli/build.rs", repo / "build.rs")
            shutil.copy2(
                family.ROOT / "crates/qwen-cli/source_identity.rs",
                repo / "source_identity.rs",
            )
            (repo / "src/main.rs").write_text(
                'fn main() { println!("{}\\n{}\\n{}\\n{}\\n{}", '
                'env!("QWEN_BUILD_COMMIT"), env!("QWEN_BUILD_DIRTY"), '
                'env!("QWEN_BUILD_SOURCE_STATE"), env!("QWEN_BUILD_STAMP_SOURCE"), '
                'env!("QWEN_BUILD_STAMP_ERROR")); }\n'
            )
            watched = repo / "watched.txt"
            watched.write_text("clean\n")
            subprocess.run(
                ["cargo", "generate-lockfile", "--quiet"], cwd=repo, check=True
            )
            self.git(repo, "add", ".")
            self.commit(repo, "initial")

            target = root / "target-main"
            clean_commit, clean_dirty, clean_state, clean_source, clean_error = (
                self.run_fixture(repo, target)
            )
            self.assertFalse(clean_dirty)
            self.assertEqual((clean_source, clean_error), ("git", "none"))

            verified = self.run_fixture(
                repo,
                target,
                {
                    "QWEN_BUILD_COMMIT": clean_commit,
                    "QWEN_BUILD_DIRTY": "0",
                    "QWEN_BUILD_SOURCE_STATE": clean_state,
                },
            )
            self.assertEqual(verified[3:], ("environment-verified", "none"))
            mismatched = self.run_fixture(
                repo,
                target,
                {
                    "QWEN_BUILD_COMMIT": clean_commit,
                    "QWEN_BUILD_DIRTY": "0",
                    "QWEN_BUILD_SOURCE_STATE": f"{family.SOURCE_STATE_PREFIX}{'f' * 64}",
                },
            )
            self.assertEqual(
                mismatched[3:],
                ("environment-mismatch", "environment_identity_mismatch"),
            )

            watched.write_text("dirty-a\n")
            _, dirty_a, state_a, _, _ = self.run_fixture(repo, target)
            watched.write_text("dirty-b\n")
            _, dirty_b, state_b, _, _ = self.run_fixture(repo, target)
            self.assertTrue(dirty_a and dirty_b)
            self.assertEqual(len({clean_state, state_a, state_b}), 3)

            self.git(repo, "add", "watched.txt")
            self.commit(repo, "tracked update")
            committed, committed_dirty, _, _, _ = self.run_fixture(repo, target)
            self.assertFalse(committed_dirty)
            self.assertNotEqual(committed, clean_commit)

            head_pointer = (repo / ".git/HEAD").read_text()
            self.commit(repo, "symbolic advance", empty=True)
            symbolic_commit, _, _, _, _ = self.run_fixture(repo, target)
            self.assertEqual((repo / ".git/HEAD").read_text(), head_pointer)
            self.assertNotEqual(symbolic_commit, committed)

            self.git(repo, "pack-refs", "--all", "--prune")
            packed_commit, _, _, _, _ = self.run_fixture(repo, target)
            self.commit(repo, "packed to loose", empty=True)
            loose_commit, _, _, _, _ = self.run_fixture(repo, target)
            self.assertNotEqual(loose_commit, packed_commit)

            self.git(repo, "checkout", "--detach", "-q")
            detached_commit, _, _, _, _ = self.run_fixture(repo, target)
            self.commit(repo, "detached advance", empty=True)
            detached_advance, _, _, _, _ = self.run_fixture(repo, target)
            self.assertNotEqual(detached_advance, detached_commit)

            linked = root / "linked"
            self.git(
                repo,
                "worktree",
                "add",
                "-b",
                "identity-linked",
                str(linked),
                "HEAD",
            )
            linked_target = root / "target-linked"
            linked_commit, linked_dirty, linked_state, _, _ = self.run_fixture(
                linked, linked_target
            )
            self.assertFalse(linked_dirty)
            self.assertEqual(linked_commit, detached_advance)
            self.assertRegex(
                linked_state,
                rf"^{family.SOURCE_STATE_PREFIX}[0-9a-f]{{64}}$",
            )
            self.commit(linked, "linked advance", empty=True)
            linked_advance, _, _, _, _ = self.run_fixture(linked, linked_target)
            self.assertNotEqual(linked_advance, linked_commit)


if __name__ == "__main__":
    unittest.main()
