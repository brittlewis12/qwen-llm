"""CPU-only analyzer checks: uv run --with numpy python -m unittest discover
-s scripts/reference/glm53 -p test_latent_quality_analysis.py
"""

import copy
import hashlib
import json
import unittest

import latent_quality_analysis as analysis


class SubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_bytes = (analysis.HERE / "quality-v1.json").read_bytes()
        cls.fixture = json.loads(cls.fixture_bytes)

    def packet(self):
        docs = analysis.selected_documents(self.fixture)
        rows = [{"event": "header", "schema": "glm53.latent_quality.v1",
                 "fixture_sha256": hashlib.sha256(self.fixture_bytes).hexdigest(),
                 "arms": list(analysis.ARMS), "tasks": list(analysis.TASKS),
                 "continuation": 64, "tool_cap": 256, "seeds": [11],
                 "workspace_rows": 2048, "lineage": "Fast", "precision": "Half",
                 "documents": [{"path": d["path"], "stratum": d["stratum"], "prefix": n} for d, n in docs]}]
        for d, n in docs:
            target = list(map(int, d["tokens"].split()))[n:n + 64]
            for arm in analysis.ARMS:
                loss, hit = (1.0, False) if arm == "half_scalar" else (1.125, True)
                common = {"path": d["path"], "prefix": n, "arm": arm}
                rows.extend({"event": "document_token", **common, "index": i,
                             "token": t, "nll": loss, "hit": hit} for i, t in enumerate(target))
                rows.append({"event": "document", **common, "stratum": d["stratum"],
                             "continuation_tokens": target, "nll": [loss] * 64, "hits": [hit] * 64})
        for task in (t for t in self.fixture["tool_tasks"] if t["id"] in analysis.TASKS):
            expected = task["expected"]
            args = {k: v for k, v in expected.items() if k != "name"}
            if task["id"] == "T03":
                args["days"] = 3
            for mode, seed in (("greedy", 0), ("sampled", 11)):
                for arm in analysis.ARMS:
                    common = {"id": task["id"], "arm": arm, "generation": {"mode": mode, "seed": seed, "cap": 256,
                              "sampler": "qwen_llm::sampling v1", "top_k": 0, "min_p": 0,
                              "temperature": 0 if mode == "greedy" else 1, "top_p": 1 if mode == "greedy" else 0.949999988079071}}
                    rows.extend({"event": "task_token", **common, "index": i, "token": t}
                                for i, t in enumerate([42, 154829]))
                    rows.append({"event": "task", **common, "expected": expected,
                                 "emitted": 1, "emitted_ids": [42], "stop": 154829, "text": arm,
                                 "calls": [{"name": expected["name"], "arguments": args}], "parse_error": None})
        rows.append({"event": "complete", "documents_per_arm": 3, "task_generations_per_arm": 8})
        return rows

    def analyze(self, rows):
        return analysis.analyze(self.encode(rows), self.fixture_bytes)

    @staticmethod
    def encode(rows):
        return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()

    def test_complete_paired_summary_and_binding(self):
        rows = self.packet()
        result = self.analyze(rows)
        self.assertEqual(result["overall"], {"mean_nll_delta": .125, "hit_delta": 192, "hit_rate_delta": 1.0})
        self.assertEqual(result["report_sha256"], hashlib.sha256(self.encode(rows)).hexdigest())
        self.assertEqual(result["fixture_sha256"], hashlib.sha256(self.fixture_bytes).hexdigest())
        self.assertTrue(all(r["text_differs"] and not r["tokens_differ"] for r in result["tools"]))
        self.assertEqual(result["tool_counts"]["greedy"], {"half_scalar": 4, "half_mma": 4})

    def test_invalid_evidence_rejected(self):
        original = self.packet()
        mutations = {
            "incomplete": lambda r: r.pop(),
            "extra boundary": lambda r: r.insert(1, r[0]),
            "duplicate result": lambda r: r.insert(-1, next(x for x in r if x["event"] == "document")),
            "missing result": lambda r: r.remove(next(x for x in r if x["event"] == "task")),
            "wrong fixture": lambda r: r[0].update(fixture_sha256="0" * 64),
            "wrong arm": lambda r: r[1].update(arm="half_other"),
            "wrong selection": lambda r: r[0]["documents"][0].update(prefix=512),
            "wrong token": lambda r: r[1].update(token=42),
            "token order": lambda r: r[1].update(index=1),
            "nonfinite token": lambda r: r[1].update(nll=float("inf")),
            "nonfinite summary": lambda r: next(x for x in r if x["event"] == "document")["nll"].__setitem__(0, float("nan")),
            "boolean loss": lambda r: next(x for x in r if x["event"] == "document")["nll"].__setitem__(0, True),
            "wrong seed": lambda r: next(x for x in r if x["event"] == "task")["generation"].update(seed=12),
            "wrong sampling preset": lambda r: next(x for x in r if x["event"] == "task")["generation"].update(temperature=2),
            "premature truncation": lambda r: next(x for x in r if x["event"] == "task").update(stop=None),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                rows = copy.deepcopy(original)
                mutate(rows)
                with self.assertRaises(ValueError):
                    self.analyze(rows)

    def test_truncation_is_reported_and_scored_wrong(self):
        rows = self.packet()
        match = lambda r: r.get("id") == "T03" and r.get("arm") == "half_mma" and r.get("generation", {}).get("mode") == "greedy"
        rows = [r for r in rows if not (match(r) and r["event"] == "task_token")]
        row = next(r for r in rows if match(r) and r["event"] == "task")
        row.update(emitted=256, emitted_ids=[42] * 256, stop=None)
        i = rows.index(row)
        rows[i:i] = [{"event": "task_token", "id": row["id"], "arm": row["arm"],
                      "generation": row["generation"], "index": j, "token": 42} for j in range(256)]
        result = self.analyze(rows)["tools"][0]
        self.assertTrue(result["arms"]["half_mma"]["truncated"])
        self.assertEqual(result["correct_delta"], -1)

    def test_fixed_document_selection(self):
        fixture = json.loads((analysis.HERE / "quality-v1.json").read_bytes())
        selected = analysis.selected_documents(fixture)
        self.assertEqual([(d["path"], n) for d, n in selected], [
            ("docs/GLM53-FLASH-PLAN.md", 2048),
            ("crates/qwen-llm/src/sampling.rs", 2048),
            ("docs/SERVE.md", 4352),
        ])
        self.assertTrue(all(len(d["tokens"].split()) >= n + 64 for d, n in selected))

    def test_uses_existing_strict_scorer(self):
        import quality_analysis
        self.assertIs(analysis.tool_correct, quality_analysis.tool_correct)
        expected = {"name": "get_weather", "city": "Lisbon"}
        row = {"stop": 154829, "calls": [
            {"name": "get_weather", "arguments": {"city": "Lisbon", "days": 3}}
        ]}
        self.assertTrue(analysis.tool_correct(row, expected, "T03"))
        row["calls"][0]["arguments"]["days"] = True
        self.assertFalse(analysis.tool_correct(row, expected, "T03"))
        row["calls"][0]["arguments"]["days"] = 3
        row["calls"] *= 2
        self.assertFalse(analysis.tool_correct(row, expected, "T03"))

    def test_currency_and_truncated_turn_remain_strict(self):
        expected = {"name": "convert_currency", "amount": 75, "from": "EUR", "to": "CAD"}
        row = {"stop": 154829, "calls": [{"name": "convert_currency", "arguments": {
            "amount": 75, "from": "euros", "to": "Canadian dollars",
        }}]}
        self.assertTrue(analysis.tool_correct(row, expected, "T10"))
        for invalid in ("dollars", "USD"):
            row["calls"][0]["arguments"]["to"] = invalid
            self.assertFalse(analysis.tool_correct(row, expected, "T10"))
        row["calls"][0]["arguments"]["to"] = "CAD"
        row["stop"] = None
        self.assertFalse(analysis.tool_correct(row, expected, "T10"))


if __name__ == "__main__":
    unittest.main()
