import json
from pathlib import Path


ROOT = Path(__file__).parent
COORDINATE_ROOT = ROOT / "artifacts" / "coordinates" / "qwen-overnight-batch-02"
GENERATED = ROOT / "plans" / "generated" / "p7-distributed-2026-09-04"
J_ARTIFACT = (
    "/Volumes/wdblack/weights-archive/workspace-lenses/qwen3.6-27b/j-lens-native-v1"
)
R_ARTIFACT = (
    "/Volumes/wdblack/weights-archive/workspace-lenses/qwen3.6-27b/r-lens-native-v1"
)
TOKEN_ID = 15191
TOKEN = " feelings"
LAYER = 49
TARGET_LAYER = 62
KAPPAS = (0.25, 0.5, 0.75, 1.0)
SITES = [
    {"label": "user-content-and", "target": 9, "twin": 10},
    {"label": "user-content-I", "target": 10, "twin": 11},
    {"label": "user-end", "target": 26, "twin": 27},
    {"label": "assistant-role", "target": 29, "twin": 30},
    {"label": "final-prefill", "target": 34, "twin": 35},
]


def load_runs() -> dict[str, dict]:
    manifest = json.loads((COORDINATE_ROOT / "manifest.json").read_text())
    return {
        row["id"]: json.loads((COORDINATE_ROOT / row["path"]).read_text())
        for row in manifest["runs"]
    }


def score(run: dict, lens: str, position: int) -> float:
    cell = next(
        cell
        for cell in run["live_readouts"]
        if cell["lens"].lower() == lens
        and cell["source_layer"] == LAYER
        and cell["phase"] == "prefill"
        and cell["index"] == position
    )
    return next(row["score"] for row in cell["scores"] if row["token_id"] == TOKEN_ID)


def plan(active_lens: str, site_rows: list[dict], multiplier: float | None) -> dict:
    lenses = [
        {
            "kind": "published_full_transport",
            "id": lens,
            "artifact": J_ARTIFACT if lens == "j" else R_ARTIFACT,
            "token_ids": [TOKEN_ID],
            "allow_unvalidated_transfer": True,
        }
        for lens in ("j", "r")
    ]
    operations = []
    for site in site_rows:
        coefficient = 1.0 if multiplier is None else multiplier * site["ratio"]
        operations.append(
            {
                "id": f"attenuate-{site['label']}",
                "scope": {
                    "layers": {"kind": "values", "values": [LAYER]},
                    "prefill": {"kind": "values", "values": [site["target"]]},
                },
                "action": {
                    "kind": "projection_ablate",
                    "direction": "stance",
                    "coefficient": coefficient,
                },
            }
        )
    return {
        "version": 2,
        "lenses": lenses,
        "directions": [
            {
                "id": "stance",
                "lens": active_lens,
                "row": {"kind": "token_id", "token_id": TOKEN_ID},
                "normalization": "unit_l2",
            }
        ],
        "operations": operations,
        "readouts": [
            {
                "id": f"{lens}-track-stance",
                "lens": lens,
                "scope": {
                    "layers": {
                        "kind": "values",
                        "values": [LAYER, TARGET_LAYER],
                    },
                    "prefill": {
                        "kind": "values",
                        "values": [site["target"] for site in site_rows],
                    },
                },
                "top_k": 1,
            }
            for lens in ("j", "r")
        ],
    }


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    target = runs["p7-target-no-thinking"]
    twin = runs["p7-twin-no-thinking"]
    jobs = []
    all_sites = {}
    arm_specs = [
        ("zero-a", 0.0),
        ("zero-b", 0.0),
        *((f"kappa-{kappa}", kappa) for kappa in KAPPAS),
        ("full-ablation", None),
    ]
    for lens in ("j", "r"):
        site_rows = []
        for site in SITES:
            p_target = score(target, lens, site["target"])
            p_twin = score(twin, lens, site["twin"])
            if p_target <= 0 or p_target <= p_twin:
                raise ValueError(f"inadmissible {lens} site {site['label']}")
            site_rows.append(
                {
                    **site,
                    "p_target": p_target,
                    "p_twin": p_twin,
                    "ratio": (p_target - p_twin) / p_target,
                }
            )
        all_sites[lens] = site_rows
        lens_dir = GENERATED / lens
        lens_dir.mkdir(exist_ok=True)
        for index, (label, multiplier) in enumerate(arm_specs):
            plan_path = lens_dir / f"arm-{index:02}-{label}.json"
            plan_path.write_text(
                json.dumps(plan(lens, site_rows, multiplier), indent=2) + "\n"
            )
            jobs.append(
                {
                    "lens": lens,
                    "index": index,
                    "label": label,
                    "kappa": multiplier,
                    "plan": str(plan_path),
                }
            )

    manifest = {
        "schema": "interpretation_elasticity.distributed_followup",
        "schema_version": 1,
        "selection_status": "fixed_before_distributed_intervention_outputs",
        "model": "Qwen3.6-27B Q8",
        "mode": "no-thinking",
        "item": "p7",
        "token_id": TOKEN_ID,
        "token": TOKEN,
        "layer": LAYER,
        "site_admission": "token_in_target_top25_under_both_j_and_r_at_every_site",
        "sites": all_sites,
        "jobs": jobs,
    }
    output = GENERATED / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(jobs)} runs -> {output}")
    for lens, sites in all_sites.items():
        print(
            lens.upper(), [(site["label"], round(site["ratio"], 6)) for site in sites]
        )


if __name__ == "__main__":
    main()
