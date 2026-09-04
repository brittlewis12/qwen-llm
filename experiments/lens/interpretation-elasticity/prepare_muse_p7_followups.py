import json
from pathlib import Path


ROOT = Path(__file__).parent
COORDINATES = ROOT / "artifacts" / "coordinates" / "muse-p7-followup"
GENERATED = ROOT / "plans" / "generated" / "muse-p7-2026-09-04"
J_ARTIFACT = "/Users/tito/models/muse-glimmer/lenses/muse-glimmer-30b-j-lens-matched-v1"
R_ARTIFACT = (
    "/Volumes/wdblack/weights-archive/jacobian-lenses/"
    "Muse-Glimmer-30B-rlens-published-v1"
)
TARGET_LAYER = 50
KAPPAS = (0.25, 0.5, 0.75, 1.0)
PROBES = {
    "empathy-final": {
        "token_id": 106507,
        "token": " empathy",
        "layer": 40,
        "sites": [{"label": "final-assistant-prefill", "target": 76, "twin": 77}],
    },
    "burnout-distributed": {
        "token_id": 146738,
        "token": " burnout",
        "layer": 46,
        "sites": [
            {"label": "user-end", "target": 74, "twin": 75},
            {"label": "assistant-start", "target": 75, "twin": 76},
        ],
    },
}


def score(run: dict, lens: str, layer: int, position: int, token_id: int) -> float:
    cell = next(
        cell
        for cell in run["live_readouts"]
        if cell["lens"].lower() == lens
        and cell["source_layer"] == layer
        and cell["index"] == position
    )
    return next(row["score"] for row in cell["scores"] if row["token_id"] == token_id)


def plan(
    probe: dict, active_lens: str, sites: list[dict], multiplier: float | None
) -> dict:
    lenses = [
        {
            "kind": "published_full_transport",
            "id": lens,
            "artifact": J_ARTIFACT if lens == "j" else R_ARTIFACT,
            "token_ids": [probe["token_id"]],
            "allow_unvalidated_transfer": True,
        }
        for lens in ("j", "r")
    ]
    operations = []
    for site in sites:
        coefficient = 1.0 if multiplier is None else multiplier * site["ratio"]
        operations.append(
            {
                "id": f"attenuate-{site['label']}",
                "scope": {
                    "layers": {"kind": "values", "values": [probe["layer"]]},
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
                "row": {"kind": "token_id", "token_id": probe["token_id"]},
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
                        "values": [probe["layer"], TARGET_LAYER],
                    },
                    "prefill": {
                        "kind": "values",
                        "values": [site["target"] for site in sites],
                    },
                },
                "top_k": 1,
            }
            for lens in ("j", "r")
        ],
    }


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    target = json.loads((COORDINATES / "target.run.json").read_text())
    twin = json.loads((COORDINATES / "twin.run.json").read_text())
    arm_specs = [
        ("zero-a", 0.0),
        ("zero-b", 0.0),
        *((f"kappa-{kappa}", kappa) for kappa in KAPPAS),
        ("full-ablation", None),
    ]
    jobs = []
    measurements = {}
    for probe_id, probe in PROBES.items():
        measurements[probe_id] = {}
        for lens in ("j", "r"):
            sites = []
            for site in probe["sites"]:
                p_target = score(
                    target, lens, probe["layer"], site["target"], probe["token_id"]
                )
                p_twin = score(
                    twin, lens, probe["layer"], site["twin"], probe["token_id"]
                )
                if p_target <= 0 or p_target <= p_twin:
                    raise ValueError(f"inadmissible {probe_id} {lens} {site['label']}")
                sites.append(
                    {
                        **site,
                        "p_target": p_target,
                        "p_twin": p_twin,
                        "ratio": (p_target - p_twin) / p_target,
                    }
                )
            measurements[probe_id][lens] = sites
            plan_dir = GENERATED / probe_id / lens
            plan_dir.mkdir(parents=True, exist_ok=True)
            for index, (label, multiplier) in enumerate(arm_specs):
                plan_path = plan_dir / f"arm-{index:02}-{label}.json"
                plan_path.write_text(
                    json.dumps(plan(probe, lens, sites, multiplier), indent=2) + "\n"
                )
                jobs.append(
                    {
                        "probe": probe_id,
                        "lens": lens,
                        "index": index,
                        "label": label,
                        "kappa": multiplier,
                        "plan": str(plan_path),
                    }
                )

    manifest = {
        "schema": "interpretation_elasticity.muse_p7_followups",
        "schema_version": 1,
        "selection_status": "fixed_before_muse_p7_intervention_outputs",
        "model": "Muse-Glimmer-30B Q8",
        "mode": "high",
        "item": "p7",
        "measurements": measurements,
        "jobs": jobs,
    }
    output = GENERATED / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(jobs)} runs -> {output}")
    for probe, lenses in measurements.items():
        for lens, sites in lenses.items():
            print(
                probe,
                lens.upper(),
                [(site["label"], round(site["ratio"], 6)) for site in sites],
            )


if __name__ == "__main__":
    main()
