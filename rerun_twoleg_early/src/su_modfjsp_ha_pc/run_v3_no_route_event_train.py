from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from su_modfjsp_ha_pc.objective_contract import read_objective_scales
from su_modfjsp_ha_pc.pgnd_v4_data import expected_context_names, sha256_file, validate_seed_role
from su_modfjsp_ha_pc.pgnd_v4_model import read_context_groups, train_v4_model
from su_modfjsp_ha_pc.pgnd_v5_proposer_ablation import ADVANTAGE_FEATURE_FIELDS, subset_v4_group

# v3 ablation (全部重跑规则_冻结v3 section 4): remove feature group (4), the route and event features of the manuscript
# (route length, edge and bottleneck counts, minimum capacity, maximum blockage probability, event-window overlap, waiting
# and travel time, active events, event-proximity pressures), in the raw, raw-base and raw-median blocks.
GROUP4 = ("candidate_route_edge_count", "candidate_route_length", "candidate_route_bottleneck_count", "candidate_route_capacity_min",
          "candidate_route_blockage_probability_max", "candidate_route_event_overlap_duration", "candidate_route_wait_time",
          "candidate_route_travel_time", "candidate_active_path_event_count", "candidate_active_agv_event_count",
          "candidate_active_machine_event_count", "candidate_machine_event_proximity_pressure",
          "candidate_path_event_proximity_pressure", "candidate_agv_event_proximity_pressure", "candidate_future_risk_pressure")


def no_route_event_feature_indexes():
    w = len(ADVANTAGE_FEATURE_FIELDS)
    drop = {ADVANTAGE_FEATURE_FIELDS.index(f) + b * w for b in range(3) for f in GROUP4}
    return tuple(i for i in range(3 * w + 15) if i not in drop)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument("--end-seed", type=int, default=119)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--groups-per-batch", type=int, default=64)
    parser.add_argument("--random-states", type=int, nargs="+", default=[320, 321, 322, 323, 324])
    parser.add_argument("--objective-scales", type=Path, required=True)
    args = parser.parse_args()
    objective_scales = read_objective_scales(args.objective_scales)
    objective_scales_sha256 = sha256_file(args.objective_scales)
    for seed in range(args.start_seed, args.end_seed + 1):
        validate_seed_role(seed, "train")

    groups = []
    context_hashes: dict[str, str] = {}
    for seed in range(args.start_seed, args.end_seed + 1):
        for name in expected_context_names(seed):
            path = args.context_dir / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            metadata_path = args.context_dir / f"{name}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("objective_scales_sha256") != objective_scales_sha256:
                raise RuntimeError(f"context objective-scale hash mismatch: {name}")
            context_hashes[name] = sha256_file(path)
            groups.extend(read_context_groups(path))
    groups = tuple(groups)
    label_manifest_sha256 = hashlib.sha256(
        json.dumps(context_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    indexes = no_route_event_feature_indexes()
    reduced_groups = tuple(subset_v4_group(group, indexes) for group in groups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "PGND-v3-no-route-event-ablation", "feature_indexes": list(indexes), "feature_width": len(indexes),
              "dropped_fields": list(GROUP4), "groups": len(groups), "epochs": args.epochs, "random_states": args.random_states,
              "objective_scales_sha256": objective_scales_sha256, "label_manifest_sha256": label_manifest_sha256,
              "context_only_members": []}
    for random_state in args.random_states:
        started = time.perf_counter()
        artifact = train_v4_model(
            reduced_groups,
            epochs=args.epochs,
            random_state=random_state,
            groups_per_batch=args.groups_per_batch,
            hidden_size=96,
        )
        path = args.output_dir / f"no_route_event_member_{random_state}.pt"
        torch.save(
            {
                "model_state": artifact.model.state_dict(),
                "scaler_mean": artifact.scaler.mean_,
                "scaler_scale": artifact.scaler.scale_,
                "feature_width": artifact.feature_width,
                "final_loss": artifact.final_loss,
                "random_state": artifact.random_state,
                "hidden_size": 96,
                "objective_scales_sha256": objective_scales_sha256,
                "objective_scale_source_sha256": objective_scales.source_sha256,
                "label_manifest_sha256": label_manifest_sha256,
            },
            path,
        )
        report["context_only_members"].append(
            {
                "random_state": random_state,
                "final_loss": artifact.final_loss,
                "wall_seconds": time.perf_counter() - started,
                "model_path": str(path),
            }
        )
        (args.output_dir / "training_report_no_route_event.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"member={random_state} loss={artifact.final_loss:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
