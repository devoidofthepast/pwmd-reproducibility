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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_report = {
        "protocol": "PGND-V4",
        "train_seeds": list(range(args.start_seed, args.end_seed + 1)),
        "groups": len(groups),
        "rows": sum(len(group.raw_advantage) for group in groups),
        "epochs": args.epochs,
        "groups_per_batch": args.groups_per_batch,
        "random_states": args.random_states,
        "cuda": torch.cuda.is_available(),
        "objective_names": list(objective_scales.values),
        "objective_scales_sha256": objective_scales_sha256,
        "objective_scale_source_sha256": objective_scales.source_sha256,
        "label_manifest_sha256": label_manifest_sha256,
        "members": [],
    }
    for random_state in args.random_states:
        started = time.perf_counter()
        artifact = train_v4_model(
            groups,
            epochs=args.epochs,
            random_state=random_state,
            groups_per_batch=args.groups_per_batch,
            hidden_size=96,
        )
        elapsed = time.perf_counter() - started
        model_path = args.output_dir / f"pgnd_v4_member_{random_state}.pt"
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
            model_path,
        )
        run_report["members"].append(
            {
                "random_state": random_state,
                "final_loss": artifact.final_loss,
                "wall_seconds": elapsed,
                "model_path": str(model_path),
            }
        )
        (args.output_dir / "training_report.json").write_text(
            json.dumps(run_report, indent=2), encoding="utf-8"
        )
        print(
            f"member={random_state} loss={artifact.final_loss:.6f} wall_seconds={elapsed:.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
