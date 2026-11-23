from __future__ import annotations
import os as _pwmd_os
_PWMD_ROOT = _pwmd_os.path.abspath(_pwmd_os.path.join(_pwmd_os.path.dirname(__file__), *['..'] * 3))


def _pwmd_path(*parts):
    return _pwmd_os.path.join(_PWMD_ROOT, *parts)



import argparse
import json
from pathlib import Path
from time import perf_counter

from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.neural_operation_rollout import _detail_row, decode_rule_operation_rollout
from su_modfjsp_ha_pc.offline_logs import write_csv
from su_modfjsp_ha_pc.path_conflict_neural_dataset import (
    _explicit_preference_id,
    _explicit_teacher_preferences,
)
from su_modfjsp_ha_pc.pgnd_v4_data import LEVELS, MODES, PREFERENCE_INDEXES
from su_modfjsp_ha_pc.pgnd_v4_online import schedules_identical
from su_modfjsp_ha_pc.pgnd_v5_exact_certificate import (
    decode_pgnd_v5_exact_certificate,
    neural_ensemble_proposal_selector,
)
from su_modfjsp_ha_pc.run_pgnd_v4_learnability_gate import _load_artifact
from su_modfjsp_ha_pc.structural import realize_structural_events


SEED = 100
MEMBER_STATES = (320, 321, 322, 323, 324)
DEFAULT_MODEL_DIR = Path(
    _pwmd_path('output', 'pgnd_v4_tiebreak_v2', 'models_train_100_119')
)
DEFAULT_OUTPUT_DIR = Path(_pwmd_path('output', 'pgnd_v5_smoke_seed100'))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    artifacts = tuple(
        _load_artifact(args.model_dir / f"pgnd_v4_member_{random_state}.pt")
        for random_state in MEMBER_STATES
    )
    loaded_states = tuple(artifact.random_state for artifact in artifacts)
    if loaded_states != MEMBER_STATES:
        raise RuntimeError(f"unexpected V4 ensemble members: {loaded_states}")
    selector = neural_ensemble_proposal_selector(artifacts)

    case_rows: list[dict[str, object]] = []
    schedule_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for level in LEVELS:
        raw_instance = generate_instance(level, "small", SEED)
        for mode in MODES:
            instance = realize_structural_events(raw_instance).instance if mode == "realized" else raw_instance
            preferences = _explicit_teacher_preferences(len(instance.objectives), max_preferences=6)
            if len(preferences) != len(PREFERENCE_INDEXES):
                raise RuntimeError(f"expected six preferences, got {len(preferences)}")
            for preference_index, preference in enumerate(preferences):
                preference_id = _explicit_preference_id(preference_index, preference)
                decision_trace: list[dict[str, object]] = []
                started = perf_counter()
                v5 = decode_pgnd_v5_exact_certificate(
                    instance=instance,
                    preference=preference,
                    level=level,
                    seed=SEED,
                    mode=mode,
                    preference_id=preference_id,
                    proposal_selector=selector,
                    trace_rows=decision_trace,
                )
                v5_wall_ms = (perf_counter() - started) * 1000.0
                started = perf_counter()
                lookahead = decode_rule_operation_rollout(
                    instance=instance,
                    preference=preference,
                    rule="lookahead",
                    random_seed=SEED,
                )
                lookahead_wall_ms = (perf_counter() - started) * 1000.0
                v5_detail = _detail_row(
                    instance=instance,
                    schedule=v5,
                    teacher_operations=(),
                    level=level,
                    seed=SEED,
                    mode=mode,
                    preference_id=preference_id,
                    preference=preference,
                    method="pgnd-v5-exact-certificate",
                )
                lookahead_detail = _detail_row(
                    instance=instance,
                    schedule=lookahead,
                    teacher_operations=(),
                    level=level,
                    seed=SEED,
                    mode=mode,
                    preference_id=preference_id,
                    preference=preference,
                    method="lookahead",
                )
                v5_value = float(v5_detail["target_weighted_value"])
                lookahead_value = float(lookahead_detail["target_weighted_value"])
                delta = v5_value - lookahead_value
                decisions = len(decision_trace)
                proposals = sum(int(row["proposal_available"]) for row in decision_trace)
                accepted = sum(int(row["accepted"]) for row in decision_trace)
                invariant_failures = sum(int(row["invariant_failure"]) for row in decision_trace)
                rollout_evaluations = sum(int(row["rollout_evaluations"]) for row in decision_trace)
                case_id = f"seed{SEED:03d}_{level}_{mode}_pref{preference_index:02d}"
                for row in decision_trace:
                    trace_rows.append(
                        {
                            "case_id": case_id,
                            "level": level,
                            "seed": SEED,
                            "mode": mode,
                            "preference_index": preference_index,
                            "preference_id": preference_id,
                            **row,
                        }
                    )
                schedule_rows.extend((v5_detail, lookahead_detail))
                case_rows.append(
                    {
                        "case_id": case_id,
                        "level": level,
                        "seed": SEED,
                        "mode": mode,
                        "preference_index": preference_index,
                        "preference_id": preference_id,
                        "v5_weighted_value": v5_value,
                        "lookahead_weighted_value": lookahead_value,
                        "v5_minus_lookahead": delta,
                        "improved": int(delta < -1.0e-9),
                        "identical_schedule": int(schedules_identical(v5, lookahead)),
                        "decisions": decisions,
                        "proposed_candidates": proposals,
                        "accepted_proposals": accepted,
                        "proposal_acceptance_rate": accepted / proposals if proposals else 0.0,
                        "proposal_rejection_rate": (proposals - accepted) / proposals if proposals else 0.0,
                        "exact_rollout_evaluations": rollout_evaluations,
                        "mean_exact_rollout_evaluations_per_decision": (
                            rollout_evaluations / decisions if decisions else 0.0
                        ),
                        "neural_scoring_ms": sum(
                            float(row["neural_scoring_ms"]) for row in decision_trace
                        ),
                        "exact_rollout_ms": sum(
                            float(row["exact_rollout_ms"]) for row in decision_trace
                        ),
                        "v5_wall_ms": v5_wall_ms,
                        "lookahead_wall_ms": lookahead_wall_ms,
                        "invariant_failures": invariant_failures,
                    }
                )
                print(
                    f"{case_id} delta={delta:.10f} accepted={accepted}/{proposals} "
                    f"rollouts/decision={rollout_evaluations / decisions if decisions else 0.0:.3f}"
                )
                if delta > 1.0e-9:
                    _write_outputs(args.output_dir, case_rows, schedule_rows, trace_rows, artifacts)
                    raise RuntimeError(f"scalarized degradation in {case_id}: {delta}")

    _write_outputs(args.output_dir, case_rows, schedule_rows, trace_rows, artifacts)
    return 0


def _write_outputs(output_dir, case_rows, schedule_rows, trace_rows, artifacts) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "case_detail.csv", tuple(case_rows))
    write_csv(output_dir / "schedule_detail.csv", tuple(schedule_rows))
    write_csv(output_dir / "decision_trace.csv", tuple(trace_rows))
    decisions = sum(int(row["decisions"]) for row in case_rows)
    proposals = sum(int(row["proposed_candidates"]) for row in case_rows)
    accepted = sum(int(row["accepted_proposals"]) for row in case_rows)
    rollout_evaluations = sum(int(row["exact_rollout_evaluations"]) for row in case_rows)
    report = {
        "protocol": "PGND-V5-neural-proposal-exact-certificate-smoke",
        "evidence_status": "implementation smoke only; not manuscript evidence",
        "seed": SEED,
        "levels": list(LEVELS),
        "modes": list(MODES),
        "preference_indexes": list(PREFERENCE_INDEXES),
        "model_members": [artifact.random_state for artifact in artifacts],
        "case_count": len(case_rows),
        "decision_count": decisions,
        "proposed_candidate_count": proposals,
        "accepted_proposal_count": accepted,
        "accepted_proposal_rate": accepted / proposals if proposals else 0.0,
        "proposal_rejection_rate": (proposals - accepted) / proposals if proposals else 0.0,
        "exact_rollout_evaluations": rollout_evaluations,
        "mean_exact_rollout_evaluations_per_decision": (
            rollout_evaluations / decisions if decisions else 0.0
        ),
        "mean_v5_minus_lookahead": (
            sum(float(row["v5_minus_lookahead"]) for row in case_rows) / len(case_rows)
            if case_rows
            else 0.0
        ),
        "max_v5_minus_lookahead": max(
            (float(row["v5_minus_lookahead"]) for row in case_rows), default=0.0
        ),
        "improved_case_count": sum(int(row["improved"]) for row in case_rows),
        "identical_case_count": sum(int(row["identical_schedule"]) for row in case_rows),
        "neural_scoring_ms": sum(float(row["neural_scoring_ms"]) for row in case_rows),
        "exact_rollout_ms": sum(float(row["exact_rollout_ms"]) for row in case_rows),
        "v5_wall_ms": sum(float(row["v5_wall_ms"]) for row in case_rows),
        "lookahead_wall_ms": sum(float(row["lookahead_wall_ms"]) for row in case_rows),
        "invariant_failure_count": sum(int(row["invariant_failures"]) for row in case_rows),
        "passed": len(case_rows) == 24
        and all(float(row["v5_minus_lookahead"]) <= 1.0e-9 for row in case_rows)
        and all(int(row["invariant_failures"]) == 0 for row in case_rows),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
