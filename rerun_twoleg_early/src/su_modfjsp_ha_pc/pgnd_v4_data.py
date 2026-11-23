from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from su_modfjsp_ha_pc.exact_advantage_learnability import AdvantageLearningRow


SEED_ROLES = {
    "train": range(600, 620),  # rerun v1 (全部重跑规则_冻结v1.md)
    "selection": range(120, 125),
    "calibration": range(125, 130),
    "confirmation": range(130, 140),
    "groupwise_calibration": range(140, 145),
    "groupwise_confirmation": range(145, 155),
}
PROTOCOL_VERSION = "PGND-V4-full-action-tiebreak-v2"
LEVELS = ("L4", "L5")
MODES = ("raw", "realized")
PREFERENCE_INDEXES = tuple(range(6))


def validate_seed_role(seed: int, role: str) -> None:
    if role not in SEED_ROLES:
        raise ValueError(f"unknown seed role: {role}")
    if seed not in SEED_ROLES[role]:
        raise ValueError(f"seed {seed} is not allocated to role {role}")


def context_name(*, seed: int, level: str, mode: str, preference_index: int) -> str:
    if level not in LEVELS or mode not in MODES or preference_index not in PREFERENCE_INDEXES:
        raise ValueError("invalid PGND-V4 context")
    return f"seed{seed:03d}_{level}_{mode}_pref{preference_index:02d}"


def expected_context_names(seed: int) -> tuple[str, ...]:
    return tuple(
        context_name(seed=seed, level=level, mode=mode, preference_index=preference_index)
        for level in LEVELS
        for mode in MODES
        for preference_index in PREFERENCE_INDEXES
    )


def validate_context_rows(rows: tuple[AdvantageLearningRow, ...], *, seed: int) -> None:
    if not rows:
        raise ValueError("context contains no labeled rows")
    if {row.seed for row in rows} != {seed}:
        raise ValueError("context contains rows from another seed")
    groups: dict[str, list[AdvantageLearningRow]] = {}
    for row in rows:
        groups.setdefault(row.group_id, []).append(row)
    for group_id, group_rows in groups.items():
        bases = [row for row in group_rows if row.is_base]
        if len(bases) != 1:
            raise ValueError(f"group {group_id!r} must contain exactly one base action")
        if bases[0].advantage != 0.0:
            raise ValueError(f"group {group_id!r} base advantage is not exactly zero")


def write_metadata_atomic(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
