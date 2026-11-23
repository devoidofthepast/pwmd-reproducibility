"""Shared paths for the rerun. v3 (全部重跑规则_冻结v3.md): random-event training data, scales and models."""
import os as _pwmd_os
_PWMD_ROOT = _pwmd_os.path.abspath(_pwmd_os.path.join(_pwmd_os.path.dirname(__file__), *['..'] * 2))


def _pwmd_path(*parts):
    return _pwmd_os.path.join(_PWMD_ROOT, *parts)


from pathlib import Path
ROOT = Path(_pwmd_path('rerun_twoleg_early'))
SRC = ROOT / "src"
SCALES = ROOT / "scales_train_600_619_v3" / "objective_scales.json"
EXPECTED_SCALES_SHA = "851ef496204dd3b81291d2864b64da634c9eb8481908b8274bf6dd3603713e52"
MODEL_DIR = ROOT / "models_physics_train_600_619_v3"
ABLATION_MODEL_DIR = ROOT / "models_ablation_train_600_619_v3"
# v1 paths (fixed event script), kept for reference: rerun_paths.py.v1
