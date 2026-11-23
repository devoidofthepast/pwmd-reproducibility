"""Record per-epoch training losses of the frozen v3 proposer members (for the training-curve figure only).
Re-runs train_v4_model with the same groups, epochs, batch size and random state as models_physics_train_600_619_v3;
the only change is that the epoch mean loss and the mean of each loss term are appended to a list. No model is saved,
and the frozen models are not touched. Parity: the last epoch loss is compared with final_loss in training_report.json.
Usage: python record_training_curves_v3.py RANDOM_STATE"""
import hashlib, inspect, json, sys
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path[:0] = [str(H / "src")]
import numpy as np
import su_modfjsp_ha_pc.pgnd_v4_model as M
from su_modfjsp_ha_pc.pgnd_v4_model import read_context_groups
from su_modfjsp_ha_pc.run_pgnd_v4_train import expected_context_names, sha256_file

rs = int(sys.argv[1])
CTX = H / "labels_train_600_619_v3"
REPORT = json.loads((H / "models_physics_train_600_619_v3" / "training_report.json").read_text())
groups, hashes = [], {}
for seed in range(600, 620):
    for name in expected_context_names(seed):
        p = CTX / f"{name}.csv"
        hashes[name] = sha256_file(p)
        groups.extend(read_context_groups(p))
groups = tuple(groups)
manifest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
assert manifest == REPORT["label_manifest_sha256"], "label manifest differs from the frozen training run"
assert len(groups) == REPORT["groups"]

src = inspect.getsource(M.train_v4_model)
for old, new in [
    ("def train_v4_model(", "def train_v4_model_logged("),
    ("            losses = regression + 0.25 * classification + 0.50 * ranking + 0.50 * best_loss\n",
     "            losses = regression + 0.25 * classification + 0.50 * ranking + 0.50 * best_loss\n"
     "            TERMS.append([float(regression.mean().item()), float(classification.mean().item()), float(ranking.mean().item()), float(best_loss.mean().item())])\n"),
    ("        final_loss = float(np.mean(epoch_losses))\n",
     "        final_loss = float(np.mean(epoch_losses))\n        HIST.append(final_loss)\n        TERM_HIST.append(np.mean(TERMS, axis=0).tolist())\n        TERMS.clear()\n"),
]:
    assert src.count(old) == 1, old[:50]
    src = src.replace(old, new)
ns = dict(M.__dict__, HIST=[], TERMS=[], TERM_HIST=[])
exec(compile(src, "train_v4_model_logged", "exec"), ns)
art = ns["train_v4_model_logged"](groups, epochs=REPORT["epochs"], random_state=rs, groups_per_batch=REPORT["groups_per_batch"], hidden_size=96)
ref = next(m["final_loss"] for m in REPORT["members"] if m["random_state"] == rs)
out = H / "training_curves_v3"
out.mkdir(exist_ok=True)
(out / f"curve_{rs}.json").write_text(json.dumps(dict(random_state=rs, epoch_loss=ns["HIST"], terms=["regression", "classification", "ranking", "best"],
                                                      term_loss=ns["TERM_HIST"], final_loss=art.final_loss, frozen_final_loss=ref,
                                                      parity_abs_diff=abs(art.final_loss - ref))))
print(rs, art.final_loss, ref, abs(art.final_loss - ref))
