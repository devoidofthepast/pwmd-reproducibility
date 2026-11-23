"""v3 step 4: main test set P (全部重跑规则_冻结v3.md section 4). One process per (block, seed, group).
Usage: python test_v3.py BLOCK SEED GROUP [--runs R]
  BLOCK: S (synthetic, 800-819), C (Mk1 and Edata7, 820-829), L (20-22 jobs, 830-839)
  GROUP: rules | pgnd | drl | ablation (S only) | reference (S only) | ea (S, first 10 seeds)
Hidden events, random per seed, strict clock, for every group except reference (events known to the simulator).
The PGND settings (M, K, k) are read from v3_final_config.json, written by the step 3 selection.
Every schedule passes the route/two-leg timing check, the hard-constraint audit and the per-leg event-window check."""
import os as _pwmd_os
_PWMD_ROOT = _pwmd_os.path.abspath(_pwmd_os.path.join(_pwmd_os.path.dirname(__file__), *['..'] * 2))


def _pwmd_path(*parts):
    return _pwmd_os.path.join(_PWMD_ROOT, *parts)


import json
import sys
from pathlib import Path
from time import perf_counter

H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / "src"), str(H / "baselines"), str(H.parent / "twoleg")]
import torch  # noqa: E402

torch.set_num_threads(1)
import drl_v3_common as C  # noqa: E402
from blind_arms import policy_decode  # noqa: E402
from run_early import check_schedule  # noqa: E402
from su_modfjsp_ha_pc.metrics import evaluate_feasibility  # noqa: E402

B = _pwmd_path('external_baselines', 'FJSP-DRL-DANIEL', 'data', 'BenchData')
CORES = {"mk1": f"{B}/Brandimarte/BrandimarteMk1.fjs", "edata7": f"{B}/Hurink_edata/HurinkEdata7.fjs"}
BLOCKS = {"S": range(800, 820), "C": range(820, 830), "L": range(830, 840)}
CFG = json.loads((H / "v3_final_config.json").read_text())


def cases_for(block, seed):
    if block == "S":
        return [("syn", c) for c in C.instances([seed])]
    if block == "C":
        return [(name, c) for name, core in CORES.items() for c in C.instances([seed], core=core)]
    return [("large", c) for c in _large(seed)]


def _large(seed):
    from event_blind import randomize_events
    from su_modfjsp_ha_pc.generator import generate_instance
    from su_modfjsp_ha_pc.structural import realize_structural_events
    out = []
    for level in ("L4", "L5"):
        raw = randomize_events(generate_instance(level, "medium", seed), seed, level)
        for mode in ("raw", "realized"):
            out.append(dict(seed=seed, level=level, mode=mode, raw=raw, true=realize_structural_events(raw).instance if mode == "realized" else raw))
    return out


def template_for(set_name, level):
    from su_modfjsp_ha_pc.generator import generate_instance
    from su_modfjsp_ha_pc.public_fjsp import load_and_extend_public_fjsp
    if set_name in CORES:
        return load_and_extend_public_fjsp(Path(CORES[set_name]), level=level, seed=600, n_agvs=3).events
    return generate_instance(level, "medium" if set_name == "large" else "small", 600).events


def checks(true, s):
    rb, tb, _, _ = check_schedule(true, s, "rerun")
    rep = evaluate_feasibility(true, s)
    hv = rep.precedence_violations + rep.machine_overlap_violations + rep.agv_overlap_violations + rep.edge_capacity_violations
    return dict(route_bad=rb, timing_bad=tb, hard_viol=hv, event_viol=rep.dynamic_event_violations)


def arms(group, block, seed):
    """(arm name, runner(set_name, case, pref_index, pref) -> (schedule, info))"""
    from rerun_paths import ABLATION_MODEL_DIR, MODEL_DIR
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_preference_id
    from su_modfjsp_ha_pc.run_pgnd_v4_learnability_gate import _load_artifact
    from su_modfjsp_ha_pc.run_pgnd_v5_smoke import MEMBER_STATES
    sc = C.scales()
    out = []

    def policy(choose_factory):
        def run(set_name, case, pi, pref):
            s, _ = policy_decode(true=case["true"], raw=case["raw"], preference=pref, choose=choose_factory(pi, case), strict=True)
            return s, {}
        return run

    if group == "rules":
        import rules_v3 as R
        for rule in R.RULES:
            out.append((f"rule-{rule}", policy(lambda pi, case, rule=rule: R.chooser(rule, seed=case["seed"]))))
        return out
    if group == "drl":
        import cheng_dqn as D
        import daniel_adapter as DA
        import hgt_adapter as HG
        import song_adapter as SO
        nets = {}
        for name, mod in (("daniel", DA), ("song", SO), ("hgt", HG)):
            m = mod.make_model()
            m.load_state_dict(torch.load(H / "drl_v3" / name / f"{name}_selected.pt"))
            m.eval()
            nets[name] = (mod, m)
            out.append((name, policy(lambda pi, case, name=name: nets[name][0].chooser(nets[name][1]))))
        dqn = []
        for i in range(6):
            q = D.network()
            q.load_state_dict(torch.load(H / "drl_v3" / "cheng" / "final" / "selected" / f"dqn_pref{i:02d}.pt"))
            q.eval()
            dqn.append(q)
        out.append(("dqn", policy(lambda pi, case: D.chooser(dqn[pi]))))
        return out
    from scenario_cert3 import decode_scen3, neural_topk_selector
    M, K, k = CFG["M"], CFG["K"], CFG["k"]

    def scen(selector, clock="strict", nogate=False, kk=K):
        def run(set_name, case, pi, pref):
            return decode_scen3(true=case["true"], raw=case["raw"], preference=pref, level=case["level"], seed=case["seed"], mode=case["mode"],
                                preference_id=_explicit_preference_id(pi, pref), selector=selector, scales=sc, clock=clock, margin=M,
                                k=kk, template=template_for(set_name, case["level"]), nogate=nogate)
        return run

    members = tuple(_load_artifact(MODEL_DIR / f"pgnd_v4_member_{s}.pt") for s in MEMBER_STATES)
    if group == "pgnd":
        return [("pgnd", scen(neural_topk_selector(members, k))), ("pgnd-nocert", scen(neural_topk_selector(members, 1), nogate=True))]
    if group == "ablation":
        from v3_selectors import context_only_topk, no_route_event_topk, ridge_topk
        return [("abl-ridge", scen(ridge_topk(ABLATION_MODEL_DIR, k))), ("abl-context-only", scen(context_only_topk(ABLATION_MODEL_DIR, k))),
                ("abl-no-route-event", scen(no_route_event_topk(ABLATION_MODEL_DIR, k)))]
    if group == "reference":
        from su_modfjsp_ha_pc.pgnd_v5_full_exact_reference import decode_pgnd_v5_full_exact_reference

        def full(set_name, case, pi, pref):
            tr = []
            s = decode_pgnd_v5_full_exact_reference(instance=case["true"], preference=pref, level=case["level"], seed=case["seed"], mode=case["mode"],
                                                    preference_id=_explicit_preference_id(pi, pref), trace_rows=tr, objective_scales=sc)
            return s, dict(rollouts=sum(int(r.get("rollout_evaluations", r.get("complete_rollout_evaluations", 0))) for r in tr))
        return [("ref-pgnd-known", scen(neural_topk_selector(members, k), clock="oracle")), ("ref-rule-known", scen(None, clock="oracle", kk=0)),
                ("ref-full-rollout-known", full)]
    if group == "ea":
        import ea_resched_v3 as E
        runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 5
        only = sys.argv[sys.argv.index("--method") + 1] if "--method" in sys.argv else None
        for m in E.METHODS:
            if only and m != only:
                continue
            for r in range(runs):
                out.append((f"ea-{m}-run{r}", lambda set_name, case, pi, pref, m=m, r=r: E.run_case(case, m, pref, sc, CFG["ea_budget"], seed=1000 * r + case["seed"])))
        return out
    raise ValueError(group)


def main():
    block, seed, group = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    assert seed in BLOCKS[block], "seed outside the frozen block"
    if group in ("ablation", "reference", "ea"):
        assert block == "S" and (group != "ea" or seed < 810)
    outdir = H / "test_v3" / "runs" / block
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"_{sys.argv[sys.argv.index('--method') + 1]}" if "--method" in sys.argv else ""
    path = outdir / f"{group}{tag}_{seed}.jsonl"
    done = {(r["case_id"], r["arm"]) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    sc, prefs = C.scales(), C.preferences()
    arm_list = arms(group, block, seed)
    with path.open("a") as f:
        for set_name, case in cases_for(block, seed):
            for pi, pref in enumerate(prefs):
                cid = f"{set_name}_seed{seed}_{case['level']}_{case['mode']}_pref{pi:02d}"
                for arm, run in arm_list:
                    if (cid, arm) in done:
                        continue
                    t0 = perf_counter()
                    s, info = run(set_name, case, pi, pref)
                    wall = perf_counter() - t0
                    rec = dict(case_id=cid, set=set_name, seed=seed, level=case["level"], mode=case["mode"], pref=pi, arm=arm,
                               value=C.partial_value(case["true"], s.operations, s.transports, pref, sc), wall_s=round(wall, 3),
                               **checks(case["true"], s), **{("ea_wall_s" if k == "wall_s" else k): v for k, v in (info or {}).items() if isinstance(v, (int, float))})
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if rec["route_bad"] or rec["timing_bad"] or rec["hard_viol"] or rec["event_viol"] or rec.get("failures"):
                        with (outdir / "check_failures.log").open("a") as g:
                            g.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
