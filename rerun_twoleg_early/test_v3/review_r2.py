"""v3 amendment 3 (全部重跑规则_v3修正案3_二审补充实验_20261118.md): extra arms for the second review round.
Usage: python review_r2.py BLOCK SEED GROUP
  GROUP drl23 (S, C, L): the four DRL methods choose the operation; the machine is re-chosen by the shared preference
        completion (Eq. 23), as for the Random rule; AGV and route as before. Inference only, no retraining.
  GROUP heu (S, C): heuristic top-2 proposers under the same certificate as pgnd:
        heu-ect  ranks the non-base candidates by projected completion (Eq. 21), ties by processing time, energy, order;
        heu-cost ranks them by the local cost of Eq. 23 with the machine fixed (the transport rule's own score).
  GROUP heuL (L): the heuristic proposer chosen on S (name in r2_heu_choice.txt), on P-L.
  GROUP rndL (L): random proposer of amendment 1 on P-L.
  GROUP mis (S): pgnd with a mis-specified scenario sampler (the true events are unchanged):
        dur0.5 / dur1.5: event durations and breakdown-recovery offsets scaled; tmax90 / tmax160: event times drawn up to
        90 / 160 instead of 120; rate0.5: each unrevealed event unit kept with prob. 0.5; rate1.5: half of the units duplicated.
Output: runs/{BLOCK}/{GROUP}_{SEED}.jsonl, through test_v3.main (same checks, resume)."""
import hashlib
import random
import sys
from pathlib import Path

sys.argv = sys.argv[:4]
import test_v3 as T  # noqa: E402

_arms = T.arms
HERE = Path(__file__).resolve().parent


def _fast_weighted_value(instance, schedule, preference, level, seed, mode, preference_id, method, objective_scales):
    """Implementation-only speed-up (amendment 3 section 7): the value that _detail_row returns when objective_scales is
    given, namely round(objective_breakdown(evaluate_objective_contract(...).raw).weighted_value, 10), without the unused
    metrics and surrogate statistics that _detail_row also computes. Parity: test_v3/r2_fast_parity.py."""
    from su_modfjsp_ha_pc.objective_contract import evaluate_objective_contract, objective_breakdown
    raw = evaluate_objective_contract(instance, schedule).raw
    return float(round(objective_breakdown(raw, preference, objective_scales).weighted_value, 10))


# heu on C (queued in the main launcher, which was started without R2_FAST_VALUE) also takes the fast path.
if __import__("os").environ.get("R2_FAST_VALUE") == "1" or (sys.argv[1:2] == ["C"] and sys.argv[3:4] == ["heu"]):
    import su_modfjsp_ha_pc.pgnd_v5_exact_certificate as _PC
    assert _PC._weighted_value.__code__.co_varnames[:9] == ("instance", "schedule", "preference", "level", "seed", "mode",
                                                             "preference_id", "method", "objective_scales")
    _PC._weighted_value = _fast_weighted_value
MIS = ("dur0.5", "dur1.5", "tmax90", "tmax160", "rate0.5", "rate1.5")
RESOURCE = ("machine_breakdown", "machine_recovery", "path_blockage", "congestion_spike", "agv_failure")


def _h(s):
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def mis_scenarios(dur=1.0, tmax=120, rate=1.0):
    """scenario_cert.scenarios with a mis-specified event process."""
    def scenarios(known, template, t, key, k):
        revealed = {e.event_id: e for e in known.events}
        machines = [m.machine_id for m in known.machines]
        edges = [e.edge_id for e in known.path_edges]
        agvs = [a.agv_id for a in known.agvs]
        tmpl = list(template)
        # pair each recovery with its breakdown (same rule as scenario_cert.scenarios)
        pairs = {}
        for e in tmpl:
            if e.event_type == "machine_recovery":
                b = next(x for x in tmpl if x.event_type == "machine_breakdown" and x.affected_entity == e.affected_entity
                         and x.event_time <= e.event_time and x.event_id not in pairs.values())
                pairs[e.event_id] = b.event_id
        rec_of = {b: r for r, b in pairs.items()}
        # rate1.5: duplicate half of the units (a breakdown with its recovery is one unit)
        units = [e for e in tmpl if e.event_type in RESOURCE and e.event_type != "machine_recovery"]
        if rate > 1.0:
            byid = {e.event_id: e for e in tmpl}
            for e in units:
                if _h(f"dup-{e.event_id}") % 2 == 0:
                    d = type(e)(f"{e.event_id}~d", e.event_time, e.event_type, e.affected_entity, e.duration, e.severity, e.scenario_id)
                    tmpl.append(d)
                    if e.event_id in rec_of:
                        r = byid[rec_of[e.event_id]]
                        rd = type(r)(f"{r.event_id}~d", r.event_time, r.event_type, r.affected_entity, r.duration, r.severity, r.scenario_id)
                        tmpl.append(rd)
                        pairs[rd.event_id] = d.event_id
        tmpl_by = {e.event_id: e for e in tmpl}
        out = []
        lo = max(5, int(t) + 1)
        for i in range(k):
            rng = random.Random(f"{key}-{i}")
            drawn = {}
            for e in tmpl:
                if e.event_type not in RESOURCE or e.event_id in revealed:
                    continue
                kind = e.event_type
                if kind == "machine_recovery":
                    bid = pairs[e.event_id]
                    b = revealed.get(bid) or drawn.get(bid)
                    if b is None:
                        continue
                    offset = (e.event_time - tmpl_by[bid].event_time) * dur
                    time = int(round(b.event_time + offset))
                    if time <= t:
                        time = int(t) + 1
                    drawn[e.event_id] = type(e)(e.event_id, time, kind, b.affected_entity, e.duration * dur if e.duration else e.duration,
                                                e.severity, e.scenario_id)
                    continue
                if rate < 1.0 and rng.random() >= rate:
                    continue
                if lo > tmax:
                    continue
                time = rng.randint(lo, tmax)
                entity = rng.choice(machines if kind == "machine_breakdown" else edges if kind in ("path_blockage", "congestion_spike") else agvs)
                d = max(1, int(round(e.duration * dur))) if e.duration else e.duration
                drawn[e.event_id] = type(e)(e.event_id, time, kind, entity, d, e.severity, e.scenario_id)
            events = tuple(sorted((*known.events, *drawn.values()), key=lambda x: (x.event_time, str(x.event_id))))
            out.append(known.replace(events=events))
        return out
    return scenarios


def mis_decoder(name):
    import scenario_cert3 as S3
    p = dict(dur=1.0, tmax=120, rate=1.0)
    if name.startswith("dur"):
        p["dur"] = float(name[3:])
    elif name.startswith("tmax"):
        p["tmax"] = int(name[4:])
    else:
        p["rate"] = float(name[4:])
    ns = dict(S3._ns)
    ns["scenarios"] = mis_scenarios(**p)
    exec(compile(S3._src, f"review_r2.mis_{name}", "exec"), ns)
    return ns["decode_scen3"]


def heuristic_topk(kind, k):
    """Top-k non-base candidates by a local score. The decoder state is read from the calling frame (event_blind.decode)."""
    from su_modfjsp_ha_pc.hierarchical_transport_attention_policy import _best_fixed_machine_preference_transport
    from su_modfjsp_ha_pc.joint_operation_machine_value_policy import _resolve_joint_candidate

    def select(rows, group, base_index):
        idx = [i for i in range(len(rows)) if i != base_index]
        if not idx:
            return None
        if kind == "ect":
            def key(i):
                r = rows[i]
                return (float(r["candidate_operation_completion_lower_bound"]), float(r["candidate_processing_time"]),
                        float(r["candidate_processing_energy"]), i)
        else:
            L = sys._getframe(1).f_locals
            known, available, pref, res = L["known"], L["available"], L["preference"], L["resources"]()
            cost = {}
            for i in idx:
                job, op, machine = _resolve_joint_candidate(known, available, rows[i])
                cost[i] = float(_best_fixed_machine_preference_transport(instance=known, preference=pref, job=job, operation=op,
                                                                          machine=machine, **res)["score"])

            def key(i):
                return (cost[i], i)
        return sorted(idx, key=key)[:k]
    return select


def drl23(choose_factory):
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _best_preference_action

    def factory(pi, case):
        inner = choose_factory(pi, case)

        def choose(known, state, operations, transports, preference):
            picked = inner(known, state, operations, transports, preference)
            job, op = picked[0], picked[1]
            m = _best_preference_action(instance=known, available=((job, op),), preference=preference,
                                        machine_available=state.machine_available, job_ready=state.job_ready,
                                        agv_available=state.agv_available, agv_node=state.agv_node,
                                        edge_reservations=state.edge_reservations)["machine"]
            return (job, op, m, *picked[3:])
        return choose
    return factory


def arms(group, block, seed):
    import torch
    from blind_arms import policy_decode
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_preference_id
    sc = T.C.scales()
    M, K, k = T.CFG["M"], T.CFG["K"], T.CFG["k"]

    def scen(selector, decoder=None):
        from scenario_cert3 import decode_scen3
        dec = decoder or decode_scen3

        def run(set_name, case, pi, pref):
            return dec(true=case["true"], raw=case["raw"], preference=pref, level=case["level"], seed=case["seed"], mode=case["mode"],
                       preference_id=_explicit_preference_id(pi, pref), selector=selector, scales=sc, clock="strict", margin=M,
                       k=K, template=T.template_for(set_name, case["level"]))
        return run

    if group == "drl23":
        out = []
        import cheng_dqn as D
        import daniel_adapter as DA
        import hgt_adapter as HG
        import song_adapter as SO
        H = T.H
        nets = {}
        for name, mod in (("daniel", DA), ("song", SO), ("hgt", HG)):
            m = mod.make_model()
            m.load_state_dict(torch.load(H / "drl_v3" / name / f"{name}_selected.pt"))
            m.eval()
            nets[name] = (mod, m)
        dqn = []
        for i in range(6):
            q = D.network()
            q.load_state_dict(torch.load(H / "drl_v3" / "cheng" / "final" / "selected" / f"dqn_pref{i:02d}.pt"))
            q.eval()
            dqn.append(q)
        facs = {n: (lambda pi, case, n=n: nets[n][0].chooser(nets[n][1])) for n in nets}
        facs["dqn"] = lambda pi, case: D.chooser(dqn[pi])
        for n, fac in facs.items():
            f23 = drl23(fac)

            def run(set_name, case, pi, pref, f23=f23):
                s, _ = policy_decode(true=case["true"], raw=case["raw"], preference=pref, choose=f23(pi, case), strict=True)
                return s, {}
            out.append((f"{n}-eq23", run))
        return out
    if group == "heu":
        assert block in ("S", "C")
        return [("heu-ect", scen(heuristic_topk("ect", k))), ("heu-cost", scen(heuristic_topk("cost", k)))]
    if group == "heuL":
        assert block == "L"
        choice = (HERE / "r2_heu_choice.txt").read_text().strip()
        assert choice in ("ect", "cost")
        return [(f"heu-{choice}", scen(heuristic_topk(choice, k)))]
    if group == "rndL":
        assert block == "L"
        from scenario_cert3 import random_topk_selector
        return [("rnd", scen(random_topk_selector(k)))]
    if group == "gate":
        # w/o certificate, gated: argmax of the ensemble mean over the whole candidate set, base action included;
        # deviate only when the argmax is not the base action (ties -> base). No rollout.
        import numpy as np
        from rerun_paths import MODEL_DIR
        from su_modfjsp_ha_pc.pgnd_v4_model import predict_v4_model
        from su_modfjsp_ha_pc.run_pgnd_v4_learnability_gate import _load_artifact
        from su_modfjsp_ha_pc.run_pgnd_v5_smoke import MEMBER_STATES
        members = tuple(_load_artifact(MODEL_DIR / f"pgnd_v4_member_{s}.pt") for s in MEMBER_STATES)

        def gated(rows, grp, base_index):
            pred = np.mean(np.asarray([predict_v4_model(a, (grp,))[0] for a in members], dtype=float), axis=0)
            if not np.all(np.isfinite(pred)):
                return None
            best = max(range(len(rows)), key=lambda i: (pred[i], i == base_index))
            return None if best == base_index else [best]

        from scenario_cert3 import decode_scen3

        def run(set_name, case, pi, pref):
            return decode_scen3(true=case["true"], raw=case["raw"], preference=pref, level=case["level"], seed=case["seed"], mode=case["mode"],
                                preference_id=_explicit_preference_id(pi, pref), selector=gated, scales=sc, clock="strict", margin=M,
                                k=K, template=T.template_for(set_name, case["level"]), nogate=True)
        return [("nocert-gated", run)]
    if group == "mis":
        assert block == "S"
        from rerun_paths import MODEL_DIR
        from scenario_cert3 import neural_topk_selector
        from su_modfjsp_ha_pc.run_pgnd_v4_learnability_gate import _load_artifact
        from su_modfjsp_ha_pc.run_pgnd_v5_smoke import MEMBER_STATES
        members = tuple(_load_artifact(MODEL_DIR / f"pgnd_v4_member_{s}.pt") for s in MEMBER_STATES)
        return [(f"pgnd-mis-{n}", scen(neural_topk_selector(members, k), mis_decoder(n))) for n in MIS]
    return _arms(group, block, seed)


T.arms = arms


def _locked_main():
    """One process per (block, group, seed): a second launch of the same job exits at once while the first is alive."""
    import os
    b, s, g = sys.argv[1], sys.argv[2], sys.argv[3]
    lock = HERE / "runs" / b / f"{g}_{s}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            pid = int(lock.read_text())
            import psutil
            if psutil.pid_exists(pid):
                print(f"locked by {pid}", flush=True)
                return
        except Exception:
            pass
    lock.write_text(str(os.getpid()))
    try:
        T.main()
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    _locked_main()
