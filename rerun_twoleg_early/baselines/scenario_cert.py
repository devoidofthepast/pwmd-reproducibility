"""Development candidate (rerun v2 draft, not frozen): sampled-event certificate for the hidden-event loop.

The decoder knows the event process but not its realization: each level has a fixed script of event types, durations and
severities (the training seeds use the same script), and the hidden-event study draws each event time uniformly in 5-120
and its target uniformly from the matching class; a recovery follows its breakdown by the script offset.
At each certified step the base action and the proposal are rolled out on K scenarios. A scenario keeps every revealed
event and draws each unrevealed resource event (breakdown, recovery, path blockage, congestion, AGV failure) from that
process conditioned on its time being later than the clock. Both branches use the same K scenarios. The proposal is
accepted if its mean scenario value beats the mean base value by the relative margin M. Structural events (new job,
rework, route branch) are not sampled. K = 0 gives the original certificate on the believed instance.
"""
import inspect
import random

import event_blind as EB

RESOURCE = ("machine_breakdown", "machine_recovery", "path_blockage", "congestion_spike", "agv_failure")


def scenarios(known, template, t, key, k):
    """K event scenarios consistent with what is revealed at clock t. template: script events of this level (training seed)."""
    revealed = {e.event_id: e for e in known.events}
    machines = [m.machine_id for m in known.machines]
    edges = [e.edge_id for e in known.path_edges]
    agvs = [a.agv_id for a in known.agvs]
    tmpl = {e.event_id: e for e in template}
    pairs = {}  # recovery id -> breakdown id, by script order and machine
    for e in template:
        if e.event_type == "machine_recovery":
            b = next(x for x in template if x.event_type == "machine_breakdown" and x.affected_entity == e.affected_entity
                     and x.event_time <= e.event_time and x.event_id not in pairs.values())
            pairs[e.event_id] = b.event_id
    out = []
    lo = max(5, int(t) + 1)
    for i in range(k):
        rng = random.Random(f"{key}-{i}")
        drawn = {}
        for e in template:
            if e.event_type not in RESOURCE or e.event_id in revealed:
                continue
            kind = e.event_type
            if kind == "machine_recovery":
                bid = pairs[e.event_id]
                b = revealed.get(bid) or drawn.get(bid)
                if b is None:
                    continue  # its breakdown cannot happen any more
                time = b.event_time + (e.event_time - tmpl[bid].event_time)
                if time <= t:
                    time = int(t) + 1
                drawn[e.event_id] = type(e)(e.event_id, time, kind, b.affected_entity, e.duration, e.severity, e.scenario_id)
                continue
            if lo > 120:
                continue
            time = rng.randint(lo, 120)
            entity = rng.choice(machines if kind == "machine_breakdown" else edges if kind in ("path_blockage", "congestion_spike") else agvs)
            drawn[e.event_id] = type(e)(e.event_id, time, kind, entity, e.duration, e.severity, e.scenario_id)
        events = tuple(sorted((*known.events, *drawn.values()), key=lambda x: (x.event_time, x.event_id)))
        out.append(known.replace(events=events))
    return out


_src = inspect.getsource(EB.decode)
_src = _src.replace("def decode(*, true, raw, preference, level, seed, mode, preference_id, selector, scales, clock, trace=None, margin=0.0, nogate=False):",
                    "def decode_scen(*, true, raw, preference, level, seed, mode, preference_id, selector, scales, clock, trace=None, margin=0.0, nogate=False, k=4, template=()):")
_old = """                    def value(j, o, m, tag):
                        schedule = _complete_schedule_after_forced_path_aware_joint_action(
                            instance=known,"""
_new = """                    scen = scenarios(known, template, t, f"scen-{seed}-{level}-{mode}-{preference_id}-{step}", k) if k > 0 else [known]

                    def value(j, o, m, tag):
                        return sum(value1(inst, j, o, m, tag) for inst in scen) / len(scen)

                    def value1(inst, j, o, m, tag):
                        schedule = _complete_schedule_after_forced_path_aware_joint_action(
                            instance=inst,"""
assert _src.count(_old) == 1
_src = _src.replace(_old, _new)
_old2 = "                        return _weighted_value(known, schedule, preference, level, seed, mode, preference_id, tag, scales)"
assert _src.count(_old2) == 1
_src = _src.replace(_old2, "                        return _weighted_value(inst, schedule, preference, level, seed, mode, preference_id, tag, scales)")
_src = _src.replace("n_rollouts += 2", "n_rollouts += 2 * len(scen)")
_ns = dict(EB.__dict__, scenarios=scenarios)
exec(compile(_src, "scenario_cert.decode_scen", "exec"), _ns)
decode_scen = _ns["decode_scen"]
