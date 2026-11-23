"""v3 step 3(c) sensitivity, report only. Run from rerun_twoleg_early with PYTHONPATH=.
Each variant of PGND (M=0.02, K=8, k=2) vs la and vs the main setting (5 members, 20 training seeds)."""
import collections, contextlib, glob, io, json, sys
sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_c1 as A
MAIN = "pgnd_M0.02_K8_k2"
for S in ("syn", "mk1", "edata7"):
    v = collections.defaultdict(dict); meta = collections.defaultdict(lambda: [0.0, 0.0, 0]); bad = 0
    files = [(f, None) for f in glob.glob(f"dev_v3/runs_Kk/{S}_*.jsonl")] + [(f, "s") for f in glob.glob(f"dev_v3/runs_sens/{S}_*.jsonl")]
    if S == "syn":
        files += [(f, None) for f in glob.glob("dev_v3/runs_M/*.jsonl")]
    for f, tag in files:
        for l in open(f):
            if not l.strip(): continue
            x = json.loads(l); a = x["arm"]
            if tag:
                a = "members1" if "_m1" in f else "members3" if "_m3" in f else "seeds5"
            elif a != "la" and a != MAIN:
                continue
            v[x["case_id"]][a] = x["value"]
            bad += x["route_bad"] + x["timing_bad"] + x["hard_viol"] + x["event_viol"] + x.get("failures", 0)
            m = meta[a]; m[0] += x["wall_s"]; m[1] += x["rollouts"]; m[2] += 1
    arms = [MAIN, "members1", "members3", "seeds5"]
    full = {c: x for c, x in v.items() if "la" in x and all(a in x for a in arms)}
    print(f"== {S}: complete cases {len(full)}, check errors {bad}")
    if not full: continue
    for a in arms:
        c = A.compare(full, a, "la"); m = meta[a]
        line = f"{a:18s} vs la {c['delta_pct']:6.2f}% W/T/L {c['WTL']:10s} p={c['p']:.4f} {c['tier']}  wall {m[0]/m[2]:6.1f}s rollouts {m[1]/m[2]:7.1f}"
        if a != MAIN:
            d = A.compare(full, a, MAIN); line += f" | vs main {d['delta_pct']:+.2f}% {d['WTL']} p={d['p']:.4f} {d['tier']}"
        print(line)
