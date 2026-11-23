"""Result figures for the second review round (2026-11-19), one shared style. Run from this folder.
Figure types follow the ASOC exemplars:
  pgnd_r2_radar      improvement over G-ECT by preference, radar chart (Cai et al. 2026, Fig. 11)
  pgnd_r2_wtl        win/tie/loss of PGND against every rule and DRL method (Lei et al. 2026, win-rate figures)
  pgnd_r2_solutions  solutions of PGND, G-ECT and the MOEAs in the makespan-energy plane (Yuan et al. 2025, Fig. 8; Cai et al. 2026, Fig. 10)
  pgnd_r2_gantt      Gantt charts of G-ECT and PGND on one case, machines and AGVs (4 of the 5 ASOC exemplars)
  pgnd_r2_budget     improvement against rollouts per case on the development seeds (parameter trend charts)
Single-column figures are 3.4 in wide and full-width figures 7.0 in wide, so that all text is 7-8 pt in print."""
import contextlib
import glob
import importlib.util
import io
import json
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

# load only the data helpers of make_tables_v9.py (V, cases, compare); stop before it writes any table file
_src = open("make_tables_v9.py", encoding="utf-8").read()
_src = _src[:_src.index('open("tables_v9_rules.tex", "w")')]
T = type("T", (), {})()
_ns = {"__name__": "make_tables_v9_helpers"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(_src, "make_tables_v9.py", "exec"), _ns)
T.V, T.cases, T.compare = _ns["V"], _ns["cases"], _ns["compare"]

# ---------------------------------------------------------------- shared style
FRAME = "#333333"
plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.6, "axes.edgecolor": FRAME, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.color": FRAME,
    "ytick.color": FRAME, "xtick.labelcolor": "black", "ytick.labelcolor": "black", "legend.handlelength": 2.2,
    "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False, "grid.color": "#dddddd",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
PGND = "#d62728"                                   # PGND in every figure
SETC = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]  # the four test sets
OTHERS = {"G-ECT": "#000000", "NSGA-II": "#1f77b4", "NSGA-III": "#2ca02c", "MOEA/D": "#ff7f0e"}
WIN, TIE, LOSS = "#2F5D8A", "#E6E0D2", "#C9A66B"  # win/tie/loss bars: navy, light sand, sand (author choice 2026-11-22)
SINGLE, FULL = 3.5, 7.2
SETS = [("syn", "Synthetic"), ("mk1", "Mk1"), ("edata7", "Edata7"), ("large", "Larger")]
PREFS = ["Balanced", "Makespan", "Tardiness", "Energy", "Path pressure", "Stability"]


def title(ax, k, text):
    ax.set_title(f"({chr(97 + k)}) {text}", loc="center", fontweight="bold")



# ---------------------------------------------------------------- radar by preference
ang = np.linspace(0, 2 * np.pi, len(PREFS), endpoint=False)
fig = plt.figure(figsize=(2.7, 2.45))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
vmax = 0
for si, (S, nm) in enumerate(SETS):
    vals = []
    for pi in range(6):
        keys = [k for k in T.cases(S, None, ("pgnd", "rule-lookahead")) if k[1].endswith(f"pref{pi:02d}")]
        vals.append(T.compare(keys, "pgnd", "rule-lookahead")["delta"])
    vmax = max(vmax, max(vals))
    v = np.r_[vals, vals[0]]
    a = np.r_[ang, ang[0]]
    ax.plot(a, v, color=SETC[si], linewidth=1.2, marker="o", markersize=3, label=nm)
    ax.fill(a, v, color=SETC[si], alpha=0.06)
ax.set_xticks(ang, PREFS)
ax.tick_params(axis="x", pad=4, labelsize=7.5)
ax.set_ylim(0, 20 * np.ceil(vmax / 20))
ax.set_yticks(np.arange(20, ax.get_ylim()[1], 20))  # no label on the outer ring, which would touch the axis labels
ax.set_yticklabels([f"{int(t)}%" for t in ax.get_yticks()], fontsize=6.5, color="#333333")
ax.set_rlabel_position(30)
ax.grid(color="#dddddd", linewidth=0.6)
ax.spines["polar"].set_color(FRAME)
ax.spines["polar"].set_linewidth(0.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, handlelength=1.1, columnspacing=0.7, handletextpad=0.3, fontsize=7)
fig.savefig("figures/pgnd_r2_radar.pdf")
plt.close(fig)

# ---------------------------------------------------------------- win/tie/loss
METHODS = [("rule-lookahead", "G-ECT"), ("rule-gest", "G-EST"), ("rule-mwkr_eat", "MWKR-EAT"), ("rule-spt", "SPT"),
           ("rule-edd", "EDD"), ("rule-fifo", "FIFO"), ("rule-mwkr", "MWKR"), ("rule-random", "Random"),
           ("daniel", "DANIEL"), ("song", "DRL-S"), ("hgt", "HGT"), ("dqn", "DQN")]
fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.7), sharey=True)
y = np.arange(len(METHODS))[::-1]
for k, (ax, (S, nm)) in enumerate(zip(axes, SETS)):
    for yi, (a, lab) in zip(y, METHODS):
        keys = T.cases(S, None, ("pgnd", a))
        c = T.compare(keys, "pgnd", a)
        n = c["n"]
        w, t, l = 100 * c["w"] / n, 100 * c["t"] / n, 100 * c["l"] / n
        ax.barh(yi, w, color=WIN, height=0.7, linewidth=0)
        ax.barh(yi, t, left=w, color=TIE, height=0.7, linewidth=0)
        ax.barh(yi, l, left=w + t, color=LOSS, height=0.7, linewidth=0)
        ax.text(w - 1.5, yi, f"{w:.0f}", va="center", ha="right", fontsize=6, color="white")
    ax.axvline(50, color="#9a9a9a", linewidth=0.5, linestyle=(0, (3, 2)))
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    title(ax, k, nm)
    ax.set_xlabel("Share of cases (%)")
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(y, [lab for _, lab in METHODS])
fig.legend(handles=[Patch(color=WIN, label="PWMD wins"), Patch(color=TIE, label="Tie"), Patch(color=LOSS, label="PWMD loses")],
           loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3)
fig.subplots_adjust(wspace=0.12, bottom=0.24)
fig.savefig("figures/pgnd_r2_wtl.pdf")
plt.close(fig)

# ---------------------------------------------------------------- solutions in the makespan-energy plane
FRONT = "../rerun_twoleg_early/test_v3/front"
pts = {}
for f in glob.glob(f"{FRONT}/det_800.jsonl") + glob.glob(f"{FRONT}/ea_*_run0_800.jsonl"):
    for l in open(f):
        r = json.loads(l)
        pts.setdefault(r["instance"], {}).setdefault(r["arm"], []).append(r["raw"])
SHOW = [("pgnd", "PWMD", PGND, "o", 18), ("rule-lookahead", "G-ECT", OTHERS["G-ECT"], "x", 16),
        ("ea-nsga2-run0", "NSGA-II", OTHERS["NSGA-II"], "^", 14), ("ea-nsga3-run0", "NSGA-III", OTHERS["NSGA-III"], "s", 12),
        ("ea-moead-run0", "MOEA/D", OTHERS["MOEA/D"], "D", 11)]
INST = [("syn_seed800_L4_raw", "Moderate setting,\nresource events only"), ("syn_seed800_L4_realized", "Moderate setting,\nall events"),
        ("syn_seed800_L5_raw", "Severe setting,\nresource events only"), ("syn_seed800_L5_realized", "Severe setting,\nall events")]
fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.1))
for k, (ax, (inst, nm)) in enumerate(zip(axes, INST)):
    for arm, lab, col, mk, sz in SHOW:
        P = np.array(pts[inst][arm])
        kw = dict(color=col, s=sz, marker=mk, label=lab, zorder=4 if arm == "pgnd" else 3)
        if mk != "x":
            kw.update(edgecolors="white", linewidths=0.4)
        else:
            kw.update(linewidths=0.9)
        ax.scatter(P[:, 0], P[:, 2], **kw)
    title(ax, k, nm)
    ax.set_xlabel("Makespan")
axes[0].set_ylabel("Energy consumption")
h, lb = axes[0].get_legend_handles_labels()
fig.legend(h, lb, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, handletextpad=0.3, columnspacing=1.2)
fig.subplots_adjust(wspace=0.42, bottom=0.27)
fig.savefig("figures/pgnd_r2_solutions.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Gantt charts
G = json.load(open("../rerun_twoleg_early/test_v3/gantt_case_s800.json"))
jobs = sorted({o["job"] for s in G["schedules"].values() for o in s["operations"]}, key=lambda j: int(j[1:]))
cmap = plt.get_cmap("tab20")
JC = {j: cmap((2 * i) % 20 if i < 10 else (2 * (i - 10) + 1) % 20) for i, j in enumerate(jobs)}
machines = sorted({o["machine"] for s in G["schedules"].values() for o in s["operations"]}, key=lambda m: int(m[1:]))
agvs = sorted({t["agv"] for s in G["schedules"].values() for t in s["transports"]}, key=lambda a: int(a[1:]))
rows = [("AGV", a) for a in agvs[::-1]] + [("M", m) for m in machines[::-1]]
one = lambda x: x[0] + str(int(x[1:]) + 1)   # 1-based labels, as in Fig. 1 and the model
ry = {r: i for i, r in enumerate(rows)}
tmax = max(o["end"] for s in G["schedules"].values() for o in s["operations"])
fig, axes = plt.subplots(2, 1, figsize=(FULL, 4.3), sharex=True)
for k, (ax, (arm, nm)) in enumerate(zip(axes, [("rule-lookahead", "G-ECT"), ("pgnd", "PWMD")])):
    s = G["schedules"][arm]
    for e in G["events"]:
        if e["duration"] <= 0:
            continue
        ent = e["entity"]
        key = ("M", ent) if ent in machines else (("AGV", ent) if ent in agvs else None)
        if key:
            ax.add_patch(Rectangle((e["time"], ry[key] - 0.45), e["duration"], 0.9, facecolor="none", edgecolor="#000000",
                                   hatch="////", linewidth=0.5, zorder=1))
    for o in s["operations"]:
        ax.barh(ry[("M", o["machine"])], o["end"] - o["start"], left=o["start"], height=0.72, color=JC[o["job"]],
                edgecolor="white", linewidth=0.4, zorder=2)
        if o["end"] - o["start"] >= 11:
            ax.text((o["start"] + o["end"]) / 2, ry[("M", o["machine"])], one(o["job"]), ha="center", va="center", fontsize=6.2, zorder=3)
    for t in s["transports"]:
        if t["end"] - t["start"] <= 0:
            continue
        y_agv = ry[("AGV", t["agv"])]
        # travel legs solid; waiting at the pickup node between them light (2026-11-22: waiting is not travel)
        for a0, a1, alpha in ((t["start"], t["empty_end"], 1.0), (t["empty_end"], t["loaded_start"], 0.28),
                              (t["loaded_start"], t["end"], 1.0)):
            if a1 > a0:
                ax.barh(y_agv, a1 - a0, left=a0, height=0.5, color=JC[t["job"]], alpha=alpha,
                        edgecolor="white", linewidth=0.3, zorder=2)
    mk = max(o["end"] for o in s["operations"])
    ax.axvline(mk, color=PGND if arm == "pgnd" else "#3c3c3c", linewidth=0.8, linestyle=(0, (4, 2)))
    ax.text(mk - 3, len(rows) - 0.6, f"$C_{{\\max}}$ = {mk}", fontsize=7, va="top", ha="right",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=4)  # left of the line, inside the frame
    ax.axhline(len(agvs) - 0.5, color="#9a9a9a", linewidth=0.5)
    ax.set_yticks(range(len(rows)), [one(r[1]) for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    title(ax, k, f"{nm} ($V$ = {s['value']:.4f})")
    ax.tick_params(axis="y", length=0)
axes[1].set_xlim(0, tmax * 1.06)
axes[1].set_xlabel("Time")
handles = [Patch(facecolor=JC[j], label=one(j)) for j in jobs] + [Patch(facecolor="none", edgecolor="#000000", hatch="////", label="Event window")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=len(handles), handlelength=1.0,
           handletextpad=0.3, columnspacing=0.7)
fig.subplots_adjust(hspace=0.28, bottom=0.14)
fig.savefig("figures/pgnd_r2_gantt.pdf")
plt.close(fig)

# ---------------------------------------------------------------- rollout budget
dev = {}
for S in ("syn", "mk1", "edata7"):
    files = glob.glob(f"../rerun_twoleg_early/dev_v3/runs_Kk/{S}_*.jsonl")
    if S == "syn":
        files += glob.glob("../rerun_twoleg_early/dev_v3/runs_M/*.jsonl")
    v, roll = {}, {}
    for f in files:
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                v.setdefault(r["case_id"], {})[r["arm"]] = r["value"]
                roll.setdefault(r["arm"], []).append(r["rollouts"])
    dev[S] = (v, roll)
ARMS = [("pgnd_M0.02_K4_k1", "$K$=4, $k$=1"), ("pgnd_M0.02_K4_k2", "$K$=4, $k$=2"), ("pgnd_M0.02_K8_k1", "$K$=8, $k$=1"),
        ("pgnd_M0.02_K8_k2", "$K$=8, $k$=2")]
# grouped bars in the style of Fig. (b) of pgnd_sens: one group per development set, the final setting in red
BARC = ["#d9d9d9", "#a6a6a6", "#595959", PGND]
fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
wbar = 0.2
budget_log = {}
for si, (S, nm) in enumerate(SETS[:3]):
    v, roll = dev[S]
    full = [c for c, x in v.items() if "la" in x and all(a in x for a, _ in ARMS)]
    for ai, (a, lab) in enumerate(ARMS):
        y = 100 * (1 - mean(v[c][a] for c in full) / mean(v[c]["la"] for c in full))
        budget_log[(S, a)] = (round(y, 2), round(mean(roll[a])))
        ax.bar(si + (ai - 1.5) * wbar, y, wbar, color=BARC[ai], edgecolor="#333333", linewidth=0.4,
               label=lab if si == 0 else None, zorder=2)
ax.set_xticks(range(3))
ax.set_xticklabels([nm for _, nm in SETS[:3]])
ax.set_xlim(-0.55, 2.55)
ax.set_ylabel("Improvement over G-ECT (%)")
ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
ax.set_ylim(0, 19.5)
ax.legend(ncol=2, loc="upper right", handlelength=1.2, columnspacing=0.8, fontsize=7.5)
print("budget", budget_log)
fig.savefig("figures/pgnd_r2_budget.pdf")
plt.close(fig)
print("saved 5 figures")
