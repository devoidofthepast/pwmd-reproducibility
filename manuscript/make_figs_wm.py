"""Additional result figures (2026-11-21), same style and data helpers as make_figs_r2.py. Run from this folder.
Figure types follow the ASOC exemplars:
  pgnd_dist   per-case improvement over G-ECT of each method, box plots (Shen et al. 2026, Fig. 8; Cai et al. 2026, Figs. 9 and 12)
  pgnd_sens   sensitivity to M, ensemble size and training seeds, and the event model (Cai et al. 2026, Fig. 7; Xiao et al. 2026, Fig. 10)
  pgnd_train  training losses of the five ensemble members (Yuan 2025, Lei 2026, Xiao 2026, Shen 2026)
pgnd_sens takes its numbers from tables_v9_sensitivity.tex, which is generated from the frozen runs; pgnd_train reads
../rerun_twoleg_early/training_curves_v3 (re-run of the frozen training with loss logging, parity checked)."""
import json
import re
from pathlib import Path

_src = open("make_figs_r2.py", encoding="utf-8").read()
exec(compile(_src[:_src.index("# ---------------------------------------------------------------- radar by preference")], "make_figs_r2_style", "exec"))

import sys
which = sys.argv[1:] or ["dist", "sens", "train"]

# ---------------------------------------------------------------- per-case improvement distribution
if "dist" in which:
    METH = [("rule-gest", "G-EST"), ("rule-mwkr_eat", "MWKR-EAT"), ("daniel", "DANIEL"), ("song", "DRL-S"), ("hgt", "HGT"),
            ("dqn", "DQN"), ("pgnd", "PWMD")]
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 2.35), sharey=False)
    for si, (S, nm) in enumerate(SETS):
        ax = axs[si]
        data, labs = [], []
        for arm, lab in METH:
            keys = [k for k in T.cases(S, None, (arm, "rule-lookahead")) if T.V[k]["rule-lookahead"] > 1e-9]  # G-ECT value 0: ratio undefined
            if not keys:
                continue
            data.append([100 * (1 - T.V[k][arm] / T.V[k]["rule-lookahead"]) for k in keys])
            labs.append(lab)
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.6, patch_artist=True,
                        medianprops=dict(color="black", linewidth=0.9), whiskerprops=dict(linewidth=0.6, color=FRAME),
                        capprops=dict(linewidth=0.6, color=FRAME), boxprops=dict(linewidth=0.6, color=FRAME))
        for b, lab in zip(bp["boxes"], labs):
            b.set_facecolor(PGND if lab == "PWMD" else "#d9d9d9")
            b.set_alpha(0.85 if lab == "PWMD" else 1.0)
        for i, d in enumerate(data):
            ax.plot(i + 1, np.mean(d), marker="^", color="white", markeredgecolor="black", markersize=3.5, markeredgewidth=0.6, zorder=5)
        ax.axhline(0, color="#777777", linewidth=0.6, linestyle=(0, (3, 2)))
        ax.set_xticks(range(1, len(labs) + 1), labs, rotation=60, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=1)
        title(ax, si, nm)
        if si == 0:
            ax.set_ylabel("Improvement over G-ECT (%)")
    fig.tight_layout(w_pad=0.8)
    fig.savefig("figures/pgnd_dist.pdf")
    plt.close(fig)

# ---------------------------------------------------------------- sensitivity
if "sens" in which:
    tex = Path("tables_v9_sensitivity.tex").read_text(encoding="utf-8")

    def row(label):
        line = next(l for l in tex.splitlines() if l.strip().startswith(label))
        return [float(x) if x != "--" else None for x in re.findall(r"(-?\d+\.\d+|--)\\?%?", line.split("&", 1)[1])]

    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw=dict(width_ratios=[1, 1.15, 1.25]))
    # (a) margin M, synthetic development seeds, K = 8, k = 1
    Ms = ["0", "0.002", "0.005", "0.01", "0.02", "0.05", "0.10"]
    ys = [row(f"$M={m}$")[0] for m in Ms]
    ax = axs[0]
    ax.plot(range(len(Ms)), ys, color=PGND, marker="o", markersize=3, linewidth=1.1)
    ax.set_xticks(range(len(Ms)), Ms, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Acceptance margin $M$")
    ax.set_ylabel("Improvement over G-ECT (%)")
    ax.set_ylim(0, 18)
    title(ax, 0, "Margin")
    # (b) ensemble size and training seeds, three development sets
    cfg = [("1 member", "1 member"), ("3 members", "3 members"), (r"\textbf{$K=8$, $k=2$}", "5 members"), ("5 training seeds", "5 train. seeds")]
    vals = [row(lab)[:3] for lab, _ in cfg]
    ax = axs[1]
    w = 0.2
    shades = ["#d9d9d9", "#a6a6a6", PGND, "#595959"]
    for ci, ((_, lab), v) in enumerate(zip(cfg, vals)):
        ax.bar(np.arange(3) + (ci - 1.5) * w, v, width=w, color=shades[ci], edgecolor=FRAME, linewidth=0.4, label=lab)
    ax.set_xticks(range(3), ["Synthetic", "Mk1", "Edata7"])
    ax.set_ylim(0, 22)
    ax.legend(loc="upper right", ncol=2, fontsize=6.5, handlelength=1.0, columnspacing=0.6, borderaxespad=0.2)
    title(ax, 1, "Ensemble and training seeds")
    # (c) event model of the scenarios, synthetic test set
    ev = [("Event process of the generator", "Generator"), (r"Durations $\times0.5$", r"Durations $\times$0.5"),
          (r"Durations $\times1.5$", r"Durations $\times$1.5"), ("Event horizon $[5,90]$", "Horizon [5, 90]"),
          ("Event horizon $[5,160]$", "Horizon [5, 160]"), (r"Event rate $\times0.5$", r"Rate $\times$0.5"), (r"Event rate $\times1.5$", r"Rate $\times$1.5")]
    ev_v = [row(lab)[0] for lab, _ in ev]
    ax = axs[2]
    yy = np.arange(len(ev))[::-1]
    ax.hlines(yy, 0, ev_v, color="#bbbbbb", linewidth=0.8)
    ax.plot(ev_v, yy, "o", color="#595959", markersize=3.5)
    ax.plot(ev_v[0], yy[0], "o", color=PGND, markersize=4)
    ax.set_yticks(yy, [l for _, l in ev])
    ax.set_xlim(0, 18)
    ax.set_xlabel("Improvement over G-ECT (%)")
    title(ax, 2, "Event model of the scenarios")
    fig.tight_layout(w_pad=0.9)
    fig.savefig("figures/pgnd_sens.pdf")
    plt.close(fig)

# ---------------------------------------------------------------- training curves
if "train" in which:
    D = Path("../rerun_twoleg_early/training_curves_v3")
    curves = [json.loads(p.read_text()) for p in sorted(D.glob("curve_*.json"))]
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.2))
    greys = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    for ci, c in enumerate(curves):
        e = np.arange(1, len(c["epoch_loss"]) + 1)
        axs[0].plot(e, c["epoch_loss"], color=greys[ci], linewidth=0.9, label=f"Member {ci + 1}")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Training loss")
    axs[0].legend(ncol=1, fontsize=6.5, handlelength=1.4)
    title(axs[0], 0, "Total loss")
    terms = np.mean([np.asarray(c["term_loss"]) for c in curves], axis=0)
    names = [r"$\mathcal{L}_{\mathrm{reg}}$", r"$\mathcal{L}_{\mathrm{pos}}$", r"$\mathcal{L}_{\mathrm{pair}}$", r"$\mathcal{L}_{\mathrm{best}}$"]
    for ti in range(4):
        axs[1].plot(np.arange(1, len(terms) + 1), terms[:, ti], color=SETC[ti], linewidth=0.9, label=names[ti])
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss term (mean of members)")
    axs[1].legend(ncol=2, fontsize=6.5, handlelength=1.4)
    title(axs[1], 1, "Loss terms")
    fig.tight_layout(w_pad=1.2)
    fig.savefig("figures/pgnd_train.pdf")
    plt.close(fig)
print("done", which)
