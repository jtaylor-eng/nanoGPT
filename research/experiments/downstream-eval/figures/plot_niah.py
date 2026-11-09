"""e005 figures: NIAH accuracy vs context length, and by depth bin at 8k. Usage: python plot_niah.py <repo root>"""
import json, sys, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1]); res = root / "results/lm1012/125m"; out = Path(__file__).parent
LENS = ["2048", "4096", "8192", "16384"]
PRIMARY = [("softmax_xs", "softmax", "#2a78d6"), ("asentmax_zi_xs", "ASEntmax", "#eb6834"),
           ("wstieltjes_q3_d2_xs", "wStieltjes q3/d2", "#1baf7a")]
OTHER = ["wstieltjes_q3_d3_xs", "wstieltjes_q3_d4_xs", "wstieltjes_q4_d2_xs", "wstieltjes_q4_d3_xs"]
load = lambda row, lr: json.load(open(res / row / f"lr{lr}" / "niah.json"))["lengths"]
se = lambda p, n: 100 * math.sqrt(max(p * (1 - p), 1e-9) / n)
x = [int(l) for l in LENS]

fig, ax = plt.subplots(figsize=(7, 4.5))
for row in OTHER:
    L = load(row, "1.2e-3")
    ax.plot(x, [100 * L[l]["acc"] for l in LENS], color="#bbbbbb", lw=1, zorder=1)
ax.plot([], [], color="#bbbbbb", lw=1, label="other wStieltjes (q3/d3, q3/d4, q4/d2, q4/d3), lr 1.2e-3")
for row, name, c in PRIMARY:
    for lr, ls in [("1.2e-3", "-"), ("6e-4", "--")]:
        L = load(row, lr)
        y = [100 * L[l]["acc"] for l in LENS]
        ax.errorbar(x, y, yerr=[se(L[l]["acc"], L[l]["n"]) for l in LENS], color=c, ls=ls, lw=2,
                    marker="o", ms=6, capsize=3, label=f"{name}, lr {lr}", zorder=3)
        if lr == "1.2e-3":
            ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color="#333333")
ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels(["2k", "4k", "8k", "16k"])
ax.axvline(2048, color="#dddddd", lw=1, zorder=0)
ax.set_xlabel("context length (train T = 2048)"); ax.set_ylabel("single-needle exact match (%)")
ax.set_title("S-NIAH, 125M / 2.5B tok, seed 1337, n = 200 per length (bars = 1 SE)", fontsize=10)
ax.grid(axis="y", color="#eeeeee"); ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=7.5, frameon=False); fig.tight_layout(); fig.savefig(out / "niah_vs_length.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4))
for row, name, c in PRIMARY:
    for lr, ls in [("1.2e-3", "-"), ("6e-4", "--")]:
        b = load(row, lr)["8192"]["acc_by_bin"]
        ax.plot([10 * i + 5 for i in range(len(b))], [100 * v for v in b], color=c, ls=ls, lw=2, marker="o", ms=6,
                label=f"{name}, lr {lr}")
ax.set_xlabel("needle depth (% of context, bin centre)"); ax.set_ylabel("exact match at 8k (%)")
ax.set_title("S-NIAH by depth at 8192, n = 20 per bin (SE up to 11 pt)", fontsize=10)
ax.grid(axis="y", color="#eeeeee"); ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=7.5, frameon=False); fig.tight_layout(); fig.savefig(out / "niah_depth.png", dpi=150)
