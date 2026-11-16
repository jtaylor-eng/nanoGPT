"""e006 figure: LAMBADA acc and ppl on the clean 125M runs (tag 125m_fw25_rng). Usage: python plot_lambada.py <repo root>"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1]); res = root / "results/lm1012/125m_fw25_rng"; out = Path(__file__).parent
MAPS = [("softmax_xs", "softmax", "#2a78d6"), ("asentmax_zi_xs", "ASEntmax", "#eb6834"),
        ("wstieltjes_q3_d2_xs", "wStieltjes\nq3/d2", "#1baf7a"), ("aswstieltjes_q3_d2_zi_xs", "AS-wStieltjes\nq3/d2", "#eda100"),
        ("wstieltjes_q3_d1.5_xs", "wStieltjes\nq3/d1.5", "#888888")]
RUNS = [("lr1.2e-3", "o", True, -0.12), ("lr1.2e-3_s1", "o", True, 0.0), ("lr1.2e-3_s2", "o", True, 0.12),
        ("lr6e-4", "v", False, 0.30), ("lr2.4e-3", "s", False, 0.42)]


def get(row, run):
    p = res / row / run / "lmeval.json"
    if not p.exists():
        return None
    L = json.load(open(p))["results"]["lambada_openai"]
    return 100 * L["acc,none"], 100 * L["acc_stderr,none"], L["perplexity,none"], L["perplexity_stderr,none"]


fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, k, ylab in [(axes[0], 0, "LAMBADA last-word accuracy (%)"), (axes[1], 2, "LAMBADA last-word perplexity")]:
    for i, (row, name, c) in enumerate(MAPS):
        seeds = []
        for run, m, filled, dx in RUNS:
            v = get(row, run)
            if v is None:
                continue
            if filled:
                seeds.append(v[k])
            ax.errorbar(i + dx, v[k], yerr=v[k + 1], fmt=m, ms=7, color=c, mfc=c if filled else "white",
                        mec=c, mew=1.5, capsize=2, lw=1, zorder=3)
        if len(seeds) == 3:
            mu = sum(seeds) / 3
            ax.hlines(mu, i - 0.22, i + 0.22, color="#333333", lw=2, zorder=4)
    ax.set_xticks(range(len(MAPS))); ax.set_xticklabels([m[1] for m in MAPS], fontsize=8)
    ax.set_ylabel(ylab); ax.grid(axis="y", color="#eeeeee"); ax.spines[["top", "right"]].set_visible(False)
h = [plt.Line2D([], [], marker="o", ls="", color="#555555", ms=7, label="lr 1.2e-3, seeds 1337/s1/s2 (left to right)"),
     plt.Line2D([], [], color="#333333", lw=2, label="3-seed mean"),
     plt.Line2D([], [], marker="v", ls="", color="#555555", mfc="white", ms=7, label="lr 6e-4"),
     plt.Line2D([], [], marker="s", ls="", color="#555555", mfc="white", ms=7, label="lr 2.4e-3")]
axes[0].legend(handles=h, fontsize=7.5, frameon=False, loc="upper right")
fig.suptitle("Zero-shot LAMBADA-openai (5153 items), clean 125M / T=2048 / 2.5B tok runs; bars = 1 SE (harness)", fontsize=10)
fig.tight_layout(); fig.savefig(out / "lambada_125m_clean.png", dpi=150)
