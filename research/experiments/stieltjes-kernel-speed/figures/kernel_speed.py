"""figures/kernel_speed.png: s/step of the 125M step, e009 part a (job 8619198, one node, a0139).

    ~/.venvs/nanogpt/bin/python research/experiments/stieltjes-kernel-speed/figures/kernel_speed.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
S = json.load(open(os.path.join(ROOT, "results/lm1012/kernel_speed/e009/a/step_summary.json")))

# (label, from-scratch run, trained run)
ROWS = [
    ("softmax_xs\nSDPA + NAPE bias, mb 16 x 16", "softmax_mb16", None),
    ("asentmax_zi_xs\nentmax kernel, mb 8 x 32", "asentmax_scratch", "asentmax_trained"),
    ("wstieltjes_q3_d2_xs\nbaseline kernel, mb 8 x 32", "wst_q3d2_scratch_ref", "wst_q3d2_trained_ref"),
    ("wstieltjes_q3_d2_xs\nfinal kernel, mb 8 x 32", "wst_q3d2_scratch_cand", "wst_q3d2_trained_cand"),
]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_SCRATCH, C_TRAINED = "#2a78d6", "#eb6834"
H = 0.26

fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for i, (label, scratch, trained) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    bars = [(scratch, C_SCRATCH)] + ([(trained, C_TRAINED)] if trained else [])
    offs = [H / 2 + 0.02, -H / 2 - 0.02] if trained else [0.0]
    for (key, col), off in zip(bars, offs):
        v = S[key]["s_per_step"]
        ax.barh(y + off, v, height=H, color=col, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        ax.text(v + 0.12, y + off, f"{v:.2f}", va="center", ha="left", fontsize=9, color=INK)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r[0] for r in reversed(ROWS)], fontsize=9, color=INK)
ax.set_xlim(0, 15.2)
ax.set_xlabel("s per step (median of 27 steps, 524,288 tokens per step)", fontsize=9, color=MUTED)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(MUTED)
ax.spines["left"].set_linewidth(0.8)
handles = [plt.Rectangle((0, 0), 1, 1, color=C_SCRATCH), plt.Rectangle((0, 0), 1, 1, color=C_TRAINED)]
ax.legend(handles, ["from scratch (iter 0)", "trained (resumed at iter 4771, lr 1.2e-3, seed 1337)"],
          loc="lower left", bbox_to_anchor=(-0.01, 0.99), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
fig.suptitle("125M / T=2048 training step on one A100 40 GB: Stieltjes kernel before and after",
             x=0.015, ha="left", fontsize=11, color=INK, y=0.975)
fig.text(0.015, 0.895, "e009 part a, job 8619198, node a0139, bf16, torch.compile, num_iter 15; outputs and gradients "
         "bit-identical to the baseline", ha="left", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.89))
out = os.path.join(HERE, "kernel_speed.png")
fig.savefig(out, facecolor=SURFACE)
print(out)
