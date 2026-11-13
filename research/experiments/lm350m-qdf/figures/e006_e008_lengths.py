"""e006-e008: 125M loss vs eval length on fineweb25B without replay (125m_fw25_rng). Reads local results/lm1012.
Top row (e006): 3 seeds {1337, 1, 2} at lr 1.2e-3, mean line + min-max band, thin lines = single seeds.
Bottom row (e007, e008): seed 1337 only; lr 2.4e-3 (dashed) and the two Stieltjes variants vs the lr 1.2e-3 rows."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "../../../../results/lm1012/125m_fw25_rng")
SETS = [("lengths_16k", "FineWeb val"), ("arxiv", "ArXiv"), ("pg19", "PG-19")]
TICKS = [512, 1024, 2048, 4096, 8192, 16384]
C = {"softmax_xs": "#4c72b0", "asentmax_zi_xs": "#dd8452", "wstieltjes_q3_d2_xs": "#55a868",
     "aswstieltjes_q3_d2_zi_xs": "#8172b3", "wstieltjes_q3_d1.5_xs": "#937860"}
NAME = {"softmax_xs": "softmax", "asentmax_zi_xs": "ASEntmax", "wstieltjes_q3_d2_xs": "wStieltjes q3/d2",
        "aswstieltjes_q3_d2_zi_xs": "AS-wStieltjes q3/d2", "wstieltjes_q3_d1.5_xs": "wStieltjes q3/d1.5"}


def curve(row, sub, name):
    L = json.load(open(f"{ROOT}/{row}/{sub}/{name}.json"))["lengths"]
    xs = sorted(L, key=int)
    return np.array([int(x) for x in xs]), np.array([L[x]["val_loss"] for x in xs])


fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
for a, (name, title) in zip(ax[0], SETS):
    for row in ["softmax_xs", "asentmax_zi_xs", "wstieltjes_q3_d2_xs"]:
        ys = []
        for sub in ["lr1.2e-3", "lr1.2e-3_s1", "lr1.2e-3_s2"]:
            x, y = curve(row, sub, name); ys.append(y)
            a.plot(x, y, color=C[row], lw=0.7, alpha=0.55)
        ys = np.array(ys)
        a.fill_between(x, ys.min(0), ys.max(0), color=C[row], alpha=0.18, lw=0)
        a.plot(x, ys.mean(0), color=C[row], lw=2, marker="o", ms=4, label=f"{NAME[row]} (mean of 3 seeds, band = range)")
    a.set_title(f"e006: 125M {title}, lr 1.2e-3, 3 seeds")
for a, (name, title) in zip(ax[1], SETS):
    for row, sub, ls, mk, lw in [
        ("softmax_xs", "lr1.2e-3", "-", "o", 1.6), ("softmax_xs", "lr2.4e-3", "--", "o", 1.6),
        ("asentmax_zi_xs", "lr1.2e-3", "-", "o", 1.6),
        ("wstieltjes_q3_d2_xs", "lr1.2e-3", "-", "o", 1.6), ("wstieltjes_q3_d2_xs", "lr2.4e-3", "--", "o", 1.6),
        ("aswstieltjes_q3_d2_zi_xs", "lr1.2e-3", "-", "s", 2), ("wstieltjes_q3_d1.5_xs", "lr1.2e-3", "-", "^", 2),
    ]:
        x, y = curve(row, sub, name)
        a.plot(x, y, color=C[row], ls=ls, marker=mk, ms=4, lw=lw, label=f"{NAME[row]} lr {sub[2:]}")
    a.set_title(f"e007 / e008: 125M {title}, seed 1337")
for a in ax.ravel():
    a.axvline(2048, color="0.7", lw=1, ls=":")
    a.set(xscale="log", xlabel="eval length (tokens)", ylabel="loss")
    a.set_xticks(TICKS); a.set_xticklabels(["512", "1k", "2k", "4k", "8k", "16k"]); a.minorticks_off()
    a.grid(alpha=0.25)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
ax[0, 0].legend(fontsize=7, loc="upper center"); ax[1, 0].legend(fontsize=7, handlelength=3.5, loc="upper center", title="ASEntmax lr 2.4e-3 diverged (no curve)", title_fontsize=7)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "e006_e008_lengths.png")
fig.savefig(out, dpi=130); print("wrote", out)
