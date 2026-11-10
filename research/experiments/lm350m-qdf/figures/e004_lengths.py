"""e004: 125M loss vs eval length, fineweb25B (1 epoch) runs vs the old 2.5-epoch fineweb10B runs. Reads local results/lm1012."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "../../../../results/lm1012")
ROWS = ["softmax_xs", "asentmax_zi_xs", "wstieltjes_q3_d2_xs"]
COLOR = {"softmax_xs": "#4c72b0", "asentmax_zi_xs": "#dd8452", "wstieltjes_q3_d2_xs": "#55a868"}
LRS = ["6e-4", "1.2e-3"]
SETS = [("lengths_16k", "FineWeb val"), ("arxiv", "ArXiv"), ("pg19", "PG-19")]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, (name, title) in zip(ax, SETS):
    for row in ROWS:
        for lr in LRS:
            for tag, ls, mk, lab in [("125m_fw25", "-", "o", "1 ep fw25B"), ("125m", ":", "x", "2.5 ep fw10B")]:
                p = f"{ROOT}/{tag}/{row}/lr{lr}/{name}.json"
                if not os.path.exists(p):
                    continue
                L = json.load(open(p))["lengths"]
                xs = sorted(L, key=int)
                a.plot([int(x) for x in xs], [L[x]["val_loss"] for x in xs], ls=ls, marker=mk, ms=4,
                       lw=2 if lr == "1.2e-3" else 1, color=COLOR[row], alpha=1 if lr == "1.2e-3" else 0.6,
                       label=f"{row} lr {lr} ({lab})")
    a.axvline(2048, color="0.7", lw=1, ls=":")
    a.set(xscale="log", xlabel="eval length (tokens)", ylabel="loss", title=f"125M {title} (seed 1337)")
    a.set_xticks([512, 1024, 2048, 4096, 8192, 16384]); a.set_xticklabels(["512", "1k", "2k", "4k", "8k", "16k"])
    a.grid(alpha=0.25)
ax[0].legend(fontsize=6)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "e004_lengths.png"
fig.savefig(out, dpi=130); print("wrote", out)
