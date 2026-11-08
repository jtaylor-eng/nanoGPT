"""e002 softmax rows: train/val vs iter (epoch-boundary plateau) and loss vs eval length. Reads results over ssh."""
import json, subprocess, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/fs/scratch/PAS2836/jacktaylor/nanogpt_results/lm1012"
LRS = ["3e-4", "6e-4", "1.2e-3"]
SHADES = {"3e-4": "#9ecae1", "6e-4": "#4292c6", "1.2e-3": "#08306b"}  # sequential: LR magnitude

def cat(path):
    return subprocess.run(["ssh", "ascend-agent", f"cat {path}"], check=True, capture_output=True, text=True).stdout

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for lr in LRS:
    ev = [json.loads(l) for l in cat(f"{R}/350m/softmax_xs/lr{lr}/metrics.jsonl").splitlines() if '"eval"' in l]
    ev = [e for e in ev if e["iter"] >= 500]
    it = [e["iter"] for e in ev]
    ax[0].plot(it, [e["val_loss"] for e in ev], color=SHADES[lr], lw=2, label=f"val lr {lr}")
    ax[0].plot(it, [e["train_loss"] for e in ev], color=SHADES[lr], lw=1.2, ls="--", label=f"train lr {lr}")
ax[0].axvline(1907, color="0.5", lw=1, ls=":"); ax[0].text(1930, 4.1, "1 epoch of\nfineweb10B (1.0B tok)", fontsize=8, color="0.3")
ax[0].set(xlabel="iter", ylabel="loss", title="350M softmax_xs: val plateaus after epoch 1", ylim=(2.8, 4.4))
ax[0].legend(fontsize=7, ncol=2)

for k, (name, title) in enumerate([("lengths_16k", "FineWeb val"), ("arxiv", "ArXiv")]):
    a = ax[k + 1]
    for lr in LRS:
        L = json.loads(cat(f"{R}/350m/softmax_xs/lr{lr}/{name}.json"))["lengths"]
        xs = sorted(L, key=int)
        a.plot([int(x) for x in xs], [L[x]["val_loss"] for x in xs], "o-", ms=5, lw=2, color=SHADES[lr], label=f"350M lr {lr}")
    L = json.loads(cat(f"{R}/125m/softmax_xs/lr1.2e-3/{name}.json"))["lengths"]
    xs = sorted(L, key=int)
    a.plot([int(x) for x in xs], [L[x]["val_loss"] for x in xs], "s--", ms=5, lw=1.5, color="0.45", label="125M lr 1.2e-3")
    a.axvline(2048, color="0.7", lw=1, ls=":")
    a.set(xscale="log", xlabel="eval length (tokens)", ylabel="loss", title=f"{title}: loss vs length (seed 1337)")
    a.set_xticks([512, 1024, 2048, 4096, 8192, 16384]); a.set_xticklabels(["512", "1k", "2k", "4k", "8k", "16k"])
    a.legend(fontsize=7)
for a in ax: a.grid(alpha=0.25)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "e002_softmax.png"
fig.savefig(out, dpi=130); print("wrote", out)
