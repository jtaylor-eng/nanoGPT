"""e003: RNG replay diagnosis. Left: periodic train/val loss vs iter with segment boundaries (350M softmax 6e-4,
125M softmax 1.2e-3). Right: loss vs eval length, 350M softmax vs 125M softmax (fineweb25B). Reads local results/lm1012."""
import json, os, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "../../../../results/lm1012")
RUNS = [("350m_fw25/softmax_xs/lr6e-4", "350M softmax lr 6e-4", "#4c72b0"),
        ("350m_fw25/softmax_xs/lr3e-4", "350M softmax lr 3e-4", "#8172b3"),
        ("125m_fw25/softmax_xs/lr1.2e-3", "125M softmax lr 1.2e-3", "#dd8452")]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for p, lab, c in RUNS:
    d = f"{ROOT}/{p}"
    ev = [json.loads(l) for l in open(f"{d}/metrics.jsonl")]
    ev = [j for j in ev if j["kind"] == "eval" and j["iter"] > 0]
    it = [j["iter"] for j in ev]
    ax[0].plot(it, [j["val_loss"] for j in ev], color=c, lw=2, label=f"{lab} val")
    ax[0].plot(it, [j["train_loss"] for j in ev], color=c, lw=1, ls="--", label=f"{lab} train")
    if os.path.exists(f"{d}/run.log"):
        for b in map(int, re.findall(r"wall limit reached at iter (\d+)", open(f"{d}/run.log").read())):
            ax[0].axvline(b, color=c, lw=0.6, alpha=0.4)
    p16 = f"{d}/lengths_16k.json"
    if os.path.exists(p16):
        L = json.load(open(p16))["lengths"]; xs = sorted(L, key=int)
        ax[1].plot([int(x) for x in xs], [L[x]["val_loss"] for x in xs], color=c, marker="o", ms=4, label=lab)
ax[0].set(ylim=(2.9, 4.2), xlabel="iter (vertical lines: segment resumes, each replays segment 1's batches)",
          ylabel="loss (12-24 iter estimate)", title="train vs val, fineweb25B, seed 1337")
ax[0].legend(fontsize=7); ax[0].grid(alpha=0.25)
ax[1].axvline(2048, color="0.7", lw=1, ls=":")
ax[1].set(xscale="log", xlabel="eval length (tokens)", ylabel="FineWeb val loss", title="loss vs length")
ax[1].set_xticks([512, 1024, 2048, 4096, 8192, 16384]); ax[1].set_xticklabels(["512", "1k", "2k", "4k", "8k", "16k"])
ax[1].legend(fontsize=7); ax[1].grid(alpha=0.25)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "e003_replay.png"
fig.savefig(out, dpi=130); print("wrote", out)
