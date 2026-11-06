---
slug: attn-viz
title: "Where does attention mass go at 16k? Support size and entropy per head vs position, and NoPE attention rows, per map (existing 125M checkpoints)"
project: nanogpt
cluster: ascend
status: active
wake: now
budget_gpu_hours: 30
spent_gpu_hours: 0
max_jobs_in_flight: 8
jobs: []
jobs_archived: []
last_wake: null
next_id: 1
model: null
effort: null
---
# Where does attention mass go at 16k?

Frontmatter: `status` active|paused|done; `wake` now|on_jobs; `jobs` in flight, finished ids move to `jobs_archived`
(never delete ids); `budget_gpu_hours` is the only compute limit (dollars autoscale).

## Goal
Figures for the write-up and a mechanism check (proposal section "Attention visualisations"). No training. Checkpoints:
`/fs/scratch/PAS2836/jacktaylor/nanogpt_results/lm1012/125m/{softmax_xs,asentmax_zi_xs,wstieltjes_q3_d2_xs}/lr1.2e-3/ckpt.pt`
(plus the q/d sweep rows if cheap). Two plots per row on ArXiv papers:
(a) support size and entropy per head as a function of query position out to 16k, one curve per head group (ALiBi
vs NoPE); (b) for a few NoPE heads, the attention row at query positions 1k, 4k, 16k, drawn against key position and,
separately, sorted by score gap from the top key with the window threshold marked.

## Theory
`eval_lengths.py --stats_lengths` logs only last-quarter means today. The per-position curve shows whether NoPE heads
sharpen, plateau, or disperse with position and whether that differs between maps. The sorted row shows the window;
the positional row shows whether softmax's dispersed mass lands on the document start (an attention sink) or on
scattered mid-document keys. Counter-argument: with 12 layers x 12 heads the picture may be head-specific and not
summarise; the head-group means in (a) guard against cherry-picking in (b).

## Hypothesis backlog
1. **e001, per-position statistics (~1 GPU-h).** `viz_attention.py`: chunked eager attention (query chunks of 512
   against all keys, fp32, one head at a time, so a 16k x 16k row block never materialises for all heads) on 8 ArXiv
   papers of >= 16k tokens from `data/arxiv_long`; per layer, head, query position: support size (sparse maps: exact
   count of nonzero weights; softmax: number of keys needed for 90 % of mass) and entropy. Save `.npz` per row under
   `$LM_RESULTS/125m/<row>/lr<lr>/attn_stats/`; plot curves averaged over papers, one line per head group, per row,
   on shared axes: `figures/support_vs_position.png`, `figures/entropy_vs_position.png`. Prediction: softmax NoPE
   entropy rises with position through 16k (70 %); windowed Stieltjes NoPE support plateaus below 64 keys (60 %);
   ASEntmax support grows but stays below 256 (55 %).
2. **e002, attention rows (~0.5 GPU-h, same pass if convenient).** Pick the 3 NoPE heads with the largest support
   growth from 2k to 16k in e001 and plot, for each row and for query positions 1k, 4k, 16k: weight vs key position
   (log y) and weight vs rank sorted by score gap, with the window threshold marked for Stieltjes:
   `figures/rows_<row>_L<layer>H<head>.png`. Prediction: softmax's NoPE mass at 16k is dominated by the first
   ~64 keys plus a flat floor (70 %); windowed Stieltjes' support at 16k lies mostly within the last 2k positions
   with a few far keys (60 %).
3. **e003, only if e002 shows a sink:** fraction of mass on the first 64 keys vs query position for every row and
   head group, one PNG. Same pass, no new compute beyond re-reading the `.npz`.

## Constraints
- Files the agent may add: `viz_attention.py`, `osc/viz_attention.sbatch`, `research/`. It may not change
  `model.py`, `train.py`, `eval_lengths.py` (import from them; subclass or wrap for weight extraction).
- Checkpoints are read-only. Figures are committed to `research/experiments/attn-viz/figures/` as PNG under 500 KB each;
  `.npz` stay on scratch.
- Per-experiment ceiling: 3 GPU-h; single `nextgen` jobs <= 2 h.
- Report: the PNGs plus one table per row of mean support and entropy at positions 512, 2k, 8k, 16k for each head group.

## Log
