---
slug: pe-ssmax-headsplit
title: "Is the sparse maps' long-context gain sparsity or position-dependent temperature, and does it live on the NoPE heads? (SSMax row + ALiBi/NoPE head split, 125M)"
project: nanogpt
cluster: ascend
status: active
wake: on_jobs
budget_gpu_hours: 150
spent_gpu_hours: 0
max_jobs_in_flight: 24
jobs: [8368600]
jobs_archived: []
last_wake: 2026-11-06
next_id: 2
model: null
effort: null
max_turns: 100
timeout_minutes: 60
---
# Is the sparse maps' long-context gain sparsity or position-dependent temperature, and does it live on the NoPE heads?

Frontmatter: `status` active|paused|done; `wake` now|on_jobs; `jobs` in flight, finished ids move to `jobs_archived`
(never delete ids); `budget_gpu_hours` is the only compute limit (dollars autoscale).

## Goal
Two confounds of the 125M result (`../../CLAUDE.md`, proposal section "Positional encodings"):
1. **Mechanism.** Every row that improves from 2k to 16k on ArXiv is sparse *and* has a position-aware scale
   (ASEntmax) or a fixed gap threshold (windowed Stieltjes). Dense softmax with the SSMax temperature
   (logits scaled by a learnable per-head s times log of the causal row length) has position-dependent
   temperature and no sparsity. If it extrapolates like the sparse rows, the story is "temperature", not "sparsity".
2. **Locus.** All rows use NAPE with 6 ALiBi + 6 NoPE heads. The claim is that the sparse advantage lives on the
   NoPE heads. Changing the split to 4/8 and 2/10 (ALiBi/NoPE) for softmax and windowed Stieltjes tests it.
Settled by: ArXiv loss at 2k and 16k, FineWeb val at 2048 and 16k, PG-19 16k, for each new row next to the existing
6/6 cells at lr 1.2e-3 (softmax 3.461 / 4.035 -> 4.353; ASEntmax 3.476 / 4.037 -> 3.874; wstieltjes_q3_d2 3.534 / 4.141 -> 3.948).

## Theory
SSMax (arXiv 2501.19399, the `scalable_softmax` paper in `Stieltjes/context/`) is the fixed-form special case of the
ASEntmax adaptive scale delta + beta (log n)^gamma. At 51M the zero-init AS-softmax row was *worse* than softmax at
4k, so a null is plausible: training at T=2048 and evaluating at 16k only raises the SSMax temperature by log(16k)/log(2k)
= 1.27x. Strongest counter-argument to the sparsity story: the ASEntmax paper lists SSMax as a baseline that
extrapolates reasonably, and Mistral Large 4 adds SSMax to its former sliding-window layers for long context.
Head split: the sparse maps' support statistics at 16k are logged on NoPE heads only; ALiBi heads are local by
construction. More NoPE heads give the map more rows where it can differ from softmax.

## Hypothesis backlog
1. **e001, implementation + probe (~0.5 GPU-h).** Add row `ssmax` to `osc/rows.sh` and `model.py`
   (`attn_scale=ssmax`: per-head learnable scalar s, logits multiplied by s * log(i+1) for causal row length i+1;
   port the init from asentmax `synthetic/src/models/architectures/sparse_gemma.py` `attn_scale_type=nakanishi`,
   which exists but was never run). Add config field `nape_n_alibi` (default n_head - n_head//2, i.e. 6 at 125M)
   threaded to the `n_scaled` line and the slope buffer in `model.py`, and row suffix `_a<N>` (= number of ALiBi
   heads) in `rows.sh`. Verify with `tests/test_attn_paths.py` plus a new test that the default split and the
   softmax row are bit-identical to before. On OSC: 40-step probe of `ssmax_xs`, `softmax_xs_a2`,
   `wstieltjes_q3_d2_xs_a2` at 125M. Prediction: ssmax step time within 3 % of softmax (80 %); no NaN (85 %).
2. **e002, SSMax row (8.5 GPU-h).** `ssmax_xs` at lr 1.2e-3, full 125M protocol. Prediction: ArXiv loss falls from
   2k to 16k (60 %) but ends >= 0.15 above ASEntmax at 16k (60 %); val@2048 within 0.01 of softmax (65 %).
   Submit in the same wake as e003.
3. **e003, head split (53 GPU-h).** `softmax_xs_a4`, `softmax_xs_a2`, `wstieltjes_q3_d2_xs_a4`,
   `wstieltjes_q3_d2_xs_a2`, lr 1.2e-3. Prediction: the Stieltjes-minus-softmax gap at ArXiv 16k widens
   monotonically from 6/6 to 2/10 (50 %); val@2048 cost of 2/10 vs 6/6 <= 0.01 for both rows (50 %).
4. **e004, only if e003 shows |delta ArXiv 16k| >= 0.03 between splits for either row (27 GPU-h):**
   `asentmax_zi_xs_a4`, `asentmax_zi_xs_a2`. Prediction: ASEntmax follows the same direction as Stieltjes (60 %).
5. **e005, only if the trend through 2/10 is monotone (26 GPU-h):** `softmax_xs_a0`, `wstieltjes_q3_d2_xs_a0`
   (all NoPE). Prediction: softmax a0 degrades at every length beyond 2k (70 %).
6. **SSMax second LR 6e-4** only if e002 diverges or its val@2048 is > 0.05 behind softmax.

## Constraints
- Files the agent may change: `model.py` (only: `attn_scale == 'ssmax'` branch, `nape_n_alibi` field and its two
  uses), `osc/rows.sh`, `tests/`, new sbatch/manifest files under `osc/`, `research/`.
- Files it may not change: `train.py`, `eval_lengths.py`, `configurator.py`, `config/*`, `stieltjes/*`.
  Existing row names must keep their exact behaviour (test it).
- Fixed protocol: `config/lm_stieltjes_125m.py`, fineweb25B, T=2048, 2.5B tokens, seed 1337, lr 1.2e-3, the four
  standard evals from `run_lm.sh`, results under `$LM_RESULTS/125m/<row>/lr<lr>/`. Micro-batch 8 / accum 32 for
  sparse rows as `run_lm.sh` already does.
- Per-experiment ceiling: 20 GPU-h per run; segments 3:30 chained with afterany (softmax 3, Stieltjes 6).
- Report: one table of all rows (existing 6/6 cells included) x {val@2048, FineWeb 16k, ArXiv 2k, ArXiv 16k,
  PG-19 16k}; one PNG `figures/arxiv_vs_length.png` with ArXiv loss vs length, one line per row, colour by map,
  linestyle by split. Copy the json results into `results/lm1012/125m/<row>/lr<lr>/` on this branch.

## Log
- 2026-11-06: e001 implemented (ssmax row, `_a<N>` split carried in `nape_slope_type` because train.py may not change; REPO/VENV override ported from lm350m-qdf), probe job 8368600 submitted (~0.5 GPU-h). Next: e002 + e003 sized from the measured step times.
