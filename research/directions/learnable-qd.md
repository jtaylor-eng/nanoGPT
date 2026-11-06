---
slug: learnable-qd
title: "Can a learned window take windowed Stieltjes off the fixed-(q,d) frontier at 125M? (AS-scale gate, per-head learnable q/d, per-query adaptive window)"
project: nanogpt
cluster: ascend
status: active
wake: now
budget_gpu_hours: 300
spent_gpu_hours: 0
max_jobs_in_flight: 36
jobs: []
jobs_archived: []
last_wake: null
next_id: 1
model: null
effort: null
max_turns: 150
timeout_minutes: 90
---
# Can a learned window take windowed Stieltjes off the fixed-(q,d) frontier at 125M?

Frontmatter: `status` active|paused|done; `wake` now|on_jobs; `jobs` in flight, finished ids move to `jobs_archived`
(never delete ids); `budget_gpu_hours` is the only compute limit (dollars autoscale).

## Goal
At 125M the five fixed-(q,d) cells trace a frontier (better at the training length <=> worse at 16k) and ASEntmax
sits off it (val@2048 3.476, ArXiv 16k 3.874; best Stieltjes cells 3.493 and 3.948). The proposal section
"Learnable q and d" (`Stieltjes/context/stieltjes_proposal.tex`) derives the gradients and defines two variants:
- **A, per-head scalars** q_h = 2 + softplus(theta_q), d_h = 1 + exp(theta_d), init (3, 2). Expected to stay on the frontier.
- **B, per-query adaptive window** d_i = 1 + delta_h (1 + beta_i (log(i+2))^gamma_i), beta_i = softplus(x_i . w_beta),
  gamma_i = tanh(x_i . w_gamma), zero-init projections, delta_h set so d_i starts near 2. The ASEntmax scale
  parametrisation moved from the temperature to the window.
Settled by: val@2048, FineWeb 16k, ArXiv 2k -> 16k, PG-19 16k at the best of 3 LRs per row, against the fixed
q3/d2 and q3/d3 cells and ASEntmax, plus the learned values themselves (q_h, d_h per layer; for B the
distribution of d_i at positions 512, 2k, 16k).

## Theory
The gate (proposal paragraph "Relation to the adaptive scale, and the cheap gate"): scaling logits by s with window d
equals window d/s to first order, so the existing AS scale on q3/d2 (`aswstieltjes_q3_d2_zi_xs`, already a valid row)
tests per-query control without new kernel work. If it reaches ASEntmax's numbers, per-query control is what
matters and B is worth building; if it does not, the heavy tail is the deficit at this size. Caution: at 51M the AS
scale on windowed Stieltjes was neutral. Variant A is expected to settle d_h in [2.1, 2.6] (gradient probe: the
model wants d wider, trained d at the wide edge of a sharp optimum) and q_h barely to move; it is the control.

## Hypothesis backlog
1. **e001, the gate (54 GPU-h).** `aswstieltjes_q3_d2_zi_xs` at lr {3e-4, 6e-4, 1.2e-3}. Existing row grammar, no
   code change; submit in the first wake. Prediction: best LR closes at least half the ArXiv-16k gap to ASEntmax,
   i.e. <= 3.911 (50 %); val@2048 within 0.02 of q3/d2 (60 %).
2. **e002, Variant A (54 GPU-h + implementation).** Row `lwstieltjes_q3_d2_xs` (l = learnable per-head q, d).
   The differentiable eager path `stieltjes_normalize_qd` exists in `stieltjes/stieltjes_eager.py`; the fused
   kernel takes q and window as scalars. First check whether the Triton kernel can take per-head q_h, d_h as runtime
   values (one extra load per head); if that is a day of work, do it; if not feasible, measure the eager path's
   throughput at 125M T=2048 in a 40-step probe and proceed only if a run fits 25 GPU-h. Log q_h, d_h per layer to
   `metrics.jsonl` at every eval. lr {3e-4, 6e-4, 1.2e-3}. Prediction (Hypothesis A, 70 %): learned d_h in
   [2.1, 2.6] on every layer, val@2048 within 0.01 of q3/d3 (3.505), ArXiv 16k within 0.02 of q3/d3 (3.985), still
   behind ASEntmax on both.
3. **e003, Variant B, only if the gate passes (best e001 ArXiv 16k <= 3.911) (60 GPU-h + 2-3 days of kernel work).**
   Row `pwstieltjes_q3_xs` (p = per-query window). Per-row d_i is one extra kernel input; the lambda bracket becomes
   per-row. Log mean and std of d_i at positions 512, 2k, 16k and the sign of gamma per head at the final eval.
   lr {3e-4, 6e-4, 1.2e-3}. Prediction: beats ASEntmax at ArXiv 16k (35 %); at val@2048 (25 %).
4. **If the gate fails at all three LRs** (ArXiv 16k > 3.93 and no val@2048 gain): write the negative up as closing
   the learnable-window question at this size, finish e002, set `status: done`. Do not build B.
5. **Seeds**: only for the single best new row, and only if it is within 0.01 of ASEntmax on either axis (2 seeds, ~40 GPU-h).
Learnable per-query q is out of scope (proposal: poorly identified against the scale).

## Constraints
- Files the agent may change: `model.py` (new attn options and parameters only), `stieltjes/stieltjes_eager.py`,
  `stieltjes/triton_stieltjes.py` (new code paths; the fixed-(q,d) path must stay bit-identical, run the existing
  tests), `osc/rows.sh`, `tests/`, `train.py` only to log the learned values, new files under `osc/` and `research/`.
- Files it may not change: `eval_lengths.py` except to read the learned values from the checkpoint, `config/*`,
  `configurator.py`, existing row semantics.
- Fixed protocol: `config/lm_stieltjes_125m.py`, fineweb25B, T=2048, 2.5B tokens, seed 1337, micro-batch 8 /
  accum 32, the four standard evals, results under `$LM_RESULTS/125m/<row>/lr<lr>/`. Selection across LRs by final
  val loss.
- Per-experiment ceiling: 25 GPU-h per run; segments 3:30 chained with afterany (6 per Stieltjes run).
- Report: table of all rows x LRs x {val@2048, FineWeb 16k, ArXiv 2k, ArXiv 16k, PG-19 16k} with the fixed cells
  and ASEntmax as reference rows; PNG `figures/arxiv_vs_length.png`; for A a PNG of d_h per layer over training;
  for B a PNG of the d_i distribution vs position. Copy result jsons into `results/lm1012/125m/` on this branch.

## Log
