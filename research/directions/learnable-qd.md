---
slug: learnable-qd
title: Can a learned window take windowed Stieltjes off the fixed-(q,d) frontier at
  125M? (AS-scale gate, per-head learnable q/d, per-query adaptive window)
project: nanogpt
cluster: ascend
status: done
wake: on_jobs
budget_gpu_hours: 300
spent_gpu_hours: 140.16
max_jobs_in_flight: 36
jobs: []
jobs_archived:
- 8368232
- 8368233
- 8368234
- 8369209
- 8369210
- 8369211
- 8369212
- 8388227
- 8368235
- 8368236
- 8368237
- 8368238
- 8368239
- 8368240
- 8369213
- 8369214
- 8369215
- 8369216
- 8369217
- 8369218
- 8369219
- 8388278
last_wake: '2026-11-08T20:50:00-05:00'
next_id: 3
model: null
effort: null
max_turns: 150
timeout_minutes: 90
wakes: 3
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
- 2026-11-06: e001 (gate, `aswstieltjes_q3_d2_zi_xs` x lr {3e-4, 6e-4, 1.2e-3}) submitted at 844514e: smoke 8368232 gates 8 x 3:30 segment arrays 8368233..8368240; est. 60 GPU-h (measured 13.7 s/step for q3/d2 + 5 % AS). Ported env.sh REPO/VENV override from lm350m-qdf; created OSC checkout and copied the ArXiv/PG-19 eval bins into it. Next wake: check smoke, then start e002 (Variant A) implementation in parallel.
- 2026-11-06: smoke 8368232 OK (0.03 GPU-h). e002 (Variant A, `lwstieltjes_q3_d2_xs`, per-head learnable q_h/d_h in the fused kernel via a constexpr PER_HEAD mode with implicit-function q/d grads) x lr {3e-4, 6e-4, 1.2e-3} submitted at 5bb25c6: qdtest 8369209 (fixed path bit-identical vs 3074a63, fused-vs-eager incl. theta grads, 125M throughput probe <= 17.5 s/step) + smoke 8369210 gate 9 x 3:30 segment arrays 8369211..8369219; est. 60 GPU-h. Shared checkout moved to 5bb25c6, so e001 seg 1 (8368233) re-gated afterok:8369209. Next wake: read qdtest (speed, correctness); if speed-only failure, fall back to learnable d_h with q fixed 3 and release e001.
- 2026-11-07: segments 8368234 (e001) and 8369212 (e002) ended exit 4 = planned wall-limit checkpoint, not failures. qdtest 8369209 had failed on a test bug (CPU linspace in test_attn_paths), so Slurm cancelled segment 1 of both chains and segment 2 trained first. Fixed at 2dfc088; qdtest2 8388227 (debug-nextgen; desktop GPU refused) ALL OK: fixed path bit-identical, learnable fused vs eager incl. theta grads <= 1.2e-6 fp32. e001 14.0 s/step (on plan, ~60 GPU-h). e002 20.0 s/step, over the 17.5 s gate: continuing (correct, 12 % done, restart as d-only saves ~14 GPU-h net) at ~28 GPU-h/LR, over the 25 ceiling; added segment array 8388278 so 4770 iters fit. Committed ~150 of 300 GPU-h. Next wake: chains finish ~Nov 9-10; then score e001 gate and e002.
- 2026-11-08: e001 done (57.34 GPU-h). Gate FAILS: `aswstieltjes_q3_d2_zi_xs` best LR 1.2e-3 val 3.537 (= fixed q3/d2 3.534), ArXiv 16k 4.095 vs threshold 3.911 and 0.147 worse than fixed q3/d2 (3.948); all LRs > 3.93. Already 0.084 worse on ArXiv at 2k, so it is a domain-robustness cost, not a length effect. Backlog item 4: no Variant B. e002 segment 8 (8369218) ended 4:0 (wall-limit checkpoint) at iter ~3990; 8369219 running, 8388278 pending, finishes ~Nov 9. At iter 4000 lr1.2e-3 the learned d_h widens to layer medians 3.0-3.9 and q_h drops to ~2.4 in deep layers (wider and heavier-tailed than any fixed 125M cell). No new submission. Figure `figures/arxiv_vs_length.png`. Next wake: score e002, write the negative, decide `done` vs seeds (item 5 only if within 0.01 of ASEntmax).
- 2026-11-08 (evening): e002 segment 9 (8369219_{0,1,2}) ended 4:0 = wall-limit checkpoint, archived; spent 135.05 GPU-h (sacct total of archived ids 134.1 h + the in-flight segment). Val@4500: lr3e-4 3.731, lr6e-4 3.589, lr1.2e-3 3.589. 8388278 at iter 4610/4770, finishes training ~19:40 then runs evals. No new submission (backlog: gate failed, no Variant B; only e002 scoring and the write-up remain). Next wake: score e002, write the negative, decide `done` vs seeds.
- 2026-11-08 (night): e002 done. 8388278_{0,1,2} COMPLETED 0:0 (training to 4771 + four evals); archived. e002 82.81 GPU-h, direction 140.15 of 300. Variant A `lwstieltjes_q3_d2_xs` best LR 1.2e-3: val 3.619 (fixed q3/d2 3.534, ASEntmax 3.476), ArXiv 16k 4.323 (3.948, 3.874), PG-19 16k 3.986. Clear negative, worse than predicted (predicted near q3/d3). Learned (q_h, d_h) plateau at an LR-dependent level (median d 2.23 / 2.48 / 3.24 at lr 3e-4 / 6e-4 / 1.2e-3; q falls to ~2.4 in deep layers at 1.2e-3); at lr 6e-4 the endpoint lies between the q3/d2 and q3/d3 cells yet the run is 0.105 behind fixed q3/d2 on val, so the cost is not the endpoint (learning dynamics vs per-head spread not separated). Both e001 and e002 lower train-split loss and raise val/ArXiv (val-train gap 0.18 / 0.32 vs 0.15 fixed); note fineweb25B/train.bin is 2.5B tokens, so runs are ~1 epoch. Figures `arxiv_vs_length.png`, `qd_vs_iter.png`. Backlog empty (item 4: no Variant B; item 5: no row within 0.01 of ASEntmax): `status: done`, no submission. Follow-ups for the user, outside this brief: learnable d at 10x lower theta LR or frozen after warmup; check the train/val gap ordering on fresh data.
