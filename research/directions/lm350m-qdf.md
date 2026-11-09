---
slug: lm350m-qdf
title: Windowed stieltjes does better at 51M than softmax and ASEntmax at 51M params,
  but not 125M. Is 125M a fluke? (trying ~350M params)
project: nanogpt
cluster: ascend
status: active
wake: on_jobs
budget_gpu_hours: 900
spent_gpu_hours: 195.57
max_jobs_in_flight: 16
jobs:
- 8365025
- 8365026
- 8365027
- 8365028
- 8365029
- 8365030
- 8365031
- 8365032
- 8365033
jobs_archived:
- 8353512
- 8365020
- 8365021
- 8365022
- 8365023
- 8365024
last_wake: '2026-11-08T12:40:01-05:00'
next_id: 3
wakes: 1
---
# Windowed stieltjes does better at 51M than softmax and ASEntmax at 51M params, but not 125M. Is 125M a fluke? (trying ~350M params)

Frontmatter fields: `status` active|paused|done. `wake` now|on_jobs (the loop runs this
direction on the next tick when `now`, otherwise only when a listed job finishes).
`jobs` = ids currently in flight; finished ids move to `jobs_archived` (never delete ids,
they feed the GPU-hour accounting). `budget_gpu_hours` is a hard stop.

## Goal
What question this direction answers and what result would settle it.
Establish better scaling understanding of current stieltjes variants, ASEntmax, and softmax. 50 and 125M have been tried. A bigger scale is in order.

## Theory
Not much developed theory yet on scaling laws of our method, not sure what to expect.

## Prior results (125M, T=2048, 2.5B tokens, 1 seed; from results/lm1012/125m)
| row | lr | val@2048 | FineWeb 16k | ArXiv 2k | ArXiv 16k | PG-19 16k |
|---|---|---|---|---|---|---|
| softmax_xs | 1.2e-3 | 3.461 | 3.437 | 4.035 | 4.353 | 3.881 |
| asentmax_zi_xs | 1.2e-3 | 3.476 | 3.458 | 4.037 | 3.874 | 3.800 |
| wstieltjes_q3_d2_xs | 1.2e-3 | 3.534 | 3.493 | 4.141 | 3.948 | 3.880 |
| wstieltjes_q4_d2_xs | 1.2e-3 | 3.493 | 3.494 | 4.125 | 3.992 | 3.873 |
| wstieltjes_q3_d3_xs | 1.2e-3 | 3.505 | 3.489 | 4.097 | 3.985 | 3.847 |
| softmax_xs | 6e-4 | 3.512 | 3.503 | 4.202 | 4.481 | 3.961 |
| asentmax_zi_xs | 6e-4 | 3.518 | 3.520 | 4.075 | 3.821 | 3.842 |
| wstieltjes_q3_d2_xs | 6e-4 | 3.541 | 3.555 | 4.150 | 4.019 | 3.915 |

125M walltime on nextgen (A100 40 GB, 0.5M tok/step, 4770 steps): softmax 8.3 h (83k tok/s, micro-batch 16),
ASEntmax 13.4 h (52k tok/s, micro-batch 8), Stieltjes q3/d2 18.1 h (38k tok/s, micro-batch 8, num_iter 15).

## Hypothesis backlog
1. e001: throughput / memory probe of the 350M shape (24L x 16H x 1024) per row and micro-batch; measured
   GPU-hour estimate for the stage. ~0.5 GPU-h.
2. e002: the stage itself. softmax_xs, asentmax_zi_xs, wstieltjes_q3_d2_xs x 3 LRs (planned 3e-4, 6e-4, 1.2e-3:
   the 125M optimum 1.2e-3 and two halvings, since the optimum moves down with width), 1 seed, 2.5B tokens
   (7 tok/param) unless the probe shows 7B tokens (20 tok/param) fits the budget. Chains of 4 h segments, one
   array per LR. The same four evals as 125M (lengths, 16k, ArXiv, PG-19).
3. If Stieltjes at 350M again trails softmax at 2048 but leads at 16k: wstieltjes_q4_d2_xs at the best LR (the
   125M q/d sweep's best train-length row) to separate the q/d choice from scale.
4. Seeds at the best LR only if the ordering is within ~0.01 nats.

Open decision for the user if the probe says the 9-run stage does not fit 500 GPU-h at 2.5B tokens:
(a) 2 LRs per row, (b) drop ASEntmax, (c) raise the budget.
Note 2026-11-05: `OSCusage -P PAS2836` shows Remaining Budget 287.8 (dollars); my 125M week cost ~40.5 for
~5650 billing-hours, so 500 GPU-h is roughly 70 dollars, but other users spent ~1040 in the last day.

## Constraints
- Files the agent may change: Same as the 50M and 125M experiments
- Fixed protocol (data, eval, seeds, metric, checkpoint selection): same data, T=2048, same metrics. 3 lr, 1 seed per variant.
- Per-experiment ceiling (GPU-hours, walltime): N/A, just fit in budget_gpu_hours 

## Decision 2026-11-09 (user): option (c)
Context: `osc/env.sh:6` sets `DATA_SRC` unconditionally and `osc/run_lm.sh:9` re-sources it, so `DATA_SRC_OVERRIDE`
never took effect. Every 125M and 350M run trained 2.5B tokens on the 1.0B-token fineweb10B shard (2.5 epochs). The 51M
stage (T=512, 7630 steps) was exactly 1 epoch. At 125M the train/val gap is softmax 0.13, ASEntmax 0.18, Stieltjes
0.20-0.27, and the train-loss ordering (q4/d3 3.230 < q3/d2 3.288 < ASEntmax 3.298 < softmax 3.332) is the reverse of
the val ordering: an overfitting signature, so the 51M -> 125M "degradation" is confounded with 1 vs 2.5 epochs.
1. Cancel every remaining e002 job (`scancel` the held arrays 8365026-8365033 and any running task). Mark e002
   `failed`, cause "2.5 epochs of fineweb10B via env.sh:6", keep all ids. The softmax rows stay recorded as a
   repeated-data data point, not as the 350M result.
2. Fix `osc/env.sh:6` to `export DATA_SRC=${DATA_SRC:-$SCRATCH/nanogpt_data/fineweb10B}` and make the submit scripts
   export `DATA_SRC=$SCRATCH/nanogpt_data/fineweb25B` (drop DATA_SRC_OVERRIDE or make run_lm.sh honour it). Smoke on
   debug-nextgen first; the smoke log must print the fineweb25B path. After segment 1 of any chain, confirm the effective
   data path and that 4770 steps = 1 epoch before the chain continues.
3. e003: the 9-run 350M stage on fineweb25B, 2.5B tokens (1 epoch), 3 LRs, as planned in backlog item 2 (~300 GPU-h).
4. e004: redo the 125M rows softmax_xs, asentmax_zi_xs, wstieltjes_q3_d2_xs at lr {6e-4, 1.2e-3} on fineweb25B
   (1 epoch), otherwise the `config/lm_stieltjes_125m.py` protocol, results under `$LM_RESULTS/125m_fw25/<row>/lr<lr>/`
   so the existing 125M results stay intact (~75 GPU-h). Hypothesis: the Stieltjes gap excess disappears at 1 epoch
   and the val ordering follows the train ordering (Stieltjes <= ASEntmax <= softmax at 2048), as at 51M. Prediction
   with confidence before submission as usual.
5. Budget raised 500 -> 900 GPU-h. Submit e003 and e004 together if `max_jobs_in_flight` allows; otherwise e004 first
   (cheaper, answers the confound directly).

## Log
One line per wake, newest last. Format: `YYYY-MM-DD HH:MM  eNNN submitted <jobids> | eNNN done <metric> | paused: <reason>`
2026-11-05 19:10  e001 submitted 8353512 (350M throughput/memory probe, nextgen, ~0.5 GPU-h) | env: scratch purge killed venvs/asentmax, built venvs/nanogpt; checkout wt/nanogpt/lm350m-qdf created at 4465367 | OSCusage remaining 287.8 dollars
2026-11-06 08:30  e001 done: tok/s softmax 32.4k / ASEntmax 21.0k / Stieltjes q3d2 15.0k at micro-batch 8 (all fit, 28-31 GB; softmax bs16 OOM), 9-run stage = 302.6 GPU-h train, fits budget | e002 submitted 8365020-8365033 (14 chained 4:15 segment arrays, 3 rows x 3e-4/6e-4/1.2e-3, 2.5B tok, est 300-360 GPU-h) at 27a87fb | env: pip entmax 1.3 added to venvs/nanogpt (missing, broke ASEntmax eager path) | OSCusage remaining 271.86 dollars
2026-11-06 20:00  e002 segment 1 (8365020_0-8) done: all 9 exit 4 = planned wall-limit checkpoint (not a failure; every e002 segment ends this way), ckpt.pt written, s/iter 16.3/25.2/35.1 matches e001; softmax 840, ASEntmax 545, Stieltjes 391 iters per segment, chain lengths suffice | segment 2 (8365021) pending on Priority | nothing new submitted (backlog 3/4 wait on e002) | ascend checkout stays at 27a87fb until the chain ends
2026-11-08 13:05  e002 softmax rows done (8365025_0-2 exit 0): best lr 1.2e-3 val@2048 3.532, ArXiv 16k 4.396, worse than 125M (3.461) | DATA BUG: run_lm.sh re-sources env.sh, which resets DATA_SRC to fineweb10B (1.0B tok), so DATA_SRC_OVERRIDE=fineweb25B was ignored; every 125M and 350M run trained 2.5 epochs on 1B tokens (val flat after epoch 1, iter 1907; 350M train/val gap 0.65) | held 8365026-8365033 (scontrol hold, reversible); 8365025_3,4,6-8 still running and will checkpoint | paused: user decision needed. (a) release the hold and finish e002 on fineweb10B (~95 GPU-h; consistent with 125M as actually run, but tests overfitting to repeated data); (b) cancel e002, fix env.sh:6 to `${DATA_SRC:-...}`, rerun the 9-run stage on fineweb25B at 1 epoch (~300 GPU-h, total ~490 of 500, so no margin); (c) (b) plus a budget raise, and also redo 125M on fineweb25B, since the 51M->125M crossover is confounded by 1 vs 2.5 epochs. Recommend (c), or (b) with 2 LRs per row if the budget stays at 500.
2026-11-09 17:38  resumed: by user
