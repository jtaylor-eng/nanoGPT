---
slug: lm350m-qdf
title: Windowed stieltjes does better at 51M than softmax and ASEntmax at 51M params,
  but not 125M. Is 125M a fluke? (trying ~350M params)
project: nanogpt
cluster: ascend
status: paused
wake: on_jobs
budget_gpu_hours: 900
spent_gpu_hours: 545.62
max_jobs_in_flight: 16
jobs: []
jobs_archived:
- 8353512
- 8365020
- 8365021
- 8365022
- 8365023
- 8365024
- 8365025
- 8365026
- 8365027
- 8365028
- 8365029
- 8365030
- 8365031
- 8365032
- 8365033
- 8453383
- 8453387
- 8453455
- 8453456
- 8453457
- 8453458
- 8453459
- 8453460
- 8453461
- 8453462
- 8453463
- 8453464
- 8453466
- 8453467
- 8453468
- 8453469
- 8496164
- 8496165
- 8496354
- 8496355
- 8496386
- 8496387
- 8496388
- 8496389
- 8496390
- 8496391
last_wake: '2026-11-11T10:00:01-05:00'
next_id: 6
wakes: 20
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
2026-11-09 18:00  e002 failed (2.5 epochs of fineweb10B via env.sh:6; 195.34 GPU-h; held arrays 8365026-33 scancelled per decision (c)) | fix at 5c5d8c0: env.sh keeps DATA_SRC, run_lm.sh data guard (MIN_TRAIN_TOKENS) + RESULTS_TAG + DIVERGED skip, mixed-config manifests | smoke debug-nextgen 8453383/8453387 ok, fineweb25B path printed (8453383_1 node fault a0339, rerun ok) | e003 (350M x9) + e004 (125M x6) submitted as one 14-segment chain 8453455-8453469 at fec70ed, segments 2-14 HELD: next wake verifies segment-1 data path, then scontrol release | est ~400 GPU-h, budget 900, spent 195.68
2026-11-09 19:05  segment 1 check: fineweb25B path + 2.5e9-token guard confirmed in all 15 run.logs | 8453455_5,_8,_10 failed on node a0339 (CUDA device busy, node fault, same as smoke 8453383_1), ~0.02 GPU-h; pending arrays 8453456-8453469 updated ExcNodeList=a0339 and released; the three tasks restart in segment 2 (enough segments left; 125M Stieltjes 1.2e-3 may need a separate eval job) | nothing new submitted
2026-11-09 22:15  segment 1 (8453455) ended: 12 tasks exit 4 = planned wall-limit checkpoint, ckpt.pt written; _5,_8,_10 = a0339 node fault (diagnosed last wake, restarted from iter 0 in segment 2) | 8453455 archived | segment 2 (8453456_0-14) running, resumes verified | nothing new submitted (backlog 3/4 wait on e003/e004)
2026-11-10 02:05  segment 2 (8453456_0-14) ended: all 15 exit 4 = planned wall-limit checkpoint, ckpt.pt written, no divergence | 8453456 archived | segment 3 (8453457) running | partial val: 125M softmax 1.2e-3 3.434 @4250, 350M softmax 6e-4 3.639 @1500 | nothing new submitted (backlog 3/4 wait on e003/e004)
2026-11-10 03:10  e004 125M softmax_xs done (8453457_0/_1 exit 0, 17.4 GPU-h): lr 1.2e-3 val@2048 3.431 (2.5-ep 3.461), FineWeb 16k 3.424, ArXiv 2k/16k 4.051/4.370 (no ArXiv gain from fresh data), PG-19 16k 3.858; lr 6e-4 3.488; figure figures/e004_lengths.png | 8453457_2-14 still running, array stays in jobs | nothing new submitted (backlog 3/4 wait on e003/e004)
2026-11-10 04:30  e003 350M asentmax_zi_xs 1.2e-3 diverged (8453457_11 exit 3, NaN at iter 1400 with no loss spike, last ckpt val 3.730 @1250; e002 same cell NaN @1710): second divergence, cell dropped, no retry (fix would change protocol); ASEntmax 350M = best of {3e-4, 6e-4}, caveat that 1.2e-3 led 3e-4 by 0.10 @1000 | prediction 'diverges again' (45 %) hit | cell cost 9.89 GPU-h | 8453457 stays in jobs (other tasks running) | nothing new submitted (backlog 3/4 wait on e003/e004)
2026-11-10 13:40  e004 125M asentmax_zi_xs done (8453458_2/_3, 27.5 GPU-h): best lr 1.2e-3 val@2048 3.448 (softmax 3.431), ArXiv 16k 3.707 (2.5-ep 3.874), PG-19 16k 3.749 | e004 wstieltjes_q3_d2_xs 6e-4 done (8453459_4, 18.7 GPU-h): val@2048 3.519 (same-lr softmax 3.488, gap +0.031, 2.5-ep +0.029), ArXiv 16k 3.792 (2.5-ep 4.019) | provisional: fresh data helps all rows ~0.03 at 2048, sparse rows gain 0.17-0.23 at ArXiv 16k, overfitting does not explain the Stieltjes 2048 deficit at 6e-4 | Stieltjes 1.2e-3 resumes @4000 in its last segment 8453460_5 (~3.4 h incl. evals) | 350M segments 3-5 all planned exit 4, no new divergence | 8453457-8453459 archived | figure e004_lengths regenerated | nothing new submitted (backlog 3/4 wait on e003)
2026-11-10 14:00  8453460_4 exit 0 (125M Stieltjes 6e-4 already done, no-op) and 8453460_11 exit 3 (DIVERGED guard, dropped 350M ASEntmax 1.2e-3 cell), both expected, ~0 GPU-h | 125M Stieltjes 1.2e-3 (8453460_5) at iter 4150, ETA ~17:00 incl. evals | 350M segment 6 running | 8453460 stays in jobs | nothing new submitted (backlog 3/4 wait on e003)
2026-11-10 16:50  e003 350M softmax_xs 3e-4/6e-4 done (8453460_6/_7, 44.7 GPU-h): val@2048 3.503/3.491, worse than 125M softmax 3.431 on every eval, train-val gap 0.32/0.50 | e004 125M Stieltjes 1.2e-3 trained: val@2048 3.486 (softmax 3.431, gap +0.055), ArXiv/PG-19 evals still running in 8453460_5 | BUG: train.py reseeds 1337 on every resume and ckpt.pt has no RNG state, so each segment replays segment 1's batch windows (CPU simulation: 99 % overlap). Unique data = one segment: 125M softmax/ASEntmax/Stieltjes 2.2x/3.5x/4.7x repeats, 350M 5.7x/8.8x/12.2x. Biased against the slow sparse kernels; confounds e002, e003, e004 and the original 125M stage (51M only if chained, unverified); other directions chaining this train.py too. Figure figures/e003_replay.png | scontrol hold 8453461-8453464, 8453466-8453469 (reversible); segment 6 running tasks left to checkpoint | paused: user decision needed. (a) fix (store/restore CPU+CUDA RNG in ckpt.pt), scancel the held arrays, rerun e004 125M x6 first (~86 GPU-h), then 350M; (b) (a) but 350M with 2 LRs per row (softmax 6e-4/1.2e-3, ASEntmax 3e-4/6e-4, Stieltjes 6e-4/1.2e-3; ~200 GPU-h, total ~290 of ~395 left); (c) release the hold and finish e003 as a repeated-data point (~110 GPU-h, not informative for scaling). Full 9-run 350M rerun + e004 rerun is ~386 GPU-h = the whole remaining budget, no margin. Recommend (b).
2026-11-10 17:33  user decision: (a) Apply the RNG fix (save and restore CPU+CUDA RNG state in ckpt.pt), scancel the held arrays 8453461-8453464 and 8453466-8453469, rerun e004 125M x6 first (~86 GPU-h), then stop for a user decision on 350M. Before submitting, test that a resumed run draws the same batch indices as an uninterrupted one. Context checked 2026-11-10 from local run.logs: the 51M T=512 stage is clean (64/64 runs, 0 resumes); 51M T=2048 has 4 seed-1337 sparse runs with one resume each (asentmax 1.5e-3/3e-3 at iter 7300/7340, wstieltjes_q3_d2 1.5e-3/3e-3 at iter 5520 of 7620); softmax and all _s1/_s2 replicates are clean.
2026-11-10 18:00  e004 done (8453460_5 exit 0, 82.42 GPU-h, replayed data): best LR 1.2e-3 all rows, val@2048 softmax 3.431 / ASEntmax 3.448 / Stieltjes q3d2 3.486, ArXiv 16k 4.370 / 3.707 / 3.981 (Stieltjes 6e-4: 3.792) | e003 failed: segment 6 (8453460_8-10,12-14) planned exit 4, held arrays 8453461-64, 8453466-69 scancelled per decision (a), 185.39 GPU-h | RNG fix aac000a (ckpt.pt stores batch-sampler CPU+CUDA RNG): tests/test_resume_rng.py tier 1 softmax/stieltjes/entmax 803/803 batches identical, losses bit-equal; tier 2 smoke 8496354->8496355 wall-stop + resume reproduces uninterrupted 8496164 losses exactly | e005 submitted 8496386-8496391 (125M x6 rerun, fw25B, tag 125m_fw25_rng, 6 x 4:15 afterany, est 78-92 GPU-h) | after e005: pause for user decision on 350M
2026-11-10 22:05  e005 segment 1 (8496386_0-5) ended: all 6 exit 4 = planned wall-limit checkpoint, ckpt.pt written, fineweb25B confirmed; iters softmax 2194, ASEntmax 1377, Stieltjes 1004 per segment (segment allotment suffices) | val@1000 lr1.2e-3 softmax 3.921 / ASEntmax 3.907 / Stieltjes 3.913 | 8496386 archived | segment 2 (8496387) running, resumed | nothing new submitted (after e005: pause for user decision on 350M)
2026-11-11 01:45  e005 segment 2 (8496387_0-5) ended: all 6 exit 4 = planned wall-limit checkpoint, RNG restore logged in every run | stop iters softmax 4335/4337, ASEntmax 2766/2713, Stieltjes 2012/2011 | val softmax 1.2e-3 3.412 @4250 (e004 replayed final 3.431), ASEntmax 3.532 @2750, Stieltjes 3.541 @2000 | 8496387 archived | segment 3 (8496388) running | nothing new submitted (after e005: pause for user decision on 350M)
2026-11-11 02:45  e005 softmax_xs done (8496388_0/_1 exit 0, 17.3 GPU-h for both softmax runs): lr 1.2e-3 val@2048 3.403 (e004 replayed 3.431), FineWeb 16k 3.415, ArXiv 2k/16k 3.951/4.269, PG-19 16k 3.843; lr 6e-4 3.472 | figure figures/e005_lengths.png | 8496388_2-5 (ASEntmax, Stieltjes) running, 8496388 stays in jobs | nothing new submitted (after e005: pause for user decision on 350M)
2026-11-11 05:45  e005 segment 3 sparse tasks (8496388_2-5) ended: all 4 exit 4 = planned wall-limit checkpoint, RNG restore logged, 15.4 GPU-h; stop iters ASEntmax 4138/4084, Stieltjes 3020/3023; 8496389_0/_1 softmax no-op exit 0 | matched-iter val vs softmax (lr 1.2e-3): ASEntmax +0.059 @4000, Stieltjes +0.035 @3000 | 8496388 archived | segment 4 (8496389_2-5) running, resumed | nothing new submitted (after e005: pause for user decision on 350M)
2026-11-11 06:10  trigger 8496389_0/_1 = softmax no-op exit 0 (already recorded last wake), ~0 GPU-h | segment 4 (8496389_2-5) running: ASEntmax @4360/4310 (ends this segment), Stieltjes @3180/3190 | nothing new submitted (after e005: pause for user decision on 350M)
2026-11-11 08:00  e005 asentmax_zi_xs done (8496389_2/_3 exit 0, 27.4 GPU-h for both runs): lr 1.2e-3 val@2048 3.405 (softmax 3.403, tie; e004 replayed 3.448), FineWeb 16k 3.415, ArXiv 2k/16k 3.901/3.604 (softmax 4.269), PG-19 16k 3.712; lr 6e-4 3.459 | 8496389_0/_1 softmax no-op | figure e005_lengths regenerated | Stieltjes (8496389_4/_5) in segment 4, ends in 8496390 | 8496389 stays in jobs | nothing new submitted (after e005: pause for user decision on 350M)
2026-11-11 09:45  e005 segment 4 Stieltjes (8496389_4/_5) ended: both exit 4 = planned wall-limit checkpoint @4019/4028, 7.7 GPU-h; 8496390_2/_3 ASEntmax no-op exit 0 | 8496389 archived | segment 5 (8496390_4/_5) resumed, RNG restore logged, ~2.9 h + evals left | matched-iter val @4000 vs softmax: Stieltjes q3d2 -0.018 (1.2e-3) / -0.033 (6e-4), ASEntmax +0.059 / +0.044 (provisional) | nothing new submitted (after e005: pause for user decision on 350M)
2026-11-11 10:05  trigger 8496390_2/_3 = ASEntmax no-op exit 0 (10 s each, TRAIN_DONE + evals present, already recorded 09:45), ~0 GPU-h | segment 5 Stieltjes (8496390_4/_5) running 40 min, resumed @4010/4020, fineweb25B confirmed | 8496390 stays in jobs, 8496391_4-5 pending | nothing new submitted (after e005: pause for user decision on 350M)
2026-11-11 12:55  e005 done (8496390_4/_5 exit 0; 8496391_4/_5 no-op exit 0; e005 82.22 GPU-h, prediction 8/9 hit): 125M unique data, lr 1.2e-3 best for all rows, val@2048 Stieltjes q3d2 3.396 / softmax 3.403 / ASEntmax 3.405 (three-way tie, spread 0.009 < +-0.05 bounce, seed 1337); ArXiv 16k 3.652 / 4.269 / 3.604; PG-19 16k 3.756 / 3.843 / 3.712; FineWeb 16k 3.455 / 3.415 / 3.415. The 125M Stieltjes train-length deficit (e004 +0.055, original stage +0.073) was replay/repeated data; ASEntmax keeps a ~0.045 lead over Stieltjes on every 16k eval | figure figures/e005_lengths.png | 8496390, 8496391 archived, no jobs in flight | paused: user decision on 350M (planned stop after e005). Remaining ~348 GPU-h. 350M costs from e001 tok/s (32.4k/21.0k/15.0k): softmax 21.4, ASEntmax 33.1, Stieltjes 46.3 GPU-h per run + ~4 % segment/eval overhead. (a) 350M 8 runs (3 LRs, minus ASEntmax 1.2e-3 which diverged twice under replay) ~280 GPU-h; (b) 350M 2 LRs per row (softmax and Stieltjes 6e-4/1.2e-3, ASEntmax 3e-4/6e-4) ~210 GPU-h; (c) 125M seeds 1 and 2 at lr 1.2e-3 for the 3 rows (backlog 4, triggered by the 0.009 spread) ~82 GPU-h; (d) (b)+(c) together ~292 GPU-h, fits with ~55 margin, 350M Stieltjes ~4 days wall incl. queue; (e) (a)+(c) ~362 GPU-h, needs ~+20 budget. Recommend (d): seeds settle whether the 125M tie is real, 2 LRs bracket the 125M optimum at 350M.
