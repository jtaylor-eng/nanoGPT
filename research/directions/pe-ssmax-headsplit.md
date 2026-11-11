---
slug: pe-ssmax-headsplit
title: Is the sparse maps' long-context gain sparsity or position-dependent temperature,
  and does it live on the NoPE heads? (SSMax row + ALiBi/NoPE head split, 125M)
project: nanogpt
cluster: ascend
status: paused
wake: on_jobs
budget_gpu_hours: 150
spent_gpu_hours: 101.54
max_jobs_in_flight: 24
jobs: []
jobs_archived:
- 8368600
- 8453490
- 8453513
- 8453514
- 8471861
- 8453515
- 8453516
- 8471995
- 8471996
- 8488648
- 8453517
- 8489898
- 8488707
- 8488708
- 8471997
- 8453518
- 8453899
- 8471998
- 8471999
- 8489957
- 8489958
- 8489959
- 8489960
last_wake: '2026-11-10T22:00:01-05:00'
next_id: 6
model: null
effort: null
max_turns: 100
timeout_minutes: 60
wakes: 15
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
- 2026-11-09 18:05  e001 done (8368600, 0.55 GPU-h): all regression tests pass, bit-identical old rows; ssmax +0.6 % step time vs softmax, a2 splits free, no NaN; 4/4 predictions hit | data bug found by lm350m-qdf (env.sh reset DATA_SRC: all earlier 125M runs, incl. the Goal's 6/6 baselines, were 2.5 epochs of fineweb10B); ported fix 5c5d8c0; baselines now lm350m-qdf e004 (`125m_fw25`, running) | smoke 8453490 (debug-nextgen) ok on fineweb25B | e002 (ssmax_xs) + e003 (softmax/wstieltjes q3d2 x a4/a2) submitted as one 6-segment 4:15 chain 8453513-8453518 at 2a32fea, est ~65 GPU-h | next: score when the chain ends; e004/e005 gated on e003.
- 2026-11-09 18:55  8453513 tasks 0/1/4 (ssmax_xs, softmax_xs_a4, wstieltjes_q3_d2_xs_a2) FAILED in 11 s on bad node a0339 (CUDA device busy/unavailable at model.to, no code fault); tasks 2/3 training at probe speed | retry in place: chain restarts them in 8453514, a0339 excluded on 8453514-8453518, spare segment 8453899 for task 4 | next: score e002/e003 when the chain ends.
- 2026-11-09 22:25  8453513 segment 1 ended: tasks 2/3 (softmax_xs_a2, wstieltjes_q3_d2_xs_a4) hit the wall limit as planned (exit 4, 3.84 h each), checkpointed at iter 2148 / 1001 and resumed in 8453514; tasks 0/1/4 (a0339 failures) restarted from scratch in 8453514 on good nodes, all 5 training at probe speed (82k / 38k tok/s); 8453513 archived (~7.7 GPU-h) | nothing submitted, chain 8453514-8453518 + 8453899 covers all runs | next: score e002/e003 when the chain ends.
- 2026-11-10 02:05  8453514 segment 2 closed: all 5 tasks exit 4:0 at 3.84-3.87 h (planned wall-limit checkpoint), resumed in 8453515; iters ssmax_xs 2170, softmax_xs_a4 2154, softmax_xs_a2 4390, wstieltjes a4 2030, a2 1020; no NaN, probe-speed throughput; 8453514 archived (19.25 GPU-h, spent 28.99 = 9.74 + sacct sum) | nothing submitted; existing chain 8453515-8453518 + 8453899 covers every run (Stieltjes a2 needs the spare) | next: score e002/e003 as runs finish (softmax_xs_a2 first, in 8453515).
- 2026-11-10 03:15  8453515_2 done: softmax_xs_a2 (2/10) finished, 0.94 h; vs 6/6 softmax_xs: val@2048 +0.042, FW 16k +0.398, ArXiv 2k -0.038, ArXiv 16k -0.206 (4.164 vs 4.370), PG-19 16k +0.007 -> more NoPE heads help ArXiv extrapolation but break FineWeb 16k; e003 interim: 'softmax worsens with NoPE' miss, 'val cost <= 0.01' miss for softmax, e004 gate hit | figure figures/arxiv_vs_length.png + plot_arxiv.py; results copied to results/lm1012/125m_fw25/ | e004 (asentmax_zi_xs_a4/a2) smoke 8471861 ok (debug-nextgen), chain 8471995-8471999 submitted at 03e9f85, ~28 GPU-h; committed total ~93 of 150 | next: score e002/e003 as the 8453515-8453518/8453899 chain finishes; e005 (a0) gated on a4 monotonicity.
- 2026-11-10 13:35  finished: softmax_xs_a4 (8.80 GPU-h) and wstieltjes_q3_d2_xs_a4 (18.78 GPU-h); softmax ArXiv 16k 6/6 -> 4/8 -> 2/10 = 4.370 -> 4.349 -> 4.164 (monotone better), FW 16k 3.424 -> 3.464 -> 3.822 (monotone worse); wstieltjes a4 ArXiv 16k 3.813 (0.536 below softmax a4, 0.106 above 6/6 ASEntmax 3.707), val@2048 3.479 | ssmax_xs segment 4 (8453516_0) OOM on resume at an eval boundary (iter 4250, fragmentation), no segment left: retry with expandable_segments (+ mb8 fallback segment), smoke 8488648 ok, submitted 8488707/8488708 at 9c4bcad (~1.3 GPU-h) | e004 segments 1-2 closed (asentmax a4/a2 at iter ~3650), running in 8471997 | archived 8453515, 8453516, 8471995, 8471996, 8488648 | next: score e002 when the retry ends; e003 needs wstieltjes a2 (8453518/8453899) and the lm350m-qdf 6/6 Stieltjes cell; e005 (a0) decision after e003 closes.
- 2026-11-10 14:15  no run finished this wake: 8453517_3 (Stieltjes a4, already scored) done; 8453517_4 (wstieltjes_q3_d2_xs_a2) planned wall-limit exit 4 at iter ~4100, resumed in 8453518_4 (iter 4120, ~2.5 h + evals left); 8453518_3 no-op; ssmax_xs retry 8488707 at iter 4570 (ends in ~25 min); e004 asentmax a4/a2 iter ~3870 in 8471997 | archived 8453517, smoke 8489898 | e005 softmax half (`softmax_xs_a0`, gate = softmax monotone through 2/10) smoke 8489898 ok (debug-nextgen), chain 8489957-8489960 submitted at b93be9e, ~9 GPU-h; committed total ~103 of 150 | next: score e002 (ssmax) and e003 (Stieltjes a2 + 6/6 cell) as they land; Stieltjes a0 (~19 GPU-h, `ROWS=wstieltjes_q3_d2_xs_a0 NSEG=6 bash osc/submit_hs_e005.sh`) only if Stieltjes 6/6 -> 4/8 -> 2/10 is monotone.
- 2026-11-10 14:25  trigger 8453518_3 = no-op task (Stieltjes a4, scored 13:35), 6 s; nothing new to score | in flight: ssmax_xs retry 8488707 TRAIN_DONE, evals running (score e002 next wake); 8453518_4 Stieltjes a2 resumed iter 4120; e004 asentmax a4/a2 in 8471997 (3.5 h into segment 3); e005 softmax_xs_a0 seg 1 in 8489957 | nothing submitted (8453518 kept in jobs, task 4 running) | next: score e002 when 8488707 ends; e003 when Stieltjes a2 lands; Stieltjes a0 decision then.
- 2026-11-10 14:45  e002 done (ssmax_xs, 8.95 GPU-h; retry 8488707 1.17 h, spare 8488708 no-op): val@2048 3.427 (softmax 3.431), ArXiv 2k/16k 3.941/4.109 vs softmax 4.051/4.370 and ASEntmax 3.972/3.707 -> log-n temperature closes 39 % of the softmax-ASEntmax 16k gap for free at train length, but the ArXiv curve still rises with length; the falling-curve signature needs sparsity and/or per-query scale; 5/6 predictions hit (miss: 'ArXiv falls 2k->16k'); backlog 6 (ssmax lr 6e-4) not triggered | archived 8488707, 8488708 | nothing submitted (Stieltjes a0 gated on e003) | in flight: Stieltjes a2 8453518_4 (+spare 8453899), e004 asentmax a4/a2 8471997-9, e005 softmax_xs_a0 8489957-60 | next: e003 when Stieltjes a2 + lm350m-qdf 6/6 cell land, then Stieltjes a0 decision.
- 2026-11-10 14:57  trigger: 8453518_3 no-op (6 s); 8471997_0/1 (e004 asentmax_zi_xs_a4/a2 segment 3) planned wall-limit exit 4 at 3.84/3.85 h, resumed in 8471998 at iter 4200/4170 of 4770 (53k/51k tok/s, ~1.7 h + evals left, no NaN) | archived 8471997 (+7.69 GPU-h, spent 94.89) | nothing finished to score, nothing submitted | in flight: Stieltjes a2 8453518_4 (+spare 8453899), e004 8471998-9, e005 softmax_xs_a0 8489957-60 | next: score e004 when 8471998 ends; e003 when Stieltjes a2 + lm350m-qdf 6/6 cell land, then Stieltjes a0 decision.
- 2026-11-10 15:05  trigger: 8453518_3 no-op again (6 s, Stieltjes a4 scored 13:35); nothing finished, nothing submitted | progress: Stieltjes a2 8453518_4 iter 4470/4770 (38k tok/s); e004 asentmax a4/a2 8471998 iter 4330/4280 (53k/51k tok/s); e005 softmax_xs_a0 8489957 iter 690 (83k tok/s); no NaN | spent_gpu_hours left at driver value 88.53 | next: score e004 when 8471998 ends (~1.3 h + evals); e003 when Stieltjes a2 + lm350m-qdf 6/6 cell land, then Stieltjes a0 decision.
- 2026-11-10 17:25  e003 done (55.00 GPU-h): Stieltjes a2 (8453518_4, 18.78 GPU-h) + 6/6 Stieltjes ref landed; ArXiv 16k Stieltjes 6/6 -> 4/8 -> 2/10 = 3.981 -> 3.813 -> 3.829, softmax 4.370 -> 4.349 -> 4.164, gap 0.389 / 0.536 / 0.335 (non-monotone); Stieltjes curve falls with length at every split, softmax rises at every split -> the signature is the map, not the split; 3/6 predictions hit (all 3 directional claims miss) | e004 interim: asentmax_zi_xs_a4 done (13.66 GPU-h), ArXiv 16k flat across splits 3.707 / 3.722 / 3.745 (a2 PG-19 eval still running in 8471998_1) | Stieltjes a0 (e005 second half) NOT submitted: gate "monotone through 2/10" not met | spare 8453899 cancelled (task done); archived 8453518, 8453899 | nothing submitted | in flight: 8471998_1 (+spare 8471999), e005 softmax_xs_a0 8489957-60 (iter 1640) | next: score e004 and e005; backlog then empty -> write the direction report and set status done.
- 2026-11-10 17:40  e004 done (27.6 GPU-h): asentmax_zi_xs_a2 finished (8471998_1, 2.27 h; spare 8471999 no-op); ASEntmax ArXiv 16k 6/6 -> 4/8 -> 2/10 = 3.707 -> 3.722 -> 3.745 (flat, inside bounce), FW 16k 3.440 -> 3.472 -> 3.606, val@2048 spread 0.02; 4/6 predictions hit (miss: "a2 improves like softmax", "val cost >= 0.02") -> the split moves softmax only; sparse-map extrapolation does not live on more NoPE heads | figure + RESULTS.md regenerated; archived 8471998, 8471999 | nothing submitted (backlog empty after e005; Stieltjes a0 gate not met) | in flight: e005 softmax_xs_a0 8489957 iter 1830 (83k tok/s, finishes in segment 2-3) | next: score e005, then write the direction report and set status done.
- 2026-11-10 18:05  trigger 8489957_0 (e005 softmax_xs_a0 segment 1): planned wall-limit exit 4 at 3.85 h, checkpoint iter 2180 of 4770, val 3.785 @2000, 83k tok/s, no NaN; resumed in 8489958 (iter 2240) | archived 8489957 (~3.85 GPU-h) | nothing finished, nothing submitted | in flight: 8489958-8489960 (finishes in segment 2-3) | next: score e005, then write the direction report and set status done.
- 2026-11-10 22:05  trigger 8489958_0 (e005 softmax_xs_a0 segment 2): planned wall-limit exit 4 at 3.84 h, checkpoint iter 4365 of 4770, val 3.585 @4250 / 3.551 @4500 (a2 final val@2048 3.473), 84k tok/s, no NaN; resumed in 8489959 (~45 min train + evals left) | archived 8489958 | nothing finished, nothing submitted | in flight: 8489959 (+spare 8489960) | next: score e005, then write the direction report and set status done.
- 2026-11-10 22:45  e005 done (softmax_xs_a0, 8.57 GPU-h; 8489959_0 0.84 h, spare 8489960 no-op): all-NoPE softmax is a cliff: val@2048 3.563 (6/6 3.431, 2/10 3.473), ArXiv 2k 4.053 -> 4k 5.005 -> 16k 7.895, FW 16k 7.645, PG-19 16k 7.258; 5/6 predictions hit (miss: cost 8.57 vs 8.6-10) -> softmax needs >= 2 ALiBi heads beyond train length; with e003/e004 the extrapolation is a property of the map, not the split | figure regenerated (y-axis clipped, a0 annotated), results copied to results/lm1012/125m_fw25/softmax_xs_a0/ | archived 8489959, 8489960 | backlog exhausted -> status paused. USER DECISION: submit Stieltjes a0 (`wstieltjes_q3_d2_xs_a0`, ~19 GPU-h of 48.5 left) despite its pre-registered gate (monotone 6/6->2/10) not being met? It would test whether a windowed sparse map avoids the all-NoPE cliff; else mark the direction done.
