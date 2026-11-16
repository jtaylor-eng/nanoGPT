---
slug: stieltjes-kernel-speed
title: How fast can the fused windowed-Stieltjes attention kernel run at 125M / T=2048
  on an A100 with unchanged outputs and gradients? (profile, then optimise the Triton
  kernel)
project: nanogpt
type: direction
cluster: ascend
status: active
wake: on_jobs
budget_gpu_hours: 15
spent_gpu_hours: 2.8
max_jobs_in_flight: 4
jobs:
- 8614758
- 8614759
jobs_archived:
- 8603546
- 8603935
- 8603936
- 8612613
- 8612640
- 8612641
- 8613034
- 8613091
- 8614500
- 8614519
- 8614520
- 8614747
last_wake: '2026-11-16T07:34:37-05:00'
next_id: 6
model: claude-fable-5-2
effort: high
max_turns: 120
timeout_minutes: 60
wakes: 6
---
# How fast can the fused windowed-Stieltjes attention kernel run at 125M / T=2048 on an A100 with unchanged outputs and gradients? (profile, then optimise the Triton kernel)

Frontmatter fields: `status` active|paused|done. `wake` now|on_jobs (the loop runs this
direction on the next tick when `now`, otherwise only when a listed job finishes).
`jobs` = ids currently in flight; finished ids move to `jobs_archived` (never delete ids,
they feed the GPU-hour accounting). `budget_gpu_hours` is a hard stop. `model`/`effort`/`max_turns`/`timeout_minutes` override the registry defaults for this direction only (null = registry). Optional `max_wakes: N` pauses the direction after N wakes.

## Goal
Minimise the training step time of the Stieltjes rows with the map itself unchanged. User decision 2026-11-15: no
fixed target, "as fast as it gets within the budget"; only changes that pass the correctness gate below count.

Measured baselines (direction lm350m-qdf, 125M = 12L x 12H x 768, T=2048, 524,288 tokens per step, micro-batch 16 x
accum 16 for softmax and 8 x 32 for the sparse rows (`osc/run_lm.sh`: micro-batch 16 does not fit 40 GB with the
sparse kernels; the memory that forces this is itself worth measuring in e001), bf16, torch.compile on, A100 40 GB `nextgen`, median `time ...ms` of the run logs, clean reruns e005-e008):

| row | s/step | GPU-h per 4770-step run |
|---|---|---|
| softmax_xs (SDPA + NAPE bias) | 6.36 | 8.7 |
| asentmax_zi_xs (`stieltjes/adasplash_entmax.py`, entmax_niter 10) | 10.0 | 13.6 |
| wstieltjes_q3_d2_xs (num_iter 15) | 13.7 | 18.7 |
| aswstieltjes_q3_d2_zi_xs (num_iter 15) | 14.0 | 19.1 |

Primary metric: s/step of `wstieltjes_q3_d2_xs` at that configuration. Secondary: fwd and bwd kernel time at the
training shape (B=8, H=12, N=2048, D=64, bf16, causal, NAPE slopes); eval forward throughput at N=16384 must not
regress. Reference points: ASEntmax 10.0 s/step, softmax 6.36. The 350M Stieltjes runs cost 48 GPU-h each, so every
10 % here is ~5 GPU-h per 350M run.

## Theory
`stieltjes/triton_stieltjes.py` (437 lines): forward = 1 row-max sweep + NUM_ITER Newton sweeps + 1 output sweep over
K, each recomputing the QK^T tile (17 sweeps at num_iter 15); backward = 3 kernels (delta; dK,dV; dQ), each recomputing
scores. bf16 tiles are 64x32 (`_pick_blocks`, measured fastest on A100 in asentmax job 7462343). Nothing has been
profiled at the LM shape: it is not known how the 7.3 s/step over softmax splits between the Newton sweeps, the output
sweep, the three backward kernels, and non-kernel overhead (`.contiguous()` copies, the `torch.compiler.disable`
boundary in `model.py:_stieltjes_fused`, the AS scale). So the first experiment is a profile, and every later
candidate is chosen from it. Strongest counter-argument: the sweeps may already be bound by the QK^T dot per tile, in
which case only removing sweeps (which touches the solver) helps, and the correctness gate limits that.

Do not rely on memory for Triton behaviour: read the docs / source of the installed version (triton 3.1.0, torch
2.5.1+cu121 in `$SCRATCH/venvs/nanogpt`) for anything you depend on (`tl.dot` precision modes, `num_warps`,
`num_stages`, `tl.range` loop options, autotuning), and treat every speed claim as unproven until it is a measured
A/B on the A100.

## Correctness gate (hard; a candidate that fails it is recorded as failed and not kept)
Build this in e001, before any kernel edit:
- Freeze the baseline: copy `stieltjes/triton_stieltjes.py` at the base commit to `stieltjes/triton_stieltjes_ref.py`
  (never edited afterwards).
- `tests/test_attn_paths.py` passes unchanged (fused vs eager `stieltjes_normalize`, outputs and gradients, fp32 and
  bf16, `SLOPES=linear` and `SLOPES=exp`).
- New `tests/test_kernel_equiv.py`: candidate vs eager AND candidate vs frozen baseline, on output and on dq, dk, dv:
  - random q/k/v at N in {64, 300, 2048, 4096}; windowed (q3/d2, q4/d2, q3/d1.5), dense q4, one non-integer q
    (e.g. 2.5); fp32 and bf16; with and without NAPE slopes; a non-multiple-of-tile N.
  - real q/k/v captured from a trained model (the `capture_qk` hook in `model.py`, as `residual_probe.py` uses it):
    checkpoint `$LM_RESULTS/125m_fw25_rng/wstieltjes_q3_d2_xs/lr1.2e-3/ckpt.pt` and the AS row
    `aswstieltjes_q3_d2_zi_xs/lr1.2e-3/ckpt.pt`, all 12 layers, FineWeb val at N=2048 and ArXiv at N=16384 (eager
    at 16k one head at a time). Trained heads have supports of ~20-40 keys; random logits do not exercise that.
  - pass = candidate-vs-eager relative error within the existing tolerances of `test_attn_paths.py` AND not more than
    1.2x the frozen baseline's error vs eager on the same input.
- Solver residual: max over rows of `|S - 1|` (`stieltjes_solver_residual`) at num_iter 15 on the real inputs is not
  worse than the frozen baseline's.
- End to end: `eval_lengths.py` loss of the trained wstieltjes_q3_d2 checkpoint at 2048 (FineWeb) and 16384 (ArXiv)
  with the candidate within 1e-3 of the frozen baseline.
Allowed: anything that passes (tiling, loop structure, launch parameters, reusing work across sweeps, merging
kernels, early exit that meets the residual bound, memory layout, the call site in `model.py`). Not allowed: changing
the map (a different normalisation, a truncated support, fewer Newton sweeps that fail the residual bound, lower
precision that fails the tolerance), or changing num_iter / q / d defaults of the rows.

## Hypothesis backlog
1. **e001, profile + gate (no kernel change, <= 2 GPU-h).** `bench_kernel.py` (start from
   `~/Desktop/research/Stieltjes/submodules/asentmax/experiments/osc/oneoff/kernel_bench.py`): median of >= 3 repeats
   of fwd kernel, each of the 3 bwd kernels, and fwd+bwd, at the training shape and at (1, 12, 16384, 64) forward
   only; fwd time vs NUM_ITER in {0, 5, 15, 30} to isolate the per-sweep cost; the entmax kernel at the same shape as
   a reference. Then a 30-step timing of the full 125M step for softmax_xs, asentmax_zi_xs, wstieltjes_q3_d2_xs
   (compile on, real FineWeb batches, micro-batches as in `osc/run_lm.sh`: 16 softmax, 8 sparse; record peak memory) with attention time separated from the rest (CUDA-synced timers
   or the torch profiler). Output: a table that accounts for the 13.7 s. Prediction: the measured 125M step
   reproduces 13.7 +- 0.7 s (80 %); the forward kernel is > 50 % of the Stieltjes attention time (60 %); attention is
   > 60 % of the Stieltjes step (65 %).
2. **e002+, one candidate per record, ordered by the e001 table** (largest measured share first). Candidate pool,
   as hypotheses to test, not instructions: per-sweep cost of the Newton loop (what is recomputed per iteration that
   could be kept); number of sweeps (better start, higher-order step, early exit; all bounded by the residual gate);
   tile sizes, `num_warps`, `num_stages` per kernel at this shape; merging backward kernels that share a grid;
   skipping tiles that cannot be in the support in the backward; copies and dtype conversions around the kernel.
   Each record: the change, the microbench A/B, the gate result. Prediction to be written per record. Overall
   pre-profile prediction (user was given these, score them at the end): >= 20 % lower s/step with the gate passed
   55 %; reaching ASEntmax's 10.0 s/step 30 %.
3. **Confirm end to end** whenever the microbench gain accumulates past 10 % and at the end: 30-step 125M timing of
   wstieltjes_q3_d2_xs, aswstieltjes_q3_d2_zi_xs and a q4 row with the candidate vs the frozen baseline in the same
   job on the same node, plus eval tok/s at 16384.
4. **Stop** when two consecutive candidates each give < 3 % on the primary metric or the budget is spent: set
   `status: done`, final table (candidate x {fwd ms, bwd ms, s/step, eval 16k tok/s, max rel err vs eager, max
   residual}) and `figures/kernel_speed.png` (bar chart of s/step: softmax, ASEntmax, baseline kernel, final kernel).
   State in the last log line the exact commit of the final kernel: the kernel's source of truth is the asentmax repo
   (header line of the file), and the user copies it back; do not push to asentmax.

## Constraints
- Files the agent may change: `stieltjes/triton_stieltjes.py`; in `model.py` only `_stieltjes_fused` and the
  Stieltjes branch of `CausalSelfAttention.forward` (softmax and entmax paths untouched); new files
  `stieltjes/triton_stieltjes_ref.py`, `bench_kernel.py`, `tests/test_kernel_equiv.py`, `osc/kernel_*.sbatch`,
  `research/`. `osc/env.sh` on this branch is stale (points at a purged venv and the old repo path): replace it once
  with the current version, `git show fork/direction/lm350m-qdf:osc/env.sh`, and pass `REPO`, `VENV`, `DATA_SRC`
  explicitly from the sbatch.
- May not change: `train.py`, `eval_lengths.py`, `stieltjes/stieltjes_eager.py`, `stieltjes/adasplash_entmax.py`,
  `osc/rows.sh`, `config/`, the venv (no Triton or torch upgrade).
- Out of scope: the entmax kernel, the PER_HEAD learnable-(q, d) kernel on `direction/learnable-qd`, dense Stieltjes
  beyond keeping it correct.
- Compute: desktop `gpu-run` for the correctness tests only (its GPU has less shared memory, so `_pick_blocks` picks
  32x32 tiles there and timings do not transfer). Every timing on an Ascend A100 40 GB: `debug-nextgen` (< 1 h, one
  GPU); `nextgen` only if a job needs more than the debug limit. A/B pairs run inside one job on one node. No
  training beyond <= 50-step timing probes. Per-experiment ceiling 2 GPU-h.
- Checkpoints under `$LM_RESULTS/125m_fw25_rng/` are read-only. Data: `$SCRATCH/nanogpt_data/fineweb25B`, ArXiv /
  PG-19 eval sets as in `osc/eval_arxiv.sbatch`. Outputs to `$LM_RESULTS/kernel_speed/`, copied into
  `results/lm1012/kernel_speed/` on this branch.

## Log
One line per wake, newest last. Format: `YYYY-MM-DD HH:MM  eNNN submitted <jobids> | eNNN done <metric> | paused: <reason>`
2026-11-15 13:30  e001 submitted 8603935 (ks_time: micro + 125M step profile) 8603936 (ks_gate: attn_paths, real-input and random equivalence, e2e eval), debug-nextgen, sha 4773794 | smoke 8603546 cancelled after 17 min (0.29 GPU-h): all stages passed except bench micro (Triton backward needs a CUDA context in the autograd thread; fixed in the harness) | no kernel change yet | desktop venv is torch 2.10 / triton 3.6, not the OSC 2.5.1 / 3.1.0: gate verdicts are taken on the A100 only
2026-11-16 01:10  e001 done: wstieltjes_q3_d2_xs 13.58 s/step (softmax_xs 6.26, asentmax_zi_xs 9.88), forward kernel 66.8 % of the step (24.1 ms x 384 calls), backward 9.8 %, non-attention 22.7 %; forward cost per Newton sweep grows with NUM_ITER (0.63 / 1.42 / 2.07 ms at 5 / 15 / 30: the unrolled tl.static_range loop); gate built and passing on the baseline (72 real + 38 random cases, e2e FineWeb 2048 3.402708, ArXiv 16k 3.651948, 7,660 tok/s) | e002 submitted 8612640 (ks_time PLAN=ab) 8612641 (ks_gate), debug-nextgen, sha 55569fb: Newton loop as a runtime tl.range loop, nothing else | smoke 8612613 (0.05 GPU-h, tier 2, desktop GPU busy): forward 22.68 -> 10.05 ms at the training shape, 173.9 -> 80.4 ms at 16k, bit-identical on 20 quick cases | spent 1.13 of 15 GPU-h
2026-11-16 02:15  e002 done: Newton loop as a runtime tl.range loop, wstieltjes_q3_d2_xs 13.69 -> 8.91 s/step (-34.9 %; aswstieltjes_q3_d2_zi_xs 13.93 -> 9.16, wstieltjes_q4_d2_xs 13.67 -> 8.86; ASEntmax 9.88, softmax 6.26 from e001), forward kernel 22.69 -> 10.00 ms and affine in NUM_ITER (0.59 ms per sweep), 16k forward 174.3 -> 79.5 ms, ArXiv 16k eval 7,654 -> 16,376 tok/s, gate PASS and bit-identical to the frozen baseline on 72 real + 38 random cases and both e2e evals (jobs 8612640, 8612641, 0.66 GPU-h); 9 of 9 predictions hit | e003 submitted 8613091 (ks_meas, debug-nextgen, sha 0b3c77a): measurement only, launch-parameter sweep of the four kernels (tiles, num_warps, num_stages) and live-tile fraction per Newton sweep on trained q/k, to choose between launch parameters and tile skipping as e004 | smoke 8613034 (0.01 GPU-h, tier 2 after tier-1 desktop checks) passed: layers 0 / 11 of wstieltjes_q3_d2_xs show 56 % / 31 % of 64x32 tiles live at the final lambda at N=2048, no tile coming back | kernel of record so far: stieltjes/triton_stieltjes.py at 55569fb | spent 1.79 of 15 GPU-h
2026-11-16 06:38  e003 done: launch parameters exhausted (default 64x32 tiles / 4 warps / 3 stages is the fastest forward, 10.10 ms; best backward variants -4.6 % and not bit-identical; 8 warps 24-56 % slower), early exit closed (lambda still moving at sweep 15 in 86 % of blocks); trained q/k of wstieltjes_q3_d2_xs lr 1.2e-3 seed 1337 leave 47 % of the swept forward tiles live at sweep 0 and 35 % at the final lambda (N=2048; 29 % at N=16384), 7.5 of 17 sweep-equivalents, no dead tile coming back in 48 cases (job 8613091, 0.08 GPU-h); 8 of 14 predictions hit | e004 submitted 8614519 (ks_time PLAN=ab_trained) 8614520 (ks_gate), debug-nextgen, sha 4d26cd9: tile skipping in the windowed forward, exact by construction (dead flag with a 2e-3 d margin, trusted only while lambda has not dropped by 1e-3 d), timed on trained q/k and on steps resumed from the trained checkpoints as well as from scratch | smoke 8614500 (0.04 GPU-h, tier 2 after tier-1 desktop checks: 480 of 480 skipcheck cases and the quick gradient set bit-identical) passed: forward on trained q/k 9.9 -> 8.7 ms (layer 0) and 6.0 ms (layer 11), 16k 77 -> 60 / 53 ms, bit-identical to the e002 kernel and the frozen baseline; resumed step 8.3-8.6 -> 7.2-7.5 s on 2 timed steps | kernel of record so far: stieltjes/triton_stieltjes.py at 55569fb (e002) until the e004 gate reports | spent 1.91 of 15 GPU-h
2026-11-16 07:34  e004 done: tile skipping in the windowed forward, wstieltjes_q3_d2_xs lr 1.2e-3 seed 1337 step resumed from the trained checkpoint 8.95 -> 7.76 s (-13.3 %; aswstieltjes_q3_d2_zi_xs 9.20 -> 8.00), from scratch 8.92 -> 8.44 (-5.4 %), against 13.69 frozen baseline / 9.88 ASEntmax / 6.26 softmax; forward kernel on trained q/k 10.18 -> 6.42 ms (12-layer mean, FineWeb N=2048), 79.8 -> 47.4 ms at 16k; ArXiv 16k eval 16,376 -> 26,524 tok/s (baseline 7,644); gate PASS, bit-identical to the e002 kernel and the frozen baseline on 72 real + 38 random + 480 skipcheck cases and both e2e evals (jobs 8614519, 8614520, 0.82 GPU-h); 11 of 13 predictions hit; both pre-profile predictions (>= 20 %, below ASEntmax) met | e005 submitted 8614758 (ks_time PLAN=ab_trained CMP=cand_nobskip) 8614759 (ks_gate), debug-nextgen, sha eb2ea77: tile skipping in the three backward kernels from the forward's flags at the final lambda (SKIP_BWD), exact by the same margin argument | smoke 8614747 (0.06 GPU-h, tier 2 after tier-1 desktop checks: 480 of 480 skipcheck cases with gradients, test_attn_paths, 38 random equivalence cases) passed: forward + backward on trained q/k 12.34 -> 10.79 ms (layer 0) and 9.60 -> 7.42 ms (layer 11), dq/dk/dv bit-identical to the e004 kernel and the frozen baseline; resumed step 7.3-7.7 -> 6.6-6.9 s on 2 timed steps | kernel of record so far: stieltjes/triton_stieltjes.py at 4d26cd9 (e004) until the e005 gate reports | spent 2.80 of 15 GPU-h
