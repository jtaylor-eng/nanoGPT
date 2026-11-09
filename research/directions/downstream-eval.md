---
slug: downstream-eval
title: Does the long-document loss advantage show up as retrieval at length and on
  LAMBADA? (needle tasks and lm-eval on the existing 125M checkpoints)
project: nanogpt
cluster: ascend
status: done
wake: on_jobs
budget_gpu_hours: 50
spent_gpu_hours: 3.48
max_jobs_in_flight: 12
jobs: []
jobs_archived:
- 8369321
- 8369150
- 8369151
- 8367629
- 8367630
- 8367868
- 8367869
- 8367871
- 8367870
- 8368482
- 8368483
- 8453350
- 8453572
last_wake: '2026-11-09T18:50:00-05:00'
next_id: 6
model: null
effort: null
wakes: 2
---
# Does the long-document loss advantage show up as retrieval at length and on LAMBADA?

Frontmatter: `status` active|paused|done; `wake` now|on_jobs; `jobs` in flight, finished ids move to `jobs_archived`
(never delete ids); `budget_gpu_hours` is the only compute limit (dollars autoscale).

## Goal
Perplexity does not pin down what a long-context mechanism buys (proposal section "Downstream evaluation"). No
training: evaluate the ten 125M checkpoints that exist on OSC under
`/fs/scratch/PAS2836/jacktaylor/nanogpt_results/lm1012/125m/<row>/lr<lr>/ckpt.pt`
(softmax_xs, asentmax_zi_xs, wstieltjes_q3_d2_xs at lr 1.2e-3 and 6e-4; wstieltjes_q3_d3, q3_d4, q4_d2, q4_d3 at 1.2e-3).
Primary rows: the three at lr 1.2e-3. Settled by: (1) LAMBADA and HellaSwag accuracy per row; (2) single-needle
retrieval accuracy at context 2k, 4k, 8k, 16k per row.

## Theory
At 125M on 2.5B tokens HellaSwag will sit near 30 % and ARC near random for every row, so they cannot separate the
maps; LAMBADA might. The evaluation matching the claim is retrieval at length (RULER S-NIAH, which the ASEntmax paper
reports in its Table 4). Strongest counter-argument: a 125M model trained at T=2048 may fail single-needle
retrieval even at 2k, in which case the task is uninformative at this size and belongs to the 350M models.

## Hypothesis backlog
1. **e001, LAMBADA + HellaSwag (~1 GPU-h).** lm-eval-harness with a thin wrapper exposing the nanoGPT model as a
   log-likelihood LM (GPT-2 tokenizer, which is what the checkpoints use). Install the harness in a NEW venv
   `$SCRATCH/venvs/nanogpt-eval` (copy the torch/triton pins from `venvs/nanogpt`); never modify `venvs/nanogpt`.
   Run the three primary checkpoints, then the rest if cheap. Prediction: LAMBADA within 1 point across the three
   primary rows (60 %); HellaSwag 28-32 % for every row (75 %).
2. **e002, single needle at length (2-4 GPU-h).** Implement `eval_niah.py`: a passkey-style task in the style of
   RULER S-NIAH-1 (one needle "the special magic number is NNNNN" inserted at a uniformly random depth in filler
   text taken from the PG-19 eval set, question at the end, greedy decode of up to 8 tokens, exact match of the
   number), >= 200 samples per context length at 2048, 4096, 8192, 16384, 10 depth bins. All ten checkpoints.
   Prediction: at 8192 the sparse rows beat softmax by >= 20 points (55 %); every row is below 50 % at 16384
   (60 %); at 2048 every row is above 80 % (50 %).
3. **e003, only if any row exceeds 50 % at 16384 on e002:** needle among distractor needles (S-NIAH-2/3 style,
   same generator with 3 decoy numbers). Same cost.
4. **If every row is below 20 % at 2048** the task is uninformative at this size: record it, set `status: done`,
   and state in the log that retrieval evals need the 350M models.

## Constraints
- Files the agent may add: `eval_downstream.py`, `eval_niah.py`, `osc/eval_niah.sbatch`, `osc/eval_lmeval.sbatch`,
  `research/`. Files it may not change: `model.py`, `train.py`, `eval_lengths.py`, `osc/env.sh`, `osc/rows.sh`.
- Checkpoints are read-only. Results go to `$LM_RESULTS/125m/<row>/lr<lr>/{lmeval.json,niah.json}` and are copied
  into `results/lm1012/125m/<row>/lr<lr>/` on this branch.
- Per-experiment ceiling: 5 GPU-h; single jobs on `nextgen`, <= 2 h walltime, no chains needed.
- Report: table rows x {LAMBADA, HellaSwag, NIAH@2k, 4k, 8k, 16k}; PNG `figures/niah_vs_length.png` (accuracy vs
  context length, one line per row) and `figures/niah_depth.png` (accuracy by depth bin at 8k) for the primary rows.

## Log
- 2026-11-06 wake 1: built venv `nanogpt-eval` (lm-eval 0.4.13, torch 2.5.1+cu121), prefetched lambada_openai/hellaswag
  and tiktoken gpt2; cloned OSC checkout. Added eval_downstream.py, eval_niah.py (teacher-forced greedy exact match,
  300/length, see e002 Change), osc/eval_{lmeval,niah}.sbatch. Smokes 8367629/8367630 cancelled while pending (script
  exit status masked python failures; fixed). Submitted at 1e4f538: smoke arrays 8367868 (lmeval) / 8367869 (niah),
  3 ckpts each; full arrays 8367870 (e001) / 8367871 (e002), 10 ckpts, afterok on the smokes. Est. ~3 GPU-h total.
  Next wake: check smoke logs (`nanogpt_results/logs/de_*`), niah gen_check agreement must be ~all; if a smoke failed,
  cancel the full array (DependencyNeverSatisfied), fix, resubmit.
- 2026-11-06 wake 2: lmeval smoke 8367868 ok (3/3, ~1 min each); full e001 array 8367870 pending (Priority). niah smoke
  8367869 exposed a scorer bug: models answer "\nNNNNN", scorer forced " NNNNN", so acc 0 everywhere and gen_check
  agreement 3/5 (greedy right on 2/5 at 2k for softmax and ASEntmax). Cancelled 8367871 and 8367869_2 before start;
  e002 failed. Fixed scoring (two-stage, greedy-equivalent) at 8e5a601 and submitted e003: smoke 8368482 (3 ckpts,
  20/length, gen_check 20) -> full 8368483 (10 ckpts, afterok). Est. ~4 GPU-h. Backlog item 3 (distractors) is now e004.
  Next wake: niah_smoke2 gen_check agreement must be >= 19/20 per row; else cancel 8368483 (second scorer failure: pause).
- 2026-11-06 wake 3: stale trigger (cancellations of 8367869_2 / 8367871, already recorded in wake 2; e002 failed).
  No new finishes: e001 full array 8367870 running (tasks 0-1 started, 2-9 pending); e003 smoke 8368482 and full
  8368483 pending. Nothing submitted (e004 is gated on e003's 16k result). Next wake: same gates as wake 2.
- 2026-11-06 wake 4: e001 done (0.59 GPU-h): LAMBADA 22.9-23.2 / HellaSwag acc_norm 29.4-29.9 for the three primary
  rows, within 1 SE; all 3 predictions true; only LR separates rows. e003 smoke gate failed (agreement 17/20 softmax,
  16/20 ASEntmax; 20/20 q3/d2): all misses are greedy-right answers "\nNNNNN" written with the needle's digit pieces
  (" 7"+"48"+"62" -> "7","48","62") while stage 2 forced the bare-number BPE ("748","62"). Different cause from e002,
  same class, so instead of pausing I removed the class: score() now IS greedy decoding (batched, early exit; ~1.15x
  e003 cost), n 300 -> 200/length for the 5 GPU-h ceiling. Cancelled 8368483 unstarted; e003 failed. Submitted e004 at
  65eac28: smoke 8369150 (3 ckpts, 20/length) -> full 8369151 (10 ckpts, afterok). Est. ~3 GPU-h. Distractors = e005.
  Smoke side evidence (n=20): answer log-prob 2k->16k softmax -4.9 -> -18.8, ASEntmax -5.2 -> -10.0, q3/d2 -3.6 -> -6.4.
  Next wake: smoke gate = batched vs single agreement >= 19/20 per row and 2k greedy correct 5/7/5 (+-1); else cancel
  8369151 and pause (a third scorer failure needs the user). Then apply backlog item 4 (all rows < 20 % at 2k -> done).
- 2026-11-06 wake 5: e004 failed: smoke 8369150_0-2 crashed in ~15 s (eval_niah.py:111 shape mismatch 2046 vs 2047);
  batched greedy loop assumed one prompt length per batch, but len(" NNNNN") in BPE varies by sample. A crash before
  scoring, not a scoring-rule failure, so retried rather than paused. 8369151 cancelled by Slurm (afterok), never ran.
  Fix at 475ab90: per-sample right-pad, read at own last position. e005 smoke 8369321 (3 ckpts, 20/length,
  niah_smoke5.json) submitted ALONE, pending at end of wake. Distractors are now e006.
  Next wake: if 8369321 crashed again for the same cause -> pause. Else gate: agreement >= 19/20 per row and 2k greedy
  correct 5/7/5 (+-1); pass -> submit full array 0-9 (`sbatch --parsable osc/eval_niah.sbatch` at 475ab90, ~2.9 GPU-h)
  and add it to e005 jobs; fail -> pause (scorer needs the user).
- 2026-11-09 wake 6: e005 smoke 8369321 done (0.09 GPU-h), gate passed: batched vs single 20/20 every row, 2k greedy
  5/7/4 (softmax/ASEntmax/q3d2, lr 1.2e-3, n=20). Smoke n=20 acc 2k/4k/8k/16k: softmax 25/5/0/0, ASEntmax 35/5/5/0,
  q3d2 20/20/10/5. Submitted full array 8453350 (10 ckpts, 200/length) at 475ab90, est. ~2 GPU-h.
  Next wake: fill e005 from niah.json x10, copy jsons to results/lm1012/125m/, draw figures/niah_vs_length.png and
  niah_depth.png, score predictions. Then e006 (distractors) only if any row > 50 % at 16k, else direction done.
- 2026-11-09 wake 7: 8453350 tasks 0-7,9 FAILED in ~10 s, all on node a0339 ("CUDA-capable device(s) busy or
  unavailable" at model load; unrelated job 8453383_1 also died there): bad node, not code. Task 8 (q4_d2) running on
  a0264 with sane numbers (2k 23.5 %). Resubmitted tasks 0-7,9 at 475ab90 with --exclude=a0339 -> 8453572.
  Next wake: if the retry hits a fresh bad node, exclude it and retry once more; a code failure -> diagnose. When all
  ten niah.json exist: fill e005, copy jsons, figures, score predictions, then e006 or done (gate: any row > 50 % at 16k).
- 2026-11-09 wake 8: e005 done (2.77 GPU-h; retry 8453572 all completed). NIAH exact match 2k/4k/8k/16k, n=200, lr 1.2e-3:
  softmax 26.5/0.5/1.0/1.0, ASEntmax 41.5/9.5/2.5/3.0, q3/d2 31.0/13.5/6.5/5.0; best at length ASEntmax lr6e-4
  46.5/25.5/12.5/6.0. Sparse rows keep a few points past 2k where softmax has ~0, but LR swings (up to 25 pt at 2k)
  exceed between-map gaps on one seed; no ASEntmax vs Stieltjes ranking. Predictions 7/10. Figures niah_vs_length.png,
  niah_depth.png. e006 gate (any row > 50 % at 16k) fails (max 6.0 %): backlog empty, status done. Retrieval at
  length needs the 350M checkpoints (lm350m-qdf); eval_niah.py runs on them unchanged. Spent 3.48 / 50 GPU-h.
