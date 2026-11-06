---
slug: downstream-eval
title: "Does the long-document loss advantage show up as retrieval at length and on LAMBADA? (needle tasks and lm-eval on the existing 125M checkpoints)"
project: nanogpt
cluster: ascend
status: active
wake: now
budget_gpu_hours: 50
spent_gpu_hours: 0
max_jobs_in_flight: 12
jobs: []
jobs_archived: []
last_wake: null
next_id: 1
model: null
effort: null
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
