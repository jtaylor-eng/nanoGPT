# nanoGPT fork: Stieltjes / entmax / NAPE language-model runs (branch `stieltjes_lm`)

Project brief: `../../CLAUDE.md` (Stieltjes). Remote `fork` = jtaylor-eng/nanoGPT; `origin` is upstream
Karpathy, read-only. Upstream files (`train.py` core loop, `model.py` GPT skeleton, `configurator.py`,
README) stay as close to upstream as the fork edits allow.

## What the fork adds
- `model.py`: `attn_type` in {softmax, entmax, stieltjes}; `pos_emb` in {learned, nape}; NAPE slopes
  (`nape_slope_type` linear = 1/h, exp = ALiBi); adaptive scale `attn_scale=as` with zero-init
  (`attn_scale_zero_init`); Stieltjes args `stieltjes_q`, `stieltjes_window` (0 = dense),
  `stieltjes_num_iter`, `stieltjes_impl`. Half the heads are NoPE under NAPE (`n_scaled`).
- `stieltjes/`: `stieltjes_eager.py` (reference + differentiable `stieltjes_normalize_qd`),
  `triton_stieltjes.py` (fused kernel, same as asentmax), `adasplash_entmax.py` (entmax kernel).
- `eval_lengths.py`: loss vs sequence length on FineWeb (512..4096 -> `lengths.json`; to 16k ->
  `lengths_16k.json`), ArXiv papers (`arxiv.json`), PG-19 (`pg19.json`); `--stats_lengths` adds
  per-head support stats. `qd_grad_probe.py`, `residual_probe.py`: diagnostics on a checkpoint.
- `config/`: `lm_stieltjes.py` 51M T=512 1B tokens; `lm_stieltjes_t2048.py` 51M T=2048;
  `lm_stieltjes_125m.py` 125M T=2048 2.5B tokens (0.5M tok/step, 4770 steps); a direction branch adds
  `lm_stieltjes_350m.py` (24L x 16H x 1024). Train metrics stream to `metrics.jsonl`, final to `final.json`.
- `tests/test_attn_paths.py`: eager vs fused paths agree (needs CUDA for the fused ones).

## Row grammar (`osc/rows.sh`, the only place a row name is parsed)
`<base>[_zi][_xs]`, base in `softmax | softmax_as | entmax | asentmax | [as][w]stieltjes[_q<Q>][_d<D>]`.
`as` = adaptive scale, `_zi` = zero-init scale (every AS row so far), `w` = windowed (`_d` default 2),
`_q` default 4, `_xs` = exp NAPE slopes (every LM row since stage 3). Example: `wstieltjes_q3_d2_xs`.

## OSC pipeline (`osc/`, all paths in `osc/env.sh`; `source` it, it also sets HF/WANDB offline)
- `REPO=$SCRATCH/nanoGPT` (main checkout) or a direction checkout under `$SCRATCH/wt/nanogpt/<slug>`
  (env.sh accepts an override on direction branches). Commit before submitting; `submit_lm.sh` refuses a dirty tree.
- `VENV`: `$SCRATCH/venvs/nanogpt` since Nov 5 2026 (torch 2.5.1+cu121, triton 3.1.0, numpy 2.0.2,
  pip `entmax` 1.3 for the eager entmax path); `venvs/asentmax` was hollowed out by the scratch purge.
- Data: `$SCRATCH/nanogpt_data/fineweb10B` (51M) and `fineweb25B` (125M+, via `DATA_SRC_OVERRIDE`),
  staged to `$TMPDIR` by `osc/stage.sh`. Long-doc eval sets: `data/arxiv_long/arxiv_long.bin`, `data/pg19_long/pg19_long.bin`.
- `run_lm.sh <row> <lr> [seed]`: train (resumes from `ckpt.pt` if valid and `TRAIN_DONE` absent) then the
  four evals, each skipped if its json exists. `CONFIG=` picks the config; results go to
  `$LM_RESULTS/<config tag>/<row>/lr<lr>[_s<seed>]/` (`$LM_RESULTS=$SCRATCH/nanogpt_results/lm1012`).
  `WALL_LIMIT_S` (default 9000) makes train.py checkpoint and exit before the segment ends.
- `lm_array.sbatch` (nextgen, 1 GPU, 3:30 default): array over lines of `$MANIFEST` = `<row> <lr> [seed]`.
  Long runs are chains of segment arrays joined with `--dependency=afterany`; each segment resumes.
  A run that finishes early just exits in its remaining segments. `probe.sbatch` + `probe_decide.py`
  pick `stieltjes_num_iter` (written to `$LM_RESULTS/NUM_ITER`, 15 at 125M+). `diag*.sbatch`, `eval_*.sbatch`: diagnostics and re-evals on existing checkpoints.
- Local copies of finished runs: `results/lm1012/<tag>/<row>/lr<lr>/` with the five json/jsonl files and
  `run.log`; figures next to them (`stage4b_qd.png`). Reports are written in `../asentmax/context/reproduction_lm_<MMDD>.md`.

## Measured costs (nextgen A100 40 GB, T=2048, 0.5M tok/step, 4770 steps)
| size | softmax | ASEntmax | windowed Stieltjes q3/d2 |
|---|---|---|---|
| 125M | 8.3 h (83k tok/s, mb16) | 13.4 h (52k, mb8) | 18.1 h (38k, mb8, num_iter 15) |
| 350M | 21.5 h (32k, mb8; mb16 OOM) | 33 h (21k, mb8) | 46 h (15k, mb8) |
Sparse kernels: 2.1x softmax per step; no speed from sparsity yet (no block skipping).

## Rules
- Never change the protocol mid-stage (data, T, tokens, eval sets, seed 1337, selection by val loss).
- Report val@2048 (or train length), FineWeb 16k, ArXiv 2k -> 16k, PG-19 16k, with row, lr, seed.
  Final-eval numbers bounce +-0.05 between periodic evals for Stieltjes rows; say so when gaps are that size.
- Add a row by extending `rows.sh` and `model.py` together; add a size by adding a `config/lm_stieltjes_<size>.py`.
