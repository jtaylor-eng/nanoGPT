# Shared OSC paths for the Stieltjes LM runs. `source osc/env.sh`
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=${REPO:-$SCRATCH/nanoGPT}         # per-direction checkouts pass REPO via sbatch --export
export VENV=${VENV:-$SCRATCH/venvs/nanogpt}   # torch 2.5.1+cu121, triton 3.1.0, numpy 2.0.2 (uv; rebuilt 2026-11-05 after the
                                              # 60-day scratch purge hollowed out the shared venvs/asentmax)
export DATA_SRC=$SCRATCH/nanogpt_data/fineweb10B
export LM_RESULTS=${LM_RESULTS:-$SCRATCH/nanogpt_results/lm1012}
export HF_HUB_OFFLINE=1 WANDB_MODE=offline
export TRITON_CACHE_DIR=$SCRATCH/.triton-cache TORCHINDUCTOR_CACHE_DIR=$SCRATCH/.inductor-cache
