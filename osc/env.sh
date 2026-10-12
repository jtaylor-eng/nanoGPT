# Shared OSC paths for the Stieltjes LM runs. `source osc/env.sh`
export SCRATCH=/fs/scratch/PAS2836/$USER
export REPO=$SCRATCH/nanoGPT
export VENV=$SCRATCH/venvs/asentmax          # torch 2.5.1+cu121, triton 3.1 (shared with the asentmax runs)
export DATA_SRC=$SCRATCH/nanogpt_data/fineweb10B
export LM_RESULTS=${LM_RESULTS:-$SCRATCH/nanogpt_results/lm1012}
export HF_HUB_OFFLINE=1 WANDB_MODE=offline
export TRITON_CACHE_DIR=$SCRATCH/.triton-cache TORCHINDUCTOR_CACHE_DIR=$SCRATCH/.inductor-cache
