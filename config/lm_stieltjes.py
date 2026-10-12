# Stieltjes-vs-softmax LM comparison (asentmax context/lm_plan_1012.md).
# ~51M params (8L x 8H x 512, head_dim 64), FineWeb-10B GPT-2 tokens, T=512, 1.0B tokens.
# Per-row attention settings + LR come from the command line (osc/run_lm.sh).
dataset = 'fineweb10B'
n_layer = 8
n_head = 8
n_embd = 512
block_size = 512
bias = False
dropout = 0.0
pos_emb = 'nape'

# 32 x 512 x 8 = 131,072 tokens/step; 7630 steps = 1.0B tokens
batch_size = 32
gradient_accumulation_steps = 8
max_iters = 7630
lr_decay_iters = 7630
warmup_iters = 400
learning_rate = 1.5e-3     # overridden per job
min_lr = 1.5e-4            # overridden per job (= lr / 10)
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0

eval_interval = 500
eval_iters = 100           # 100 x 32 x 512 = 1.6M val tokens per periodic eval
final_eval_iters = 800     # 13M val tokens for the selection number (final.json)
log_interval = 20
always_save_checkpoint = True   # keep last (cosine-decayed) model; selection = final val loss
compile = True
wandb_log = False
