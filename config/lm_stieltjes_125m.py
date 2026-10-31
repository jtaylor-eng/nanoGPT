# Stage 4: 125M (GPT-2 small shape), T=2048, 2.5B FineWeb tokens (20 tok/param), NAPE exp slopes on 6 of 12 heads.
# 0.5M tokens per step (16 x 2048 x 16 accum); 4770 steps. Same optimizer settings as the 51M runs.
dataset = 'fineweb10B'
n_layer = 12
n_head = 12
n_embd = 768
block_size = 2048
bias = False
dropout = 0.0
pos_emb = 'nape'
batch_size = 16
gradient_accumulation_steps = 16
max_iters = 4770
lr_decay_iters = 4770
warmup_iters = 250
learning_rate = 1.2e-3     # overridden per job
min_lr = 1.2e-4            # overridden per job (= lr / 10)
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
eval_interval = 250
eval_iters = 12                   # 12 x 16 x 2048 = 393k val tokens per periodic eval
final_eval_iters = 100            # 3.3M val tokens for the selection number
log_interval = 10
always_save_checkpoint = True
compile = True
wandb_log = False
