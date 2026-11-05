# Stage 5: 350M (GPT-2 medium shape), T=2048, FineWeb, NAPE exp slopes on 8 of 16 heads.
# 0.5M tokens per step (micro-batch x accum = 256 sequences of 2048; micro-batch set per row by run_lm.sh from the
# e001 probe). Default token budget 2.5B (4770 steps, 7 tok/param: the same fineweb25B data as the 125M runs);
# 20 tok/param would be 7B tokens = 13,350 steps (override max_iters / lr_decay_iters). Same optimizer settings
# as the 51M / 125M runs.
dataset = 'fineweb10B'
n_layer = 24
n_head = 16
n_embd = 1024
block_size = 2048
bias = False
dropout = 0.0
pos_emb = 'nape'
batch_size = 8
gradient_accumulation_steps = 32
max_iters = 4770
lr_decay_iters = 4770
warmup_iters = 250
learning_rate = 6e-4       # overridden per job
min_lr = 6e-5              # overridden per job (= lr / 10)
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
eval_interval = 250
eval_iters = 24                   # 24 x 8 x 2048 = 393k val tokens per periodic eval
final_eval_iters = 200            # 3.3M val tokens for the selection number
log_interval = 10
always_save_checkpoint = True
compile = True
wandb_log = False
