# Stage 3: same model and token budget as lm_stieltjes.py, trained at T=2048 (the ASEntmax paper's LM context
# length). Tokens per step unchanged (8 x 2048 x 8 = 131,072), so max_iters / schedule / LR grid carry over.
# Keep in sync with lm_stieltjes.py except the five lines marked T2048.
dataset = 'fineweb10B'
n_layer = 8
n_head = 8
n_embd = 512
block_size = 2048                 # T2048
bias = False
dropout = 0.0
pos_emb = 'nape'
batch_size = 8                    # T2048
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
eval_iters = 25                   # T2048: 25 x 8 x 2048 = 410k val tokens per periodic eval
final_eval_iters = 200            # T2048: 3.3M val tokens for the selection number
log_interval = 20
always_save_checkpoint = True
compile = True
wandb_log = False
