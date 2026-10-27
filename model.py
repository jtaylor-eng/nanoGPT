"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Stieltjes / NAPE / adaptive-scale additions (fork: jtaylor-eng/nanoGPT, branch stieltjes_lm) ---
# Semantics match jtaylor-eng/asentmax synthetic/src/models/architectures/sparse_gemma.py:
#   NAPE: first n_head//2 heads ALiBi with linear slopes 1/(1..H/2), the rest NoPE (no positional signal).
#   Stieltjes: w_j = [(lam - s_j)^-q - d^-q]_+ with lam solved per row (window d=0 -> dense).
#   AS ('adapt-softplus-tanh'): NoPE-head queries scaled by 1 + softplus(Wb x) * log(pos+2)^(2 tanh(Wg x)).
from stieltjes.stieltjes_eager import stieltjes_normalize, stieltjes_normalize_qd
try:
    from stieltjes.triton_stieltjes import stieltjes_attention as _triton_stieltjes
except Exception:  # no triton (CPU-only box)
    _triton_stieltjes = None
try:
    from stieltjes.adasplash_entmax import sparse_attn as _adasplash_entmax   # AdaSplash alpha-entmax (NAPE slopes)
except Exception:
    _adasplash_entmax = None
try:
    from entmax import entmax_bisect as _entmax_bisect                        # eager reference (pip entmax)
except Exception:
    _entmax_bisect = None


def nape_slopes(n_head, slope_type='linear'):
    """ALiBi slopes for the first n_head//2 heads, zeros (NoPE) for the rest.
    'linear': 1/h (asentmax synthetic-task NAPE; very local, ~e^-1 per token on head 0).
    'exp':    2^(-8h/n_alibi), the original ALiBi geometric slopes spread over the ALiBi heads (0.25 .. 2^-8 for 4
              heads); the paper's LM runs use 'original exponential ALiBi slopes' (arXiv 2506.16640 App. G.3)."""
    n_alibi = n_head // 2
    h = torch.arange(1, n_alibi + 1, dtype=torch.float32)
    if slope_type == 'linear':
        s = 1.0 / h
    elif slope_type == 'exp':
        s = torch.exp2(-8.0 * h / n_alibi)
    else:
        raise ValueError(slope_type)
    return torch.cat((s, torch.zeros(n_head - n_alibi)))


def nape_bias(slopes, T, device, dtype, causal_fill=None):
    """(1, H, T, T) additive bias: slope_h * min(j - i, 0); optionally causal_fill above the diagonal.
    Built head by head in the target dtype so long-context eval (16k: 8 x T x T) does not allocate fp32 temporaries."""
    i = torch.arange(T, device=device)
    diff = (i[None, :] - i[:, None]).clamp(max=0)
    out = torch.empty(1, slopes.numel(), T, T, device=device, dtype=dtype)
    for h, s in enumerate(slopes.tolist()):
        out[0, h] = (diff * s).to(dtype) if s != 0 else 0.0
    if causal_fill is not None:
        out.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), 1).view(1, 1, T, T), causal_fill)
    return out


@torch.compiler.disable
def _entmax_fused(q, k, v, alpha, niter, slopes):
    # AdaSplash applies 1/sqrt(D) internally; slopes are added to the scaled scores as in make_bias_window.
    B, _, T, _ = q.shape
    varlen = torch.full((B,), T, dtype=torch.long, device=q.device)   # kernel needs varlen unless T is a power of 2
    return _adasplash_entmax(q.contiguous(), k.contiguous(), v.contiguous(), alpha=alpha, is_causal=True, varlen=varlen,
                             niter=niter, alibi_slopes=None if slopes is None else slopes.to(q.device, q.dtype))


@torch.compiler.disable
def _stieltjes_fused(q, k, v, sm_scale, sq, num_iter, window, slopes):
    return _triton_stieltjes(q.contiguous(), k.contiguous(), v.contiguous(), causal=True, sm_scale=sm_scale,
                             stieltjes_q=sq, num_iter=num_iter, window=window, alibi_slopes=slopes)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.attn_type = config.attn_type
        self.pos_emb = config.pos_emb
        self.head_dim = config.n_embd // config.n_head
        if self.attn_type not in ('softmax', 'stieltjes', 'entmax'):
            raise ValueError(f"attn_type {self.attn_type!r}")
        self.entmax_alpha = float(config.entmax_alpha)
        self.entmax_niter = int(config.entmax_niter)
        if self.pos_emb not in ('learned', 'nape'):
            raise ValueError(f"pos_emb {self.pos_emb!r}")
        if self.pos_emb == 'nape':
            self.register_buffer('nape_slopes', nape_slopes(config.n_head, config.nape_slope_type), persistent=False)
        self.stieltjes_q = float(config.stieltjes_q)
        self.stieltjes_window = None if config.stieltjes_window <= 0 else float(config.stieltjes_window)
        self.stieltjes_num_iter = int(config.stieltjes_num_iter)
        self.stieltjes_impl = config.stieltjes_impl
        # adaptive length scale on the NoPE heads (all heads if no NAPE)
        self.attn_scale = config.attn_scale
        if self.attn_scale not in ('none', 'as'):
            raise ValueError(f"attn_scale {self.attn_scale!r}")
        if self.attn_scale == 'as':
            self.n_scaled = config.n_head - config.n_head // 2 if self.pos_emb == 'nape' else config.n_head
            self.scale_beta = nn.Linear(config.n_embd, self.n_scaled, bias=True)
            self.scale_gamma = nn.Linear(config.n_embd, self.n_scaled, bias=True)
            self.attn_scale_zero_init = float(config.attn_scale_zero_init)
        self.collect_stats = False   # set by eval_lengths.py: materialise P and record entropy/support
        self.last_stats = None
        self.capture_qk = False      # set by residual_probe.py: keep the (scaled) q, k of the last forward
        self.last_qk = None
        # diagnostics (eval only): force the per-head eager path; per-head q / window overrides (tensors of shape (H,)
        # or None); cap NoPE heads (slope == 0) to the last nope_cap keys (0 = no cap)
        self.force_per_head = False
        self.head_q = None
        self.head_window = None
        self.nope_cap = 0
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def reset_scale_params(self):
        """AS zero-init (asentmax attn_scale_zero_init): scaler = 1 + eps for every query at step 0."""
        if self.attn_scale != 'as' or self.attn_scale_zero_init <= 0:
            return
        eps = self.attn_scale_zero_init
        with torch.no_grad():
            self.scale_beta.weight.zero_(); self.scale_gamma.weight.zero_(); self.scale_gamma.bias.zero_()
            self.scale_beta.bias.fill_(math.log(math.expm1(eps)))

    def _apply_scale(self, q, x):
        B, H, T, D = q.shape
        log_pos = torch.log(torch.arange(2, T + 2, device=q.device, dtype=torch.float32)).view(1, 1, T, 1)
        beta = F.softplus(self.scale_beta(x).float()).view(B, T, self.n_scaled, 1).transpose(1, 2)
        gamma = 2.0 * torch.tanh(self.scale_gamma(x).float()).view(B, T, self.n_scaled, 1).transpose(1, 2)
        scaler = (1.0 + beta * log_pos ** gamma).to(q.dtype)
        return torch.cat((q[:, :H - self.n_scaled], q[:, H - self.n_scaled:] * scaler), dim=1)

    def _scores(self, q, k):
        T = q.shape[2]
        s = (q.float() @ k.float().transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if self.pos_emb == 'nape':
            s = s + nape_bias(self.nape_slopes, T, q.device, torch.float32)
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        return s.masked_fill(~causal, float('-inf') if self.attn_type in ('softmax', 'entmax') else -1e30)

    def _probs(self, q, k):
        s = self._scores(q, k)
        if self.attn_type == 'softmax':
            return F.softmax(s, dim=-1)
        if self.attn_type == 'entmax':
            return _entmax_bisect(s, alpha=self.entmax_alpha, dim=-1, n_iter=50)
        return stieltjes_normalize(s, q=self.stieltjes_q, num_iter=self.stieltjes_num_iter, window=self.stieltjes_window)

    def _probs_head(self, q, k, h):
        # q, k: (B, 1, T, D) for head h; same math as _probs but with that head's slope only
        T = q.shape[2]
        s = (q.float() @ k.float().transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if self.pos_emb == 'nape':
            s = s + nape_bias(self.nape_slopes[h:h + 1], T, q.device, torch.float32)
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        if self.nope_cap > 0 and self.pos_emb == 'nape' and float(self.nape_slopes[h]) == 0.0:
            causal = causal & torch.ones(T, T, dtype=torch.bool, device=q.device).triu(-(self.nope_cap - 1))
        s = s.masked_fill(~causal, float('-inf') if self.attn_type in ('softmax', 'entmax') else -1e30)
        if self.attn_type == 'softmax':
            return F.softmax(s, dim=-1)
        if self.attn_type == 'entmax':
            return _entmax_bisect(s, alpha=self.entmax_alpha, dim=-1, n_iter=50)
        if self.head_q is not None or self.head_window is not None:
            # per-head q / d as tensors (differentiable: used by the q/d gradient probe)
            hq = self.head_q[h] if self.head_q is not None else torch.tensor(self.stieltjes_q, device=s.device)
            hd = self.head_window[h] if self.head_window is not None else torch.tensor(self.stieltjes_window or 0.0, device=s.device)
            return stieltjes_normalize_qd(s, hq, hd, num_iter=self.stieltjes_num_iter)
        return stieltjes_normalize(s, q=self.stieltjes_q, num_iter=self.stieltjes_num_iter, window=self.stieltjes_window)

    def _head_stats(self, p):
        # p: (B, 1, T, T) fp32, causal. Scalars per stat: mean over all rows and over last-quarter rows.
        T = p.shape[-1]
        ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)              # (B,1,T)
        nnz = (p > 0).sum(-1).float()
        pmax = p.amax(-1)
        tail = slice(3 * T // 4, T)
        out = {k: v.mean().cpu() for k, v in (('entropy', ent), ('support', nnz), ('pmax', pmax))}
        out.update({k + '_tail': v[:, :, tail].mean().cpu() for k, v in (('entropy', ent), ('support', nnz), ('pmax', pmax))})
        return out

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if self.attn_scale == 'as':
            q = self._apply_scale(q, x)
        if self.capture_qk:
            self.last_qk = (q.detach(), k.detach())

        if self.collect_stats or self.force_per_head:
            # materialised eager path, one head at a time (a 16k x 16k fp32 map is 1 GB per head)
            ys, stats = [], []
            for h in range(self.n_head):
                att = self._probs_head(q[:, h:h + 1], k[:, h:h + 1], h)
                if self.collect_stats:
                    stats.append(self._head_stats(att.detach()))
                ys.append(att.to(v.dtype) @ v[:, h:h + 1])
                del att
            y = torch.cat(ys, dim=1)
            if self.collect_stats:
                self.last_stats = {key: torch.stack([s[key] for s in stats]) for key in stats[0]}
        elif self.attn_type in ('stieltjes', 'entmax') and (self.stieltjes_impl == 'eager' or not q.is_cuda):
            att = self._probs(q, k)
            y = self.attn_dropout(att).to(v.dtype) @ v
        elif self.attn_type == 'entmax':
            slopes = self.nape_slopes if self.pos_emb == 'nape' else None
            y = _entmax_fused(q, k, v, self.entmax_alpha, self.entmax_niter, slopes)
        elif self.attn_type == 'stieltjes':
            slopes = self.nape_slopes if self.pos_emb == 'nape' else None
            y = _stieltjes_fused(q, k, v, 1.0 / math.sqrt(self.head_dim), self.stieltjes_q,
                                 self.stieltjes_num_iter, self.stieltjes_window, slopes)
        elif self.pos_emb == 'nape':
            # softmax + NAPE: additive bias with the causal -inf folded in (mem-efficient SDPA backend)
            mask = nape_bias(self.nape_slopes, T, q.device, q.dtype, causal_fill=float('-inf'))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0)
        elif self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention (stock nanoGPT path, learned positions + softmax only)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attn_type: str = 'softmax'      # 'softmax' | 'stieltjes' | 'entmax'
    entmax_alpha: float = 1.5       # alpha-entmax (AdaSplash kernel; eager entmax_bisect for stats)
    entmax_niter: int = 10          # AdaSplash Halley-bisection iterations for tau
    pos_emb: str = 'learned'        # 'learned' (GPT-2 wpe) | 'nape' (half ALiBi / half NoPE heads, no wpe)
    nape_slope_type: str = 'linear' # 'linear' (1/h) | 'exp' (2^-8h/n_alibi, original ALiBi)
    stieltjes_q: float = 4.0
    stieltjes_window: float = 0.0   # d; 0 = dense
    stieltjes_num_iter: int = 30
    stieltjes_impl: str = 'triton'  # 'triton' | 'eager'
    attn_scale: str = 'none'        # 'none' | 'as' (adapt-softplus-tanh on NoPE heads)
    attn_scale_zero_init: float = 0.0  # >0: zero-init the AS projections, scaler = 1 + eps at step 0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        mods = dict(wte = nn.Embedding(config.vocab_size, config.n_embd))
        if config.pos_emb == 'learned':
            mods['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(dict(
            **mods,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        for block in self.transformer.h:
            block.attn.reset_scale_params()

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and 'wpe' in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_all_logits=False):
        device = idx.device
        b, t = idx.size()
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.config.pos_emb == 'learned':
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            tok_emb = tok_emb + self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # NAPE: no absolute positions, so t may exceed block_size (length-extrapolation eval)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        elif return_all_logits:
            logits = self.lm_head(x)
            loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if 'wpe' not in self.transformer:
            return
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
