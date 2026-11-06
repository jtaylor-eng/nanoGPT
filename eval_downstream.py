"""
lm-eval-harness zero-shot tasks on a nanoGPT checkpoint (fork addition, direction downstream-eval).

The checkpoint is wrapped as a log-likelihood LM (GPT-2 BPE via tiktoken, the tokenizer the FineWeb shards use).
Batches are right-padded to a multiple of 64 tokens: attention is causal and NAPE / AS depend only on
positions <= t, so padding after the last real token cannot change any scored log-prob.

    python eval_downstream.py --out_dir=<run> [--tasks=lambada_openai,hellaswag] [--limit=0] [--batch_size=64]
writes <run>/<out_name> (default lmeval.json)
"""
import os, sys, json, time
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT


def load_model(out_dir, device='cuda'):
    ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location='cpu', weights_only=False)
    args = dict(ckpt['model_args'])
    model = GPT(GPTConfig(**args))
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()})
    model.to(device).eval()
    return model, args, ckpt['iter_num']


def hidden(model, idx):
    """GPT.forward up to ln_f (so the caller can apply lm_head to only the positions it needs)."""
    x = model.transformer.drop(model.transformer.wte(idx))
    for block in model.transformer.h:
        x = block(x)
    return model.transformer.ln_f(x)


def main():
    from lm_eval import simple_evaluate
    from lm_eval.api.model import TemplateLM
    import tiktoken

    cfg = dict(out_dir='out', tasks='lambada_openai,hellaswag', limit=0, batch_size=64, out_name='lmeval.json',
               max_len=2048, device='cuda')
    for arg in sys.argv[1:]:
        k, v = arg.lstrip('-').split('=', 1)
        cfg[k] = type(cfg[k])(v)

    model, args, it = load_model(cfg['out_dir'], cfg['device'])
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    enc = tiktoken.get_encoding('gpt2')

    class NanoLM(TemplateLM):
        @property
        def eot_token_id(self):
            return enc.eot_token

        def tok_encode(self, string, add_special_tokens=None, **kw):
            return enc.encode_ordinary(string)

        def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kw):
            out = [None] * len(requests)
            items = sorted(((i, (c + d)[-(cfg['max_len'] + 1):], len(d)) for i, (_, c, d) in enumerate(requests)),
                           key=lambda t: -len(t[1]))
            bs = cfg['batch_size']
            for s in range(0, len(items), bs):
                chunk = items[s:s + bs]
                Lp = (max(len(t) for _, t, _ in chunk) - 1 + 63) // 64 * 64
                x = torch.full((len(chunk), Lp), enc.eot_token, dtype=torch.long)
                for j, (_, toks, _) in enumerate(chunk):
                    x[j, :len(toks) - 1] = torch.tensor(toks[:-1])
                with torch.no_grad(), ctx:
                    h = hidden(model, x.to(cfg['device']))
                for j, (i, toks, nc) in enumerate(chunk):
                    n = len(toks) - 1                      # logits at position p predict toks[p + 1]
                    with torch.no_grad(), ctx:
                        lp = F.log_softmax(model.lm_head(h[j, n - nc:n]).float(), dim=-1)
                    tgt = torch.tensor(toks[-nc:], device=lp.device)
                    out[i] = (lp.gather(-1, tgt[:, None]).sum().item(), bool((lp.argmax(-1) == tgt).all()))
            return out

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise NotImplementedError

        def generate_until(self, requests, disable_tqdm=False):
            raise NotImplementedError

    t0 = time.time()
    r = simple_evaluate(model=NanoLM(), tasks=cfg['tasks'].split(','), num_fewshot=0,
                        limit=cfg['limit'] or None, log_samples=False, bootstrap_iters=1000)
    res = {'iter': it, 'model_args': args, 'eval_cfg': cfg, 'eval_s': time.time() - t0,
           'results': r['results'], 'n_samples': r.get('n-samples')}
    for task, m in r['results'].items():
        print(task, {k: v for k, v in m.items() if isinstance(v, float)}, flush=True)
    with open(os.path.join(cfg['out_dir'], cfg['out_name']), 'w') as f:
        json.dump(res, f, indent=1, default=str)
    print('wrote', os.path.join(cfg['out_dir'], cfg['out_name']))


if __name__ == '__main__':
    main()
