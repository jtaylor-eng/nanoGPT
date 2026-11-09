"""Mean periodic-eval train/val loss over iters >= 2500 per run (smooths the +-0.05 eval bounce)."""
import json, os, statistics as st

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'lm1012', '125m')
print('| row | lr | n evals | train | val | val-train |')
print('|---|---|---|---|---|---|')
for row in sorted(os.listdir(ROOT)):
    for lr in sorted(os.listdir(os.path.join(ROOT, row))):
        p = os.path.join(ROOT, row, lr, 'metrics.jsonl')
        if not os.path.exists(p):
            continue
        ev = [json.loads(l) for l in open(p)]
        ev = {e['iter']: e for e in ev if e.get('kind') == 'eval' and e['iter'] >= 2500}.values()
        if not ev:
            continue
        t = st.mean(e['train_loss'] for e in ev)
        v = st.mean(e['val_loss'] for e in ev)
        print(f'| `{row}` | {lr[2:]} | {len(ev)} | {t:.3f} | {v:.3f} | {v - t:.3f} |')
