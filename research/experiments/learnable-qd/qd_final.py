"""Final learned q_h, d_h per layer (median [min, max] over heads) for lwstieltjes runs."""
import json, os, statistics as st

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'lm1012', '125m', 'lwstieltjes_q3_d2_xs')
for lr in sorted(os.listdir(ROOT)):
    recs = [json.loads(l) for l in open(os.path.join(ROOT, lr, 'metrics.jsonl'))]
    recs = [r for r in recs if r.get('kind') == 'qd']
    r = recs[-1]
    print(f'## {lr} iter {r["iter"]}  keys {sorted(k for k in r if k not in ("kind", "iter"))}')
    q, d = r['q'], r['d']
    allq = [x for l in q for x in l]
    alld = [x for l in d for x in l]
    print(f'all heads: q median {st.median(allq):.2f} [{min(allq):.2f}, {max(allq):.2f}]  '
          f'd median {st.median(alld):.2f} [{min(alld):.2f}, {max(alld):.2f}]')
    for i, (ql, dl) in enumerate(zip(q, d)):
        print(f'  L{i:2d} q {st.median(ql):.2f} [{min(ql):.2f}, {max(ql):.2f}]  d {st.median(dl):.2f} [{min(dl):.2f}, {max(dl):.2f}]')
