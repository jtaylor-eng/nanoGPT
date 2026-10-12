"""probe_decide.py <probe_dir> <out_file>: NUM_ITER = 15 iff every Stieltjes row's max |S-1| at
num_iter=15 is < 1e-4 at every probed length (user decision e, 2026-10-12); else 30."""
import json, sys, glob, os
pdir, out = sys.argv[1], sys.argv[2]
worst, n = 0.0, 0
for f in sorted(glob.glob(os.path.join(pdir, 'res_*/residual.json'))):
    r = json.load(open(f))
    for L, by_it in r['residual'].items():
        m = max(p['max'] for p in by_it['15']); worst = max(worst, m); n += 1
        print(f"{os.path.basename(os.path.dirname(f))} L={L}: max|S-1| @15 = {m:.2e}, @30 = {max(p['max'] for p in by_it['30']):.2e}")
ni = 15 if (n >= 6 and worst < 1e-4) else 30
open(out, 'w').write(f"{ni}\n")
print(f"DECISION num_iter={ni} (worst residual @15 = {worst:.2e} over {n} row x length cells; threshold 1e-4)")
