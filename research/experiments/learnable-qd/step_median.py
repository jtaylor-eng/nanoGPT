"""Median train step time (ms) from nanoGPT run.log files: python step_median.py <run.log>..."""
import re, statistics, sys

for f in sys.argv[1:]:
    t = [float(m.group(2)) for m in re.finditer(r"^iter (\d+): loss [\d.]+, time ([\d.]+)ms", open(f).read(), re.M)
         if int(m.group(1)) % 10 == 0 and int(m.group(1)) > 20]
    print(f"{f}  n={len(t)}  median_ms={statistics.median(t):.0f}" if t else f"{f}  no iters")
