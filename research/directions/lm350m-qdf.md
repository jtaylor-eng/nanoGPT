---
slug: lm350m-qdf
title: "Windowed stieltjes does better at 51M than softmax and ASEntmax at 51M params, but not 125M. Is 125M a fluke? (trying ~350M params)"
project: nanogpt
cluster: ascend
status: active
wake: now
budget_gpu_hours: 100
spent_gpu_hours: 0
max_jobs_in_flight: 4
jobs: []
jobs_archived: []
last_wake: null
next_id: 1
---
# Windowed stieltjes does better at 51M than softmax and ASEntmax at 51M params, but not 125M. Is 125M a fluke? (trying ~350M params)

Frontmatter fields: `status` active|paused|done. `wake` now|on_jobs (the loop runs this
direction on the next tick when `now`, otherwise only when a listed job finishes).
`jobs` = ids currently in flight; finished ids move to `jobs_archived` (never delete ids,
they feed the GPU-hour accounting). `budget_gpu_hours` is a hard stop.

## Goal
What question this direction answers and what result would settle it.

## Theory
Why it might work. Cite the proposal / paper sections. Note the strongest counter-argument.

## Hypothesis backlog
Ordered. The agent takes from the top unless results change the plan, and records why
when it deviates.
1.
2.

## Constraints
- Files the agent may change:
- Fixed protocol (data, eval, seeds, metric, checkpoint selection):
- Per-experiment ceiling (GPU-hours, walltime):

## Log
One line per wake, newest last. Format: `YYYY-MM-DD HH:MM  eNNN submitted <jobids> | eNNN done <metric> | paused: <reason>`
