# PatternBoost Fixed-Timeout Protocol

This note records the corrected PatternBoost recreation protocol for the
no-three-in-line fixed-timeout experiments.

## Purpose

The earlier diagnostic run used `max_epochs=20`. That was enough to verify the
file-descriptor fix, but it was not a true fixed-timeout experiment because a
grid size could finish before using the allotted 120 minutes.

The corrected protocol is:

- run `N=10..19` sequentially;
- give each grid size an end-to-end timeout of 7200 seconds;
- include initial heap generation and PatternBoost training in that timeout;
- set PatternBoost `max_epochs=1000`, so the wrapper timeout, not a small
  generation cap, controls the run;
- stop early only if post-saturation reaches the target `2n`;
- record process status separately from mathematical success.

## Status Semantics

`status` is process-level:

- `completed`: the subprocess exited normally;
- `timeout`: the wrapper killed the subprocess group at 7200 seconds;
- `failed_return_*`: the subprocess exited nonzero.

`result_status` is result-level:

- `solved`: `best_post_saturation_score >= target_points`;
- `completed_not_solved`: process completed but did not reach `2n`;
- `timeout`: the grid size used the whole timeout without normal completion;
- `failed_return_*`: the run failed.

This distinction matters: a completed PatternBoost run is not necessarily a
recreated no-three-in-line solution.

On future launches, a wrapper timeout is also written into the relevant grid
log as:

```text
Wrapper timeout reached after <seconds> seconds; terminated subprocess group.
```

and the CSV `termination_reason` is set to `wrapper_timeout`.

## Corrected Runner

The corrected runner is:

```text
./run_patternboost_timeout.py
```

Important defaults after checking the archived Apocrita `N20` run:

```text
PATTERNBOOST_GRID_SIZES=10-19
PATTERNBOOST_TIMEOUT_SECONDS=7200
PATTERNBOOST_MAX_EPOCHS=1000
PATTERNBOOST_MAX_STEPS=2000
PATTERNBOOST_INITIAL_JOBS=20
PATTERNBOOST_INITIAL_JOB_TARGET_TRAINING_SIZE=100
PATTERNBOOST_INITIAL_PARALLELISM=10
PATTERNBOOST_TARGET_TRAINING_SIZE=2000
PATTERNBOOST_BATCH_SIZE=500
PATTERNBOOST_INITIAL_KEEP_BEST_FRACTION=0.1
PATTERNBOOST_TRAIN_KEEP_BEST_FRACTION=0.1
PATTERNBOOST_NUM_WORKERS=0
PATTERNBOOST_STOP_ON_TARGET=1
```

The Apocrita-derived settings reflect the archived run artifacts:

- `apocrita/N20/initial-generation_job*.log` shows 20 initial shards, each with
  `target_training_size=100`, `batch_size=500`, and `keep_best_fraction=0.1`;
- each shard saved 800 symmetrized examples in the old code path, i.e. 100 base
  constructions times 8 symmetries;
- the aggregated `apocrita/N20/training_sets/N20_gen0.txt` has 16,000 lines,
  consistent with about 2,000 base constructions times 8 symmetries;
- subsequent Apocrita generation files remain around 16,000 lines, while the
  training log reports about 20,000 generated samples per generation. This is
  consistent with a 2,000-construction base pool and `keep_best_fraction=0.1`;
- the Apocrita training log trains through step 2000 before sampling each
  generation.

The run below was launched before this Apocrita correction and should be treated
as superseded if the goal is faithful Apocrita-style recreation:

```text
/Users/pranavr/Developer/Work/STRIDE/fixed_timeout_runs/patternboost/20260624_200537
```

Use the current `run_patternboost_timeout.py` for the next launch; it creates
Apocrita-style initial-generation shards, concatenates them into `N*_gen0.txt`,
and then starts PatternBoost training from the aggregated pool.

Operational commands for checking, stopping, auditing, and tabulating the
background run are recorded in:

```text
patternboost_background_commands.md
```

## Final Audits

After the detached run finishes, use the normal runtime audit:

```text
./audit_patternboost_run.py /Users/pranavr/Developer/Work/STRIDE/fixed_timeout_runs/patternboost/20260624_200537 --expected-sizes 10-19
```

This checks that the run has all expected rows, process statuses are clean,
`num_workers=0`, logs contain no `Too many open files`, `Traceback`, or
`OSError`, and every row is either solved or ended by the wrapper timeout.
An unsolved `completed_not_solved` row fails this audit because it means the
generation cap, not the fixed wall-clock timeout, ended the run.

Use the strict recreation audit when the claim is that PatternBoost reproduced
the `2n` target for every size:

```text
./audit_patternboost_run.py /Users/pranavr/Developer/Work/STRIDE/fixed_timeout_runs/patternboost/20260624_200537 --expected-sizes 10-19 --require-solved
```

The strict audit fails if any expected size is missing or has not reached `2n`.

Diagnostic runs that intentionally use a smaller generation cap can be audited
with:

```text
./audit_patternboost_run.py <run_dir> --expected-sizes <sizes> --allow-generation-exhausted
```

Do not use that flag for the final fixed-timeout comparison.

To print a manuscript-ready Markdown table from a finished summary CSV:

```text
./summarize_patternboost_results.py /Users/pranavr/Developer/Work/STRIDE/fixed_timeout_runs/patternboost/20260624_200537
```

Helper behavior can be checked without touching the live run:

```text
./test_patternboost_helpers.py
```

This creates temporary synthetic run folders and verifies that:

- solved rows pass normal and strict audits;
- timeout rows pass normal audit but fail strict solved audit;
- unsolved generation-exhausted rows fail normal audit;
- unsolved generation-exhausted rows pass only with `--allow-generation-exhausted`;
- the Markdown summary helper reports solved rows correctly.
- timeout cleanup is safe when the child process exits before the process
  group can be signaled.

## Implementation Fixes

The file-descriptor crash was caused by repeated DataLoader worker creation on
macOS. The fix is:

- default PatternBoost `--num_workers` to `0`;
- pass `--num_workers 0` from the fixed-timeout runner;
- explicitly shut down `InfiniteDataLoader`;
- clear DataLoader references after each generation;
- disable CPU/MPS pin-memory usage;
- terminate subprocess groups on wrapper timeout.

PatternBoost now also supports:

```text
--stop_on_target
```

which stops after a generation whose post-saturation best score reaches
`max_points`, i.e. `2n` in these runs.
