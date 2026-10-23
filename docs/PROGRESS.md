# Progress Log

## Stage 4 Pilot - 2026-10-23

Completed the Stage 4 pilot only: four reward variants, seed 42, 100,000
timesteps each, three workers, CPU device, and one torch/BLAS thread per
worker. The full 20-run matrix was not started.

- Queue result: 4/4 jobs succeeded in 441 seconds (7.3 minutes).
- Whole-training ground-truth success rates: sparse 0.0%, dense 95.9%,
  potential 91.1%, hybrid 78.1%.
- Final logged rolling ground-truth success rates at 100,000 steps: sparse
  0%, dense 100%, potential 100%, hybrid 0%. Hybrid reached 100% early and
  then collapsed, so the whole-training aggregate must not be interpreted as
  stable final performance.
- Threshold crossings (25% / 50% / 75% / 90%): sparse not reached; dense
  15,000 / 18,432 / 20,000 / 20,000; potential 26,624 / 28,672 / 30,000 /
  32,768; hybrid 14,336 / 15,000 / 18,432 / 18,432 timesteps.
- Saved the ground-truth learning curve to
  `runs/pilot_ground_truth_success_curve.png` and threshold data to
  `runs/pilot_learning_metrics.json`.

### Encoding repair

The failed prior pilot was caused by the parent queue printing a Unicode
right-arrow status marker to a Windows cp1252 console. The exception occurred
before its child training job was launched and was caught as a queue job
exception. Direct sparse training at seed 42 for 2,000 requested timesteps
(one 2,048-step PPO rollout) completed successfully and produced no traceback.

Configured stdout/stderr as UTF-8 at CLI/direct-module entry points, retained
UTF-8 environment variables for worker subprocesses, made all project file
opens explicit UTF-8, and changed status/log configuration text to ASCII where
practical. CPU device selection is explicit and shared by every reward variant.
PPO hyperparameters, architecture, reward definitions, and the ground-truth
success evaluator were not changed.

### Change-review limitation

This workspace has no `.git` directory. `git status`, `git log -5`, and
`git diff` all cannot run, so prior edits were reviewed against the documented
handoff and current files rather than a commit diff. Retained the existing
ThreadPoolExecutor runner, subprocess UTF-8 environment setup, UTF-8 decoding,
and ASCII queue messages because they are required for the repair. No
documented prior edit was reverted: none was extraneous to the runner or
encoding fix.
