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

## Stage 4 Full Matrix - 2026-10-25

Stage 4 is complete. Ran the full controlled matrix: four reward formulations
by five seeds (0 through 4), 200,000 timesteps per run, three CPU workers, and
one torch/BLAS thread per worker. All 20 queue jobs completed with status
`ok`. The shared 200,000-step budget was selected because the pilot's sparse
seed 42 had not learned at 100,000 steps, while the prior sparse baseline had
required 200,000 steps.

### Core results

Whole-training ground-truth success rate (mean +/- standard deviation):

- Sparse: 38.4% +/- 47.1%; 2 of 5 seeds reached 90% success.
- Dense: 96.8% +/- 0.7%; 5 of 5 seeds reached 90% success.
- Potential: 75.5% +/- 37.9%; 4 of 5 seeds reached 90% success.
- Hybrid: 95.5% +/- 1.9%; 5 of 5 seeds reached 90% success.

Mean steps to 90% success among seeds that reached it: sparse 32,500 (2/5),
dense 22,432 (5/5), potential 28,778 (4/5), and hybrid 20,000 (5/5).
Normalized ground-truth success AUC: sparse 0.343, dense 0.892, potential
0.666, hybrid 0.869. Dense is the most stable formulation; hybrid has the
fastest threshold crossings but seed 0 collapsed from successful behavior to a
0% final logged rolling success rate. Sparse and potential also have failed
learning seeds, so aggregate means must be reported with variance rather than
only their successful seeds.

Saved five-seed mean +/- standard deviation plots and machine-readable results
under `runs/stage4_analysis/`:

- `stage4_success_rate.png`
- `stage4_invalid_action_rate.png`
- `stage4_subgoal_completion_rate.png`
- `stage4_metrics.json`
- `stage4_summary.csv`

Added `reward_rl.visualization.aggregate_stage4`, which reads only logged
ground-truth evaluator metrics for success, computes per-seed threshold
crossings and normalized AUC, and writes the aggregate plots and CSV. No PPO
hyperparameters, reward definitions, environment dynamics, or ground-truth
success evaluator logic were changed.

## Stage 5 Reward-Hacking Stress Tests - 2026-10-25

Stage 5 is complete. Added and ran a reproducible audit against the actual
dense and hybrid reward configurations. The audit constructs adversarial
scripted trajectories, accumulates the real reward components, and judges
every result only through the unchanged reward-blind evaluator.

Three distinct stress scenarios were tested, each with a genuine successful
control trajectory:

- Repeated subgoal attempts: pick a key or open a door once, then repeat the
  interaction through timeout. Neither reward re-pays the completed event;
  repeated interactions are invalid and receive the configured penalty.
- Movement oscillation: move back and forth for the full episode with no net
  progress. Dense receives zero reward and hybrid receives -0.60 from its
  step cost and timeout penalty. Neither formulation pays for motion alone.
- Incomplete progress: complete each increasing prefix of keys and doors,
  then use valid neutral movement instead of reaching the goal. The final
  prefix completes all six intermediate subgoals but still fails ground-truth
  success. It earns +1.50 dense proxy reward and +1.01 hybrid proxy reward,
  demonstrating that intermediate reward can diverge from completion.

Across all audit episodes, proxy reward versus true success was correlated but
not identical: dense Pearson r=0.590 and Spearman rho=0.679; hybrid Pearson
r=0.699 and Spearman rho=0.679. The audit therefore finds no repeat-event or
oscillation farming vulnerability in the implemented rewards, but it records
the residual partial-progress proxy risk and keeps the ground-truth evaluator
as the required selection metric.

Saved outputs under `runs/stage5_reward_hacking/`:

- `repeated_subgoal_attempt_proxy_vs_success.png`
- `movement_oscillation_proxy_vs_success.png`
- `incomplete_progress_proxy_vs_success.png`
- `audit_results.json`
- `audit_episodes.csv`

Added `reward_rl.analysis.reward_hacking_audit` to regenerate the analysis.
The test suite remains 82 passed. No environment dynamics, PPO
hyperparameters, reward configuration, or success evaluator logic changed.
