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

### Core results - SUPERSEDED FOR FINAL-POLICY CLAIMS

**SUPERSEDED:** The following whole-training, episode-weighted aggregates are
historical learning-curve metrics only. They are not final-checkpoint success
rates and can mask late collapse. Use the deterministic final-checkpoint table
in the 2026-10-26 verification section, and its forthcoming 10-seed extension,
for all final-policy claims.

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
## Verification addendum - 2026-10-25

Read-only verification of existing Stage 4 and Stage 5 artifacts. No source, config,
evaluator, environment, or training changes were made, and no new training was launched.

- The Stage 4 headline success rates are whole-training, on-policy episode aggregates
  from `SuccessRateLogger.get_results()`, not deterministic final-checkpoint scores.
  The values in `runs/stage4_analysis/stage4_metrics.json` are unchanged, but their
  provenance should be described precisely.
- Full-matrix seeds were 0 through 4; they are distinct from the Stage 4 pilot seed 42.
- Hybrid seed 0 is the observed late collapse: rolling success fell from 1.00 at
  170k steps to 0.00 by 185k, with mean episode length reaching the 100-step limit
  and mean evaluator subgoals settling near 4/7. Invalid actions declined, so the
  saved logs do not support an invalid-action farming explanation.
- Potential seed 1 failed throughout: no successes, episodes at the 100-step limit,
  and roughly 2/7 evaluator subgoals late in training. Its potential is
  `(keys_collected + doors_open) / 6`; reward shaping gamma and PPO gamma are both
  0.99. A clean shaping event is about +0.157 to +0.165 versus terminal +1.0.
- Existing deterministic evaluation episodes (4,000 per variant) show proxy-return
  versus success point-biserial correlations: sparse 1.0000, dense 0.8238,
  potential 0.9981, hybrid 0.9288. No evaluation episode had above-median return
  while `success=False`. Plots and source metrics are in
  `runs/stage4_reward_vs_success/`.
- A rolling-log signal screen found reward increases with flat/falling success or
  subgoals in dense, potential, and hybrid, but not sparse. This is a diagnostic
  flag, not proof of reward hacking because component events are not retained at
  episode granularity in the training logs.
- Stage 5 control trajectories remain ordered correctly: dense successful 2.50 vs
  incomplete 1.50; hybrid successful 3.221667 vs incomplete 1.011667.

## Stage 4 Final Checkpoint Verification and Stage 5b - 2026-10-26

Stage 4 was re-reported using deterministic evaluation of each saved final model:
200 episodes per seed, using the unchanged ground-truth evaluator. This is the
primary headline and replaces whole-training episode aggregates as the measure of
final policy quality. Results are in `runs/stage4_final_checkpoint/`.

- Seeds at or above 90 percent final success: sparse 2/5, dense 4/5,
  potential 4/5, hybrid 3/5. Mean final success and bootstrap 95 percent CI:
  sparse 0.40 [0.00, 0.80], dense 0.80 [0.40, 1.00], potential 0.80
  [0.40, 1.00], hybrid 0.60 [0.20, 1.00].
- Whole-training success remains a secondary episode-weighted statistic only:
  sparse 0.384, dense 0.968, potential 0.755, hybrid 0.955. It masks late
  failures, notably dense seed 3 and hybrid seeds 0 and 4.
- Hybrid seed 0 (160k to 190k) had mean entropy loss -0.229, mean approximate
  KL 0.0759 with a 0.8689 spike, clip fraction 0.0931, value loss 0.00374,
  and explained variance 0.972. This is a late optimization instability/collapse,
  not merely a stuck high-entropy policy.
- Potential seed 1 was an exploration failure/stuck policy: entropy loss fell from
  -1.601 to -0.526, while KL remained modest (mean 0.00942), clip fraction mean
  0.100, value loss fell to near zero, and explained variance ended at 0.9999.

Stage 5b added two explicitly labelled adversarial rewards only; the existing four
reward configs, PPO settings, environment dynamics, and success evaluator were not
modified. All 10 new jobs (2 variants x 5 seeds, 200k steps, 3 workers) completed.

- `naive_distance`: pays +0.1 for every one-cell decrease in Manhattan distance to
  the next required target, with no away-move penalty. A 100-step forward/backward
  loop earns 5.0, above successful-control return 3.0. Final deterministic success
  was 0/5 seeds; all 1,000 evaluation episodes exceeded the successful-control
  median reward while failing. Final proxy returns were 5.2 to 5.6.
- `potential_distance`: uses the same scale and Phi = -0.1 times distance, with
  shaping gamma 1.0 so a fixed-target oscillation telescopes to exactly 0.0. PPO
  gamma remains 0.99. The unit test verifies non-positive oscillation return.
  Final deterministic success was 1/5 seeds; no final evaluation episode exceeded
  the successful-control median while failing.
- Scripted audit, scatter plots, final-checkpoint tables, action JSON, and GIF
  replays are saved under `runs/stage5b_reward_hacking/` and
  `runs/stage5b_final_checkpoint/`. The naive-distance replay demonstrates the
  reward-hacking incentive. Test suite: 85 passed.

## Ten-Seed Final Checkpoint Extension and Distance-Weight Sweep - 2026-11-01

Ran one resumable 35-job queue at 200,000 steps per job, three workers, CPU,
and one OMP/BLAS thread per worker. The queue completed 35/35 jobs with valid
`results.json` plus `completion.json` markers. Existing evaluator, dynamics,
PPO settings, and the six pre-existing reward configs were not changed.

All final-policy results below use saved final checkpoints, deterministic
actions, and 200 evaluator episodes per seed. This supersedes every
whole-training success figure as a final-policy claim. Whole-training values
remain secondary, episode-weighted learning-curve aggregates only.

| Variant | Final success mean (bootstrap 95 percent CI) | Seeds >=90 percent | Median | Mean steps to 90 percent among learned |
| --- | --- | --- | --- | --- |
| sparse | 0.40 [0.10, 0.70] | 4/10 | 0.00 | 50,554 |
| dense | 0.90 [0.70, 1.00] | 9/10 | 1.00 | 22,275 |
| potential | 0.70 [0.40, 1.00] | 7/10 | 1.00 | 28,711 |
| hybrid | 0.80 [0.50, 1.00] | 8/10 | 1.00 | 20,710 |

Failure taxonomy: `never_learned` means rolling ground-truth success never
reached 90 percent; `learned_then_collapsed` means it reached 90 percent but
deterministic final success was below 90 percent.

| Variant | Never learned | Learned then collapsed | Learned and retained |
| --- | --- | --- | --- |
| sparse | 1, 2, 4, 5, 6, 8 | none | 0, 3, 7, 9 |
| dense | none | 3 | 0, 1, 2, 4, 5, 6, 7, 8, 9 |
| potential | 1, 5 | 9 | 0, 2, 3, 4, 6, 7, 8 |
| hybrid | none | 0, 4 | 1, 2, 3, 5, 6, 7, 8, 9 |

Full per-seed checkpoint metrics, taxonomy, and threshold steps are in
`runs/stage4_10seed_final_checkpoint/final_checkpoint_per_seed.csv`.

Added naive-distance `c=0.02` and `c=0.05` configs and a new
`potential_distance_gamma099` control. The pre-existing gamma=1.0
`potential_distance` remains labelled legacy and was not modified.

For the 100-step scripted oscillation there are 50 distance-decreasing moves.
The successful path has 20 decreases plus terminal reward 1.0, so the analytic
break-even is `50c = 1 + 20c`, or `c = 1/30 = 0.033333`. The scripted audit
and unit test match this value. Therefore c=0.02 is below break-even; c=0.05
and c=0.10 are above it.

| c | Final success | Mean final proxy return | Hacking incidence | Scripted successful control | Failed minus control gap |
| --- | --- | --- | --- | --- | --- |
| 0.02 | 0.60 (3/5) | 1.26 | 0.00 | 1.40 | not applicable (success variance) |
| 0.05 | 0.00 (0/5) | 2.45 | 0.80 | 2.00 | +0.45 |
| 0.10 | 0.00 (0/5) | 5.38 | 1.00 | 3.00 | +2.38 |

For zero-variance success (`c=0.05` and `c=0.10`), final-report JSON writes
`point_biserial_proxy_vs_success: null`, not NaN, and reports the mean failed
proxy-return gap to scripted successful control. The gamma-matched potential
control uses `F=0.99*Phi(next)-Phi(current)` and an affine baseline of 22.0,
which preserves distance differences while making fixed-target oscillation
negative (-2.05 in the scripted audit). It achieved 0.60 final success (3/5)
with zero hacking-incidence episodes.

Saved audit, final-checkpoint tables, scatter plots, and the weight-sweep plot
under `runs/stage5c_weight_sweep_audit/` and `runs/stage5c_weight_sweep_final/`.
The full suite passed: 86 tests.

## Stage 6 Zero-Shot Evaluation - 2026-11-01

Added three solver-validated unseen 5x17 layouts without changing base dynamics:
L1 minor start move (shortest solution 25), L2 one relocated key/door pair (32),
and L3 all key/door pairs relocated (28). Added an evaluation-only executed-action
noise wrapper that records the action actually passed to the unchanged environment
and its episode record. No training was launched and no existing checkpoint/config,
PPO setting, evaluator, or environment dynamics was modified.

Evaluated all 40 saved core final checkpoints for 100 deterministic-policy episodes
per condition. Primary results use the 28 learned-and-retained models. Base success
was 100% for each retained variant; L1 was sparse 75.0%, dense 88.9%, potential
71.4%, hybrid 87.5%; L2 and L3 were 0% for every variant. Therefore no variant
generalized above 10% to L2 or L3. At 20% executed-action noise on base, success
was sparse 99.5%, dense 99.3%, potential 99.0%, hybrid 95.9%.

Saved per-seed data, both aggregate views and bootstrap CIs under
`runs/stage6_zero_shot/`, plus `success_vs_noise.png`, `success_by_layout.png`,
failed-episode action JSON, and a collapsed-policy GIF. The collapsed replays for
dense seed 3, hybrid seed 4, and potential seed 9 all complete six subgoals then
loop rather than taking the final route to the goal. Full claims, Phi conventions,
weight-sweep context, and caveats are in `docs/FINDINGS.md`.
