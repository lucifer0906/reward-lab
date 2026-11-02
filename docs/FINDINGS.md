# Findings

## Claims, evidence, and caveats

| Claim | Status | Evidence | Caveat |
| --- | --- | --- |
| Final retained-policy reliability differs by reward. | Robust | Ten-seed final-checkpoint evaluation: sparse 4/10, dense 9/10, potential 7/10, hybrid 8/10 seeds at or above 90% success. | These are one fixed layout and PPO setup, not a universal ranking. |
| Collapse is not specific to dense or hybrid shaping. | Robust | Dense seed 3, hybrid seeds 0 and 4, and **potential seed 9** learned then collapsed. | A final checkpoint is necessary to detect it; episode-weighted training aggregates can hide it. |
| Larger naive distance weights create a reward-hacking incentive. | Robust | The analytic break-even is c=1/30=0.0333. At c=0.02, 3/5 retained success; at c=0.05 and c=0.10, 0/5 retained success and hacking incidence was 0.80 and 1.00. | This establishes the incentive in this 100-step gridworld, not every distance bonus. |
| Retained policies tolerate execution noise on the base layout. | Suggestive | At 20% action replacement, primary-view success: sparse 99.5%, dense 99.3%, potential 99.0%, hybrid 95.9%. | The wrapper records executed actions; this tests stochastic action execution, not observation noise or new dynamics. |
| Retained policies do not generalize to the moderate or major layouts. | Suggestive | L2 and L3 success is 0.0% for every core variant in the primary view. | The layouts preserve size, observation shape, three pairs, and task order, but are only three hand-designed tests. |

Detailed Stage 4 data: `runs/stage4_10seed_final_checkpoint/final_checkpoint_per_seed.csv`.
Weight sweep: `runs/stage5c_weight_sweep_final/final_checkpoint_summary.csv`.

## Potential definitions

`potential` uses Phi(s) = (keys collected + doors opened) / (2 * pairs), with
Phi(start)=0 and F = 0.99 Phi(s') - Phi(s). It has no spatial target-distance
term and reaches one after all six subgoals. Its terminal success reward is
separate.

`potential_distance_gamma099` uses Phi(s) = 0.1 * (22 - distance to the next
required target) and F = 0.99 Phi(s') - Phi(s). The constant offset 22 changes
the discounted-return convention (and makes fixed-target oscillation slightly
negative) without changing distance differences. Once all doors are open, the
next-required target becomes the goal; terminal success is a separate reward
component. Unlike progress-count Phi, this Phi contains designer
knowledge of the required key-door order and the next target, so it is a
control/incentive diagnostic rather than a reward-blind representation.

## Stage 6 protocol and zero-shot results

Loaded exactly the 40 saved core final checkpoints; no retraining. The policy
action is deterministic. Each model/condition uses 100 episodes. Confidence
intervals are bootstrap 95% CIs over seeds. The primary view includes the 28
`learned_and_retained` models; all-40 is secondary. Complete per-seed and both
views: `runs/stage6_zero_shot/per_seed.csv` and `runs/stage6_zero_shot/summary.csv`.

Important sampling caveat: with this deterministic policy and deterministic
environment, repeated no-noise episodes from a fixed seed are identical. Thus
the per-seed no-noise success value is binary and seeds, not the repeated
episodes, are the independent sample size. The action-noise episodes are real
stochastic replicates because the wrapper samples the executed action.

| Primary view: variant (seed count) | Base success [CI] | L1 minor [CI] | L2 moderate [CI] | L3 major [CI] |
| --- | --- | --- | --- | --- |
| sparse (4) | 100.0 [100.0, 100.0] | 75.0 [25.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| dense (9) | 100.0 [100.0, 100.0] | 88.9 [66.7, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| potential (7) | 100.0 [100.0, 100.0] | 71.4 [42.9, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| hybrid (8) | 100.0 [100.0, 100.0] | 87.5 [62.5, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

The base reference, episode length, mean subgoals, all-40 view, and CIs are in
the cited CSV. Plots: `runs/stage6_zero_shot/success_vs_noise.png` and
`runs/stage6_zero_shot/success_by_layout.png`.

Follow-up noise evaluation (28 retained models, 100 stochastic episodes/seed)
extends the base-layout result: at p=0.3/0.4/0.5 respectively, success was
sparse 99.5/99.0/96.3%, dense 99.4/97.9/96.3%, potential 99.1/98.4/94.7%,
and hybrid 96.6/97.1/94.1%; seed-bootstrap CIs are in
`runs/stage6_followup/noise_summary.csv`. Stochastic-policy evaluation of the
four deterministic collapses gave dense-3 46%, hybrid-0 0%, hybrid-4 100%, and
potential-9 100% success (100 episodes each), versus 0% deterministic final
success for all four. This indicates action-distribution support for a valid
route in three collapses, not reliable deterministic retention.

### Why

For each unseen layout, two failed replay episodes were saved per variant when
available in `runs/stage6_zero_shot/failure_replays.json`. Across retained
models, failures are predominantly a memorized-route loop: the policy revisits
a small region or follows the old route without completing the relocated next
target. L1 often still works because it changes only the start; L2/L3 do not.

Collapsed base-policy replays show dense seed 3, hybrid seed 4, and potential
seed 9 complete six subgoals then repeat an alternating two-action route instead
of reaching the goal. Actions and a GIF are saved as
`runs/stage6_zero_shot/collapsed_replay_actions.json` and
`runs/stage6_zero_shot/collapsed_replay.gif`.

## Stage 7 pre-registered horizon predictions

The horizon levels are solver-validated at 11 steps (two pairs), 51 steps
(five pairs), and 102 steps (six pairs), with limits of 36, 204, and 408.
Before any Stage 7 training, the predictions are: sparse will be near zero at
51+ steps; potential may degrade because its per-event signal is 1/(2*pairs);
dense and hybrid may learn more often but their fixed partial-credit sums grow
relative to the terminal reward; and learned-then-collapsed runs may become
more common. These are predictions, not results.

The observation vector contains normalized agent position; held/collected-key
and opened-door flags; the absolute positions of every key and door; goal
position; action mask; previous action; and remaining time. It does **not**
encode the wall layout. A policy can in principle adapt to relocated
keys/doors/goal from this vector, but cannot infer relocated walls except from
the current action mask and its interaction history; that limitation is
consistent with the Stage 6 L2/L3 failures.

## Stage 7 horizon scaling results

All 36 final-matrix jobs completed (three seeds per cell; 200,000 steps each).
Final saved checkpoints were judged by the unchanged evaluator with 200
deterministic episodes and 100 stochastic-policy episodes. The deterministic
comparison, including the existing 10-seed base point, is in
`runs/stage7/final/comparison_with_base.csv`; plots and per-seed taxonomy are
under `runs/stage7/final/`.

| Horizon (pairs) | Sparse | Dense | Potential | Hybrid |
| --- | --- | --- | --- | --- |
| H10, 11 steps (2) | 33% [0,100] | 100% [100,100] | 100% [100,100] | 100% [100,100] |
| Base, 26 steps (3; 10 seeds) | 40% [10,70] | 90% [70,100] | 70% [40,100] | 80% [50,100] |
| H50, 51 steps (5) | 0% [0,0] | 67% [0,100] | 67% [0,100] | 100% [100,100] |
| H100, 102 steps (6) | 0% [0,0] | 100% [100,100] | 33% [0,100] | 67% [0,100] |

The pre-registered sparse prediction was confirmed: it deteriorated to zero
at H50/H100. The potential-degradation prediction was supported at H100
(33%, mean checkpoint-based steps-to-90 80k), but is preliminary. Dense did
not show monotonic deterioration (100% at H100); hybrid was strongest at H50
and tied dense at H10, but fell to 67% at H100. The prediction that dense and
hybrid partial-credit advantages would systematically grow is therefore
not supported at n=3. Collapse/retention failures occurred: H10
sparse-2 (stochastic success 55%), H50 dense-2 (38%), H100 potential-0
(100%), and H100 hybrid-1 (100%) were learned checkpoint policies whose final
deterministic action was failing. The last two are stochastic-rescued.

The H10 rolling-training discrepancy is unexplained. The training callback
records only completed (`done`) episodes, including time-limit truncations; it
does not score partial episodes. Although its trailing 100-episode window
differs from checkpoint evaluation, stochastic checkpoint evaluation was also
100%, so action sampling does not explain the discrepancy. Headline numbers
use checkpoint evaluation. A reported 10k steps-to-90 means at or before the
first saved 10k checkpoint, not an exact crossing time.

Limitations: Stage 7 has only three seeds per cell, so its wide bootstrap CIs
often span 0--100%; the 10-seed base point has better precision but a different
seed count. Only the sparse decline with horizon is robust at three seeds; the
other strategy comparisons are preliminary.

## Baselines

| Layout | Random policy (1,000 episodes) | Solver success / steps |
| --- | --- | --- |
| Base | 0.0%, 100.0 mean steps | 100%, 26 |
| H10 | 0.4%, 36.0 mean steps | 100%, 11 |
| H50 | 0.0%, 204.0 mean steps | 100%, 51 |
| H100 | 0.0%, 408.0 mean steps | 100%, 102 |

The random policy uses the unchanged action space and evaluator; the scripted
planner is the exact BFS solver. Raw results: `runs/baselines/baseline_summary.csv`.

## Stage 8 analysis (final saved checkpoints only)

All 118 Stage 8 completion markers and `results.json` files were present,
readable, and internally complete (50 ablations, 56 extra-horizon seeds, and
12 sparse long-budget runs). No training was run for this analysis. Final
checkpoints used 200 deterministic and 100 stochastic episodes. Every 10k
intermediate checkpoint used one deterministic and 20 stochastic episodes.
On five sampled deterministic checkpoints, all repeated episodes had identical
success and proxy return; therefore the deterministic per-seed result is one
policy/layout outcome, not 200 independent observations. A 10k threshold means
at or before the first saved checkpoint.

### Base-layout ablations (five seeds)

The table is final deterministic success; brackets are seed-bootstrap 95% CIs.
`n90` is retained seeds at or above 90%. Point-biserial and the strict
above-scripted-success-control failed-episode incidence are in
`runs/stage8/analysis/ablation_summary.csv`; every incidence was 0.0 here.
That absence does not make removal of terminal success safe: it only says this
specific high-proxy failure criterion did not occur in final evaluations.

| Variant | n90 | Mean [CI] | Steps-to-90 | Verdict |
| --- | ---: | --- | ---: | --- |
| dense (parent; 10-seed reference 9/10, 0.90) | 4/5 | 0.80 [0.40,1.00] | 32k | Five-seed control. |
| dense no door reward | 4/5 | 0.80 [0.40,1.00] | 85k | Door credit is removable here but learning is slower. |
| dense no key reward | 3/5 | 0.60 [0.20,1.00] | 30k | Key credit helps retention. |
| dense no invalid penalty | 3/5 | 0.60 [0.20,1.00] | 30k | Invalid-action penalty helps retention. |
| dense no terminal success | 1/5 | 0.20 [0.00,0.60] | 32k | Hacking/goal-misalignment test: terminal success is essential. |
| dense subgoal x0.5 | 2/5 | 0.40 [0.00,0.80] | 38k | Weaker progress reward harms retention. |
| dense subgoal x2 | 4/5 | 0.80 [0.40,1.00] | 24k | No retention gain; may learn faster. |
| hybrid (parent; 10-seed reference 8/10, 0.80) | 3/5 | 0.60 [0.20,1.00] | 24k | Five-seed control. |
| hybrid no invalid penalty | 4/5 | 0.80 [0.40,1.00] | 30k | Removable without visible harm at n=5. |
| hybrid no step/timeout penalty | 4/5 | 0.80 [0.40,1.00] | 24k | Removable without visible harm at n=5. |
| potential (parent; 10-seed reference 7/10, 0.70) | 4/5 | 0.80 [0.40,1.00] | 35k | Five-seed control. |
| potential x3 | 5/5 | 1.00 [1.00,1.00] | 30k | Stronger valid Phi signal helped in this sample. |
| potential noisy | 4/5 | 0.80 [0.40,1.00] | 38k | Misleading-signal test: noise did not reduce retention here, but this is not valid potential shaping. |

### Extended horizon seeds and long sparse budget

| Horizon | Sparse | Dense | Potential | Hybrid |
| --- | --- | --- | --- | --- |
| H50 (10 seeds) | 4/10, 0.40 [0.10,0.70] | 8/10, 0.80 [0.50,1.00] | 6/10, 0.60 [0.30,0.90] | 7/10, 0.70 [0.40,1.00] |
| H100 (10 seeds) | 0/10, 0.00 [0.00,0.00] | 10/10, 1.00 [1.00,1.00] | 5/10, 0.50 [0.20,0.80] | 9/10, 0.90 [0.70,1.00] |

At H50, none of the six pairwise retained-count tests survives Holm correction.
At H100, sparse is lower than dense (Holm p=0.000065) and hybrid (0.000595);
all other comparisons are not significant after Holm correction. Thus H100
sparse failure and dense reliability are robust in this 10-seed comparison;
the remaining ordering is suggestive. H10 remains preliminary (three seeds).
The machine-readable bootstrap differences and Fisher tests are
`runs/stage8/analysis/pairwise_comparisons.csv`.

Sparse did not become broadly viable at one million steps: base seed 6 was
slow and reached 90% at 960k (five other base seeds stayed stuck); all three
H50 runs stayed stuck; H100 seed 1 was slow and reached 90% at 570k (two stayed
stuck). No long-budget seed learned and then failed. This supports a real
long-horizon sparse-credit problem rather than only an insufficient 200k budget.

Updated Stage 4 checkpoint-only steps-to-90 for all 40 base runs are in
`runs/stage8/analysis/stage4_steps_to_90_checkpoint.csv`. All Stage 8 analysis
tables and plots are in `runs/stage8/analysis/`, including success and
steps-to-90 versus horizon, the ablation chart, and sparse 200k-versus-1M plot.

### Stage 8b statistical corrections

Only two Stage 8 conclusions are robust at their tested sample sizes: removing
the terminal success reward harms dense retention, and at H100 sparse is below
dense and hybrid. A one-seed-of-five difference such as 3/5 versus 4/5 is not
detectable (two-sided Fisher exact p=1.00), so the other n=5 ablation apparent
differences are exploratory rather than evidence that a component is removable
or necessary. The sparse horizon pattern is suggestive, not robust: base 4/10,
H50 4/10, H100 0/10; base versus H100 has two-sided Fisher p=0.087. Each
horizon is one hand-built layout, so layout changes and path length are
confounded.

The new reward-indistinguishable-failure metric counts failed final episodes
whose proxy return is at least 90% of scripted successful-control return. It is
75% (600/800 failed episodes) for `dense_no_terminal_success`, 0% for
`potential_noisy` (0/200), and 0% for every core cell and other ablation in
this deterministic final-checkpoint set. Thus the no-terminal condition is a
genuine reward/goal mismatch under a less strict test; noisy Phi did not show
that particular failure mode here. Details are in
`runs/stage8/analysis/reward_indistinguishable_summary.csv`.

An exploratory Cochran-Mantel-Haenszel pooling of the base, H50, and H100
retained-seed tables estimates odds ratios relative to sparse of 16.83 for
dense (95% CI 4.66--60.80, p=0.000001), 4.57 for potential (1.43--14.64,
p=0.0085), and 9.00 for hybrid (2.87--28.26, p=0.000048). Dense versus
potential has OR 5.74 (1.43--22.98, p=0.0086); dense versus hybrid and hybrid
versus potential are not significant. These pooled tests are descriptive, not
new causal layout-generalization claims, because the three strata are different
hand-built layouts. Full tables and homogeneity tests are in
`runs/stage8/analysis/stratified_retained_seed_tests.csv`.

Sparse logger data show that most final failures never logged a single success:
all H100 200k failures, all H50 long-budget failures, and five of six base
long-budget failures have no nonzero logged success window. The late retained
base seed 6 first logged success at 940k and H100 seed 1 at 542,720. H10 seed
2 did log success at 4,096 but ultimately failed, showing a rare learn-then-
fail exception. The callback did not persist individual episode timestamps, so
these are first *logged* nonzero-success steps, not exact first successful
episode times; see `runs/stage8/analysis/sparse_first_logged_success.csv`.

## Stage 8b final checkpoint evaluation

All 70 Stage 8b `results.json` files and completion markers were readable. The
saved final checkpoints were evaluated with 200 deterministic and 100
stochastic episodes; each 10k intermediate checkpoint used one deterministic
and 20 stochastic episodes. Outputs are in `runs/stage8b/analysis/`.

| Claim | Label | Evidence and caveat |
| --- | --- | --- |
| Potential x3 improves standard potential at H50/H100. | Not detectable | H50: 8/10 versus 6/10, Fisher p=0.628; H100: 9/10 versus 5/10, raw p=0.141 and Holm p=0.282. |
| Potential x3 matches dense. | Not detectable | H50: both 8/10; H100: 9/10 versus 10/10, Fisher p=1.0. |
| Removing terminal success reduces dense retention. | Suggestive | 3/10 retained versus dense 9/10; raw Fisher p=0.0198, 10-way Holm p=0.198. It does not meet the predeclared robust threshold. |
| Any other tested ablation changes retention. | Not detectable | All remaining raw Fisher p values are at least 0.141 and all Holm p values are 1.0. |

Potential x3 final retained counts were H50 8/10 (mean 0.80, CI
0.50--1.00; mean steps-to-90 15k) and H100 9/10 (0.90, CI 0.70--1.00;
22k). It had two stochastic rescues at H50 and none at H100. At H100 the
standard-potential comparison is suggestive in direction but not Holm-significant.

The pooled base/H50/H100 CMH comparison estimates potential x3 versus sparse:
OR 9.00 (95% CI 2.87--28.26, p=0.000048); versus dense: 0.43
(0.09--1.98, p=0.281); versus potential: 2.58 (0.83--8.03, p=0.099); versus
hybrid: 1.00 (0.28--3.59, p=1.00). These are descriptive because the three
strata are different hand-built layouts.
