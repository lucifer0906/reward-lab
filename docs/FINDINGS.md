# Findings

## Claims, evidence, and caveats

| Claim | Evidence | Caveat |
| --- | --- | --- |
| Final retained-policy reliability differs by reward. | Ten-seed final-checkpoint evaluation: sparse 4/10, dense 9/10, potential 7/10, hybrid 8/10 seeds at or above 90% success. | These are one fixed layout and PPO setup, not a universal ranking. |
| Collapse is not specific to dense or hybrid shaping. | Dense seed 3, hybrid seeds 0 and 4, and **potential seed 9** learned then collapsed. | A final checkpoint is necessary to detect it; episode-weighted training aggregates can hide it. |
| Larger naive distance weights create a reward-hacking incentive. | The analytic break-even is c=1/30=0.0333. At c=0.02, 3/5 retained success; at c=0.05 and c=0.10, 0/5 retained success and hacking incidence was 0.80 and 1.00. | This establishes the incentive in this 100-step gridworld, not every distance bonus. |
| Retained policies tolerate execution noise on the base layout. | At 20% action replacement, primary-view success: sparse 99.5%, dense 99.3%, potential 99.0%, hybrid 95.9%. | The wrapper records executed actions; this tests stochastic action execution, not observation noise or new dynamics. |
| Retained policies do not generalize to the moderate or major layouts. | L2 and L3 success is 0.0% for every core variant in the primary view. | The layouts preserve size, observation shape, three pairs, and task order, but are only three hand-designed tests. |

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
refuted for this small sample. Collapse/retention failures occurred: H10
sparse-2 (stochastic success 55%), H50 dense-2 (38%), H100 potential-0
(100%), and H100 hybrid-1 (100%) were learned checkpoint policies whose final
deterministic action was failing. The last two are stochastic-rescued.

The H10 rolling-training discrepancy is not an evaluator issue. The training
callback records only completed (`done`) episodes, including time-limit
truncations; it does not score partial episodes. However, it records actions
sampled from PPO's stochastic training policy in a trailing 100-episode
window, whereas saved-checkpoint evaluation uses deterministic argmax actions.
Thus the 78% H10 rolling value is exploratory-action noise, while every tested
deterministic checkpoint succeeded.

Limitations: Stage 7 has only three seeds per cell, so its wide bootstrap CIs
often span 0--100%; the 10-seed base point has better precision but a different
seed count. Results establish a useful scaling signal, not a definitive ranking.
