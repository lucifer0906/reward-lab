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

| Primary view: variant (seed count) | Base success [CI] | L1 minor [CI] | L2 moderate [CI] | L3 major [CI] |
| --- | --- | --- | --- | --- |
| sparse (4) | 100.0 [100.0, 100.0] | 75.0 [25.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| dense (9) | 100.0 [100.0, 100.0] | 88.9 [66.7, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| potential (7) | 100.0 [100.0, 100.0] | 71.4 [42.9, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| hybrid (8) | 100.0 [100.0, 100.0] | 87.5 [62.5, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

The base reference, episode length, mean subgoals, all-40 view, and CIs are in
the cited CSV. Plots: `runs/stage6_zero_shot/success_vs_noise.png` and
`runs/stage6_zero_shot/success_by_layout.png`.

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
