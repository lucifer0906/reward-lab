# Reward-Shaping Recipe v1 (long-horizon, subgoal-structured tasks)

Scope: one task family (key-door gridworld), PPO (Stable-Baselines3), 200,000-step
budget per run, CPU, one hand-built layout per horizon. Ground truth is always the
reward-blind evaluator, judged on final saved checkpoints.

Evidence labels used below:
- ROBUST: survives multiple-comparison correction, or is exact by construction/analysis.
- SUGGESTIVE: consistent direction, but not significant after correction or small n.
- NOT DETECTABLE: no effect distinguishable from noise at 10 seeds.

## The recipe (8 rules)

1. **Define success first and evaluate only on it.** [ROBUST]
   Reward and success can diverge: training-time averages hid late collapses, and
   with the terminal reward removed, 85.7% of failed episodes earned at least 90% of
   the success return (reward could not tell failure from success).

2. **Keep a terminal success reward.** [SUGGESTIVE, mechanism clear]
   Dense without terminal success: 3/10 seeds retained vs 9/10 for dense
   (raw p = 0.0198, Holm over 10 comparisons = 0.198).

3. **Add progress shaping tied to verified subgoal events.** [ROBUST]
   Retained seeds (of 10) on base / H50 / H100: sparse 4 / 4 / 0; dense 9 / 8 / 10;
   hybrid 8 / 7 / 9; potential 7 / 6 / 5; potential x3 7 / 8 / 9.
   At H100, sparse vs dense Holm p = 0.000065 and vs hybrid 0.000595. Pooled
   (stratified) potential x3 vs sparse: odds ratio 9.00, 95% CI [2.87, 28.26].
   Sparse failures mostly never logged a single success (no learning signal), and
   1,000,000 steps rescued only 2 of 12 stuck sparse runs.

4. **If the progress signal depends on position/distance or on repeatable events,
   use the potential form F = gamma*Phi(s') - Phi(s) (gamma matched to the
   learner) and audit it before training.** [ROBUST for the exploit; the safety of
   the potential form rests on theory (Ng, Harada, Russell 1999) plus zero high-return
   failures observed here]
   A non-potential distance bonus c per one-cell decrease became exploitable above the
   analytic break-even c = 1/30 (farming return 50c vs genuine path 20c + 1). Observed:
   c = 0.02: 3/5 solved, no hacking; c = 0.05 and 0.10: 0/5 solved, proxy return
   2.45 and 5.38 vs 2.0 and 3.0 for a genuine solution. The same signal in potential
   form (gamma 0.99) had no failed episode above the success return (3/5 solved).

5. **Make total shaping comparable to or larger than the terminal reward.** [SUGGESTIVE]
   Count-based potential (total shaping about 1x terminal): 5/10 at H100. Scaled x3:
   9/10 at H100 and 8/10 at H50; pooled odds ratio vs unscaled potential 2.58
   (CI [0.83, 8.03], not significant). Dense subgoal rewards total 1.5x terminal on
   the base layout and 3x on H100 (10/10 at H100); halving them gave 5/10 on base
   (not significant vs 9/10).

6. **Start without extra penalty terms.** [NOT DETECTABLE]
   Removing the invalid-action penalty or the step/timeout penalty from hybrid gave
   9/10 vs 8/10; removing key reward, door reward or the invalid penalty from dense gave
   7/10 vs 9/10 each. No consistent effect, so add penalties only for a measured need.
   Noisy potential (sigma 0.05, not a valid potential) retained 9/10; larger noise was
   not tested.

7. **Evaluate with at least 10 seeds, final checkpoints, deterministic and stochastic
   actions.** [ROBUST]
   Small samples misled: hybrid 3/3 at H50 became 7/10 with 10 seeds, potential x3 5/5
   on base became 7/10. Several deterministic failures succeeded under sampled actions
   (for example potential seed 9, hybrid seed 4 on base), so report both.

8. **Do not expect reward design to give cross-layout generalization.** [ROBUST here]
   Retained policies scored 71-89% when only the start moved, 0% when keys and doors
   were relocated (all variants). Robustness to action noise was high (94-96% at 50%
   random-action replacement).

## Decision guide

| Situation | Recommendation | Evidence |
| --- | --- | --- |
| Short task (about 3 subgoal pairs, about 25 steps) | Any shaped reward works; choose the simplest (dense) | base: dense 9/10, hybrid 8/10, potential 7/10, sparse 4/10 |
| Long task (5-6 pairs, 50-100 steps) | Dense, hybrid or scaled potential; do not use sparse alone | H100: sparse 0/10, dense 10/10, hybrid 9/10, potential x3 9/10 |
| Progress signal uses distance or repeatable events | Potential form, scaled; pass the audit below | break-even c = 1/30; hacking above it |
| Cannot define a progress function | Not covered by these experiments | Phi here uses designer knowledge of subgoal count and order |

## Pre-training reward-hacking audit

1. List every reward component, its trigger event, and whether it can repeat.
2. Compute the largest return a non-completing loop can earn within the episode limit
   (for example oscillation) and the return of a genuine completion. Require
   loop return < completion return.
3. Run scripted loop and partial-progress trajectories through the real reward code and
   the reward-blind evaluator.
4. After training, report proxy return vs success, and the fraction of failed episodes
   whose proxy return is at least 90% of the success return.
5. Prefer the potential form for any position- or distance-based signal.

## Limitations

- One task family and one algorithm (PPO); no other environments or algorithms tested.
- Each horizon is a single hand-built layout, so layout and length are confounded.
- 200,000-step budget; some "stuck" or collapsed runs might change with other budgets
  (the sparse 1M-step check covers only 12 runs).
- Ablation effects are mostly not detectable at 10 seeds; only the terminal-reward
  result is suggestive.
- H10 has 3 seeds and is preliminary.
- Potential Phi uses designer knowledge of the subgoal structure.
- Not tested: stochastic training dynamics, observation noise, larger progress-signal
  noise, normalized vs unnormalized components, other potential formulations beyond
  scaling and noise.

## Evidence files

- docs/FINDINGS.md, docs/DEFINITION_OF_DONE.md
- runs/stage4_10seed_final_checkpoint/final_checkpoint_per_seed.csv
- runs/stage5c_weight_sweep_final/ and runs/stage5b_reward_hacking/
- runs/stage6_zero_shot/
- runs/stage7/final/
- runs/stage8/analysis/ (pairwise and stratified tests, sparse long-budget)
- runs/stage8b/analysis/
