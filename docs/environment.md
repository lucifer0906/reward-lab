# Environment Specification: Key-Door-Chain

Founder spec Sections 3–6. This document defines the MDP and the **true task objective** *independently of any reward function*.

## MDP  M = (S, A, P, R, γ)

| Element | Definition | Code |
|---|---|---|
| **S** (state) | `(agent_pos, keys_held[n], keys_collected[n], doors_open[n], step_count, at_goal)` | `GridState` in [dynamics.py](../reward_rl/envs/dynamics.py) |
| **A** (actions) | `{UP, DOWN, LEFT, RIGHT, INTERACT}`, fixed for all experiments | `Action` |
| **P** (transitions) | Deterministic; `transition(layout, s, a) → (s', events)` | `transition()` |
| **R** (reward) | **Not part of the env.** Injected via `reward_fn` (Stage 2+). Stage 1 returns 0. | `reward_rl/rewards/` |
| **γ** | Set in training configs (Stage 2+) | — |

### Transition rules
- **Move**: enter the target cell unless it is a wall or out of bounds (`wall`) or a closed door (`locked_door`). Entering the goal sets `at_goal`.
- **INTERACT**, in priority order:
  1. standing on an uncollected key → pick it up (`keys_held[i] = keys_collected[i] = 1`);
  2. next to a closed door *i* while holding key *i* → open it. **The key is consumed.**
  3. otherwise invalid (`no_key_for_door` if next to a closed door, else `nothing_to_interact`).
- Invalid actions leave the state unchanged but **still consume a step**.
- A key can never be picked up twice, so the env itself has no "re-pick" loop. Any reward-hacking loop must come from a reward design flaw, which is what Stage 5 tests.

### Terminal conditions
- `terminated`: agent reaches the goal.
- `truncated`: `step_count == max_steps` (default 100) without reaching the goal.

## True task objective (ground truth)
An episode is a **success** if and only if **all** of these hold:
1. The recorded trajectory replays exactly through the pure dynamics (tamper check).
2. Doors were opened in the order 0, 1, …, n−1.
3. Each door was opened only after its key was collected.
4. The agent reached the goal within `max_steps`.
5. All n doors are open.

Implemented in [success_evaluator.py](../reward_rl/evaluation/success_evaluator.py). It receives an `EpisodeRecord`, which contains **no reward information**. It also reports diagnostics: subgoals completed (out of 2n+1), invalid-action rate, door order, and planning efficiency (`optimal_length / episode_length`).

## Observation (structured, no pixels; Section 6)

| Key | Space | Spec requirement |
|---|---|---|
| `agent_pos` | MultiDiscrete[H, W] | relevant observations |
| `keys_held` | MultiBinary(n) | resource status |
| `keys_collected`, `doors_open` | MultiBinary(n) | completed subgoals / task progress |
| `key_positions`, `door_positions`, `goal_pos` | MultiDiscrete | relevant observations (needed for unseen-layout generalization in Stage 6) |
| `action_mask` | MultiBinary(5) | available actions |
| `prev_action` | Discrete(6), 5 = none | previous actions |
| `time_remaining` | Box[0, 1] | termination information |

## Base layout ([base_3keys.yaml](../configs/env/base_3keys.yaml))
```
#################
#S..#...#...#..G#
#...A.b.B...C...#
#a..#...#.c.#...#
#################
```
- 3 key→door pairs, **optimal solution 26 steps** (verified by BFS and by a hand-traced script), budget 100.
- Key `a` is in a dead-end corner. Fetching it has no visible benefit until door A, which is the long-horizon credit-assignment problem (Section 4).
- The layout is **fixed** for Experiments 1–7. Unseen layouts are used only for zero-shot generalization (Stage 6).

> **Constraint for Stage 6:** observation shapes depend on grid size (H, W) and number of pairs n. Unseen generalization layouts must keep the same H, W and n as the base layout so trained policies can be applied directly.
