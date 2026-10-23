# Project Context & Agent Handoff Guide

> **Note for AI Agent (OpenAI Codex / VS Code):** This document contains full technical context, architectural decisions, experiment status, known failure analysis, and next steps for the project. Every claim in this document has been verified against the physical filesystem, runtime logs, and test executions on the Windows machine.

---

## A. Project Overview

- **Sponsorship & Origin:** This is an industry-sponsored research and engineering project from **SingleCore Labs** (Founder, CEO & CTO: **Manav Sutar**). The company builds reinforcement learning infrastructure for enterprise agentic AI (specifically an "RL Lab" platform for environments, reward modeling, and evaluation). Multiple university teams received the problem statement: *"Reward Model Design for Long-Horizon Agent Planning (RL for Agentic AI)"*.
- **Core Objective:** Design, implement, and benchmark four reward formulations—**sparse**, **dense**, **potential-based shaping**, and **hybrid**—on a long-horizon key-door gridworld. Evaluate sample efficiency, detect and diagnose reward hacking, quantify reward-to-success correlation, test zero-shot generalization to unseen layouts, and deliver a validated reward-shaping decision recipe and report to be presented directly back to SingleCore Labs leadership.
- **Hardware & Runtime Environment:** Local execution on Windows 11 with an **Intel Core i5 CPU (4 physical cores)**. Compute is **strictly CPU-only** (no CUDA/GPU). Due to the small neural network (MLP with two hidden layers of 64 units) and CPU-bound Python environment stepping, GPU execution (such as Google Colab T4) was evaluated and rejected because CPU-GPU tensor transfer latency exceeds any compute acceleration.
- **Core Tech Stack:** Python 3.13.7, PyTorch 2.13.0+cpu, Gymnasium 1.4.0, Stable-Baselines3 (SB3) 2.9.0, PyYAML 6.0.3, NumPy 2.3.3, Pandas 2.3.3, Matplotlib 3.10.6, pytest 8.4.2.
- **Chosen Strategic Differentiators:**
  1. *Deep Reward-Hacking Stress Testing:* Constructing distinct exploit scenarios against dense and hybrid rewards to measure proxy reward divergence from true task success (directly aligning with SingleCore Labs' core product mission).
  2. *Zero-Shot Generalization on Unseen Layouts:* Evaluating policies trained on the base fixed layout across 2–3 new layouts without retraining.
  - *Deferred Stretch Items:* Hand-coded heuristic planner baseline and phase-dependent adaptive rewards ($R_{\text{adaptive}}$) are deferred to focus solo builder bandwidth on the two main differentiators.

---

## B. Planning and Specification Files

### Existing Files in Repository
- [project_requirements.md](file:///c:/Users/cheta/Desktop/agentic_RL/project_requirements.md): The verbatim 28-section project instructions from SingleCore Labs founder Manav Sutar, defining the problem, required candidates ($R_{\text{sparse}}$, $R_{\text{dense}}$, $R_{\text{potential}}$, $R_{\text{hybrid}}$, $R_{\text{adaptive}}$), evaluation metrics, required plots, ablations, and definition of done.
- [chat.md](file:///c:/Users/cheta/Desktop/agentic_RL/chat.md): The initial kickoff communication log containing the founder's letter, problem analysis, and the initial proposal for the key-door gridworld task.
- [project_spec_reward_shaping.md](file:///c:/Users/cheta/Desktop/agentic_RL/project_spec_reward_shaping.md): Master specification and build plan divided into Part 1 (Founder requirements verbatim) and Part 2 (Implementation decisions, compute constraints, differentiator priorities, and 12-stage exit criteria).
- [docs/environment.md](file:///c:/Users/cheta/Desktop/agentic_RL/docs/environment.md): Formal MDP specification $M = (S, A, P, R, \gamma)$, state and action definitions, transition dynamics, ground-truth task objective rules, structured observation dictionary, and base layout diagram.
- [configs/env/base_3keys.yaml](file:///c:/Users/cheta/Desktop/agentic_RL/configs/env/base_3keys.yaml): ASCII layout grid definition ($17 \times 5$, 3 key-door pairs, start at (1,1), goal at (1,15), optimal length 26 steps, step limit 100).
- [configs/sparse.yaml](file:///c:/Users/cheta/Desktop/agentic_RL/configs/sparse.yaml): Experiment configuration for $R_{\text{sparse}}$ with fixed PPO hyperparameters (200k steps, lr=3e-4, n_steps=2048, batch_size=64, 5 seeds: 0–4).
- [configs/dense.yaml](file:///c:/Users/cheta/Desktop/agentic_RL/configs/dense.yaml): Experiment configuration for $R_{\text{dense}}$ (+0.2 key pickup, +0.3 door opened, -0.05 invalid action, identical PPO hyperparameters).
- [configs/potential.yaml](file:///c:/Users/cheta/Desktop/agentic_RL/configs/potential.yaml): Experiment configuration for $R_{\text{potential}}$ ($\gamma=0.99$ potential shaping, identical PPO hyperparameters).
- [configs/hybrid.yaml](file:///c:/Users/cheta/Desktop/agentic_RL/configs/hybrid.yaml): Experiment configuration for $R_{\text{hybrid}}$ (dense + validity -0.05 + step cost -0.005 + potential $\gamma=0.99$ + terminal success +1.0 / timeout -0.1).
- [pyproject.toml](file:///c:/Users/cheta/Desktop/agentic_RL/pyproject.toml): Build system and pytest configuration (`testpaths = ["tests"]`, `pythonpath = ["."]`).
- [requirements.txt](file:///c:/Users/cheta/Desktop/agentic_RL/requirements.txt): Pinned package versions for Python 3.13.7 Windows CPU environment.
- [run_experiments.py](file:///c:/Users/cheta/Desktop/agentic_RL/run_experiments.py): Multi-worker experiment queue runner with CPU thread pinning for Stage 4 pilot and full matrix runs.

### Missing Files
- [docs/PROGRESS.md](file:///c:/Users/cheta/Desktop/agentic_RL/docs/PROGRESS.md): **MISSING.** Specified in the development plan to track stage-by-stage results, findings, and decisions. The next agent must create this file when concluding Stage 4.
- **Git Repository (`.git`):** **MISSING.** Git is not initialized in `c:\Users\cheta\Desktop\agentic_RL`. Running `git status` returns `fatal: not a git repository`. Changes cannot be tracked via `git diff` or `git log`.
- **Founder's Original PDF / External Attachments:** The raw external attachment mentioned in `chat.md` is preserved textually as `project_requirements.md`, but original binary attachments (e.g. PDFs) are not in the repo root.
- **CLI Commands `compare` and `reward_audit`:** While `reward_rl.train` and `reward_rl.evaluate` exist, `reward_rl.compare` and `reward_rl.reward_audit` from Section 25 have not yet been implemented (scheduled for Stage 9).

---

## C. Locked Design Decisions

1. **Task Environment (Key-Door-Chain Gridworld):**
   - *Decision:* 2D discrete gridworld with 3 sequential key-door pairs on a fixed layout ([configs/env/base_3keys.yaml](file:///c:/Users/cheta/Desktop/agentic_RL/configs/env/base_3keys.yaml)), 26-step shortest path verified by BFS solver, maximum episode budget of 100 steps.
   - *Rationale:* Provides a strict dependency chain where earlier subgoals (e.g. key `a` tucked into a dead-end pocket) provide no immediate spatial proximity to the final goal. Walking into walls or trying locked doors yields natural invalid actions needed to benchmark invalid action rates.
2. **Key Consumption & Environment Exploit Prevention:**
   - *Decision:* Keys are consumed upon unlocking a door and cannot be picked up twice.
   - *Rationale:* The environment dynamics themselves contain no reward-farming loopholes. Any reward hacking observed during experiments is guaranteed to arise entirely from reward shaping design flaws (e.g. step-level potential oscillation, proxy-objective divergence).
3. **Observation Design:**
   - *Decision:* Structured dictionary observation space (`agent_pos`, `keys_held`, `keys_collected`, `doors_open`, `key_positions`, `door_positions`, `goal_pos`, `action_mask`, `prev_action`, `time_remaining`) flattened via `FlattenObsWrapper` for SB3.
   - *Rationale:* Section 6 explicitly forbids pixel/image state to keep compute lightweight and avoid image representation confounders. Including `key_positions` and `door_positions` allows zero-shot generalization testing on unseen layouts in Stage 6. Action masking explicitly exposes available actions.
4. **Controlled Experimental Variables (Fixed PPO Hyperparameters):**
   - *Decision:* Identical PPO hyperparameters across all 4 variants: MLP [64, 64], learning rate $3 \times 10^{-4}$, $n\_steps=2048$, $batch\_size=64$, $n\_epochs=10$, $\gamma=0.99$, $\lambda=0.95$, clip range 0.2, entropy coefficient 0.01, value function coefficient 0.5. Evaluated with 5 seeds (0, 1, 2, 3, 4).
   - *Rationale:* Founder Section 3 requires holding the environment, algorithm, and hyperparameters constant so that the reward formulation is the sole independent variable.
5. **Layout Randomness Control:**
   - *Decision:* Fixed base layout for all core training (Stages 2–4). Unseen layouts used strictly during zero-shot evaluation (Stage 6).
   - *Rationale:* Randomized layouts during training introduce high variance in layout difficulty, confounding whether sample efficiency gains stem from reward shaping or layout luck. Unseen layouts in Stage 6 must keep the identical grid size ($17 \times 5$) and key count (3 pairs) to preserve observation tensor shapes.
6. **Ground-Truth Evaluator Incorruptibility:**
   - *Decision:* Ground-truth evaluator ([reward_rl/evaluation/success_evaluator.py](file:///c:/Users/cheta/Desktop/agentic_RL/reward_rl/evaluation/success_evaluator.py)) replays trajectories from scratch through pure dynamics, inspects events, and validates strict subgoal ordering. It never imports, inspects, or receives reward signals. Tested by AST analysis in [tests/test_success_evaluator.py](file:///c:/Users/cheta/Desktop/agentic_RL/tests/test_success_evaluator.py).
   - *Rationale:* Founder's "critical implementation rule": never optimize reward before defining true success. Evaluator must NEVER be modified to "make things pass".

---

## D. Stage Plan and Current Status

| Stage | Name | Status | Exit Criteria & Verification Evidence |
|---|---|---|---|
| **0** | **Environment Setup** | **DONE** | Python virtual environment active (`.venv`), PyTorch 2.13.0+cpu, SB3 2.9.0, Gymnasium 1.4.0 verified. Pinned `requirements.txt` and `pyproject.toml`. Skeleton folders matching Section 24 created. |
| **1** | **Core Env + Ground-Truth Evaluator** | **DONE** | Gymnasium environment implemented ([reward_rl/envs/key_door_env.py](file:///c:/Users/cheta/Desktop/agentic_RL/reward_rl/envs/key_door_env.py)), BFS solver verifies 26-step optimal solution ([reward_rl/envs/solver.py](file:///c:/Users/cheta/Desktop/agentic_RL/reward_rl/envs/solver.py)). Ground-truth evaluator built and verified reward-blind. Unit tests pass (82/82). Random agent solved 0/1000 episodes (confirming long-horizon hardness). |
| **2** | **Sparse Baseline + Pipeline Check** | **DONE** | $R_{\text{sparse}}$ implemented ([reward_rl/rewards/sparse.py](file:///c:/Users/cheta/Desktop/agentic_RL/reward_rl/rewards/sparse.py)). PPO trained seed 0 for 200k timesteps (`runs/sparse/seed_0/results.json` shows 97.3% ground-truth success, 26.2 avg length, 431.9s runtime). Checkpoints and TensorBoard logs verified. |
| **3** | **Dense, Potential, Hybrid Rewards** | **DONE** | Implemented $R_{\text{dense}}$, $R_{\text{potential}}$, $R_{\text{hybrid}}$ with per-component TensorBoard decomposition callbacks. Pipeline validation runs completed on seed 0 (200k steps): Dense 97.7% (373s), Potential 97.6% (482s), Hybrid 96.0% (358s, with temporary mid-training policy collapse 100% -> 11% -> 100%). All 82 tests pass. |
| **4** | **Full Matrix (4 variants x 5 seeds) + Core Metrics** | **IN PROGRESS (BLOCKED)** | Runner script `run_experiments.py` created for parallel execution (3 workers, thread pinning). Pilot run (seed 42, 100k timesteps) attempted to calibrate final timestep budget, but failed with `'charmap' codec can't encode character '\u2192'` in `run_experiments.py`. Exit criteria: 20 successful runs (4 variants x 5 seeds), metrics computed (steps to 25/50/75/90%, AUC, invalid-action rate, subgoal rate), and comparison plots generated. |
| **5** | **Reward-Hacking Stress Tests** | **NOT STARTED** | Exit criteria: 2–3 distinct exploit scenarios against dense/hybrid variants (e.g. step oscillation, key re-touching proxy), proxy reward vs true success scatter plots, Pearson/Spearman reward-success correlation analysis. |
| **6** | **Generalization on Unseen Layouts** | **NOT STARTED** | Exit criteria: 2–3 new $17 \times 5$ layouts created with 3 key-door pairs, zero-shot evaluation of existing trained models without retraining, comparison of generalization drop across reward types. |
| **7** | **Horizon Scaling (5–6 Keys)** | **NOT STARTED** | Exit criteria: Scaled grid layout with 5–6 sequential key-door pairs (~60–100+ steps), retrain 4 variants (3 seeds), analyze whether dense/potential advantages shrink or grow. |
| **8** | **Ablations** | **NOT STARTED** | Exit criteria: At minimum one ablation completed (e.g. remove invalid action penalty -0.05 from hybrid reward, evaluate impact on training reward vs true success). |
| **9** | **Analysis, Recipe, and Report** | **NOT STARTED** | Exit criteria: Metrics consolidation, written "reward-shaping recipe" decision tree with explicit boundary conditions and failure modes, CLI commands `reward_rl.compare` and `reward_rl.reward_audit` implemented. |
| **10** | **Presentation Prep for SingleCore Labs** | **NOT STARTED** | Exit criteria: Slide deck matching founder's 28-section document structure, visual replays/GIFs of agent behavior and reward hacking, rehearsal notes. |
| **11** | **Replay Viewer Web Demo (Stretch)** | **NOT STARTED** | Exit criteria: Interactive web-based or lightweight browser playback of agent gridworld episodes (ASCII viewer currently exists in `reward_rl/envs/demo.py`). |

---

## E. What Exists in the Repository

### Annotated File Tree
```
agentic_RL/
├── AGENTS.md                          # Quick instruction card for coding agents (<60 lines)
├── chat.md                            # Kickoff conversation & founder's email
├── project_requirements.md            # Verbatim 28-section SingleCore Labs requirements doc
├── project_spec_reward_shaping.md     # Full design spec, decisions, and stage exit criteria
├── pyproject.toml                     # Package & test configuration
├── requirements.txt                   # Pinned Python package dependencies
├── run_experiments.py                 # Multi-worker parallel runner with thread pinning
├── configs/
│   ├── dense.yaml                     # Dense reward training config (200k steps, seeds 0-4)
│   ├── hybrid.yaml                    # Hybrid reward training config
│   ├── potential.yaml                 # Potential shaping config
│   ├── sparse.yaml                    # Sparse baseline config
│   └── env/
│       └── base_3keys.yaml            # 17x5 ASCII grid layout (3 key-door pairs)
├── docs/
│   ├── environment.md                 # Formal MDP specification & ground-truth rules
│   ├── PROJECT_CONTEXT.md             # This document (complete handoff guide)
│   └── PROGRESS.md                    # [MISSING - to be created by next agent]
├── reward_rl/                         # Core Python package
│   ├── __init__.py                    # Package root
│   ├── train.py                       # CLI wrapper: python -m reward_rl.train
│   ├── evaluate.py                    # CLI wrapper: python -m reward_rl.evaluate
│   ├── agents/                        # Agent wrappers (stub initialized)
│   ├── analysis/                      # Audit & correlation scripts (stub initialized)
│   ├── visualization/                 # Plotting utilities (stub initialized)
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── dynamics.py                # Pure MDP transition functions, GridState, Action
│   │   ├── key_door_env.py            # Gymnasium environment class (KeyDoorEnv)
│   │   ├── layout.py                  # ASCII layout parser & validation
│   │   ├── solver.py                  # BFS optimal path solver (verifies 26 steps)
│   │   ├── wrappers.py                # FlattenObsWrapper for SB3 MLP policy
│   │   └── demo.py                    # ASCII demo script of episode rollout
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py                # Policy evaluation script & checkpoint loader
│   │   └── success_evaluator.py       # Ground-truth success evaluator (strictly reward-blind)
│   ├── rewards/
│   │   ├── __init__.py                # Reward factory: make_reward()
│   │   ├── sparse.py                  # R_sparse implementation
│   │   ├── dense.py                   # R_dense implementation (+per-component tracking)
│   │   ├── potential.py               # R_potential (F = gamma*Phi' - Phi)
│   │   └── hybrid.py                  # R_hybrid implementation
│   └── training/
│       ├── __init__.py
│       ├── callbacks.py               # Subgoal, invalid action, and reward decomposition callbacks
│       └── train.py                   # PPO training loop with checkpointing & seed management
├── runs/                              # Saved checkpoints, TensorBoard logs, and results
│   ├── queue_results_pilot.json       # Record of failed Stage 4 pilot queue run
│   ├── sparse/ (seed_0, seed_42)
│   ├── dense/ (seed_0, seed_42)
│   ├── potential/ (seed_0, seed_42)
│   └── hybrid/ (seed_0, seed_42)
└── tests/                             # Pytest test suite (82 tests passing)
    ├── conftest.py                    # Shared fixtures & 26-step hand-traced test trajectory
    ├── test_dense_reward.py
    ├── test_dynamics.py
    ├── test_env.py
    ├── test_hybrid_reward.py
    ├── test_layout.py
    ├── test_potential_reward.py
    ├── test_reward_factory.py
    ├── test_sparse_reward.py
    ├── test_success_evaluator.py      # Includes AST reward-blindness assertion
    └── test_wrapper_and_training.py
```

### Key Modules and Responsibilities
- `reward_rl.envs.dynamics`: Functional core of the gridworld. Contains `GridState`, `Action` enum (UP, DOWN, LEFT, RIGHT, INTERACT), `StepEvents`, `transition(layout, state, action)`, and `EpisodeRecord`. Completely stateless and independent of Gym or PyTorch.
- `reward_rl.envs.key_door_env`: Custom `gymnasium.Env` wrapper implementing `reset()`, `step()`, dictionary observation space, and action masking.
- `reward_rl.evaluation.success_evaluator`: Deterministic referee. Receives `EpisodeRecord`, replays each action step-by-step through `transition()`, verifies that keys were collected before doors were opened, checks that doors were unlocked in strict numerical order 0, 1, 2, and confirms goal arrival within `max_steps`. Computes `planning_efficiency` ($26 / \text{steps}$) and `invalid_action_rate`.
- `reward_rl.envs.solver`: Breadth-First Search (BFS) graph search algorithm over `(agent_pos, keys_held_bitmask, doors_open_bitmask)`. Verifies layout solvability and outputs optimal step length.
- `reward_rl.rewards`: Implements the reward variants. Each step returns a scalar reward and a decomposed dictionary of components (`key_pickup`, `door_opened`, `invalid_action`, `step_cost`, `potential_shaping`, `terminal_success`, `timeout_penalty`).
- `reward_rl.training.train`: Handles PPO setup, seed iteration, environment creation, callback attachment, TensorBoard logging, checkpoint saving, and JSON results dumping.
- `run_experiments.py`: Manages parallel execution of multiple training jobs across worker threads/processes, enforcing single-thread PyTorch/BLAS pinning per worker to prevent CPU thread thrashing.

### Test Suite Status
- Real execution output of `.\.venv\Scripts\python -m pytest`:
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0
  rootdir: C:\Users\cheta\Desktop\agentic_RL
  configfile: pyproject.toml
  testpaths: tests
  plugins: anyio-4.10.0, hydra-core-1.3.4, langsmith-0.4.33, cov-7.0.0
  collected 82 items

  tests\test_dense_reward.py .......                                       [  8%]
  tests\test_dynamics.py .........                                         [ 19%]
  tests\test_env.py ..........                                             [ 31%]
  tests\test_hybrid_reward.py ........                                     [ 41%]
  tests\test_layout.py .........                                           [ 52%]
  tests\test_potential_reward.py ......                                    [ 59%]
  tests\test_reward_factory.py .........                                   [ 70%]
  tests\test_sparse_reward.py .......                                      [ 79%]
  tests\test_success_evaluator.py ...........                              [ 92%]
  tests\test_wrapper_and_training.py ......                                [100%]

  ============================= 82 passed in 54.39s =============================
  ```
- **Pass/Fail Count:** **82 passed, 0 failed.**

### Environment & Virtualenv on Windows
- **Python Version:** `3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)]`
- **Key Package Versions:**
  - `torch`: `2.13.0+cpu`
  - `gymnasium`: `1.4.0`
  - `stable-baselines3`: `2.9.0`
  - `numpy`: `2.3.3`
  - `pytest`: `8.4.2`
- **Virtualenv Activation on Windows:**
  - PowerShell: `.\.venv\Scripts\Activate.ps1`
  - Direct interpreter invocation: `.\.venv\Scripts\python.exe <args>`

---

## F. Current Blocker: Stage 4 Pilot Run Failure

### 1. Run Setup
To determine the final training timestep budget for the full 20-run experiment matrix, a pilot queue was launched:
- **Command:** `.\.venv\Scripts\python run_experiments.py --mode pilot --workers 3 --timesteps 100000`
- **Configuration:** 4 jobs (sparse, dense, potential, hybrid), seed 42, 100,000 timesteps override, 3 parallel workers on CPU.

### 2. Verbatim Error Output
From [runs/queue_results_pilot.json](file:///c:/Users/cheta/Desktop/agentic_RL/runs/queue_results_pilot.json):
```json
{
  "mode": "pilot",
  "workers": 3,
  "timesteps": 100000,
  "results": [
    {
      "status": "exception",
      "config": "configs/potential.yaml",
      "seed": 42,
      "error": "'charmap' codec can't encode character '\\u2192' in position 28: character maps to <undefined>"
    },
    {
      "status": "exception",
      "config": "configs/dense.yaml",
      "seed": 42,
      "error": "'charmap' codec can't encode character '\\u2192' in position 24: character maps to <undefined>"
    },
    {
      "status": "exception",
      "config": "configs/sparse.yaml",
      "seed": 42,
      "error": "'charmap' codec can't encode character '\\u2192' in position 25: character maps to <undefined>"
    },
    {
      "status": "exception",
      "config": "configs/hybrid.yaml",
      "seed": 42,
      "error": "'charmap' codec can't encode character '\\u2192' in position 25: character maps to <undefined>"
    }
  ]
}
```
Console summary output:
```text
Queue complete: 0/4 succeeded in 386s. Results file: runs/queue_results_pilot.json
```

### 3. Chronology of What Was Attempted
1. **Runner Architecture Switch:** Switched from `concurrent.futures.ProcessPoolExecutor` to `concurrent.futures.ThreadPoolExecutor` managing independent `subprocess.run()` calls. This was done to allow per-subprocess environment variable setting (pinning CPU threads to 1 per worker) without Windows multiprocessing pickling/forking issues.
2. **Subprocess Decoding:** Replaced raw text decoding with `result.stdout.decode("utf-8", errors="replace")` and `result.stderr.decode("utf-8", errors="replace")`.
3. **Environment UTF-8 Flags:** Injected `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` into `env` passed to `subprocess.run()`.
4. **Removal of Non-ASCII Characters in Print Statements:**
   - In `run_experiments.py`, replaced Unicode right arrow (`→`) in completion print statements with ASCII `->` (`print(f"  [DONE]  {variant_name} seed={seed} -> {gt_rate:.1%} success ({elapsed:.0f}s)", flush=True)`).
   - In `run_experiments.py`, line 80 start print was modified to `print(f"  [START] {variant_name} seed={seed} (timesteps={timesteps or 'from config'})", flush=True)`.

### 4. Working Tree State of Edits (Verified Without Git)
Because `git` is not initialized (`fatal: not a git repository`), `git status` / `git diff` cannot be used. Physical file inspection confirms:
- **`run_experiments.py`:** Contains the switch to `ThreadPoolExecutor`, binary decode with fallback, `PYTHONUTF8=1` in subprocess env, and ASCII `->` on line 103.
- **`reward_rl/training/train.py`:** Still contains Unicode `±` on line 224 (`print(f"  Success rate: {np.mean(rates):.1%} ± {np.std(rates):.1%}")`).
- **`configs/potential.yaml`:** Still contains Unicode Greek letters on line 6 (`description: "R_potential: sparse + potential-based shaping F(s)=γΦ(s')-Φ(s) (Section 9)"`).
- **`reward_rl/rewards/potential.py`:** Still contains Unicode right arrow (`→`) on line 31 (`# The agent must still discover the correct key→door ordering itself.`).

### 5. Honest Assessment of Root Cause & Exact Character Positions
The character positions in the error message explain the failure precisely:
- String: `  [START] potential seed=42 → ...` -> Prefix length before `→` is exactly **28 characters**:
  ` ` (2) + `[START]` (7) + ` ` (1) + `potential` (9) + ` ` (1) + `seed=42` (7) + ` ` (1) = **28**.
- String: `  [START] dense seed=42 → ...` -> Prefix length before `→` is:
  ` ` (2) + `[START]` (7) + ` ` (1) + `dense` (5) + ` ` (1) + `seed=42` (7) + ` ` (1) = **24**.
- String: `  [START] sparse seed=42 → ...` -> Prefix length before `→` is:
  ` ` (2) + `[START]` (7) + ` ` (1) + `sparse` (6) + ` ` (1) + `seed=42` (7) + ` ` (1) = **25**.
- String: `  [START] hybrid seed=42 → ...` -> Prefix length before `→` is:
  ` ` (2) + `[START]` (7) + ` ` (1) + `hybrid` (6) + ` ` (1) + `seed=42` (7) + ` ` (1) = **25**.

The exception was thrown by Python's standard `print()` in the **parent runner process** on Windows, where standard output was attached to a console using Windows codepage 1252 (`charmap`). Codepage 1252 does not support `\u2192`. Because the `print()` was inside `try: ... except Exception as e:` inside `run_single_job`, the exception was caught, recorded as `"status": "exception"`, and aborted that job's execution or reporting.

**Has the pilot been re-run since the latest edits to `run_experiments.py`?**
**NO.** The timestamp on `runs/queue_results_pilot.json` is `23-10-2026 00:27`, whereas `run_experiments.py` was last saved at `23-10-2026 22:28`. The pilot queue has NOT been executed since line 80 was updated to remove the arrow.

### 6. Suggested First Steps for the Next Agent
1. **Sanity Check Single Job Directly (No Queue):** Run one job with a tiny step budget in the active terminal to confirm stdout formatting and catch any raw tracebacks:
   ```powershell
   .\.venv\Scripts\python -m reward_rl.train --config configs/sparse.yaml --seed 99 --output runs/debug
   ```
2. **Grep and Clean Non-ASCII Text Across Python Files:**
   Check for remaining non-ASCII characters in source code:
   - Line 224 of [reward_rl/training/train.py](file:///c:/Users/cheta/Desktop/agentic_RL/reward_rl/training/train.py): Replace `±` with `+/-`.
   - Line 6 of [configs/potential.yaml](file:///c:/Users/cheta/Desktop/agentic_RL/configs/potential.yaml): Replace `γ` and `Φ` with ASCII `gamma` and `Phi`.
3. **Audit File I/O Encoding:**
   Inspect all `open(..., "w")` calls (e.g. line 213 in [reward_rl/training/train.py](file:///c:/Users/cheta/Desktop/agentic_RL/reward_rl/training/train.py)) and ensure `encoding="utf-8"` is explicitly provided.
4. **Enforce UTF-8 on Windows Console Entry Points:**
   Add at the very top of `run_experiments.py` and `reward_rl/training/train.py`:
   ```python
   import sys
   if hasattr(sys.stdout, "reconfigure"):
       sys.stdout.reconfigure(encoding="utf-8")
   if hasattr(sys.stderr, "reconfigure"):
       sys.stderr.reconfigure(encoding="utf-8")
   ```
5. **Re-run the Pilot Queue:**
   ```powershell
   .\.venv\Scripts\python run_experiments.py --mode pilot --workers 3 --timesteps 100000
   ```

---

## G. Remaining Work

### 1. Stage 4 Pilot Analysis & Budget Setting
- Inspect learning curves from the pilot (seed 42, 100k steps for all 4 variants).
- Prior test runs showed that on seed 0:
  - Dense reached optimal 26 steps around ~30k–40k steps.
  - Sparse reached optimal 26 steps around ~50k–70k steps.
  - Hybrid recovered to 26 steps around ~80k steps.
- If all variants plateau by 40k–50k steps, select a budget approximately $2\times$ to $3\times$ the steps needed to reach 90% success (e.g. 100,000 timesteps). Keep this identical budget across all variants.
- The 5 seeds for the final matrix must be seeds `[0, 1, 2, 3, 4]`, distinct from pilot seed `42`.

### 2. Stage 4 Full Matrix Execution
- Execute 20 training runs: 4 variants $\times$ 5 seeds = 20 jobs.
- Command:
  ```powershell
  .\.venv\Scripts\python run_experiments.py --mode full --workers 3 --timesteps <BUDGET>
  ```
- Estimated wall-clock runtime: ~50–60 minutes on 3 CPU workers (compared to ~2.5 hours sequential).
- Verify each run writes to `runs/<variant>/seed_<seed>/` with `results.json`, `eval_logs/evaluations.npz`, and `final_model.zip`.

### 3. Implementation of Required Metrics and Graphs (Founder Sections 18 & 20)
| Metric / Graph | Founder Section | Implementation Status | Action Needed |
|---|---|---|---|
| Success rate vs. steps | Section 20 | **Missing** | Create plotting script in `reward_rl/visualization/` |
| Mean episode reward vs. steps | Section 20 | **Missing** | Extract from TB logs / eval_logs |
| Steps to 25%, 50%, 75%, 90% success | Section 18 | **Missing** | Compute threshold crossings from `evaluations.npz` |
| Area Under Learning Curve (AUC) | Section 14, 18 | **Missing** | Numerical trapezoidal integration over success rate curve |
| Invalid-action rate over training | Section 18, 20 | Partial | Logged in TensorBoard; need aggregation plot |
| Subgoal completion rate | Section 18 | Partial | Logged in TensorBoard; need comparison table |
| Seed variance (mean $\pm$ std / CI) | Section 18 | Partial | Calculated in train summary; need shaded error bands on curves |
| Reward-to-success correlation | Section 17, 18 | **Missing** | Compute scatter plot & Pearson/Spearman $r$ |

### 4. Overview of Stages 5 through 11
- **Stage 5 (Reward Hacking):** Construct 2–3 exploit wrappers/scenarios (e.g. step oscillation near goal, re-touching key locations if allowed, partial distance gaming). Plot proxy reward vs true task success showing clear divergence.
- **Stage 6 (Generalization):** Write 2–3 new $17 \times 5$ layout YAML files in `configs/env/` (same dimensions and 3 keys). Evaluate checkpoints zero-shot using `python -m reward_rl.evaluate`.
- **Stage 7 (Horizon Scaling):** Create a 5-key layout config. Retrain 4 variants across 3 seeds. Analyze whether potential shaping scales better than hand-crafted dense rewards.
- **Stage 8 (Ablations):** Remove invalid action penalty from $R_{\text{hybrid}}$ and evaluate impact on learning dynamics.
- **Stage 9 (Recipe & CLI):** Implement `python -m reward_rl.compare` and `python -m reward_rl.reward_audit`. Draft the validated reward shaping recipe.
- **Stage 10 (Presentation Prep):** Prepare slide deck structured according to Manav Sutar's 28-section document.
- **Stage 11 (Web Demo - Stretch):** Build a browser-based episode visualizer if time permits.

---

## H. Hard Rules for the Next Agent

1. **Evaluator Sanctity:** Never modify [reward_rl/evaluation/success_evaluator.py](file:///c:/Users/cheta/Desktop/agentic_RL/reward_rl/evaluation/success_evaluator.py) logic or allow it to import any reward modules. Tests in [tests/test_success_evaluator.py](file:///c:/Users/cheta/Desktop/agentic_RL/tests/test_success_evaluator.py) strictly enforce reward blindness via AST parsing.
2. **Controlled Independent Variable:** Never alter PPO hyperparameters, network architecture, or training parameters between reward variants. Reward formulation must remain the only variable under study.
3. **Reward Decomposition:** Every reward function must output decomposed components, and training callbacks must log each component independently. Never log only a single aggregated scalar reward.
4. **Scope Discipline:** Do not introduce features outside the stage plan (no continuous control, no image observations, no physics engines). The 2D discrete gridworld is locked.
5. **ASCII & UTF-8 Safety on Windows:** All strings printed to stdout/stderr must use standard ASCII (use `->` instead of `→`, `+/-` instead of `±`). All file read/write operations must explicitly declare `encoding="utf-8"`.
6. **Founder CLI & Package Architecture:** Retain the prescribed package structure and CLI entrypoints: `python -m reward_rl.train`, `python -m reward_rl.evaluate`, `python -m reward_rl.compare`, and `python -m reward_rl.reward_audit`.
7. **Progress Documentation:** Maintain and update [docs/PROGRESS.md](file:///c:/Users/cheta/Desktop/agentic_RL/docs/PROGRESS.md) at the completion of each stage with experiments executed, numerical metrics, and architectural decisions.

---

## I. Assumptions Made & Unfound Files

1. **Git Tracking:** Assumed that the absence of a `.git` folder was intentional or due to an uninitialized local folder. No remote git URL or commit history exists in the workspace.
2. **Founder's Binary Attachments:** Assumed that the markdown text in `project_requirements.md` and `chat.md` contains the complete, unabridged content of the founder's original requirements document, as no standalone PDF was found.
3. **CPU Core Allocation:** Assumed 3 parallel workers for `run_experiments.py` on the 4-core Intel i5, leaving 1 physical core free for OS responsiveness and file I/O.
4. **Existing Seed 42 Results:** Individual run directories for `seed_42` exist in `runs/` from prior direct or partial invocations, but `run_experiments.py` failed during the pilot queue aggregation. The pilot queue itself must be re-run cleanly to generate a valid `runs/queue_results_pilot.json`.
