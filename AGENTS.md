# AGENTS.md — Agent Handoff Guide

## Project Summary
This is a sponsored industry project from SingleCore Labs (founder Manav Sutar): "Reward Model Design for Long-Horizon Agent Planning" (RL for Agentic AI). The objective is to design, train, and benchmark four reward formulations (sparse, dense, potential-based shaping, and hybrid) on a fixed 3-key sequential key-door Gymnasium gridworld using PPO (Stable-Baselines3). The agent must measure sample efficiency, evaluate zero-shot layout generalization, conduct deep reward-hacking stress tests, and deliver a validated reward-shaping recipe to present back to the company. Execution runs CPU-only on Windows (Intel i5, 4 cores).

Full context: [docs/PROJECT_CONTEXT.md](file:///c:/Users/cheta/Desktop/agentic_RL/docs/PROJECT_CONTEXT.md)

## Hard Rules for Any Agent Working on This Project
1. **Never modify the ground-truth success evaluator logic** ([reward_rl/evaluation/success_evaluator.py](file:///c:/Users/cheta/Desktop/agentic_RL/reward_rl/evaluation/success_evaluator.py)) or let it import any reward code. Tests enforce strict reward blindness via AST.
2. **Do not change PPO hyperparameters or network architecture between reward variants.** Reward formulation must remain the only independent variable.
3. **Log every reward component separately (reward decomposition).** Never log only a single aggregated scalar reward.
4. **Do not add features outside the stage plan.** Do not build complex or continuous environments; keep the 2D discrete gridworld.
5. **Keep all stdout/stderr prints and log files strictly ASCII-safe.** Always specify `encoding="utf-8"` on every file `open()`.
6. **Preserve founder's package structure and CLI entrypoints:** `python -m reward_rl.train`, `python -m reward_rl.evaluate`, `python -m reward_rl.compare`, and `python -m reward_rl.reward_audit`.
7. **Update [docs/PROGRESS.md](file:///c:/Users/cheta/Desktop/agentic_RL/docs/PROGRESS.md) at the end of each stage** (create it if missing) with what was done, key metrics, and architectural decisions.

## Setup & Exact Commands (Windows PowerShell)
```powershell
# 1. Activate existing virtual environment
.\.venv\Scripts\Activate.ps1
# Alternative direct interpreter: .\.venv\Scripts\python.exe

# 2. Run unit & integration test suite (82 tests currently passing)
.\.venv\Scripts\python -m pytest

# 3. Run a single training job (e.g. dense reward, seed 0, 200k steps)
.\.venv\Scripts\python -m reward_rl.train --config configs/dense.yaml --seed 0

# 4. Run a fast test training job (tiny step count for debugging)
.\.venv\Scripts\python -m reward_rl.train --config configs/sparse.yaml --seed 99 --output runs/debug

# 5. Run parallel experiment queue (Stage 4 pilot: 4 jobs, 3 workers)
.\.venv\Scripts\python run_experiments.py --mode pilot --workers 3 --timesteps 100000

# 6. Run full Stage 4 matrix (20 jobs: 4 variants x 5 seeds, requires --timesteps)
.\.venv\Scripts\python run_experiments.py --mode full --workers 3 --timesteps <BUDGET>
```
