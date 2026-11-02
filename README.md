# Reward Lab: Long-Horizon Reward Design

This repository benchmarks sparse, dense, potential-based, and hybrid rewards
for PPO in a sequential key-door Gymnasium gridworld. Ground-truth success is
checked by a reward-blind referee, not by training reward.

## Setup

Use Python 3.13.7 on Windows CPU. Create/activate a virtual environment and
install the pinned dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows terminals must use UTF-8. PowerShell users can set
`$env:PYTHONUTF8='1'` and `$env:PYTHONIOENCODING='utf-8'`; the package entry
points also reconfigure standard streams to UTF-8.

## Founder Section 25 commands

```powershell
# Train one configured run (200k steps in configs/sparse.yaml).
python -m reward_rl.train --config configs/sparse.yaml --seed 0 --output runs/sparse

# Evaluate a saved checkpoint with the reward-blind referee.
python -m reward_rl.evaluate --checkpoint runs/sparse/seed_0/checkpoints/final_model.zip --episodes 200 --output runs/example_evaluation.json

# Aggregate existing core seed runs (requires their TensorBoard logs).
python -m reward_rl.compare --seeds 0 --timesteps 200000 --output-dir runs/example_compare

# Run the scripted reward-hacking audit.
python -m reward_rl.reward_audit --output-dir runs/example_reward_audit
```

## Repository map

- `configs/`: locked core reward/PPO configs and environment layouts.
- `reward_rl/envs/`: gridworld, solver, wrappers, and dynamics.
- `reward_rl/evaluation/`: reward-blind success evaluator and CLI evaluation.
- `reward_rl/rewards/`: sparse, dense, potential, and hybrid rewards.
- `reward_rl/training/`: PPO training, callbacks, and resumable queues.
- `reward_rl/analysis/`: checkpoint, reward-hacking, and statistical analyses.
- `runs/`: checkpoints, TensorBoard logs, saved evaluations, tables, and figures.
- `docs/`: findings, progress log, recipe, and completion audit.

## Reproducing headline tables

Existing-run analyses are reproducible without retraining:

```powershell
python -m reward_rl.analysis.stage8_summarize
python -m reward_rl.analysis.stage8b_summarize
python -m reward_rl.analysis.final_auc_convergence
```

Use the core configs in `configs/{sparse,dense,potential,hybrid}.yaml`; the
Stage 8b potential scale experiment uses `configs/stage8b/`.

## Results and limits

See [the recipe](docs/RECIPE.md), [findings](docs/FINDINGS.md), and the
[definition-of-done audit](docs/DEFINITION_OF_DONE.md). Headline claims use
deterministic final-checkpoint evaluation and seed-level statistics. Main
limitations are hand-built layouts (layout and horizon are confounded), only
three H10 seeds, limited noisy-signal settings, no aggregate reward-vs-step
graph, and no universal reward-ranking claim outside this task.
