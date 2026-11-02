# Definition of Done Audit

This is a documentation-only audit against founder Sections 18--27. Statuses
mean: **done** has direct saved evidence; **partial** has useful evidence but a
scope, replication, or artifact gap; **not done** has no adequate evidence.

## Section 27 checklist

| Item | Status | Evidence | Honest gap |
| --- | --- | --- | --- |
| Reproducible long-horizon Gymnasium environment | done | `reward_rl/envs/`, `configs/env/`, `tests/test_env.py` | Fixed hand-built layouts only. |
| Reward-independent success evaluator | done | `reward_rl/evaluation/success_evaluator.py`, `tests/test_success_evaluator.py` | None identified. |
| Sparse, dense, potential, hybrid rewards | done | `reward_rl/rewards/`, `configs/{sparse,dense,potential,hybrid}.yaml` | None. |
| Reward components independently logged | done | `reward_rl/training/callbacks.py`, TensorBoard logs in `runs/*/tb_logs/` | No consolidated component plot for every experiment. |
| PPO trains all primary variants | done | `runs/{sparse,dense,potential,hybrid}/seed_*/results.json` | Results are task-specific. |
| Multiple random seeds evaluated | done | `runs/stage4_10seed_final_checkpoint/`, `runs/stage8/analysis/` | H10 remains only three seeds. |
| Quantitative sample efficiency | partial | `runs/stage8/analysis/stage4_steps_to_90_checkpoint.csv`, `runs/stage7/final/steps_to_90_vs_horizon.png` | 25/50/75 thresholds and a common all-variant sample-efficiency summary are absent. |
| Long-horizon scaling | partial | `runs/stage8/analysis/horizon_comparison_summary.csv`, Stage 7 layouts | One hand-built layout per length confounds layout and horizon. |
| Reward-hacking tests | done | `runs/stage5_reward_hacking/audit_results.json`, `runs/stage5c_weight_sweep_audit/audit_results.json` | Scripted stress cases are not an exhaustive exploit search. |
| Reward-to-success correlation | done | `runs/stage4_reward_vs_success/*_proxy_reward_vs_success.png`, `runs/stage8/analysis/reward_indistinguishable_summary.csv` | Training-time correlation remains observational. |
| Ablations identify useful components | partial | `runs/stage8b/analysis/ablation_10seed_summary.csv`, `ablation_10seed_tests.csv` | No effect survives 10-way Holm; terminal removal is suggestive only. |
| Validated reward-shaping recipe with limitations | partial | `docs/FINDINGS.md` | No explicit decision-tree recipe document yet. |

## Section 19 experiments

| Experiment | Status | Evidence | Gap |
| --- | --- | --- | --- |
| 1. Sparse vs dense | done | `runs/stage4_10seed_final_checkpoint/final_checkpoint_summary.csv` | Fixed base task. |
| 2. Sparse vs potential | done | `runs/stage4_10seed_final_checkpoint/final_checkpoint_summary.csv` | Fixed base task. |
| 3. Dense component ablation | done | `runs/stage8b/analysis/ablation_10seed_summary.csv` | No Holm-robust component effect. |
| 4. Weight sensitivity | done | `runs/stage5c_weight_sweep_final/final_checkpoint_summary.csv` | Focused on distance/progress weights. |
| 5. Horizon scaling | partial | `runs/stage8b/analysis/horizon_comparison_summary.csv` | Layout and horizon confounded. |
| 6. Noisy/misleading progress signals | done | `runs/stage8b/analysis/ablation_10seed_summary.csv` (`potential_noisy`) | Single noise scale. |
| 7. Reward-hacking stress test | done | `runs/stage5c_weight_sweep_audit/audit_results.json` | Scripted scenarios. |
| 8. Multiple task distributions | partial | `runs/stage6_zero_shot/summary.csv` | Three hand-designed unseen layouts, not a sampled distribution. |

## Section 18 metrics and Section 20 graphs

| Requirement | Status | Evidence file | Gap |
| --- | --- | --- | --- |
| Final success / seed variance | done | `runs/stage8b/analysis/horizon_comparison_summary.csv` | H10 only n=3. |
| Steps to target success | partial | `runs/stage8b/analysis/per_seed_evaluations.csv` | Only 90% threshold. |
| AUC / episodes to convergence | not done | -- | No consolidated artifact. |
| Episode length / planning efficiency | partial | `runs/stage8b/analysis/raw_evaluations.json`, `runs/stage6_zero_shot/per_seed.csv` | Planning-efficiency comparison not consolidated. |
| Invalid action / subgoal rate | partial | `runs/stage4_analysis/stage4_invalid_action_rate.png`, TensorBoard logs | No all-stage aggregate. |
| Reward-success correlation / hacking incidence | done | `runs/stage4_reward_vs_success/`, `runs/stage8/analysis/reward_indistinguishable_summary.csv` | Correlation is task-specific. |
| Success vs environment steps graph | done | `runs/stage7/final/learning_curve_h50.png` | Not one pooled all-stage graph. |
| Mean reward vs steps graph | not done | -- | Missing. |
| Sample-efficiency comparison graph | partial | `runs/stage8/analysis/steps_to_90_vs_horizon.png` | It is horizon-oriented, not one base comparison. |
| Threshold graph | partial | `runs/stage8/analysis/stage4_steps_to_90_checkpoint.csv` | 25/50/75 omitted. |
| Reward vs success scatter | done | `runs/stage4_reward_vs_success/dense_proxy_reward_vs_success.png` | Additional variants in same directory. |
| Horizon learning curves | done | `runs/stage7/final/learning_curve_h10.png` | n=3 at H10. |
| Component-contribution graph | partial | TensorBoard component scalars in `runs/*/tb_logs/` | No saved aggregate plot. |
| Invalid-action graph | done | `runs/stage4_analysis/stage4_invalid_action_rate.png` | Stage 4 only. |
| Reward-hacking examples | done | `runs/stage5c_weight_sweep_audit/*_proxy_vs_success.png`, `runs/stage5c_weight_sweep_final/*_hacking_replay.gif` | Scripted cases. |

## Section 23 reproducibility

| Requirement | Status | Evidence / missing item |
| --- | --- | --- |
| Pinned versions | done | `requirements.txt`, `pyproject.toml`. |
| Seeds recorded | done | Config `meta.seeds`, run directory names, and results JSON. |
| Environment/reward configs stored | done | `configs/` and per-run `config.yaml`. |
| Checkpoints saved | done | `runs/**/checkpoints/*.zip`. |
| Reward components logged separately | done | `reward_rl/training/callbacks.py` and TensorBoard logs. |
| Evaluations stored separately | done | `runs/*/eval_logs/`, `runs/stage8*/analysis/`. |
| At least three seeds | done | Stage 4/8 use 10; Stage 7 H10 has 3 and is labelled preliminary. |
| README with Section 25 commands | not done | No root `README.md` exists. The commands exist in `AGENTS.md` and package entrypoints, but the requested README artifact is missing. |
