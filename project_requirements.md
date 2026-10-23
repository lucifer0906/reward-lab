# Reward Model Design for Long-Horizon Agent Planning

## RL for Agentic AI — End-to-End Project Instructions

**Objective:** Design and benchmark sparse, dense, and shaped reward functions for long-horizon agentic planning

|task performance.|tasks, then identify a reward-shaping recipe that improves sample efficiency without reward hacking or sacrificing final|
|---|---|
|Project Area|RL for Agentic AI|
|Project|Reward Model Design for Long-Horizon Agent Planning|
|Core Stack|Python, PyTorch, Stable-Baselines3, custom Gymnasium environments|
|Primary Outcome|Comparative reward study and validated reward-shaping recipe|

## 1. Problem Definition

•Long-horizon agents may need dozens or hundreds of decisions before reaching a final objective. A sparse terminal reward provides little information about which intermediate actions were useful. •Dense reward shaping can accelerate learning, but poorly designed shaping can create shortcuts, reward hacking, myopic behavior, or policies that optimize the proxy rather than the actual objective. •The central research problem is to determine which reward formulation provides useful learning signals while preserving alignment with final task success.

## 2. Research Question

•How do sparse, dense, and potential-based reward-shaping strategies affect sample efficiency, convergence, final task success, planning quality, robustness, and reward hacking in long-horizon agentic environments?

## 3. Formalize the RL Environment

•Model the environment as a Markov Decision Process M = (S, A, P, R, γ). •Define state, action, transition dynamics, terminal conditions, and the true task objective independently from the reward implementation. •The same environment and task distribution must be used across reward variants so that the reward formulation is the main experimental variable.

## 4. Choose a Long-Horizon Planning Task

•Start with a controlled planning environment containing multiple dependent subgoals. •Example task structure: discover information → collect resources → satisfy prerequisites → execute actions → verify completion. •Use configurable horizon lengths such as 10, 25, 50, and 100+ decisions. •Include tasks where early actions may have no immediate visible benefit but are necessary for later success.

## 5. Define the True Objective First

•Create a ground-truth task-success evaluator before implementing dense rewards. •The evaluator should determine whether the final objective was completed correctly, independently of how many intermediate rewards were received. •This evaluator becomes the reference metric for detecting reward hacking.

## 6. Define the Agent State and Action Space

•State should include task progress, available actions, completed subgoals, relevant observations, resource status, previous actions, and termination information. •Begin with structured states. Add richer observations only after the reward benchmark is stable. •Keep the action space fixed across reward experiments whenever possible.

## 7. Implement the Sparse Reward Baseline

•Use a terminal reward for successful completion and a corresponding failure/timeout outcome. •Avoid intermediate rewards in the first baseline. •Record how many episodes, environment steps, and samples are required to reach predefined success thresholds.

## 8. Design Dense Reward Variants

•Create interpretable intermediate rewards for meaningful progress: subgoal completion, valid state transitions, information acquisition, resource efficiency, and successful prerequisite completion. •Avoid rewarding superficial activity such as number of actions, text length, or repeated exploration unless those behaviors are genuinely part of the objective. •Every dense component must have a clear relationship to the final task objective.

## 9. Potential-Based Reward Shaping

•Implement potential-based shaping as an important control condition. •Use F(s) = γΦ(s') − Φ(s), where Φ represents estimated progress toward the goal. •Compare this formulation with manually designed dense rewards and the sparse baseline. •Document how the potential function is constructed and ensure it does not directly leak the final answer.

## 10. Reward Function Candidates

•R_sparse: terminal success/failure only. •R_dense: weighted progress signals at intermediate steps. •R_potential: sparse reward plus potential-based shaping. •R_hybrid: progress + validity + efficiency + terminal success. •R_adaptive: reward weights that vary by planning phase, if justified by experiments.

## 11. Reward Decomposition

•Log every reward component separately rather than only storing the final scalar reward.

•Example components: subgoal_progress, prerequisite_completion, invalid_action_penalty, resource_efficiency, step_cost, terminal_success. •This makes it possible to diagnose whether a policy is learning the intended behavior or exploiting one component.

## 12. Implement Baseline Agents

•Random policy. •Simple heuristic planner. •Sparse-reward RL baseline. •Dense-reward RL baseline. •Potential-based shaping baseline. •Hybrid reward baseline. •Use the same Stable-Baselines3 algorithm, network architecture, and training budget where practical.

## 13. Stable-Baselines3 Training

•Start with PPO because it provides a strong general-purpose on-policy baseline for custom environments. •Use deterministic evaluation separately from stochastic training. •Track policy loss, value loss, entropy, episode reward, episode length, success rate, and environment steps. •Keep hyperparameters fixed across reward experiments before performing reward-specific tuning.

## 14. Sample Efficiency Measurement

•Define sample efficiency as task performance achieved per environment interaction. •Measure environment steps required to reach 25%, 50%, 75%, and 90% success thresholds where feasible. •Report area under the learning curve as an aggregate sample-efficiency metric. •Use multiple random seeds and report mean plus variability.

## 15. Long-Horizon Scaling

•Evaluate reward formulations at increasing horizon lengths. •Measure whether dense shaping continues to help as the number of decisions increases. •Identify cases where dense rewards improve short-horizon learning but fail on long-horizon planning.

## 16. Reward Hacking Tests

•Construct adversarial or misleading intermediate states that allow a poorly designed reward to be increased without completing the true task. •Examples include repeatedly completing an already-completed subgoal, collecting irrelevant resources, taking unnecessary actions, or exploiting an incomplete progress metric. •Compare proxy reward against true task success to identify divergence.

## 17. Reward-Performance Correlation

•Measure the correlation between cumulative training reward and actual task success. •A reward function should not be considered successful merely because its scalar reward increases. •Investigate cases with high reward but low task success and low reward but successful task completion.

## 18. Required Evaluation Metrics

•Final task success rate. •Environment steps to target success. •Area under the learning curve. •Episodes to convergence. •Average episode length. •Planning efficiency. •Invalid-action rate. •Subgoal completion rate. •Reward-to-success correlation. •Reward-hacking incidence. •Performance variance across random seeds.

## 19. Required Experiments

•Experiment 1: sparse vs. dense reward. •Experiment 2: sparse vs. potential-based shaping. •Experiment 3: dense reward component ablation. •Experiment 4: reward weight sensitivity. •Experiment 5: horizon scaling. •Experiment 6: noisy or misleading progress signals. •Experiment 7: reward hacking stress test. •Experiment 8: same reward design across multiple task distributions.

## 20. Required Graphs

•Task success rate vs. environment steps. •Mean episode reward vs. environment steps. •Sample efficiency comparison across reward functions. •Steps to reach each success threshold. •Reward vs. true task success scatter plot. •Learning curves across horizon lengths. •Reward-component contribution over training. •Invalid-action rate over training. •Reward-hacking examples: proxy reward vs. true success.

## 21. Ablation Studies

•Remove each dense reward component individually. •Remove terminal success reward. •Change potential-function formulation. •Change reward weights. •Remove efficiency penalties. •Remove invalid-action penalties. •Compare normalized vs. unnormalized reward components. •Compare short and long horizons.

## 22. Robustness and Generalization

•Test unseen task layouts. •Vary task horizon. •Vary action availability. •Add stochastic transitions. •Add irrelevant observations. •Change resource constraints. •Test reward formulations under distribution shift.

## 23. Reproducibility

•Pin Python and package versions. •Record all random seeds. •Store environment and reward configurations. •Save model checkpoints. •Log reward components independently. •Store evaluation results separately from training results. •Run at least three seeds for preliminary experiments and increase the number for final claims.

## 24. Recommended Repository Structure

•configs/ — reward and training configurations •envs/ — custom Gymnasium environments •rewards/ — sparse, dense, potential, and hybrid reward implementations •agents/ — policy wrappers and evaluation agents •training/ — Stable-Baselines3 training scripts •evaluation/ — metrics, success evaluators, and benchmark runners •analysis/ — reward correlation and reward-hacking analysis •visualization/ — learning curves and diagnostics

•tests/ — unit and integration tests •runs/ — checkpoints and experiment outputs

## 25. Suggested CLI

•python -m reward_rl.train --config configs/sparse.yaml •python -m reward_rl.train --config configs/dense.yaml •python -m reward_rl.train --config configs/potential.yaml •python -m reward_rl.evaluate --checkpoint runs/best_model •python -m reward_rl.compare --experiments sparse dense potential hybrid •python -m reward_rl.reward_audit --experiment runs/experiment_name

## 26. Final Architecture

**Task Generator** → **Custom Gymnasium Environment** → **State** → **Policy** → **Action** → **Transition** → **Reward** **Decomposition** → **Scalar Reward** → **PPO Update** → **Evaluation** → **Reward Audit**

## 27. Definition of Done

•A reproducible long-horizon Gymnasium environment is implemented. •A ground-truth task-success evaluator exists independently of reward design. •Sparse, dense, potential-based, and hybrid rewards are implemented. •Reward components are logged independently. •Stable-Baselines3 PPO trains all primary reward variants. •Multiple random seeds are evaluated. •Sample efficiency is measured quantitatively. •Long-horizon scaling experiments are completed. •Reward-hacking tests are performed. •Reward-to-success correlation is analyzed. •Ablation studies identify useful reward components. •A validated reward-shaping recipe is documented with its limitations.

**28. Recommended Development Order** 1Define one long-horizon task and its true success evaluator. 2Build and validate the custom Gymnasium environment. 3Implement the sparse reward baseline. 4Train and measure the baseline policy. 5Add dense reward components one at a time. 6Implement potential-based shaping. 7Run controlled sample-efficiency experiments. 8Run ablations and reward-hacking tests. 9Scale the horizon and test generalization. 10Select and document the reward-shaping recipe based on measured evidence. **Critical implementation rule:** Never optimize the reward before defining the true task objective. The final task-success evaluator is the ground truth; intermediate reward is only a learning signal. A reward function that produces higher reward but worse task success is a failed reward design.