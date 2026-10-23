# Reward Model Design for Long-Horizon Agent Planning
## Complete Project Spec & Build Plan

**Context for the coding assistant reading this:** This is a sponsored industry project from SingleCore Labs (a startup building RL infrastructure for enterprise agentic AI — their core product is literally an "RL Lab" for environments/rewards/evaluation). The founder (Manav Sutar) has given multiple student teams the same detailed problem statement. This doc has two parts: (1) the founder's original requirements, verbatim, and (2) our specific implementation decisions and build plan on top of it. Follow both — Part 1 is what we're graded on, Part 2 is how we've chosen to execute it. Where Part 2 makes a choice, that choice is final — don't re-litigate it, just build it.

---

# PART 1: FOUNDER'S ORIGINAL REQUIREMENTS (VERBATIM)

## Objective
Design and benchmark sparse, dense, and shaped reward functions for long-horizon agentic planning tasks, then identify a reward-shaping recipe that improves sample efficiency without reward hacking or sacrificing final task performance.

**Core Stack:** Python, PyTorch, Stable-Baselines3, custom Gymnasium environments
**Primary Outcome:** Comparative reward study and validated reward-shaping recipe

## 1. Problem Definition
- Long-horizon agents may need dozens or hundreds of decisions before reaching a final objective. A sparse terminal reward provides little information about which intermediate actions were useful.
- Dense reward shaping can accelerate learning, but poorly designed shaping can create shortcuts, reward hacking, myopic behavior, or policies that optimize the proxy rather than the actual objective.
- The central research problem is to determine which reward formulation provides useful learning signals while preserving alignment with final task success.

## 2. Research Question
How do sparse, dense, and potential-based reward-shaping strategies affect sample efficiency, convergence, final task success, planning quality, robustness, and reward hacking in long-horizon agentic environments?

## 3. Formalize the RL Environment
- Model the environment as a Markov Decision Process M = (S, A, P, R, γ).
- Define state, action, transition dynamics, terminal conditions, and the true task objective independently from the reward implementation.
- The same environment and task distribution must be used across reward variants so that the reward formulation is the main experimental variable.

## 4. Choose a Long-Horizon Planning Task
- Start with a controlled planning environment containing multiple dependent subgoals.
- Example task structure: discover information → collect resources → satisfy prerequisites → execute actions → verify completion.
- Use configurable horizon lengths such as 10, 25, 50, and 100+ decisions.
- Include tasks where early actions may have no immediate visible benefit but are necessary for later success.

## 5. Define the True Objective First
- Create a ground-truth task-success evaluator before implementing dense rewards.
- The evaluator should determine whether the final objective was completed correctly, independently of how many intermediate rewards were received.
- This evaluator becomes the reference metric for detecting reward hacking.

## 6. Define the Agent State and Action Space
- State should include task progress, available actions, completed subgoals, relevant observations, resource status, previous actions, and termination information.
- Begin with structured states. Add richer observations only after the reward benchmark is stable.
- Keep the action space fixed across reward experiments whenever possible.

## 7. Implement the Sparse Reward Baseline
- Use a terminal reward for successful completion and a corresponding failure/timeout outcome.
- Avoid intermediate rewards in the first baseline.
- Record how many episodes, environment steps, and samples are required to reach predefined success thresholds.

## 8. Design Dense Reward Variants
- Create interpretable intermediate rewards for meaningful progress: subgoal completion, valid state transitions, information acquisition, resource efficiency, and successful prerequisite completion.
- Avoid rewarding superficial activity such as number of actions, text length, or repeated exploration unless those behaviors are genuinely part of the objective.
- Every dense component must have a clear relationship to the final task objective.

## 9. Potential-Based Reward Shaping
- Implement potential-based shaping as an important control condition.
- Use F(s) = γΦ(s') − Φ(s), where Φ represents estimated progress toward the goal.
- Compare this formulation with manually designed dense rewards and the sparse baseline.
- Document how the potential function is constructed and ensure it does not directly leak the final answer.

## 10. Reward Function Candidates
- R_sparse: terminal success/failure only.
- R_dense: weighted progress signals at intermediate steps.
- R_potential: sparse reward plus potential-based shaping.
- R_hybrid: progress + validity + efficiency + terminal success.
- R_adaptive: reward weights that vary by planning phase, if justified by experiments.

## 11. Reward Decomposition
- Log every reward component separately rather than only storing the final scalar reward.
- Example components: subgoal_progress, prerequisite_completion, invalid_action_penalty, resource_efficiency, step_cost, terminal_success.
- This makes it possible to diagnose whether a policy is learning the intended behavior or exploiting one component.

## 12. Implement Baseline Agents
- Random policy.
- Simple heuristic planner.
- Sparse-reward RL baseline.
- Dense-reward RL baseline.
- Potential-based shaping baseline.
- Hybrid reward baseline.
- Use the same Stable-Baselines3 algorithm, network architecture, and training budget where practical.

## 13. Stable-Baselines3 Training
- Start with PPO because it provides a strong general-purpose on-policy baseline for custom environments.
- Use deterministic evaluation separately from stochastic training.
- Track policy loss, value loss, entropy, episode reward, episode length, success rate, and environment steps.
- Keep hyperparameters fixed across reward experiments before performing reward-specific tuning.

## 14. Sample Efficiency Measurement
- Define sample efficiency as task performance achieved per environment interaction.
- Measure environment steps required to reach 25%, 50%, 75%, and 90% success thresholds where feasible.
- Report area under the learning curve as an aggregate sample-efficiency metric.
- Use multiple random seeds and report mean plus variability.

## 15. Long-Horizon Scaling
- Evaluate reward formulations at increasing horizon lengths.
- Measure whether dense shaping continues to help as the number of decisions increases.
- Identify cases where dense rewards improve short-horizon learning but fail on long-horizon planning.

## 16. Reward Hacking Tests
- Construct adversarial or misleading intermediate states that allow a poorly designed reward to be increased without completing the true task.
- Examples include repeatedly completing an already-completed subgoal, collecting irrelevant resources, taking unnecessary actions, or exploiting an incomplete progress metric.
- Compare proxy reward against true task success to identify divergence.

## 17. Reward-Performance Correlation
- Measure the correlation between cumulative training reward and actual task success.
- A reward function should not be considered successful merely because its scalar reward increases.
- Investigate cases with high reward but low task success and low reward but successful task completion.

## 18. Required Evaluation Metrics
- Final task success rate.
- Environment steps to target success.
- Area under the learning curve.
- Episodes to convergence.
- Average episode length.
- Planning efficiency.
- Invalid-action rate.
- Subgoal completion rate.
- Reward-to-success correlation.
- Reward-hacking incidence.
- Performance variance across random seeds.

## 19. Required Experiments
- Experiment 1: sparse vs. dense reward.
- Experiment 2: sparse vs. potential-based shaping.
- Experiment 3: dense reward component ablation.
- Experiment 4: reward weight sensitivity.
- Experiment 5: horizon scaling.
- Experiment 6: noisy or misleading progress signals.
- Experiment 7: reward hacking stress test.
- Experiment 8: same reward design across multiple task distributions.

## 20. Required Graphs
- Task success rate vs. environment steps.
- Mean episode reward vs. environment steps.
- Sample efficiency comparison across reward functions.
- Steps to reach each success threshold.
- Reward vs. true task success scatter plot.
- Learning curves across horizon lengths.
- Reward-component contribution over training.
- Invalid-action rate over training.
- Reward-hacking examples: proxy reward vs. true success.

## 21. Ablation Studies
- Remove each dense reward component individually.
- Remove terminal success reward.
- Change potential-function formulation.
- Change reward weights.
- Remove efficiency penalties.
- Remove invalid-action penalties.
- Compare normalized vs. unnormalized reward components.
- Compare short and long horizons.

## 22. Robustness and Generalization
- Test unseen task layouts.
- Vary task horizon.
- Vary action availability.
- Add stochastic transitions.
- Add irrelevant observations.
- Change resource constraints.
- Test reward formulations under distribution shift.

## 23. Reproducibility
- Pin Python and package versions.
- Record all random seeds.
- Store environment and reward configurations.
- Save model checkpoints.
- Log reward components independently.
- Store evaluation results separately from training results.
- Run at least three seeds for preliminary experiments and increase the number for final claims.

## 24. Recommended Repository Structure
- configs/ — reward and training configurations
- envs/ — custom Gymnasium environments
- rewards/ — sparse, dense, potential, and hybrid reward implementations
- agents/ — policy wrappers and evaluation agents
- training/ — Stable-Baselines3 training scripts
- evaluation/ — metrics, success evaluators, and benchmark runners
- analysis/ — reward correlation and reward-hacking analysis
- visualization/ — learning curves and diagnostics
- tests/ — unit and integration tests
- runs/ — checkpoints and experiment outputs

## 25. Suggested CLI
- python -m reward_rl.train --config configs/sparse.yaml
- python -m reward_rl.train --config configs/dense.yaml
- python -m reward_rl.train --config configs/potential.yaml
- python -m reward_rl.evaluate --checkpoint runs/best_model
- python -m reward_rl.compare --experiments sparse dense potential hybrid
- python -m reward_rl.reward_audit --experiment runs/experiment_name

## 26. Final Architecture
Task Generator → Custom Gymnasium Environment → State → Policy → Action → Transition → Reward Decomposition → Scalar Reward → PPO Update → Evaluation → Reward Audit

## 27. Definition of Done
- A reproducible long-horizon Gymnasium environment is implemented.
- A ground-truth task-success evaluator exists independently of reward design.
- Sparse, dense, potential-based, and hybrid rewards are implemented.
- Reward components are logged independently.
- Stable-Baselines3 PPO trains all primary reward variants.
- Multiple random seeds are evaluated.
- Sample efficiency is measured quantitatively.
- Long-horizon scaling experiments are completed.
- Reward-hacking tests are performed.
- Reward-to-success correlation is analyzed.
- Ablation studies identify useful reward components.
- A validated reward-shaping recipe is documented with its limitations.

## 28. Recommended Development Order (Founder's Version)
1. Define one long-horizon task and its true success evaluator.
2. Build and validate the custom Gymnasium environment.
3. Implement the sparse reward baseline.
4. Train and measure the baseline policy.
5. Add dense reward components one at a time.
6. Implement potential-based shaping.
7. Run controlled sample-efficiency experiments.
8. Run ablations and reward-hacking tests.
9. Scale the horizon and test generalization.
10. Select and document the reward-shaping recipe based on measured evidence.

**Critical implementation rule (founder's exact words):** Never optimize the reward before defining the true task objective. The final task-success evaluator is the ground truth; intermediate reward is only a learning signal. A reward function that produces higher reward but worse task success is a failed reward design.

---

# PART 2: OUR IMPLEMENTATION DECISIONS

## Team context
- Solo core builder (me), using AI-assisted coding (Antigravity/Opus) to accelerate implementation. Rest of team handles presentation/other work.
- First RL project — no prior RL experience, learning concepts just-in-time as needed.
- Compute: free Colab / CPU only. No GPU access assumed.
- Timeline: 2-3 weeks available, but want to move in STAGES not fixed week-blocks, so faster progress can compress the timeline.
- This is a sponsored project (SingleCore Labs, founder Manav Sutar) — multiple teams received the identical spec above. We need to be differentiated AND we need a clean final presentation/report to deliver back to the company explaining what we built and what we found.

## Why we're differentiating on depth, not on environment complexity
The founder's own doc (Part 1, Section 3) explicitly states the environment must be held CONSTANT across reward variants — it is the experimental control, not the variable under study. Building a fancier/more complex task environment than other teams would be optimizing something the founder explicitly said doesn't matter, and is also the most obvious idea every other team will independently have. Instead, we differentiate on:
1. **Depth of reward-hacking analysis** (Part 1 Section 16-17) — most teams will do this shallowly or skip it; we go deep with multiple distinct exploit types. This is literally what SingleCore Labs' RL Lab product exists to solve, so it's the highest-signal thing we can show the founder.
2. **Generalization testing** (Part 1 Section 22) — zero-shot evaluation on unseen layouts using already-trained agents, cheap to add once the core pipeline works.
3. **Rigor over flash** — proper multi-seed statistics, a real decision-rule "recipe" (not just "dense is better"), honest ablations.

## Locked environment design

**Task type:** 2D grid world — agent moves on a grid, must find keys and open doors in a required sequence to reach a goal.

**Why a grid (not an abstract/symbolic state):**
- Gives invalid actions for free and meaningfully (walking into a wall, trying a locked door) — the doc requires tracking "invalid-action rate" (Section 18) and using invalid-action penalties as a reward component (Section 8) and ablation target (Section 21). An abstract state has no natural notion of an invalid action.
- Visually demoable — judges and the founder can immediately understand an agent navigating a grid with keys/doors; an abstract state vector is not intuitive to present.
- Makes reward-hacking exploits concrete and convincing (e.g., "agent walks back and forth between two cells to repeatedly re-trigger a reward" is vivid and easy to show in a replay/GIF).
- Minimal extra implementation cost with AI-assisted coding — the difference between grid state and abstract state is a small amount of code, not a real time cost.

**Base difficulty:** 3 sequential key→door pairs (~20-30 step episodes).
- Hard enough that sparse reward genuinely struggles (if too easy, sparse succeeds fine too and the comparison has no story).
- Small enough to be solo-debuggable — you must be able to mentally trace what the agent is doing when something breaks. This is the #1 risk for a solo first-time RL builder: an overly complex environment makes it impossible to tell whether poor results are "bad reward design" (the actual research question) or "broken environment code" (a bug).
- Will be scaled up later (to 5-6 keys) specifically and only for the horizon-scaling experiment (Section 15) — this is a controlled, deliberate increase, not the starting difficulty.

**Layout randomness:** FIXED layout for all core experiments (Experiments 1-7). Randomized/unseen layouts ONLY used for the final generalization test (Section 22), using already-trained agents in evaluation mode (no additional training).
- Reasoning: this is the most important technical decision in the whole build. If the layout is randomized during core training, you cannot isolate whether training outcomes are due to the reward formulation (what we're studying) or due to random layout difficulty (an uncontrolled second variable). This directly follows the founder's own stated requirement in Section 3 that environment and task distribution must be held constant so reward formulation is the main experimental variable.

**Required state/action design (per Section 6):**
- State must include: task progress, available actions, completed subgoals, relevant observations (agent position, key positions, door states), resource status (keys held), previous action, termination info.
- Keep state structured (e.g., Dict or MultiDiscrete space — positions, keys-held as binary flags, doors-open as binary flags) — NOT pixel/image observations. Image-based state is unnecessary complexity and incompatible with CPU-only/free Colab constraints.
- Action space fixed across all reward experiments: movement actions (up/down/left/right) + interact action (pick up key / open door).

## Locked reward variants (per Section 10)

Implement all four of:
1. **R_sparse** — terminal reward only (success/fail/timeout), zero intermediate signal.
2. **R_dense** — hand-designed intermediate rewards: small reward for each key pickup, each door opened, each valid prerequisite completion. MUST log each component separately (Section 11) — this is required, not optional, because it's how reward-hacking diagnosis works later.
3. **R_potential** — sparse terminal reward + potential-based shaping using F(s) = γΦ(s') − Φ(s), where Φ(s) is a progress estimate (e.g., number of subgoals completed, or distance-to-next-required-key). Must NOT leak the final answer directly (per Section 9's explicit warning).
4. **R_hybrid** — dense progress signals + validity penalties + efficiency penalties + terminal success, combined.

R_adaptive (Section 10's fifth, optional candidate) is deferred — only build if time remains after the core four and both differentiators are done.

## Locked training setup

- Algorithm: PPO via Stable-Baselines3 (per Section 13's explicit recommendation).
- Same hyperparameters, same network architecture, same training budget across all 4 reward variants — only the reward function changes between runs. This isolation is the entire point of the experiment.
- 5 random seeds per reward variant (increased from the doc's stated minimum of 3, since this run on CPU is lightweight and AI-assisted coding frees up time for the extra runs — gives more statistically trustworthy results).
- Track per Section 13: policy loss, value loss, entropy, episode reward, episode length, success rate, environment steps.
- Deterministic evaluation kept separate from stochastic training evaluation, as required.

## Baseline agents (per Section 12)

Build: random policy, sparse-RL, dense-RL, potential-RL, hybrid-RL.
DEFER: heuristic/rule-based planner — build only if time remains at the end. Not core to the research question (which is about reward formulation, not about whether RL beats hand-coded planning), so it's correctly lower priority than the reward-hacking and generalization differentiators.

## Differentiator build order (sequenced, not parallel — solo builder)

Build these ONLY after the core 4-reward-variant comparison is fully working (environment validated, all 4 variants trained, 5 seeds each, core graphs produced). Do not parallelize — as a solo builder, partially-done differentiators are worse than one fully-done differentiator.

1. **Reward-hacking stress tests first** (addresses Section 16-17, 19's Experiment 7, and is our #1 priority differentiator). Design at least 2-3 DISTINCT exploit scenarios against the dense and hybrid reward variants — e.g.:
   - Repeatedly "picking up" an already-held key if the environment allows re-triggering that reward component.
   - Oscillating between two cells near a reward-bearing transition to farm step-wise reward without net progress.
   - Exploiting an incomplete progress metric (e.g., moving toward a key without ever actually acquiring it, if partial-distance reward isn't capped correctly).
   For each exploit, produce the required scatter plot (Section 20): proxy/training reward vs. true task success, showing divergence when hacking occurs.

2. **Generalization test second** (addresses Section 22). Using agents ALREADY TRAINED on the fixed base layout, evaluate (no further training) on 2-3 new hand-designed layouts not seen during training. This reuses existing trained models and environment code — cheap to add once the core pipeline works. Report success rate drop (if any) per reward variant — this is itself a research finding (e.g., "potential-based shaping generalizes better than hand-dense because it doesn't overfit to the specific key coordinates").

## Deferred / optional (only if time remains after both differentiators are complete)
- R_adaptive reward variant (Section 10).
- Heuristic planner baseline (Section 12).
- Full ablation sweep across ALL components (Section 21) — do at minimum one meaningful ablation (e.g., remove invalid-action penalty from hybrid, show effect) but the full sweep across every component is lower priority than the two main differentiators.
- Experiment 8 (same reward design across multiple task distributions, Section 19) — the generalization test above partially covers this; a full second task-distribution is a stretch goal only.

## Reproducibility requirements (per Section 23) — non-negotiable, build in from the start
- Pin Python/package versions (requirements.txt or similar) from day one.
- Record all random seeds used.
- Store environment and reward configs as YAML files (per Section 24/25's suggested structure) — do not hardcode reward weights in Python.
- Save model checkpoints for every trained agent.
- Log reward components independently (ties back to Section 11 requirement).
- Store evaluation results in files separate from training logs.

## Repository structure — use founder's exact structure (Section 24)
configs/, envs/, rewards/, agents/, training/, evaluation/, analysis/, visualization/, tests/, runs/ — use these exact folder names. This is a low-effort, high-signal way to show the founder we followed his spec precisely.

## CLI — use founder's exact suggested commands (Section 25) as our actual entry points
python -m reward_rl.train --config configs/sparse.yaml, etc. — implement these as literal working commands, not just directory names.

---

# BUILD STAGES (not fixed weeks — move to the next stage as soon as the current one's exit criteria are met)

## Stage 0: Environment setup
**Exit criteria:** Python environment set up, Gymnasium + Stable-Baselines3 + PyTorch installed and verified working on Colab (CPU), repo skeleton created with the folder structure from Section 24, requirements.txt pinned.

## Stage 1: Core environment + ground-truth evaluator
**Exit criteria:**
- Custom Gymnasium env class built: 2D grid, 3 sequential key→door pairs, fixed layout.
- `reset()`, `step()`, observation space, action space all implemented and manually tested (e.g., random actions run without crashing, episode terminates correctly on success/failure/timeout).
- A ground-truth task-success evaluator exists as SEPARATE code from any reward logic — it only checks "did the agent reach the goal with correct key/door sequence," nothing else. This must be built and validated BEFORE any reward code is written (per the founder's "critical implementation rule").
- Confirm: you can manually walk the agent through a correct solution (scripted actions) and the evaluator correctly reports success; you can walk it through an incorrect/incomplete attempt and it correctly reports failure.

## Stage 2: Sparse reward baseline + first training run
**Exit criteria:**
- R_sparse implemented (terminal reward only).
- PPO (via SB3) successfully trains on R_sparse for at least one seed — confirm training runs end-to-end without crashing, loss values are logging, and a saved checkpoint can be reloaded and evaluated.
- Don't worry yet about whether it actually learns well — this stage is about pipeline correctness, not results quality.

## Stage 3: Dense, potential-based, and hybrid reward variants
**Exit criteria:**
- R_dense implemented with per-component logging (subgoal_progress, prerequisite_completion, invalid_action_penalty, etc. logged separately, per Section 11).
- R_potential implemented using F(s) = γΦ(s') − Φ(s), with Φ(s) documented (what progress function you chose and why it doesn't leak the answer).
- R_hybrid implemented combining dense + potential + efficiency/validity penalties.
- All three train successfully end-to-end (pipeline correctness check, same as Stage 2) — at least 1 seed each before committing to the full 5-seed runs.

## Stage 4: Full core experiment matrix (5 seeds × 4 variants)
**Exit criteria:**
- All 4 reward variants (sparse, dense, potential, hybrid) trained with 5 seeds each = 20 total training runs.
- Required metrics (Section 18) computed for all variants: success rate, steps to success thresholds (25/50/75/90%), AUC of learning curve, episode length, invalid-action rate, subgoal completion rate.
- Required core graphs produced (Section 20): success rate vs. steps, mean episode reward vs. steps, sample-efficiency comparison, steps-to-threshold comparison — all with multi-seed mean ± variance shown, not single noisy runs.
- This is your main comparative result — Experiments 1 and 2 (sparse vs dense, sparse vs potential) from Section 19 are satisfied here.

## Stage 5: Reward-hacking stress tests (DIFFERENTIATOR #1)
**Exit criteria:**
- At least 2-3 distinct adversarial exploit scenarios constructed and run against dense and hybrid variants (see Part 2's "Differentiator build order" above for example exploit types).
- For each exploit: proxy/training reward vs. true task success scatter plot produced (Section 20's explicit requirement), clearly showing divergence where it occurs.
- Reward-to-success correlation computed quantitatively (Section 17) — not just visual inspection.
- This satisfies Section 19's Experiment 7 (reward hacking stress test) with genuine depth, not a token example.

## Stage 6: Generalization test (DIFFERENTIATOR #2)
**Exit criteria:**
- 2-3 new hand-designed grid layouts built (unseen during training).
- All previously-trained agents (from Stage 4, no retraining) evaluated zero-shot on these new layouts.
- Success rate comparison: base layout vs. unseen layouts, per reward variant — report which reward formulation generalizes best/worst and why (this becomes part of your final "recipe").
- Satisfies Section 22 (robustness/generalization) and partially addresses Section 19's Experiment 8.

## Stage 7: Horizon scaling (if time allows, otherwise minimum viable version)
**Exit criteria:**
- Scale task to 5-6 sequential key-door pairs (longer horizon).
- Retrain all 4 reward variants at this scale (can reduce to 3 seeds here if time-constrained, since this is a secondary experiment, not the core comparison).
- Compare whether dense/potential-based advantages shrink or grow with horizon length — directly satisfies Section 15 and Section 19's Experiment 5.

## Stage 8: Ablations (minimum viable, expand if time allows)
**Exit criteria (minimum):**
- At least one meaningful ablation completed — recommended: remove invalid-action penalty from the hybrid reward, show the effect on both training reward and true success rate.
**Exit criteria (stretch, if time remains):**
- Additional ablations per Section 21 — remove terminal reward, vary potential-function formulation, reward weight sensitivity (Section 19's Experiment 4).

## Stage 9: Analysis, recipe, and report writing
**Exit criteria:**
- All results consolidated into the metrics table and graphs required by Sections 18/20.
- Written "reward-shaping recipe" — NOT just "dense is better," but a genuine decision rule, e.g.: "For long-horizon tasks (>X subgoals) with unknown subgoal ordering, use potential-based shaping because it matched dense reward's sample efficiency while showing zero reward-hacking incidence in our stress tests; use hand-designed dense reward only when subgoal completion is guaranteed to be non-repeatable, since our tests show it is vulnerable to [specific exploit] otherwise."
- Recipe explicitly documents its own limitations (per Section 27's definition-of-done requirement) — be honest about what wasn't tested (e.g., stochastic transitions, if you didn't get to that).
- Repository cleaned up to match Section 24's folder structure, with configs/ containing the actual YAML files used for every experiment (reproducibility).

## Stage 10: Presentation prep for SingleCore Labs
**Exit criteria:**
- Since this is a sponsored project being presented back to the founder, prepare a presentation that mirrors HIS document's structure — this makes it trivially easy for him to see you addressed every section he asked for. Suggested structure:
  1. Research question (his Section 2, restated)
  2. Environment design + why (grid, 3-key chain, fixed layout — brief justification)
  3. Core comparison results (Stage 4 graphs) — the headline sample-efficiency finding
  4. Reward-hacking deep dive (Stage 5) — this is your strongest differentiator, give it real presentation time, use the scatter plots, make the exploit concrete and demoable if possible (e.g., a short GIF/replay of the agent farming reward)
  5. Generalization results (Stage 6) — second differentiator
  6. Horizon scaling + ablations (Stages 7-8) — brief, supporting evidence
  7. Final recipe + limitations (Stage 9) — the actual deliverable he asked for in "Primary Outcome"
  8. Repo walkthrough — show the clean structure matching his Section 24, and the working CLI commands matching his Section 25
- Rehearse explaining the reward-hacking exploits in plain language — this is the part most likely to impress a founder whose company's product is specifically built around this exact problem, so it needs to be the most polished, clearly-explained part of the presentation, not a rushed afterthought.
