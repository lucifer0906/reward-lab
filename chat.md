# Reward Model Design for Long-Horizon Agent Planning ## RL for Agentic AI — End-to-End Project Instructions **Objective:** Design and benchmark sparse, dense, and shaped reward functions for long-horizon agentic planning |task performance.|tasks, then identify a reward-shaping recipe that

PASTED

I’m sharing the project proposal for our RL for Agentic AI initiative: “Reward Model Design for Long-Horizon Agent Planning.”
The objective is to design and benchmark different reward strategies for long-horizon agentic planning, where sparse and delayed rewards make it difficult for reinforcement learning agents to learn effective planning behavior.
The project will focus on:

Sparse vs. dense reward design
Potential-based reward shaping
Custom Gymnasium environments
PyTorch and Stable-Baselines3
PPO-based policy training
Reward decomposition and analysis
Sample-efficiency benchmarking
Reward-hacking detection
Long-horizon planning evaluation
Ablation and robustness studies

The evaluation will focus on task success rate, sample efficiency, convergence speed, reward-to-success correlation, planning efficiency, robustness, and reward-hacking incidence.
I have attached the detailed project instruction document covering the environment design, reward formulations, training methodology, experiments, evaluation metrics, ablations, stress testing, and definition of done.
The intended outcome is a validated reward-shaping methodology for long-horizon agentic planning, supported by quantitative experiments demonstrating how different reward designs affect learning efficiency and final task performance.
I would appreciate your feedback on the proposed reward architecture, evaluation methodology, and experimental framework.
Best regards,
Manav Sutar
Founder CEO CTO 
SingleCore Labs

these are the instructions given by the company for the project, now i want you to go through them understand what exaclty they need, and according to those needs how should we proceed, like what project to select to make the RL agent with the tasks told by them, and then noting down the results as they need.

This doc is way more rigorous than the one-line PS — good, now there's no ambiguity. Let me break down what they actually want and give you a concrete plan.

What they're really asking for

This isn't "build an RL agent." It's "prove, with evidence, when dense reward shaping helps and when it lies to you." The whole document keeps circling back to one thing: reward hacking detection. Sections 5, 9, 16, 17, 20, 27 all hammer the same point — a reward that goes up but doesn't produce real task success is a failed design. That's the actual research contribution they want, not just "dense beats sparse, here's a chart."

So your deliverable isn't a comparison. It's a diagnostic framework: environment → 4 reward variants → PPO training → but every result gets checked against a separate, incorruptible ground-truth success evaluator that the reward function itself has zero influence over.

Step 1 — Pick the task (this decision matters most)

You need a task with real dependency structure — later steps only make sense if earlier ones happened, and early actions look useless in isolation. That's what makes sparse reward hard and dense reward exploitable.

My recommendation: a "Key-Door-Chain" gridworld.

Grid world, agent must: find Key A → open Door A → find Key B (only reachable after Door A) → open Door B → ... → reach Goal.
Configurable horizon: 3 keys = ~25 steps, 6 keys = ~100+ steps (satisfies their horizon-scaling requirement, section 15).
Has a natural reward hacking trap built in: if you give "+1 for picking up a key," an agent can drop and re-pick the same key infinitely (they explicitly mention this exact exploit in section 16).
Ground-truth evaluator is trivial and airtight: "did agent reach Goal with correct sequence of doors opened" — a pure boolean/step-counter independent of any reward code.

Alternatives to consider but I'd deprioritize: a text-based "task checklist" agent (harder to build clean MDP formalism in Gymnasium), or a resource-gathering/crafting env (good but more implementation overhead than you need for a hackathon-scale project). Key-Door-Chain gives you maximum research signal for minimum environment-engineering time — which matters because most of your budget should go to the experiments, not the env.

Step 2 — Build order (follow their section 28 literally, they'll notice if you don't)
Define task + ground-truth evaluator FIRST, before any reward code exists.
Gymnasium env: state = position, keys held, doors opened, step count. Action space fixed: {up, down, left, right, pick up, use key}.
R_sparse: +1 on Goal, 0 otherwise (or small negative step cost + terminal).
Train PPO on sparse — this is your baseline, record steps-to-threshold.
R_dense: +small reward per key pickup, per door opened, prerequisite completion — but log each component separately (section 11 — non-negotiable, this is how you'll later prove/disprove hacking).
R_potential: Φ(s) = some progress estimate (e.g., number of doors opened + normalized distance to next required key) — implement F(s)=γΦ(s')−Φ(s) exactly as specified.
R_hybrid: dense + potential + efficiency/invalid-action penalties combined.
Run all 4 under identical PPO hyperparams, ≥3 seeds each.
Step 3 — The experiments that actually prove something

Don't try to run all 8 of their listed experiments if you're time-constrained — rank by what demonstrates the core thesis:

Sparse vs dense vs potential vs hybrid — sample efficiency curves (steps to 25/50/75/90% success) — this is your headline result.
Reward-hacking stress test — deliberately let the dense agent exploit repeated key pickups; show reward climbing while true success stalls (section 16+17 combined). This is your most compelling graph — a founder evaluating this will care about this one graph more than all the others combined, because it's the one that shows you understood the actual risk, not just the technique.
Horizon scaling — same 4 rewards at 3-key vs 6-key chains — show if dense/potential's advantage shrinks or grows with horizon.
One ablation — remove one dense component (say, drop the invalid-action penalty) and show it degrades or gets exploited.

That's a tight, defensible 4-experiment core. If you have spare time, add reward-weight sensitivity (#4 in their list) as a bonus.

Step 4 — What "noting down results" should look like

Match their Required Evaluation Metrics (section 18) and Required Graphs (section 20) as literal checkboxes — build a results table with exactly those columns per reward variant:

Metric	R_sparse	R_dense	R_potential	R_hybrid
Final success rate				
Steps to 50%/90% success				
AUC of learning curve				
Reward-to-success correlation				
Reward-hacking incidence				
Invalid-action rate				

Then the reward-shaping recipe is your conclusion paragraph: "For long-horizon tasks with N+ dependent subgoals, use potential-based shaping with Φ = [x], because it matched dense reward's sample efficiency but showed zero reward-hacking incidence, while dense reward hacked in Y% of hacking-stress-test episodes."

Repo structure

Use their section 24 folder layout verbatim (configs/, envs/, rewards/, agents/, training/, evaluation/, analysis/, visualization/, tests/, runs/) and their CLI commands in section 25 as your actual train.py/evaluate.py/compare.py entry points. This is a strong signal to them that you followed spec precisely — worth the low effort to just copy their naming.

Want me to scaffold this repo (Gymnasium env + reward variants + SB3 training script skeleton) right now so you have a working baseline to iterate on?