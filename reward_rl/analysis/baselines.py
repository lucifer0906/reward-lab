"""Random and solver baselines for the fixed layouts."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from reward_rl.envs.key_door_env import KeyDoorChainEnv
from reward_rl.envs.solver import solve

ROOT=Path(__file__).resolve().parents[2]
LAYOUTS={'base':'configs/env/base_3keys.yaml','h10':'configs/env/stage7_h10_5keys.yaml','h50':'configs/env/stage7_h50_5keys.yaml','h100':'configs/env/stage7_h100_6keys.yaml'}
def main():
 rows=[]
 for name,path in LAYOUTS.items():
  env=KeyDoorChainEnv.from_config(path); successes=[]; lengths=[]
  for ep in range(1000):
   _,_=env.reset(seed=ep); done=False
   while not done:
    _,_,term,trunc,info=env.step(int(env.action_space.sample()));done=term or trunc
   successes.append(bool(info['evaluation']['success']));lengths.append(info['evaluation']['episode_length'])
  plan=solve(env.layout); rows.extend([{'layout':name,'baseline':'random','episodes':1000,'success_rate':float(np.mean(successes)),'mean_steps':float(np.mean(lengths))},{'layout':name,'baseline':'solver','episodes':1,'success_rate':1.0,'mean_steps':len(plan)}]);env.close()
 out=ROOT/'runs/baselines';out.mkdir(parents=True,exist_ok=True)
 with open(out/'baseline_summary.csv','w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(f'Saved baselines to {out}')
if __name__=='__main__':main()
