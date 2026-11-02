"""Read-only Stage 8 checkpoint evaluation batch and statistical report."""
from __future__ import annotations
import csv, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, pointbiserialr
from reward_rl.analysis.stage6_zero_shot import evaluate_model, bootstrap_ci
from reward_rl.envs.key_door_env import KeyDoorChainEnv
from reward_rl.envs.solver import solve
from reward_rl.rewards import make_reward
from reward_rl.training.train import load_config

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'runs/stage8/analysis'
CORE=('sparse','dense','potential','hybrid')
LAY={'base':'configs/env/base_3keys.yaml','h50':'configs/env/stage7_h50_5keys.yaml','h100':'configs/env/stage7_h100_6keys.yaml'}

def control(config,layout):
 cfg=load_config(config); raw=KeyDoorChainEnv.from_config(layout,reward_fn=make_reward(cfg['reward'],max_steps=KeyDoorChainEnv.from_config(layout).max_steps)); raw.reset(seed=0); total=0.0
 for a in solve(raw.layout):
  _,r,t,x,_=raw.step(a);total+=r
  if t or x:break
 raw.close();return total
def one(task):
 os.environ['OMP_NUM_THREADS']='1'; run=Path(task['run']); cfg=str(run/'config.yaml') if (run/'config.yaml').exists() else task['config']; layout=task['layout']; seed=task['seed']
 answer={'id':task['id'],'kind':task['kind'],'variant':task['variant'],'horizon':task['horizon'],'seed':seed,'run':str(run)}
 if task.get('final'):
  det,_=evaluate_model(run/'checkpoints/final_model.zip',cfg,layout,0,200,seed,True); stoch,_=evaluate_model(run/'checkpoints/final_model.zip',cfg,layout,0,100,seed,False)
  values=[r['proxy_return'] for r in det];success=[float(r['success']) for r in det]; c=control(cfg,layout); hacked=[(r['proxy_return']>c and not r['success']) for r in det]; answer.update(final_success=float(np.mean(success)),stochastic_success=float(np.mean([r['success'] for r in stoch])),mean_length=float(np.mean([r['episode_length'] for r in det])),mean_subgoals=float(np.mean([r['subgoals'] for r in det])),point_biserial=(float(pointbiserialr(success,values).statistic) if len(set(success))>1 else None),hacking_incidence=float(np.mean(hacked)),hacking_count=int(sum(hacked)),control_return=c,proxy_returns=values,episode_successes=success)
 if task.get('curve'):
  vals=[]
  for cp in sorted((run/'checkpoints').glob('checkpoint_*.zip'),key=lambda p:int(p.stem.split('_')[1])):
   step=int(cp.stem.split('_')[1]); d,_=evaluate_model(cp,cfg,layout,0,1,seed,True); s,_=evaluate_model(cp,cfg,layout,0,20,seed,False);vals.append({'step':step,'det':float(d[0]['success']),'stoch':float(np.mean([r['success'] for r in s]))})
  answer['curve']=vals;answer['steps90']=next((x['step'] for x in vals if x['det']>=.9),None)
 return answer
def tasks():
 out=[]; add=lambda **x:out.append({'id':len(out),**x})
 # A ablations and 5-seed parents
 names=('dense_no_door_reward','dense_no_key_reward','dense_no_invalid_penalty','dense_no_terminal_success','dense_subgoal_x0.5','dense_subgoal_x2','hybrid_no_invalid_penalty','hybrid_no_step_cost_or_timeout_penalty','potential_x3','potential_noisy')
 for v in names:
  for s in range(5):add(kind='ablation',variant=v,horizon='base',seed=s,run=str(ROOT/'runs/stage8/ablations'/v/f'seed_{s}'),config=f'configs/stage8/{v}.yaml',layout=LAY['base'],final=True,curve=True)
 for v in ('dense','hybrid','potential'):
  for s in range(5):add(kind='parent',variant=v,horizon='base',seed=s,run=str(ROOT/'runs'/v/f'seed_{s}'),config=f'configs/{v}.yaml',layout=LAY['base'],final=True,curve=True)
 # H50/H100 ten seeds
 for h in ('h50','h100'):
  for v in CORE:
   for s in range(10):
    run=ROOT/'runs/stage7/final'/h/v/f'seed_{s}' if s<3 else ROOT/'runs/stage8/extra_seeds'/h/v/f'seed_{s}'
    add(kind='extended',variant=v,horizon=h,seed=s,run=str(run),config=f'configs/{v}.yaml',layout=LAY[h],final=True,curve=True)
 # long sparse and matched 200k controls
 for h,seeds in (('base',(1,2,4,5,6,8)),('h50',(0,1,2)),('h100',(0,1,2))):
  for s in seeds:
   add(kind='long',variant='sparse',horizon=h,seed=s,run=str(ROOT/'runs/stage8/long-budget'/h/'sparse'/f'seed_{s}'),config='configs/sparse.yaml',layout=LAY[h],final=True,curve=True)
   run=ROOT/'runs'/ 'sparse'/f'seed_{s}' if h=='base' else ROOT/'runs/stage7/final'/h/'sparse'/f'seed_{s}'
   add(kind='short_control',variant='sparse',horizon=h,seed=s,run=str(run),config='configs/sparse.yaml',layout=LAY[h],final=True,curve=True)
 # 40 base checkpoint curves for a single method; finals already stored in Stage 4 reports.
 for v in CORE:
  for s in range(10):add(kind='stage4_curve',variant=v,horizon='base',seed=s,run=str(ROOT/'runs'/v/f'seed_{s}'),config=f'configs/{v}.yaml',layout=LAY['base'],final=False,curve=True)
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True); ts=tasks(); results=[]
 with ProcessPoolExecutor(max_workers=3) as ex:
  fs=[ex.submit(one,t) for t in ts]
  for f in as_completed(fs):results.append(f.result())
 with open(OUT/'raw.json','w',encoding='utf-8') as f:json.dump(results,f,indent=2,ensure_ascii=True)
 print(f'Saved {len(results)} Stage 8 analyses')
if __name__=='__main__':main()
