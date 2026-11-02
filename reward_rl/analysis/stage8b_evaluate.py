"""Detached, read-only evaluation batch for completed Stage 8b checkpoints."""
from __future__ import annotations
import json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

from reward_rl.analysis.stage6_zero_shot import evaluate_model

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/stage8b/analysis'

def evaluate(task):
 os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
 run=Path(task['run']);cfg=str(run/'config.yaml');layout=task['layout'];seed=task['seed']
 det,_=evaluate_model(run/'checkpoints/final_model.zip',cfg,layout,0,200,seed,True)
 sto,_=evaluate_model(run/'checkpoints/final_model.zip',cfg,layout,0,100,seed,False)
 curve=[]
 for cp in sorted((run/'checkpoints').glob('checkpoint_*.zip'),key=lambda p:int(p.stem.split('_')[1])):
  step=int(cp.stem.split('_')[1]);d,_=evaluate_model(cp,cfg,layout,0,1,seed,True);s,_=evaluate_model(cp,cfg,layout,0,20,seed,False)
  curve.append({'step':step,'deterministic_success':float(d[0]['success']),'stochastic_success':float(np.mean([x['success'] for x in s]))})
 return {'kind':task['kind'],'variant':task['variant'],'horizon':task['horizon'],'seed':seed,'run':str(run),'final_success':float(np.mean([x['success'] for x in det])),'stochastic_final_success':float(np.mean([x['success'] for x in sto])),'mean_episode_length':float(np.mean([x['episode_length'] for x in det])),'mean_subgoals':float(np.mean([x['subgoals'] for x in det])),'proxy_returns':[x['proxy_return'] for x in det],'episode_successes':[float(x['success']) for x in det],'curve':curve,'steps_to_90':next((x['step'] for x in curve if x['deterministic_success']>=.9),None)}

def tasks():
 out=[]
 for h,layout in (('h50','configs/env/stage7_h50_5keys.yaml'),('h100','configs/env/stage7_h100_6keys.yaml')):
  for seed in range(10):out.append({'kind':'potential_x3','variant':'potential_x3','horizon':h,'seed':seed,'layout':layout,'run':str(ROOT/'runs/stage8b/potential_x3'/h/f'seed_{seed}')})
 names=('dense_no_door_reward','dense_no_key_reward','dense_no_invalid_penalty','dense_no_terminal_success','dense_subgoal_x0.5','dense_subgoal_x2','hybrid_no_invalid_penalty','hybrid_no_step_cost_or_timeout_penalty','potential_x3','potential_noisy')
 for name in names:
  for seed in range(5,10):out.append({'kind':'ablation_extension','variant':name,'horizon':'base','seed':seed,'layout':'configs/env/base_3keys.yaml','run':str(ROOT/'runs/stage8b/ablations'/name/f'seed_{seed}')})
 return out

def main():
 OUT.mkdir(parents=True,exist_ok=True);ts=tasks();result=[]
 with ProcessPoolExecutor(max_workers=3) as ex:
  futures=[ex.submit(evaluate,t) for t in ts]
  for future in as_completed(futures):result.append(future.result())
 with open(OUT/'raw_evaluations.json','w',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=True,indent=2)
 print(f'Completed {len(result)} Stage 8b checkpoint evaluations')

if __name__=='__main__':main()
