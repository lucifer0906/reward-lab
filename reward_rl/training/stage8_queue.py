"""One resumable, priority-ordered Stage 8 queue (A then B then C)."""
from __future__ import annotations
import json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
PYTHON=ROOT/'.venv/Scripts/python.exe'
CORE=('sparse','dense','potential','hybrid')

def complete(run:Path, steps:int)->bool:
 p=run/'completion.json'
 if not p.exists(): return False
 with open(p,encoding='utf-8') as f: marker=json.load(f)
 return marker.get('status')=='complete' and marker.get('timesteps')==steps

def derived_config(variant:str, layout:str, steps:int, path:Path, name:str)->None:
 with open(ROOT/'configs'/f'{variant}.yaml',encoding='utf-8') as f: cfg=yaml.safe_load(f)
 cfg['meta']['name']=name;cfg['env']['config']=layout;cfg['training']['total_timesteps']=steps
 with open(path,'w',encoding='utf-8') as f:yaml.safe_dump(cfg,f,sort_keys=False,allow_unicode=False)

def run(job:dict)->dict:
 run_dir=ROOT/job['run_dir'];run_dir.mkdir(parents=True,exist_ok=True);steps=job['steps']
 if complete(run_dir,steps):return {**job,'status':'skipped'}
 config=run_dir/'config.yaml'
 if job.get('source'):
  with open(ROOT/job['source'],encoding='utf-8') as f:cfg=yaml.safe_load(f)
  with open(config,'w',encoding='utf-8') as f:yaml.safe_dump(cfg,f,sort_keys=False,allow_unicode=False)
 else: derived_config(job['variant'],job['layout'],steps,config,job['name'])
 env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','TF_ENABLE_ONEDNN_OPTS':'0'})
 t=time.time();p=subprocess.run([str(PYTHON),'-m','reward_rl.train','--config',str(config),'--seed',str(job['seed']),'--output',str(run_dir.parent)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE);elapsed=time.time()-t
 if p.returncode:return {**job,'status':'error','elapsed_seconds':elapsed,'stderr':p.stderr.decode('utf-8',errors='replace')[-500:]}
 with open(run_dir/'completion.json','w',encoding='utf-8') as f:json.dump({'status':'complete','timesteps':steps},f,indent=2,ensure_ascii=True)
 return {**job,'status':'complete','elapsed_seconds':elapsed}

def jobs():
 base=Path('runs/stage8');out=[]
 for name in ('dense_no_door_reward','dense_no_key_reward','dense_no_invalid_penalty','dense_no_terminal_success','dense_subgoal_x0.5','dense_subgoal_x2','hybrid_no_invalid_penalty','hybrid_no_step_cost_or_timeout_penalty','potential_x3','potential_noisy'):
  for seed in range(5):out.append({'group':'A','name':f'stage8_{name}','variant':name,'seed':seed,'steps':200000,'source':f'configs/stage8/{name}.yaml','run_dir':str(base/'ablations'/name/f'seed_{seed}')})
 for horizon,layout in (('h50','configs/env/stage7_h50_5keys.yaml'),('h100','configs/env/stage7_h100_6keys.yaml')):
  for variant in CORE:
   for seed in range(3,10):out.append({'group':'B','name':f'stage8_{horizon}_{variant}','variant':variant,'seed':seed,'steps':200000,'layout':layout,'run_dir':str(base/'extra_seeds'/horizon/variant/f'seed_{seed}')})
 for horizon,layout,seeds in (('base','configs/env/base_3keys.yaml',(1,2,4,5,6,8)),('h50','configs/env/stage7_h50_5keys.yaml',(0,1,2)),('h100','configs/env/stage7_h100_6keys.yaml',(0,1,2))):
  for seed in seeds:out.append({'group':'C','name':f'stage8_long_budget_{horizon}_sparse','variant':'sparse','seed':seed,'steps':1000000,'layout':layout,'run_dir':str(base/'long-budget'/horizon/'sparse'/f'seed_{seed}')})
 return out

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--resume',action='store_true');a=p.parse_args()
 all_results=[]
 for group in ('A','B','C'):
  group_jobs=[j for j in jobs() if j['group']==group]
  with ThreadPoolExecutor(max_workers=a.workers) as ex: results=[f.result() for f in as_completed([ex.submit(run,j) for j in group_jobs])]
  all_results.extend(results)
  with open(ROOT/'runs/stage8'/f'queue_{group}.json','w',encoding='utf-8') as f:json.dump(results,f,indent=2,ensure_ascii=True)
 with open(ROOT/'runs/stage8/queue_all.json','w',encoding='utf-8') as f:json.dump(all_results,f,indent=2,ensure_ascii=True)
 print('Saved Stage 8 queue results')
if __name__=='__main__':main()
