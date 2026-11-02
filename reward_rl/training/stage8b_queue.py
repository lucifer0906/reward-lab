"""One resumable, priority-ordered Stage 8b queue (A then B)."""
from __future__ import annotations
import json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
PYTHON=ROOT/'.venv/Scripts/python.exe'
ABLATIONS=('dense_no_door_reward','dense_no_key_reward','dense_no_invalid_penalty','dense_no_terminal_success','dense_subgoal_x0.5','dense_subgoal_x2','hybrid_no_invalid_penalty','hybrid_no_step_cost_or_timeout_penalty','potential_x3','potential_noisy')

def complete(run:Path, steps:int)->bool:
 p=run/'completion.json'
 if not p.exists():return False
 with open(p,encoding='utf-8') as f: marker=json.load(f)
 return marker.get('status')=='complete' and marker.get('timesteps')==steps

def run(job):
 run_dir=ROOT/job['run_dir'];run_dir.mkdir(parents=True,exist_ok=True)
 if complete(run_dir,job['steps']):return {**job,'status':'skipped'}
 with open(ROOT/job['source'],encoding='utf-8') as f:cfg=yaml.safe_load(f)
 with open(run_dir/'config.yaml','w',encoding='utf-8') as f:yaml.safe_dump(cfg,f,sort_keys=False,allow_unicode=False)
 env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','TF_ENABLE_ONEDNN_OPTS':'0'})
 started=time.time();p=subprocess.run([str(PYTHON),'-m','reward_rl.train','--config',str(run_dir/'config.yaml'),'--seed',str(job['seed']),'--output',str(run_dir.parent)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE);elapsed=time.time()-started
 if p.returncode:return {**job,'status':'error','elapsed_seconds':elapsed,'stderr':p.stderr.decode('utf-8',errors='replace')[-500:]}
 with open(run_dir/'completion.json','w',encoding='utf-8') as f:json.dump({'status':'complete','timesteps':job['steps']},f,indent=2,ensure_ascii=True)
 return {**job,'status':'complete','elapsed_seconds':elapsed}

def jobs():
 out=[]
 for horizon,source in (('h50','configs/stage8b/potential_x3_h50.yaml'),('h100','configs/stage8b/potential_x3_h100.yaml')):
  for seed in range(10):out.append({'group':'A','variant':'potential_x3','horizon':horizon,'seed':seed,'steps':200000,'source':source,'run_dir':f'runs/stage8b/potential_x3/{horizon}/seed_{seed}'})
 for name in ABLATIONS:
  for seed in range(5,10):out.append({'group':'B','variant':name,'horizon':'base','seed':seed,'steps':200000,'source':f'configs/stage8/{name}.yaml','run_dir':f'runs/stage8b/ablations/{name}/seed_{seed}'})
 return out

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--resume',action='store_true');a=p.parse_args(); all_results=[]
 for group in ('A','B'):
  queued=[j for j in jobs() if j['group']==group]
  with ThreadPoolExecutor(max_workers=a.workers) as ex:results=[f.result() for f in as_completed([ex.submit(run,j) for j in queued])]
  all_results.extend(results)
  with open(ROOT/'runs/stage8b'/f'queue_{group}.json','w',encoding='utf-8') as f:json.dump(results,f,indent=2,ensure_ascii=True)
 with open(ROOT/'runs/stage8b/queue_all.json','w',encoding='utf-8') as f:json.dump(all_results,f,indent=2,ensure_ascii=True)
 print('Saved Stage 8b queue results')

if __name__=='__main__':main()
