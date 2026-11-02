"""Read-only Stage 8b statistical corrections over saved Stage 4--8 records."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from statsmodels.stats.contingency_tables import StratifiedTable
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/stage8/analysis'
CORE=('sparse','dense','potential','hybrid')

def write(path, rows):
 if not rows:return
 with open(path,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)

def cmh(rows, a, b):
 tables=[]
 for h in ('base','h50','h100'):
  aa=[r for r in rows if r['horizon']==h and r['variant']==a]
  bb=[r for r in rows if r['horizon']==h and r['variant']==b]
  tables.append(np.array([[sum(r['retained'] for r in aa),sum(not r['retained'] for r in aa)], [sum(r['retained'] for r in bb),sum(not r['retained'] for r in bb)]]))
 stat=StratifiedTable(tables)
 lo,hi=stat.oddsratio_pooled_confint()
 return {'comparison':f'{a}_vs_{b}','common_odds_ratio':float(stat.oddsratio_pooled),'or_ci_low':float(lo),'or_ci_high':float(hi),'cmh_p':float(stat.test_null_odds().pvalue),'homogeneity_p':float(stat.test_equal_odds().pvalue)}

def first_logged_success(run):
 events=list(run.glob('tb_logs/**/*.tfevents.*'))
 if not events:return None
 acc=EventAccumulator(str(events[0]));acc.Reload()
 if 'ground_truth/success_rate' not in acc.Tags().get('scalars',[]):return None
 return next((int(x.step) for x in acc.Scalars('ground_truth/success_rate') if x.value>0),None)

def main():
 with open(OUT/'raw.json',encoding='utf-8') as f: raw=json.load(f)
 final=[r for r in raw if 'final_success' in r]
 # Metric for every stored final episode (all Stage 8 ablations and horizon core cells).
 metric=[]
 for r in final:
  values=r.get('proxy_returns',[]); outcome=r.get('episode_successes',[]);control=r.get('control_return')
  failed=[x for x,y in zip(values,outcome) if not y]
  metric.append({'kind':r['kind'],'variant':r['variant'],'horizon':r['horizon'],'seed':r['seed'],'failed_episodes':len(failed),'control_proxy_return':control,'reward_indistinguishable_failure_fraction':(sum(x>=.9*control for x in failed)/len(failed) if failed else 0.0)})
 # Base core 10-seed results were stored as deterministic per-seed summaries.
 controls={}
 with open(ROOT/'runs/stage4_10seed_final_checkpoint/final_checkpoint_summary.csv',newline='',encoding='utf-8') as f:
  for row in csv.DictReader(f):controls[row['variant']]=float(row['successful_control_median_proxy_return'])
 with open(ROOT/'runs/stage4_10seed_final_checkpoint/final_checkpoint_per_seed.csv',newline='',encoding='utf-8') as f:
  for row in csv.DictReader(f):
   failed=float(row['final_success_rate'])<1
   metric.append({'kind':'core_base_stage4','variant':row['variant'],'horizon':'base','seed':int(row['seed']),'failed_episodes':200 if failed else 0,'control_proxy_return':controls[row['variant']],'reward_indistinguishable_failure_fraction':float(float(row['mean_proxy_return'])>=.9*controls[row['variant']]) if failed else 0.0})
 write(OUT/'reward_indistinguishable_failures.csv',metric)
 summary=[]
 for kind,v,h in sorted({(r['kind'],r['variant'],r['horizon']) for r in metric}):
  data=[r for r in metric if (r['kind'],r['variant'],r['horizon'])==(kind,v,h)]
  failures=sum(r['failed_episodes'] for r in data)
  summary.append({'kind':kind,'variant':v,'horizon':h,'seeds':len(data),'failed_episodes':failures,'reward_indistinguishable_failure_fraction':sum(r['reward_indistinguishable_failure_fraction']*r['failed_episodes'] for r in data)/failures if failures else 0.0})
 write(OUT/'reward_indistinguishable_summary.csv',summary)

 # Retained-seed data stratified by the three layouts.
 retained=[]
 with open(ROOT/'runs/stage4_10seed_final_checkpoint/final_checkpoint_per_seed.csv',newline='',encoding='utf-8') as f:
  for row in csv.DictReader(f):retained.append({'horizon':'base','variant':row['variant'],'seed':int(row['seed']),'retained':float(row['final_success_rate'])>=.9})
 for row in final:
  if row['kind']=='extended':retained.append({'horizon':row['horizon'],'variant':row['variant'],'seed':row['seed'],'retained':row['final_success']>=.9})
 cmh_rows=[cmh(retained,a,b) for a,b in [('dense','sparse'),('potential','sparse'),('hybrid','sparse'),('dense','hybrid'),('dense','potential'),('hybrid','potential')]]
 write(OUT/'stratified_retained_seed_tests.csv',cmh_rows)

 # All saved sparse training runs. This is first *logged* nonzero success window;
 # callbacks did not persist individual episode event steps.
 sparse=[]
 def add(h,budget,seed,run,final_state):
  sparse.append({'horizon':h,'budget_steps':budget,'seed':seed,'run':str(run),'first_logged_success_step':first_logged_success(run),'final_state':final_state})
 stage4={}
 with open(ROOT/'runs/stage4_10seed_final_checkpoint/final_checkpoint_per_seed.csv',newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   if r['variant']=='sparse':stage4[int(r['seed'])]=float(r['final_success_rate'])
 for s,x in stage4.items():add('base',200000,s,ROOT/'runs/sparse'/f'seed_{s}','retained' if x>=.9 else 'failed')
 # Stage 7 final includes 200k sparse runs at all horizons, and Stage 8 adds H50/H100 seeds 3-9.
 for h in ('h10','h50','h100'):
  for s in range(3):
   row=next(r for r in final if r['kind']=='extended' and r['horizon']==h and r['variant']=='sparse' and r['seed']==s) if h in ('h50','h100') else None
   if row is None:
    with open(ROOT/'runs/stage7/final/final_per_seed.csv',newline='',encoding='utf-8') as f: row=next(r for r in csv.DictReader(f) if r['horizon']==h and r['variant']=='sparse' and int(r['seed'])==s)
    state='retained' if float(row['deterministic_success'])>=.9 else 'failed'
   else: state='retained' if row['final_success']>=.9 else 'failed'
   add(h,200000,s,ROOT/'runs/stage7/final'/h/'sparse'/f'seed_{s}',state)
 for h in ('h50','h100'):
  for s in range(3,10):
   row=next(r for r in final if r['kind']=='extended' and r['horizon']==h and r['variant']=='sparse' and r['seed']==s)
   add(h,200000,s,ROOT/'runs/stage8/extra_seeds'/h/'sparse'/f'seed_{s}','retained' if row['final_success']>=.9 else 'failed')
 for h,seeds in [('base',(1,2,4,5,6,8)),('h50',(0,1,2)),('h100',(0,1,2))]:
  for s in seeds:
   row=next(r for r in final if r['kind']=='long' and r['horizon']==h and r['seed']==s)
   add(h,1000000,s,ROOT/'runs/stage8/long-budget'/h/'sparse'/f'seed_{s}','retained' if row['final_success']>=.9 else 'failed')
 for r in sparse:r['ever_logged_success']=r['first_logged_success_step'] is not None
 write(OUT/'sparse_first_logged_success.csv',sparse)
 print('Saved Stage 8b read-only analyses')

if __name__=='__main__':main()
