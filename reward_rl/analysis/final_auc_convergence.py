"""Read-only AUC and convergence summary from saved checkpoint evaluations."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from reward_rl.analysis.stage6_zero_shot import bootstrap_ci

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/final_summary';CORE=('sparse','dense','potential','hybrid')
def save(path,rows):
 with open(path,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
def episodes(run,step):
 events=list(Path(run).glob('tb_logs/**/*.tfevents.*'))
 if not events or step is None:return None
 a=EventAccumulator(str(events[0]));a.Reload();tag='ground_truth/total_episodes'
 if tag not in a.Tags().get('scalars',[]):return None
 pts=a.Scalars(tag); return next((int(p.value) for p in pts if p.step>=step),int(pts[-1].value) if pts else None)
def add(rows,h,v,s,curve,run):
 curve=sorted(curve);steps=np.array([0]+[x[0] for x in curve]);ys=np.array([0]+[x[1] for x in curve]);last=int(steps[-1]);auc=float(np.trapezoid(ys,steps)/last) if last else None
 hits={f'steps_to_{pct}':next((int(x) for x,y in curve if y>=pct/100),None) for pct in (25,50,75,90)}
 rows.append({'horizon':h,'variant':v,'seed':s,'final_step':last,'normalised_deterministic_auc':auc,**hits,'training_episodes_to_90':episodes(run,hits['steps_to_90']),'run':str(run)})
def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=[]
 old=json.load(open(ROOT/'runs/stage8/analysis/raw.json',encoding='utf-8'))
 for r in old:
  if r.get('kind')=='stage4_curve':add(rows,'base',r['variant'],r['seed'],[(x['step'],x['det']) for x in r['curve']],ROOT/'runs'/r['variant']/f"seed_{r['seed']}")
  if r.get('kind')=='extended':
   run=ROOT/'runs/stage7/final'/r['horizon']/r['variant']/f"seed_{r['seed']}" if r['seed']<3 else ROOT/'runs/stage8/extra_seeds'/r['horizon']/r['variant']/f"seed_{r['seed']}"
   add(rows,r['horizon'],r['variant'],r['seed'],[(x['step'],x['det']) for x in r['curve']],run)
 # H10 uses its saved deterministic checkpoint curve.
 h10={}
 with open(ROOT/'runs/stage7/final/learning_curves.csv',newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   if r['horizon']=='h10':h10.setdefault((r['variant'],int(r['seed'])),[]).append((int(r['step']),float(r['deterministic_success'])))
 for (v,s),curve in h10.items():add(rows,'h10',v,s,curve,ROOT/'runs/stage7/final/h10'/v/f'seed_{s}')
 # Potential x3 is available at base/H50/H100, not H10.
 for r in old:
  if r.get('kind')=='ablation' and r['variant']=='potential_x3':add(rows,'base','potential_x3',r['seed'],[(x['step'],x['det']) for x in r['curve']],ROOT/'runs/stage8/ablations/potential_x3'/f"seed_{r['seed']}")
 new=json.load(open(ROOT/'runs/stage8b/analysis/raw_evaluations.json',encoding='utf-8'))
 for r in new:
  if r['variant']=='potential_x3':
   h=r['horizon'];add(rows,h,'potential_x3',r['seed'],[(x['step'],x['deterministic_success']) for x in r['curve']],r['run'])
 save(OUT/'auc_convergence_per_run.csv',rows)
 summary=[]
 for h in ('base','h10','h50','h100'):
  for v in (*CORE,'potential_x3'):
   data=[r for r in rows if r['horizon']==h and r['variant']==v]
   if not data:continue
   au=[r['normalised_deterministic_auc'] for r in data if r['normalised_deterministic_auc'] is not None];ep=[r['training_episodes_to_90'] for r in data if r['training_episodes_to_90'] is not None]
   row={'horizon':h,'variant':v,'n_runs':len(data),'n_auc':len(au),'mean_normalised_auc':float(np.mean(au)) if au else None,'auc_ci_low':bootstrap_ci(au)[0] if au else None,'auc_ci_high':bootstrap_ci(au)[1] if au else None,'n_converged_90':len(ep),'mean_training_episodes_to90':float(np.mean(ep)) if ep else None,'episodes_ci_low':bootstrap_ci(ep)[0] if ep else None,'episodes_ci_high':bootstrap_ci(ep)[1] if ep else None,'missing_convergence_runs':len(data)-len(ep)}
   for pct in (25,50,75,90):
    values=[r[f'steps_to_{pct}'] for r in data if r[f'steps_to_{pct}'] is not None];row[f'n_reached_{pct}']=len(values);row[f'mean_steps_to_{pct}']=float(np.mean(values)) if values else None
   summary.append(row)
 save(OUT/'auc_convergence_summary.csv',summary)
 fig,ax=plt.subplots(figsize=(9,5));x=np.arange(4);hs=('base','h10','h50','h100')
 for v in CORE:
  vals=[next((r['mean_normalised_auc'] for r in summary if r['horizon']==h and r['variant']==v),np.nan) for h in hs];ax.plot(x,vals,marker='o',label=v)
 ax.set(xticks=x,xticklabels=['Base','H10','H50','H100'],ylabel='Normalised deterministic AUC',ylim=(-.05,1.05));ax.legend();fig.tight_layout();fig.savefig(OUT/'normalised_auc_by_horizon.png',dpi=160);plt.close(fig)
 print(f'Saved {len(rows)} per-run AUC records')
if __name__=='__main__':main()
