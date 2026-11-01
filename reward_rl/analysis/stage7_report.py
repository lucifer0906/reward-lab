"""Checkpoint-only Stage 7 final analysis."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from reward_rl.analysis.stage6_zero_shot import bootstrap_ci, evaluate_model

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/stage7/final'
LAY={'h10':('configs/env/stage7_h10_5keys.yaml',11,2),'h50':('configs/env/stage7_h50_5keys.yaml',51,5),'h100':('configs/env/stage7_h100_6keys.yaml',102,6)}
VAR=('sparse','dense','potential','hybrid')

def mean(rows,key): return float(np.mean([r[key] for r in rows]))
def rolling(run):
    d=next((run/'tb_logs').glob('*'))
    a=EventAccumulator(str(d)); a.Reload()
    pts=a.Scalars('ground_truth/success_rate')
    return next((int(p.step) for p in pts if p.value>=.9),None)

def main():
    final=[]; curves=[]
    for h,(layout,length,pairs) in LAY.items():
      for v in VAR:
       for seed in range(3):
        run=OUT/h/v/f'seed_{seed}'; cfg=str(run/'stage7_config.yaml')
        det,_=evaluate_model(run/'checkpoints/final_model.zip',cfg,layout,0,200,seed,True)
        sto,_=evaluate_model(run/'checkpoints/final_model.zip',cfg,layout,0,100,seed,False)
        final.append({'horizon':h,'length':length,'pairs':pairs,'variant':v,'seed':seed,
          'deterministic_success':mean(det,'success'),'stochastic_success':mean(sto,'success'),
          'mean_episode_length':mean(det,'episode_length'),'mean_subgoals':mean(det,'subgoals'),
          'rolling_steps_to_90':rolling(run)})
        checkpoints=sorted((run/'checkpoints').glob('checkpoint_*.zip'),key=lambda p:int(p.stem.split('_')[1]))
        for cp in checkpoints:
          step=int(cp.stem.split('_')[1]); rows,_=evaluate_model(cp,cfg,layout,0,1,seed,True)
          curves.append({'horizon':h,'length':length,'variant':v,'seed':seed,'step':step,'deterministic_success':mean(rows,'success')})
    for row in final:
      cr=[x for x in curves if x['horizon']==row['horizon'] and x['variant']==row['variant'] and x['seed']==row['seed']]
      row['deterministic_steps_to_90']=next((x['step'] for x in cr if x['deterministic_success']>=.9),None)
      learned=row['deterministic_steps_to_90'] is not None
      row['taxonomy']='never_learned' if not learned else ('learned_then_deterministic_failure' if row['deterministic_success']<.9 else 'learned_and_retained')
      row['stochastic_rescued']=bool(row['taxonomy']=='learned_then_deterministic_failure' and row['stochastic_success']>=.9)
    summary=[]
    for h,(layout,length,pairs) in LAY.items():
      for v in VAR:
       rows=[r for r in final if r['horizon']==h and r['variant']==v]; vals=[r['deterministic_success'] for r in rows]; ci=bootstrap_ci(vals)
       summary.append({'horizon':h,'length':length,'pairs':pairs,'variant':v,'n_seeds':3,'mean_success':float(np.mean(vals)),'ci_low':ci[0],'ci_high':ci[1],
        'mean_stochastic_success':mean(rows,'stochastic_success'),'mean_episode_length':mean(rows,'mean_episode_length'),'mean_subgoals':mean(rows,'mean_subgoals'),
        'seeds_at_90':sum(x>=.9 for x in vals),'mean_deterministic_steps_to_90':float(np.mean([r['deterministic_steps_to_90'] for r in rows if r['deterministic_steps_to_90'] is not None])) if any(r['deterministic_steps_to_90'] is not None for r in rows) else None})
    for name,data in [('final_per_seed.csv',final),('learning_curves.csv',curves),('summary.csv',summary)]:
      with open(OUT/name,'w',newline='',encoding='utf-8') as f:
       w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    fig,ax=plt.subplots(figsize=(8,5))
    for v in VAR:
      rows=sorted([r for r in summary if r['variant']==v],key=lambda r:r['length']);ax.errorbar([r['length'] for r in rows],[r['mean_success'] for r in rows],yerr=[[r['mean_success']-r['ci_low'] for r in rows],[r['ci_high']-r['mean_success'] for r in rows]],marker='o',label=v)
    ax.set(xlabel='Shortest solution length',ylabel='Final deterministic success',ylim=(-.05,1.05),title='Stage 7 success vs horizon');ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/'success_vs_horizon.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for v in VAR:
      rows=sorted([r for r in summary if r['variant']==v and r['mean_deterministic_steps_to_90'] is not None],key=lambda r:r['length']);ax.plot([r['length'] for r in rows],[r['mean_deterministic_steps_to_90'] for r in rows],marker='o',label=v)
    ax.set(xlabel='Shortest solution length',ylabel='Checkpoint-evaluated steps to 90%',title='Stage 7 sample efficiency');ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/'steps_to_90_vs_horizon.png',dpi=160);plt.close(fig)
    for h in LAY:
      fig,ax=plt.subplots(figsize=(8,5))
      for v in VAR:
       data=[r for r in curves if r['horizon']==h and r['variant']==v]; xs=sorted(set(r['step'] for r in data));ys=[np.mean([r['deterministic_success'] for r in data if r['step']==x]) for x in xs];ax.plot(xs,ys,label=v)
      ax.set(xlabel='Training steps',ylabel='Deterministic checkpoint success',ylim=(-.05,1.05),title=f'Stage 7 learning curves {h}');ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/f'learning_curve_{h}.png',dpi=160);plt.close(fig)
    with open(OUT/'report.json','w',encoding='utf-8') as f:json.dump({'summary':summary,'final':final},f,indent=2,ensure_ascii=True)
    print('Saved Stage 7 report')
if __name__=='__main__':main()
