"""Append the existing 10-seed base result to Stage 7 horizon comparisons."""
from __future__ import annotations
import ast, csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/stage7/final'

def main():
    rows=[]
    with open(OUT/'summary.csv',newline='',encoding='utf-8') as f:
      for r in csv.DictReader(f): rows.append(r)
    with open(ROOT/'runs/stage4_10seed_final_checkpoint/final_checkpoint_summary.csv',newline='',encoding='utf-8') as f:
      for r in csv.DictReader(f):
       ci=ast.literal_eval(r['bootstrap_95pct_ci_mean_success'])
       rows.append({'horizon':'base','length':'26','pairs':'3','variant':r['variant'],'n_seeds':'10',
                    'mean_success':r['mean_final_success_rate'],'ci_low':str(ci[0]),'ci_high':str(ci[1]),
                    'mean_stochastic_success':'','mean_episode_length':'','mean_subgoals':'',
                    'seeds_at_90':r['seeds_at_or_above_90pct'],'mean_deterministic_steps_to_90':r['mean_steps_to_90pct_among_learned']})
    with open(OUT/'comparison_with_base.csv','w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(8,5))
    for v in ('sparse','dense','potential','hybrid'):
      x=sorted([r for r in rows if r['variant']==v],key=lambda r:int(r['length']))
      y=[float(r['mean_success']) for r in x];lo=[float(r['mean_success'])-float(r['ci_low']) for r in x];hi=[float(r['ci_high'])-float(r['mean_success']) for r in x]
      ax.errorbar([int(r['length']) for r in x],y,yerr=[lo,hi],marker='o',label=v)
    ax.set(xlabel='Shortest solution length',ylabel='Final deterministic success',ylim=(-.05,1.05),title='Base plus Stage 7 success vs horizon');ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(OUT/'success_vs_horizon_with_base.png',dpi=160)
    print('Saved base-inclusive Stage 7 comparison')
if __name__=='__main__':main()
