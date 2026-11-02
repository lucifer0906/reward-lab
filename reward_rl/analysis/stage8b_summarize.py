"""Aggregate completed Stage 8b evaluations without altering saved runs."""
from __future__ import annotations
import csv,json
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable
import matplotlib.pyplot as plt
from reward_rl.analysis.stage6_zero_shot import bootstrap_ci
from reward_rl.envs.key_door_env import KeyDoorChainEnv
from reward_rl.envs.solver import solve
from reward_rl.rewards import make_reward
from reward_rl.training.train import load_config

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'runs/stage8b/analysis'; CORE=('sparse','dense','potential','hybrid')

def save(path,rows):
 if rows:
  with open(path,'w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=sorted({k for x in rows for k in x}));w.writeheader();w.writerows(rows)
def load(path):
 with open(path,encoding='utf-8') as f:return json.load(f)
def ci(xs):return [float(x) for x in bootstrap_ci([float(x) for x in xs],draws=4000)]
def control(config,layout):
 cfg=load_config(config);e=KeyDoorChainEnv.from_config(layout,reward_fn=make_reward(cfg['reward'],max_steps=KeyDoorChainEnv.from_config(layout).max_steps));e.reset(seed=0);total=0.
 for a in solve(e.layout):
  _,r,t,x,_=e.step(a);total+=r
  if t or x:break
 e.close();return total
def tax(r):
 curve=r.get('curve',[]);learned=any(x.get('deterministic_success',x.get('det',0))>=.9 for x in curve);final=r.get('final_success',0)
 if not learned:return 'never_learned'
 if final>=.9:return 'learned_and_retained'
 return 'learned_then_failure_stochastic_rescue' if r.get('stochastic_final_success',r.get('stochastic_success',0))>=.9 else 'learned_then_failure'
def holm(rows):
 ordered=sorted(enumerate(rows),key=lambda x:x[1]['fisher_p']);last=0.;n=len(rows)
 for rank,(i,r) in enumerate(ordered):
  v=max(last,min(1.,(n-rank)*r['fisher_p']));rows[i]['holm_p']=v;last=v
  rows[i]['label']='robust' if v<.05 else ('suggestive' if r['fisher_p']<.1 else 'not_detectable')

def main():
 old=load(ROOT/'runs/stage8/analysis/raw.json');new=load(OUT/'raw_evaluations.json')
 # Existing 10-seed core base final rows and their checkpoint-derived steps.
 core=[]; controls={}
 with open(ROOT/'runs/stage4_10seed_final_checkpoint/final_checkpoint_summary.csv',newline='',encoding='utf-8') as f:
  for x in csv.DictReader(f):controls[x['variant']]=float(x['successful_control_median_proxy_return'])
 stage4_steps={(r['variant'],r['seed']):r.get('steps90') for r in old if r.get('kind')=='stage4_curve'}
 with open(ROOT/'runs/stage4_10seed_final_checkpoint/final_checkpoint_per_seed.csv',newline='',encoding='utf-8') as f:
  for x in csv.DictReader(f):
   core.append({'variant':x['variant'],'horizon':'base','seed':int(x['seed']),'final_success':float(x['final_success_rate']),'stochastic_final_success':None,'proxy_returns':[float(x['mean_proxy_return'])]*200,'episode_successes':[float(x['final_success_rate'])]*200,'steps_to_90':stage4_steps.get((x['variant'],int(x['seed']))),'curve':[]})
 # Horizon core evaluations were already checkpoint-evaluated in Stage 8.
 for r in old:
  if r.get('kind')=='extended':
   core.append({'variant':r['variant'],'horizon':r['horizon'],'seed':r['seed'],'final_success':r['final_success'],'stochastic_final_success':r['stochastic_success'],'proxy_returns':r['proxy_returns'],'episode_successes':r['episode_successes'],'steps_to_90':r.get('steps90'),'curve':r.get('curve',[])})
 # Match field names for newly evaluated rows.
 for r in new:r['steps90']=r.get('steps_to_90')
 # Potential x3 new horizons and 10-seed base ablations.
 p3=[r for r in new if r['kind']=='potential_x3']
 names=('dense_no_door_reward','dense_no_key_reward','dense_no_invalid_penalty','dense_no_terminal_success','dense_subgoal_x0.5','dense_subgoal_x2','hybrid_no_invalid_penalty','hybrid_no_step_cost_or_timeout_penalty','potential_x3','potential_noisy')
 ablations=[]
 for r in old:
  if r.get('kind')=='ablation':ablations.append(r)
 for r in new:
  if r['kind']=='ablation_extension':ablations.append(r)
 configs={n:f'configs/stage8/{n}.yaml' for n in names};layouts={n:'configs/env/base_3keys.yaml' for n in names}
 # per-run standardized outputs
 per=[]
 for r in ablations:
  c=control(configs[r['variant']],layouts[r['variant']]);failed=[v for v,s in zip(r['proxy_returns'],r['episode_successes']) if not s]
  per.append({'family':'ablation','variant':r['variant'],'horizon':'base','seed':r['seed'],'final_success':r['final_success'],'stochastic_success':r.get('stochastic_final_success',r.get('stochastic_success')),'steps_to_90':r.get('steps_to_90',r.get('steps90')),'taxonomy':tax(r),'failed_episodes':len(failed),'reward_indistinguishable_fraction':sum(v>=.9*c for v in failed)/len(failed) if failed else 0.})
 for r in p3:
  per.append({'family':'potential_x3','variant':'potential_x3','horizon':r['horizon'],'seed':r['seed'],'final_success':r['final_success'],'stochastic_success':r['stochastic_final_success'],'steps_to_90':r['steps_to_90'],'taxonomy':tax(r),'failed_episodes':sum(not s for s in r['episode_successes']),'reward_indistinguishable_fraction':0.})
 save(OUT/'per_seed_evaluations.csv',per)
 # Parent summary for ablation comparisons.
 parent={'dense':[r for r in core if r['horizon']=='base' and r['variant']=='dense'],'hybrid':[r for r in core if r['horizon']=='base' and r['variant']=='hybrid'],'potential':[r for r in core if r['horizon']=='base' and r['variant']=='potential']}
 parent_for={n:('hybrid' if n.startswith('hybrid') else 'potential' if n.startswith('potential') else 'dense') for n in names}
 ab_summary=[];tests=[]
 for n in names:
  rows=[r for r in per if r['family']=='ablation' and r['variant']==n]; pv=parent[parent_for[n]]; retained=sum(r['final_success']>=.9 for r in rows); pret=sum(r['final_success']>=.9 for r in pv)
  p=float(fisher_exact([[retained,10-retained],[pret,10-pret]]).pvalue);steps=[r['steps_to_90'] for r in rows if r['steps_to_90'] is not None]
  failed=sum(r['failed_episodes'] for r in rows)
  ab_summary.append({'variant':n,'parent':parent_for[n],'retained_seeds_ge90':retained,'mean_final_success':float(np.mean([r['final_success'] for r in rows])),'ci_low':ci([r['final_success'] for r in rows])[0],'ci_high':ci([r['final_success'] for r in rows])[1],'mean_steps_to90':float(np.mean(steps)) if steps else None,'steps_to90_n':len(steps),'never_learned':sum(r['taxonomy']=='never_learned' for r in rows),'learned_retained':sum(r['taxonomy']=='learned_and_retained' for r in rows),'learned_then_failure':sum(r['taxonomy'].startswith('learned_then_failure') for r in rows),'stochastic_rescues':sum(r['taxonomy'].endswith('rescue') for r in rows),'reward_indistinguishable_failure_fraction':sum(r['reward_indistinguishable_fraction']*r['failed_episodes'] for r in rows)/failed if failed else 0.})
  tests.append({'variant':n,'parent':parent_for[n],'ablation_retained':retained,'parent_retained':pret,'fisher_p':p})
 holm(tests)
 for a,t in zip(ab_summary,tests):a.update({'fisher_p':t['fisher_p'],'holm_p':t['holm_p'],'effect_label':t['label']})
 save(OUT/'ablation_10seed_summary.csv',ab_summary);save(OUT/'ablation_10seed_tests.csv',tests)
 # Horizon results and pre-specified tests.
 horizon=[];p3tests=[]
 for h in ('h50','h100'):
  p=[r for r in p3 if r['horizon']==h];combined=core+p3
  for v in (*CORE,'potential_x3'):
   rows=[r for r in combined if r['horizon']==h and r['variant']==v];vals=[r['final_success'] for r in rows];steps=[r.get('steps_to_90',r.get('steps90')) for r in rows if r.get('steps_to_90',r.get('steps90')) is not None]
   horizon.append({'horizon':h,'variant':v,'n':len(rows),'retained_seeds_ge90':sum(x>=.9 for x in vals),'mean_success':float(np.mean(vals)),'ci_low':ci(vals)[0],'ci_high':ci(vals)[1],'mean_steps_to90':float(np.mean(steps)) if steps else None,'steps_to90_n':len(steps),'never_learned':sum(tax(r)=='never_learned' for r in rows),'learned_retained':sum(tax(r)=='learned_and_retained' for r in rows),'learned_then_failure':sum(tax(r).startswith('learned_then_failure') for r in rows),'stochastic_rescues':sum(tax(r).endswith('rescue') for r in rows)})
  for target in ('potential','dense'):
   a=np.asarray([r['final_success'] for r in p]);b=np.asarray([r['final_success'] for r in core if r['horizon']==h and r['variant']==target]);rng=np.random.default_rng(80);d=np.asarray([rng.choice(a,len(a),True).mean()-rng.choice(b,len(b),True).mean() for _ in range(10000)])
   p3tests.append({'horizon':h,'comparison':f'potential_x3_vs_{target}','difference_mean':float(a.mean()-b.mean()),'diff_ci_low':float(np.percentile(d,2.5)),'diff_ci_high':float(np.percentile(d,97.5)),'fisher_p':float(fisher_exact([[sum(a>=.9),sum(a<.9)],[sum(b>=.9),sum(b<.9)]]).pvalue)})
 # Holm is per horizon (two pre-specified tests).
 for h in ('h50','h100'):holm([r for r in p3tests if r['horizon']==h])
 save(OUT/'horizon_comparison_summary.csv',horizon);save(OUT/'potential_x3_prespecified_tests.csv',p3tests)
 # Pooled tests includes x3 on base/h50/h100. Use only core/x3 available cells.
 pooled=[]
 basep3=[r for r in per if r['family']=='ablation' and r['variant']=='potential_x3'];rpool=core+[{**r,'variant':'potential_x3','horizon':'base'} for r in basep3]+p3
 for a,b in [('potential_x3','sparse'),('potential_x3','dense'),('potential_x3','potential'),('potential_x3','hybrid')]:
  tables=[]
  for h in ('base','h50','h100'):
   aa=[r for r in rpool if r['horizon']==h and r['variant']==a];bb=[r for r in rpool if r['horizon']==h and r['variant']==b];tables.append(np.array([[sum(r['final_success']>=.9 for r in aa),sum(r['final_success']<.9 for r in aa)],[sum(r['final_success']>=.9 for r in bb),sum(r['final_success']<.9 for r in bb)]]))
  st=StratifiedTable(tables);lo,hi=st.oddsratio_pooled_confint();pooled.append({'comparison':f'{a}_vs_{b}','common_odds_ratio':float(st.oddsratio_pooled),'ci_low':float(lo),'ci_high':float(hi),'cmh_p':float(st.test_null_odds().pvalue)})
 save(OUT/'stratified_potential_x3_tests.csv',pooled)
 # concise plots
 fig,ax=plt.subplots(figsize=(8,4));x=np.arange(2);width=.16
 for i,v in enumerate((*CORE,'potential_x3')):
  vals=[next(r['mean_success'] for r in horizon if r['horizon']==h and r['variant']==v) for h in ('h50','h100')];ax.bar(x+(i-2)*width,vals,width,label=v)
 ax.set(xticks=x,xticklabels=['H50','H100'],ylim=(0,1.05),ylabel='Final deterministic success');ax.legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(OUT/'potential_x3_horizon_comparison.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,4));ax.bar(range(len(ab_summary)),[r['mean_final_success'] for r in ab_summary]);ax.set(xticks=range(len(ab_summary)),xticklabels=[r['variant'] for r in ab_summary],ylabel='Mean final success',ylim=(0,1.05));plt.setp(ax.get_xticklabels(),rotation=45,ha='right');fig.tight_layout();fig.savefig(OUT/'ablation_10seed_comparison.png',dpi=160);plt.close(fig)
 print('Saved Stage 8b summaries')
if __name__=='__main__':main()
