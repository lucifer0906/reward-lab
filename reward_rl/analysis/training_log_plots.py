"""Plot retained training-log reward and component aggregates, read-only."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/final_summary'
def events(run,tag):
 files=list(Path(run).glob('tb_logs/**/*.tfevents.*'))
 if not files:return None
 a=EventAccumulator(str(files[0]));a.Reload()
 if tag not in a.Tags().get('scalars',[]):return None
 x=a.Scalars(tag);return np.array([z.step for z in x]),np.array([z.value for z in x])
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=list(csv.DictReader(open(OUT/'auc_convergence_per_run.csv',newline='',encoding='utf-8')))
 variants=('sparse','dense','potential','hybrid'); horizons=('base','h10','h50','h100')
 fig,axes=plt.subplots(4,4,figsize=(14,12),sharex=False)
 for i,v in enumerate(variants):
  for j,h in enumerate(horizons):
   ax=axes[i,j];data=[]
   for r in rows:
    if r['variant']==v and r['horizon']==h:
     e=events(r['run'],'rollout/ep_rew_mean')
     if e:data.append(e)
   if data:
    xmax=max(x[-1] for x,_ in data);grid=np.linspace(0,xmax,201);arr=np.vstack([np.interp(grid,x,y,left=y[0],right=y[-1]) for x,y in data]);ax.plot(grid,arr.mean(0));ax.fill_between(grid,arr.mean(0)-arr.std(0),arr.mean(0)+arr.std(0),alpha=.25)
   else:ax.text(.5,.5,'No retained reward scalar',ha='center',transform=ax.transAxes)
   ax.set_title(f'{v} / {h} (n={len(data)})');ax.set(xlabel='Environment steps',ylabel='Mean episode reward');ax.grid(alpha=.25)
 fig.suptitle('Training reward by variant and layout; bands are seed standard deviation. Raw reward scales differ across reward functions and are not comparable across variants.',fontsize=10);fig.tight_layout();fig.savefig(OUT/'mean_episode_reward_vs_steps.png',dpi=160);plt.close(fig)
 # Base-only reward components: callback logs windowed per-step means, not episode-level returns.
 fig,axes=plt.subplots(1,3,figsize=(15,4))
 report=[]
 for ax,v in zip(axes,('dense','hybrid','potential')):
  vr=[r for r in rows if r['variant']==v and r['horizon']=='base'];tags=set()
  for r in vr:
   files=list(Path(r['run']).glob('tb_logs/**/*.tfevents.*'))
   if files:
    a=EventAccumulator(str(files[0]));a.Reload();tags.update(t for t in a.Tags().get('scalars',[]) if t.startswith('reward_components/'))
  for tag in sorted(tags):
   ds=[events(r['run'],tag) for r in vr];ds=[d for d in ds if d]
   if not ds:continue
   xmax=max(x[-1] for x,_ in ds);grid=np.linspace(0,xmax,201);arr=np.vstack([np.interp(grid,x,y,left=y[0],right=y[-1]) for x,y in ds]);ax.plot(grid,arr.mean(0),label=tag.split('/',1)[1]);report.append({'variant':v,'component':tag.split('/',1)[1],'seed_logs':len(ds),'granularity':'running per-step window mean; episode-level component returns not retained'})
  ax.set(title=f'{v} base components',xlabel='Environment steps',ylabel='Component mean / step');ax.grid(alpha=.25);ax.legend(fontsize=7)
 fig.suptitle('Base-layout reward-component contribution over training; episode-level component returns were not retained.',fontsize=10);fig.tight_layout();fig.savefig(OUT/'base_reward_component_contributions.png',dpi=160);plt.close(fig)
 with open(OUT/'reward_component_log_coverage.csv','w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['variant','component','seed_logs','granularity']);w.writeheader();w.writerows(report)
 print('Saved training-log figures')
if __name__=='__main__':main()
