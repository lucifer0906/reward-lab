"""Summarise the read-only Stage 8 checkpoint evaluations."""
from __future__ import annotations

import csv, json, math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact, pointbiserialr

from reward_rl.analysis.stage6_zero_shot import bootstrap_ci

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/stage8/analysis"
CORE = ("sparse", "dense", "potential", "hybrid")


def ci(values):
    values = [float(v) for v in values]
    if not values:
        return (float("nan"), float("nan"))
    return tuple(float(x) for x in bootstrap_ci(values, draws=4000))


def taxonomy(row):
    learned = row.get("steps90") is not None
    retained = row.get("final_success", 0.0) >= 0.9
    if not learned:
        return "never_learned"
    if retained:
        return "learned_and_retained"
    return "learned_then_deterministic_failure" + ("_stochastic_rescue" if row.get("stochastic_success", 0.0) >= 0.9 else "")


def holm(pairs):
    ordered = sorted(enumerate(pairs), key=lambda x: x[1]["fisher_p"])
    n = len(pairs)
    previous = 0.0
    for rank, (index, pair) in enumerate(ordered):
        adjusted = max(previous, min(1.0, (n - rank) * pair["fisher_p"]))
        pairs[index]["fisher_holm"] = adjusted
        pairs[index]["significant_holm_0_05"] = adjusted < 0.05
        previous = adjusted


def group(rows, name, reference=""):
    vals = [r["final_success"] for r in rows]
    low, high = ci(vals)
    successes = sum(v >= .9 for v in vals)
    steps = [r["steps90"] for r in rows if r.get("steps90") is not None]
    pooled_y = [y for r in rows for y in r.get("episode_successes", [])]
    pooled_x = [x for r in rows for x in r.get("proxy_returns", [])]
    pb = None if len(set(pooled_y)) < 2 else float(pointbiserialr(pooled_y, pooled_x).statistic)
    tax = {key: sum(taxonomy(r).startswith(key) for r in rows) for key in ("never_learned", "learned_and_retained", "learned_then_deterministic_failure")}
    return {"name": name, "n": len(rows), "seeds_ge_90": successes, "mean_final_success": float(np.mean(vals)), "ci_low": low, "ci_high": high,
            "mean_steps90": float(np.mean(steps)) if steps else None, "steps90_n": len(steps), "never_learned": tax["never_learned"],
            "learned_and_retained": tax["learned_and_retained"], "learned_then_deterministic_failure": tax["learned_then_deterministic_failure"],
            "stochastic_rescues": sum("stochastic_rescue" in taxonomy(r) for r in rows), "point_biserial": pb,
            "hacking_incidence": float(sum(r.get("hacking_count", 0) for r in rows) / max(1, 200 * len(rows))), "reference_10_seed": reference}


def save_csv(path, records):
    if not records:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in records for k in r}))
        writer.writeheader(); writer.writerows(records)


def main():
    with open(OUT / "raw.json", encoding="utf-8") as f:
        raw = json.load(f)
    final = [r for r in raw if "final_success" in r]
    for r in final:
        r["taxonomy"] = taxonomy(r)
    save_csv(OUT / "final_per_run.csv", final)
    curves = [{"kind": r["kind"], "variant": r["variant"], "horizon": r["horizon"], "seed": r["seed"], **point}
              for r in raw for point in r.get("curve", [])]
    save_csv(OUT / "intermediate_checkpoint_evaluations.csv", curves)

    stage4_reference = {}
    ref_path = ROOT / "runs/stage4_10seed_final_checkpoint/final_checkpoint_summary.csv"
    with open(ref_path, newline="", encoding="utf-8") as f:
        for x in csv.DictReader(f):
            stage4_reference[x["variant"]] = f"{x['seeds_at_or_above_90pct']}/10, {float(x['mean_final_success_rate']):.3f}"

    abl_rows = [r for r in final if r["kind"] in ("ablation", "parent")]
    ablation_summary = []
    for v in sorted({r["variant"] for r in abl_rows}):
        ablation_summary.append(group([r for r in abl_rows if r["variant"] == v], v, stage4_reference.get(v, "")))
    save_csv(OUT / "ablation_summary.csv", ablation_summary)

    ext = [r for r in final if r["kind"] == "extended"]
    extended_summary = []
    comparisons = []
    for h in ("h50", "h100"):
        data = {v: [r for r in ext if r["horizon"] == h and r["variant"] == v] for v in CORE}
        extended_summary += [dict(group(data[v], f"{h}_{v}"), horizon=h, variant=v) for v in CORE]
        horizon_pairs = []
        for a, b in combinations(CORE, 2):
            av = np.asarray([r["final_success"] for r in data[a]]); bv = np.asarray([r["final_success"] for r in data[b]])
            rng = np.random.default_rng(802 + len(horizon_pairs)); diffs = [float(np.mean(rng.choice(av, len(av), True))-np.mean(rng.choice(bv, len(bv), True))) for _ in range(10000)]
            aa, bb = sum(av >= .9), sum(bv >= .9)
            p = float(fisher_exact([[aa, len(av)-aa], [bb, len(bv)-bb]]).pvalue)
            horizon_pairs.append({"horizon": h, "comparison": f"{a}_minus_{b}", "mean_success_difference": float(np.mean(av)-np.mean(bv)), "diff_ci_low": float(np.percentile(diffs,2.5)), "diff_ci_high": float(np.percentile(diffs,97.5)), "fisher_p":p,"a_ge90":int(aa),"b_ge90":int(bb)})
        holm(horizon_pairs); comparisons += horizon_pairs
    save_csv(OUT / "extended_seed_summary.csv", extended_summary); save_csv(OUT / "pairwise_comparisons.csv", comparisons)

    long = [r for r in final if r["kind"] == "long"]
    short = {(r["horizon"],r["seed"]):r for r in final if r["kind"] == "short_control"}
    budget = []
    for r in long:
        baseline=short[(r["horizon"],r["seed"])]
        if r["final_success"] >= .9 and baseline["final_success"] < .9: label="slow_learns_eventually"
        elif r["final_success"] < .9 and r.get("steps90") is None: label="stuck"
        elif r["final_success"] < .9: label="learned_then_failed"
        else: label="already_retained_at_200k"
        budget.append({"horizon":r["horizon"],"seed":r["seed"],"success_200k":baseline["final_success"],"success_1m":r["final_success"],"steps90_1m":r.get("steps90"),"classification":label})
    save_csv(OUT / "sparse_long_budget_comparison.csv", budget)

    stage4 = [r for r in raw if r["kind"] == "stage4_curve"]
    s4out=[]
    for r in stage4:s4out.append({"variant":r["variant"],"seed":r["seed"],"steps90":r.get("steps90")})
    save_csv(OUT / "stage4_steps_to_90_checkpoint.csv", s4out)

    # Figures.
    plt.style.use("seaborn-v0_8-whitegrid")
    horizons=["h10","base","h50","h100"]; xs=np.array([11,26,51,102]); labels=["H10\n11", "Base\n26", "H50\n51", "H100\n102"]
    stage7_rows=[]
    with open(ROOT/'runs/stage7/final/final_per_seed.csv', newline='', encoding='utf-8') as f:
        stage7_rows=list(csv.DictReader(f))
    # Base values are the existing ten-seed final evaluations.
    base={v:float(stage4_reference[v].split(", ")[1]) for v in CORE}
    fig,ax=plt.subplots(figsize=(9,5))
    for v in CORE:
        vals=[np.nan,base[v],np.nan,np.nan]
        for i,h in ((2,"h50"),(3,"h100")):
            row=next(x for x in extended_summary if x["horizon"]==h and x["variant"]==v);vals[i]=row["mean_final_success"]
        h10=[float(x['deterministic_success']) for x in stage7_rows if x['horizon']=='h10' and x['variant']==v]
        vals[0]=float(np.mean(h10))
        ax.plot(xs,vals,marker='o',label=v)
    ax.plot(xs,[.004,0,0,0],color='gray',ls=':',label='random baseline');ax.axhline(1,color='black',ls='--',label='solver')
    ax.set(xlabel='Shortest solution length',ylabel='Final deterministic success',xticks=xs,xticklabels=labels,ylim=(-.05,1.05));ax.legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(OUT/'success_vs_horizon.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5))
    for v in CORE:
        vals=[]
        for h in horizons:
            rows=stage4 if h=='base' else ([r for r in raw if r['kind']=='extended' and r['horizon']==h and r['variant']==v] if h in ('h50','h100') else [])
            if h=='base': rows=[r for r in stage4 if r['variant']==v]
            if h=='h10':
                vals.append(np.mean([float(x['deterministic_steps_to_90']) for x in stage7_rows if x['horizon']=='h10' and x['variant']==v and x['deterministic_steps_to_90']]))
            else:
                vals.append(np.mean([r['steps90'] for r in rows if r.get('steps90') is not None]) if rows else np.nan)
        ax.plot(xs,vals,marker='o',label=v)
    ax.set(xlabel='Shortest solution length',ylabel='Checkpoint steps to 90%',xticks=xs,xticklabels=labels);ax.legend();fig.tight_layout();fig.savefig(OUT/'steps_to_90_vs_horizon.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(12,5)); names=[x['name'] for x in ablation_summary]; ax.bar(range(len(names)),[x['mean_final_success'] for x in ablation_summary]);ax.set(xticks=range(len(names)),xticklabels=names,ylabel='Mean final success',ylim=(0,1.05));plt.setp(ax.get_xticklabels(),rotation=45,ha='right');fig.tight_layout();fig.savefig(OUT/'ablation_bar.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); names=[f"{r['horizon']}-{r['seed']}" for r in budget];ax.scatter(range(len(budget)),[r['success_200k'] for r in budget],label='200k');ax.scatter(range(len(budget)),[r['success_1m'] for r in budget],label='1M');ax.set(xticks=range(len(budget)),xticklabels=names,ylabel='Final success',ylim=(-.05,1.05));ax.legend();plt.setp(ax.get_xticklabels(),rotation=45,ha='right');fig.tight_layout();fig.savefig(OUT/'sparse_200k_vs_1m.png',dpi=160);plt.close(fig)
    print('Wrote summaries and plots to', OUT)

if __name__ == '__main__':
    main()
