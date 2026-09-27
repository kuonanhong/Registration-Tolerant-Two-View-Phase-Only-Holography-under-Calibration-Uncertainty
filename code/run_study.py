"""Execute an independent, fully specified reconstruction of the manuscript study.

This cannot recover missing original targets, initialization and RNG streams.
Every output is newly computed and labelled independent_manuscript_reconstruction.
No submitted/published numerical result is substituted into calculated outputs.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import scipy
from scipy.stats import t
from model import DEFAULT,Problem,design_scenarios,eval_scenarios,gs,optimize
from verify import verify

def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def aggregate(rows):
    out={}
    for metric in ('nmse','correct_power','wrong_power','background_power'):
        x=np.array([r[metric] for r in rows]);out[metric]=float(x.mean())
    out['max_nmse']=float(max(r['nmse'] for r in rows));out['min_correct_power']=float(min(r['correct_power'] for r in rows))
    return out

def summarize(seed_rows):
    group={}
    for r in seed_rows:group.setdefault((r['method'],r['radius'],r['calibration'],r['quantized']),[]).append(r)
    summary=[]
    for (method,radius,calibration,quantized),rows in group.items():
        d=dict(method=method,radius=radius,calibration=calibration,quantized=quantized,seeds=len(rows))
        for metric in ('nmse','correct_power','wrong_power','background_power','max_nmse','min_correct_power'):
            x=np.array([r[metric] for r in rows]);d[metric]={'mean':float(x.mean()),'ci95_halfwidth':float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None}
        summary.append(d)
    primary={r['method']:r for r in summary if r['radius']==1 and r['calibration'] and r['quantized']}
    paired=[]
    n={r['seed']:r for r in seed_rows if r['method']=='Nominal' and r['radius']==1 and r['calibration'] and r['quantized']}
    for meth in primary:
        if meth=='Nominal':continue
        vals=[100*(1-r['nmse']/n[r['seed']]['nmse']) for r in seed_rows if r['method']==meth and r['radius']==1 and r['calibration'] and r['quantized']]
        x=np.array(vals);paired.append({'method':meth,'mean_reduction_percent':float(x.mean()),'ci95_halfwidth':float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None,'wrong_power_ratio_of_means':primary[meth]['wrong_power']['mean']/primary['Nominal']['wrong_power']['mean']})
    return {'provenance':'independent_manuscript_reconstruction','summary':summary,'primary':primary,'paired_changes':paired}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=Path('results_reconstructed'));ap.add_argument('--n',type=int,default=128);ap.add_argument('--seeds',nargs='+',type=int,default=DEFAULT['seeds']);ap.add_argument('--family',choices=['symbols','lines','texture'],default='symbols');ap.add_argument('--no-penalty',action='store_true');ap.add_argument('--no-ru',action='store_true');ap.add_argument('--updates',type=int,default=250);ap.add_argument('--warmup',type=int,default=40);ap.add_argument('--radii',type=int,nargs='+',default=[0,1,2,3]);a=ap.parse_args()
    if 0 not in a.radii or 1 not in a.radii:raise ValueError('radii must include 0 and 1 for ablation')
    a.out.mkdir(parents=True,exist_ok=True)
    cfg={**DEFAULT,'n':a.n,'seeds':a.seeds,'target_family':a.family,'updates':a.updates,'warmup':a.warmup,'robust_updates':a.updates//5,'evaluation_radii':a.radii,'implementation':'independent_manuscript_reconstruction','original_data_available':False,'target_rasterization':'model.py targets(); new explicit rasterization, not recovered original','initialization':'default_rng(optimizer_seed).uniform(0,1,(2,N,N)); shared GS warm start','learning_rate_index':'k=0,...,K-1; alpha=.035*(.25+.75*(1-k/K))'}
    (a.out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');p=Problem(cfg)
    scenarios=design_scenarios(cfg);regonly=design_scenarios(cfg,True)
    write_csv(a.out/'scenarios_design.csv',[dict(dx=int(s[0]),dy=int(s[1]),gamma=s[2],b=s[3]) for s in scenarios])
    np.savez_compressed(a.out/'targets.npz',local=p.local,full=p.target,masks=p.mask,aberration=p.a)
    validation=verify();(a.out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    if not validation['passed']:raise RuntimeError('Numerical verification failed')
    raw=[];seedrows=[];histories=[];scenario_rows=[];start=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    for seed in a.seeds:
        st=time.perf_counter();rng=np.random.default_rng(seed);initial=rng.uniform(0,1,(2,a.n,a.n));warm=gs(initial,p.target,cfg['warmup'])
        commands={'GS':gs(warm,p.target,cfg['updates'])}
        objectives={'Nominal':(p.objective(np.array([[0,0,cfg['gamma0'],cfg['b0']]])),cfg['updates']), 'R-F':(p.objective(scenarios),cfg['robust_updates']), 'Collapsed':(p.objective(regonly).collapse(),cfg['updates'])}
        if not a.no_ru:objectives['R-U']=(p.objective(scenarios),cfg['updates'])
        if not a.no_penalty:
            for weight in (32,128):objectives[f'R-F-w{weight}']=(p.objective(scenarios,weight),cfg['robust_updates'])
        for method,(obj,iters) in objectives.items():
            commands[method],history=optimize(obj,warm,iters,cfg['learning_rate'])
            histories.extend(dict(seed=seed,method=method,**h) for h in history)
        np.savez_compressed(a.out/f'commands_seed{seed}.npz',initial=initial,warm=warm,**commands)
        cases=[(r,True,True) for r in a.radii]+[(r,False,True) for r in (0,1)]+[(1,True,False)]
        for radius,calibration,quantized in cases:
            evals=eval_scenarios(seed,radius,cfg,calibration)
            scenario_rows.extend(dict(seed=seed,radius=radius,calibration=calibration,quantized=quantized,scenario=k,dx=int(s[0]),dy=int(s[1]),gamma=s[2],b=s[3]) for k,s in enumerate(evals))
            for method,g in commands.items():
                # Exploratory penalty sensitivity only needs the primary case.
                if method.startswith('R-F-w') and (radius!=1 or not calibration or not quantized):continue
                rows=p.metrics(g,evals,quantized)
                meta=dict(seed=seed,method=method,radius=radius,calibration=calibration,quantized=quantized)
                raw.extend(dict(**meta,**r) for r in rows);seedrows.append(dict(**meta,**aggregate(rows)))
        print(f'seed={seed} seconds={time.perf_counter()-st:.2f}',flush=True)
        # Checkpoint after each seed; safe to inspect during long runs.
        write_csv(a.out/'raw_metrics.csv',raw);write_csv(a.out/'seed_metrics.csv',seedrows);write_csv(a.out/'histories.csv',histories);write_csv(a.out/'scenarios_evaluation.csv',scenario_rows)
        (a.out/'summary.json').write_text(json.dumps(summarize(seedrows),indent=2)+'\n')
    env={'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-start,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'command':sys.argv,'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},'note':'Whole CPU study timing, not per-frame holographic display latency; diagnostic forward calls not included in reported algorithmic FFT budgets.'}
    (a.out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    print(json.dumps(summarize(seedrows)['primary'],indent=2),flush=True)
if __name__=='__main__':main()
