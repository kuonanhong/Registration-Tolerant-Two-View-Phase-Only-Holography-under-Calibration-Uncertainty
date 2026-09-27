"""Check saved raw metrics, aggregation, and an archived-command forward rerun."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from model import Problem,eval_scenarios
from run_study import summarize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=Path('results_reconstructed'));a=ap.parse_args();p=a.results
    cfg=json.loads((p/'config.json').read_text());raw=list(csv.DictReader((p/'raw_metrics.csv').open()));seedrows=list(csv.DictReader((p/'seed_metrics.csv').open()));reported=json.loads((p/'summary.json').read_text())
    for r in seedrows:
        for k in r:
            if k in ('seed','radius'):r[k]=int(r[k])
            elif k in ('calibration','quantized'):r[k]=r[k]=='True'
            elif k!='method':r[k]=float(r[k])
    regroup=summarize(seedrows)
    assert regroup==reported,'summary differs from saved seed metrics'
    eta=cfg['eta'];finite=True;bounded=True;max_sum=0
    for r in raw:
        vals=[float(r[k]) for k in ['nmse','correct_power','wrong_power','background_power']]
        finite &= bool(np.all(np.isfinite(vals)));bounded &= bool(min(vals)>=0 and vals[1]+vals[2]<=eta+1e-12 and vals[3]<=vals[1]+1e-12);max_sum=max(max_sum,vals[1]+vals[2])
    seed=cfg['seeds'][0];problem=Problem(cfg);z=np.load(p/f'commands_seed{seed}.npz');scenarios=eval_scenarios(seed,1,cfg,True);rerun=problem.metrics(z['R-F'],scenarios,True)
    saved=[r for r in raw if int(r['seed'])==seed and r['method']=='R-F' and r['radius']=='1' and r['calibration']=='True' and r['quantized']=='True']
    error=max(abs(float(x[k])-y[k]) for x,y in zip(saved,rerun) for k in ['nmse','correct_power','wrong_power','background_power'])
    command_bounds=all(np.isfinite(z[k]).all() and np.min(z[k])>=0 and np.max(z[k])<=1 for k in z.files)
    result={'provenance':'new reconstruction outputs','raw_rows':len(raw),'seed_rows':len(seedrows),'summary_regeneration_exact':True,'all_metrics_finite':finite,'all_power_bounds_satisfied':bounded,'largest_addressed_plus_wrong_power':max_sum,'archived_command_rerun_max_abs_metric_difference':error,'command_bounds_satisfied':command_bounds,'passed':bool(finite and bounded and error<1e-12 and command_bounds)}
    (p/'output_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']
if __name__=='__main__':main()
