"""Regenerate cross-run sensitivity and uploaded-versus-new comparison tables.

All supplementary targets are independently optimized. They are not held-out
content generalization; N=256 changes physical aperture at fixed pitch.
"""
import argparse,csv,json
from pathlib import Path
from reported_tables import PRIMARY

def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));a=ap.parse_args();rows=[]
    for label,name in [('symbols128','results_reconstructed'),('symbols256','results_panel256'),('lines128','results_lines'),('texture128','results_texture')]:
        path=a.root/name/'summary.json'
        if not path.exists():continue
        d=json.loads(path.read_text())
        for m in ['Nominal','R-F','Collapsed']:
            s=d['primary'][m];rows.append(dict(dataset=label,method=m,nmse_mean=s['nmse']['mean'],nmse_ci95=s['nmse']['ci95_halfwidth'],correct_power_mean=s['correct_power']['mean'],wrong_power_mean=s['wrong_power']['mean'],seeds=s['seeds']))
    target=a.root/'results_reconstructed';target.mkdir(parents=True,exist_ok=True);write(target/'content_sensitivity_summary.csv',rows)
    d=json.loads((target/'summary.json').read_text());comp=[]
    for reported in PRIMARY:
        method=reported[0];fresh=d['primary'][method]
        for metric,index in [('nmse',1),('correct_power',3),('wrong_power',5)]:
            old=reported[index];new=fresh[metric]['mean'];comp.append(dict(method=method,metric=metric,uploaded_reported_rounded_value=old,new_independent_reconstruction_value=new,new_minus_reported=new-old,status='not_an_exact_original_reproduction'))
    write(target/'reported_vs_reconstructed.csv',comp)
    print(f'Wrote {len(rows)} sensitivity rows and {len(comp)} provenance comparison rows')
if __name__=='__main__':main()
