"""Machine-readable transcription of UPLOADED manuscript values, not recomputation.

The original package contains no raw seed-level data. These numbers must never be
represented as newly verified model output. Use run_study.py for fresh computation.
"""
import argparse,csv,json
from pathlib import Path

PRIMARY=[['GS',.255392,.003352,.808111,.001411,.002596,.000105],['Nominal',.220155,.000044,.604258,.000266,.001610,.000052],['R-U',.182907,.002322,.599506,.000388,.006394,.000135],['R-F',.183169,.001389,.604563,.000461,.006744,.000126]]
ABLATION=[[0,False,.000041,.056758,.055109],[0,True,.000078,.057062,.055901],[1,False,.220123,.182722,.182215],[1,True,.220155,.183169,.182907]]
SCENARIOS=[[0,0,1.15,.08],[-1,-1,1.07,.05],[-1,1,1.23,.05],[1,-1,1.07,.11],[1,1,1.23,.11]]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('results_reported'));a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    d={'provenance':'transcribed_from_uploaded_manuscript_NOT_recomputed','raw_data_available':False,'primary_columns':['method','nmse','nmse_ci95','correct_power','correct_power_ci95','wrong_power','wrong_power_ci95'],'primary':PRIMARY,'ablation_columns':['radius','calibration','Nominal','R-F','R-U'],'ablation':ABLATION,'design_scenarios':SCENARIOS,'reported_paired_reduction_percent':16.800,'reported_spill_ratio':4.189,'rounded_mean_reduction_percent':100*(1-.183169/.220155),'rounded_spill_ratio':.006744/.001610,'note':'Ratio of rounded means is only an arithmetic consistency check. It does not recover the mean of six paired percentage reductions or its CI.'}
    (a.out/'reported_values.json').write_text(json.dumps(d,indent=2)+'\n')
    for name,header,rows in [('table1',['dx','dy','gamma','b'],SCENARIOS),('table2',d['primary_columns'],PRIMARY),('table3',d['ablation_columns'],ABLATION)]:
        with (a.out/f'{name}.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(header);w.writerows(rows)
    print(json.dumps(d,indent=2))
if __name__=='__main__':main()
