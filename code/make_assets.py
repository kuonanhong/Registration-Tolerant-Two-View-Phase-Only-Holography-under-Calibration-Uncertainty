"""Generate every numerical figure and Tables 1--3 from NEW saved run outputs.

Original uploaded figures cannot be numerically regenerated without their missing
command arrays/targets/raw metrics; they are retained only as source documents.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from model import Problem,eval_scenarios

COLORS={'GS':'#81929A','Nominal':'#128799','R-U':'#BB5035','R-F':'#B58C22','Collapsed':'#7967A2'}
LABELS={'GS':'Ideal-model GS','Nominal':'Calibrated nominal','R-U':'R-U: equal updates','R-F':'R-F: equal FFT budget','Collapsed':'Collapsed target'}
ORDER=['GS','Nominal','R-U','R-F','Collapsed']

def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))

def save(fig,out,name):
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight');fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)

def plot_angular_channels(p,commands,out,info):
    """First seed, R-F, nominal calibration; absolute incident-power intensity."""
    s=np.array([[0,0,p.config['gamma0'],p.config['b0']]])
    g=np.round(255*commands['R-F'])/255;_,_,_,I=p.forward(g,s)
    n=p.n;angle=np.degrees(np.arcsin(p.config['wavelength_m']*(np.arange(n)-n//2)/(n*p.config['pitch_m'])))
    # Image extent uses sample edges, avoiding off-by-half-pixel center displacement.
    edges=np.degrees(np.arcsin(p.config['wavelength_m']*(np.arange(n+1)-n//2-.5)/(n*p.config['pitch_m'])))
    fig,ax=plt.subplots(figsize=(5.8,4.5));im=ax.imshow(10*np.log10(np.maximum(I[0,0],1e-12)),origin='lower',extent=[edges[0],edges[-1],edges[0],edges[-1]],cmap='magma',vmin=-65,vmax=-22,interpolation='nearest')
    for j,cx in enumerate((n//4,3*n//4)):
        lo=cx-n//8;hi=cx+n//8;yl=n//2-n//8;yh=n//2+n//8
        ax.add_patch(Rectangle((edges[lo],edges[yl]),edges[hi]-edges[lo],edges[yh]-edges[yl],fill=False,ec='#5AE5C1' if j==0 else '#72C6FF',lw=1.5,ls='-' if j==0 else '--'))
    ax.set(xlabel='Horizontal diffraction angle [deg]',ylabel='Vertical diffraction angle [deg]',title='R-F, slot 1: absolute angular power')
    fig.colorbar(im,ax=ax,label=r'$10\log_{10}(I/P_{\rm inc})$ [dB/sample]');save(fig,out,'angular_channels')
    info['angular_channels']={'function':'make_assets.py::plot_angular_channels','inputs':['config.json',f'commands_seed{p.config["seeds"][0]}.npz'],'method':'R-F','seed':p.config['seeds'][0],'slot':1,'gamma':float(s[0,2]),'b':float(s[0,3]),'quantized':True,'normalization':'unit incident power; not per-image peak','power_sum':float(I[0,0].sum())}

def plot_reconstruction_comparison(p,commands,out,info):
    seed=p.config['seeds'][0];scenarios=eval_scenarios(seed,1,p.config,True);index=np.where((scenarios[:,0]==1)&(scenarios[:,1]==1))[0][0];s=scenarios[index:index+1]
    methods=[m for m in ORDER if m in commands];fig,axs=plt.subplots(2,len(methods)+1,figsize=(2*(len(methods)+1),4.5),constrained_layout=True);vmax=p.local.max()
    for j in range(2):
        axs[j,0].imshow(p.local[j],cmap='magma',vmin=0,vmax=vmax);axs[j,0].set_title(f'Target / slot {j+1}');axs[j,0].axis('off')
    for col,method in enumerate(methods,1):
        _,_,_,I=p.forward(np.round(255*commands[method])/255,s)
        for j in range(2):
            mask=np.roll(p.mask[j],(1,1),axis=(-2,-1)).astype(bool);crop=I[0,j][mask].reshape(p.local[j].shape);crop/=crop.sum()
            axs[j,col].imshow(crop,cmap='magma',vmin=0,vmax=vmax);axs[j,col].set_title(method);axs[j,col].axis('off')
    fig.suptitle('Independent reconstruction: held-out +1,+1 registration; own-power-normalized shape',fontsize=11);save(fig,out,'reconstruction_comparison')
    info['reconstruction_comparison']={'function':'make_assets.py::plot_reconstruction_comparison','inputs':['config.json',f'commands_seed{seed}.npz','targets.npz'],'seed':seed,'evaluation_scenario_index':int(index),'scenario':s[0].tolist(),'normalization':'each crop divided by its own addressed-window power; common target scale'}

def plot_robustness_sweep(summary,out,info):
    fig,axs=plt.subplots(1,3,figsize=(11.8,3.6),constrained_layout=True)
    for method in ORDER:
        rows=sorted([r for r in summary['summary'] if r['method']==method and r['calibration'] and r['quantized']],key=lambda r:r['radius'])
        if not rows:continue
        for ax,key,title in zip(axs,['nmse','correct_power','wrong_power'],['Mean shape NMSE','Addressed-window power / incident power','Wrong-window power / incident power']):
            ax.errorbar([r['radius'] for r in rows],[r[key]['mean'] for r in rows],yerr=[r[key]['ci95_halfwidth'] or 0 for r in rows],label=LABELS[method],color=COLORS[method],marker='o',ms=4,lw=1.5,capsize=2);ax.set(xlabel='Registration radius [Fourier samples]',ylabel=title,xticks=[r['radius'] for r in rows]);ax.grid(alpha=.16)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False);save(fig,out,'robustness_sweep')
    info['robustness_sweep']={'function':'make_assets.py::plot_robustness_sweep','inputs':['summary.json','seed_metrics.csv'],'selection':'calibration=True, quantized=True; all radii','interval':'Student t, across seed-level means, not individual scenarios'}

def plot_collapsed_diagnostic(summary,out,info):
    methods=['Nominal','R-F','Collapsed'];fig,axs=plt.subplots(1,2,figsize=(8.0,3.6),constrained_layout=True)
    for ax,key,title in zip(axs,['nmse','wrong_power'],['Mean shape NMSE','Wrong-window power / incident power']):
        rows=[summary['primary'][m] for m in methods];ax.bar(range(3),[r[key]['mean'] for r in rows],yerr=[r[key]['ci95_halfwidth'] or 0 for r in rows],color=[COLORS[m] for m in methods],capsize=3,width=.70);ax.set(xticks=range(3),xticklabels=['Nominal','R-F\n(5 scenarios)','Collapsed\ntarget'],ylabel=title);ax.grid(axis='y',alpha=.16)
    fig.suptitle('Mechanism control: 250 optimization FFT pairs per slot',fontsize=11);save(fig,out,'collapsed_diagnostic')
    info['collapsed_diagnostic']={'function':'make_assets.py::plot_collapsed_diagnostic','inputs':['summary.json','seed_metrics.csv'],'selection':'r=1, calibration=True, quantized=True; Nominal/R-F/Collapsed'}

def plot_optimization_history(histories,out,info,seed):
    fig,axs=plt.subplots(1,2,figsize=(8.0,3.5),constrained_layout=True)
    for method in ['Nominal','R-U','R-F','Collapsed']:
        rows=[r for r in histories if r['method']==method and int(r['seed'])==seed]
        if not rows:continue
        for ax,key in zip(axs,['update','fft_pairs_per_slot']):
            ax.semilogy([int(r[key]) for r in rows],[float(r['loss']) for r in rows],color=COLORS[method],label=LABELS[method],lw=1.8);ax.set_ylabel('Own training objective');ax.grid(alpha=.16)
    axs[0].set_xlabel('Command updates after common warm start');axs[1].set_xlabel('Optimization FFT pairs per slot after warm start');axs[1].legend(fontsize=8);save(fig,out,'optimization_history')
    info['optimization_history']={'function':'make_assets.py::plot_optimization_history','inputs':['histories.csv'],'seed':seed,'warning':'Different objectives/constant offsets; vertical values are not common performance scores. Counts exclude diagnostic loss-only calls.'}

def tex_table(path,caption,columns,rows,label,star=False):
    env='table*' if star else 'table';align='l'+'c'*(len(columns)-1)
    lines=[r'\begin{'+env+r'}[t]\centering\small',r'\caption{'+caption+r'}\label{'+label+'}',r'\begin{tabular}{'+align+r'}\toprule',' & '.join(columns)+r'\\\midrule']
    lines+=[' & '.join(row)+r'\\' for row in rows];lines += [r'\bottomrule\end{tabular}\end{'+env+'}']
    path.write_text('\n'.join(lines)+'\n')

def generate_tables(config,summary,design,out,info):
    scenario_rows=[[str(int(float(r['dx']))),str(int(float(r['dy']))),f"{float(r['gamma']):.2f}",f"{float(r['b']):.2f}"] for r in design]
    tex_table(out/'generated_scenario_table_EN.tex','Prescribed design scenarios; shifts are Fourier samples. Generated by code/make\\_assets.py from scenarios\\_design.csv. These values are assumptions, not observations.',['$\\delta_x$','$\\delta_y$','$\\gamma$','$b$'],scenario_rows,'tab:scenarios')
    rows=[]
    for method in ORDER:
        if method not in summary['primary']:continue
        r=summary['primary'][method];vals=[]
        for metric in ['nmse','correct_power','wrong_power']:
            m=r[metric];vals.append(f"${m['mean']:.6f}\\pm {m['ci95_halfwidth']:.6f}$" if m['ci95_halfwidth'] is not None else f"${m['mean']:.6f}$")
        rows.append([method]+vals)
    tex_table(out/'generated_primary_table_EN.tex','New independent reconstruction at $r=1$: means $\\pm$ 95\\% confidence half-widths across seed-level means. Source: code/make\\_assets.py, summary.json. This is not a recovery of the uploaded numerical archive.',['Method','Shape NMSE','Correct power','Wrong-window power'],rows,'tab:main',True)
    rows=[]
    for r in (0,1):
        for calibration in (False,True):
            vals=[]
            for method in ['Nominal','R-F','R-U','Collapsed']:
                z=next((s for s in summary['summary'] if s['method']==method and s['radius']==r and s['calibration']==calibration and s['quantized']),None)
                vals.append(f"{z['nmse']['mean']:.6f}" if z else '--')
            rows.append([str(r),'Yes' if calibration else 'No']+vals)
    tex_table(out/'generated_ablation_table_EN.tex','Evaluation-only ablation for the new reconstruction; fixed optimized commands and eight-bit quantization. Source: code/make\\_assets.py and seed\\_metrics.csv.',['$r$','Calibration','Nominal','R-F','R-U','Collapsed'],rows,'tab:ablation',True)
    with (out/'table1.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['dx','dy','gamma','b']);w.writerows(scenario_rows)
    with (out/'table2.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','nmse_mean','nmse_ci95','correct_power_mean','correct_power_ci95','wrong_power_mean','wrong_power_ci95'])
        for method in ORDER:
            if method in summary['primary']:
                r=summary['primary'][method];w.writerow([method]+[r[k][v] for k in ['nmse','correct_power','wrong_power'] for v in ['mean','ci95_halfwidth']])
    with (out/'table3.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['radius','calibration','Nominal','R-F','R-U','Collapsed']);w.writerows(rows)
    changes={r['method']:r for r in summary['paired_changes']};rf=changes['R-F']
    (out/'generated_values.tex').write_text('\\newcommand{\\BudgetReduction}{'+f"{rf['mean_reduction_percent']:.3f}"+'}\n\\newcommand{\\SpillRatio}{'+f"{rf['wrong_power_ratio_of_means']:.3f}"+'}\n')
    info['table1']={'generator':'make_assets.py::generate_tables','input':'scenarios_design.csv','origin':'prescribed configuration; not derived by fitting'}
    info['table2']={'generator':'run_study.py::aggregate/summarize + make_assets.py::generate_tables','input':'raw_metrics.csv -> seed_metrics.csv -> summary.json','aggregation':'scenario+slot mean within seed, then Student-t CI across seeds'}
    info['table3']={'generator':'same as table2','input':'raw_metrics.csv evaluation toggles; optimizers fixed','aggregation':'means across seed-level NMSE, calibration and registration toggled; quantized=True'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=Path('results_reconstructed'));ap.add_argument('--out',type=Path,default=Path('artifacts_reconstructed'));a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    config=json.loads((a.results/'config.json').read_text());summary=json.loads((a.results/'summary.json').read_text());p=Problem(config);seed=config['seeds'][0]
    with np.load(a.results/'targets.npz') as saved_targets:
        if not all(np.array_equal(saved_targets[k],v) for k,v in [('local',p.local),('full',p.target),('masks',p.mask),('aberration',p.a)]):
            raise ValueError('Saved targets differ from current model/config; regenerate or use the archived model version.')
    with np.load(a.results/f'commands_seed{seed}.npz') as z:commands={k:z[k] for k in z.files}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42})
    info={'provenance':'All assets from independent reconstruction; not the unavailable original run.', 'input_roles':'inputs lists direct files and upstream provenance; summary.json is the direct plot input for aggregate panels, seed_metrics.csv is its upstream statistical source; targets.npz is read to verify exact target/model agreement.'}
    plot_angular_channels(p,commands,a.out,info);plot_reconstruction_comparison(p,commands,a.out,info);plot_robustness_sweep(summary,a.out,info);plot_collapsed_diagnostic(summary,a.out,info);plot_optimization_history(read_csv(a.results/'histories.csv'),a.out,info,seed);generate_tables(config,summary,read_csv(a.results/'scenarios_design.csv'),a.out,info)
    (a.out/'asset_provenance.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':main()
