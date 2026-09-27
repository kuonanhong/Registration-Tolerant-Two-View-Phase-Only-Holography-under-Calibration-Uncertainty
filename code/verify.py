"""Numerical checks for the independent implementation (not physical validation)."""
import argparse,json
from pathlib import Path
import numpy as np
from model import DEFAULT,Problem,fft,adjoint,design_scenarios

def verify():
    rng=np.random.default_rng(91822); out={'implementation':'independent_manuscript_reconstruction'}
    out['discrete_checks']=[]
    for n in (32,64,128,256):
        u=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));U=fft(u)
        power=abs(np.sum(abs(u)**2)-np.sum(abs(U)**2))/np.sum(abs(u)**2)
        inv=np.linalg.norm(adjoint(U)-u)/np.linalg.norm(u)
        y,x=np.indices((n,n));blaze=fft(np.exp(2j*np.pi*(3*x+5*y)/n)/n)
        peak=np.unravel_index(np.argmax(abs(blaze)**2),blaze.shape)
        out['discrete_checks'].append(dict(n=n,parseval_relative_error=float(power),inverse_relative_error=float(inv),blaze_peak=list(map(int,peak)),expected_peak=[n//2+5,n//2+3]))
    cfg={**DEFAULT,'n':32};p=Problem(cfg);g=rng.uniform(.15,.85,(2,32,32));d=rng.normal(size=g.shape);d/=np.linalg.norm(d)
    obj=p.objective(design_scenarios(cfg));loss,grad=obj.loss_grad(g);eps=1e-5
    fd=(obj.loss_grad(g+eps*d,False)-obj.loss_grad(g-eps*d,False))/(2*eps);analytic=float((grad*d).sum())
    out['directional_derivative']={'finite_difference':fd,'analytic':analytic,'relative_error':abs(fd-analytic)/max(abs(fd),abs(analytic)),'step':eps}
    objr=p.objective(design_scenarios(cfg,True));collapsed=objr.collapse()
    lr,gr=objr.loss_grad(g);lc,gc=collapsed.loss_grad(g)
    out['collapsed_identity']={'loss_absolute_error':abs(lr-lc),'gradient_relative_error':float(np.linalg.norm(gr-gc)/np.linalg.norm(gr))}
    _,_,_,I=p.forward(g,np.array([[0,0,1.15,.08]]));selfobj=p.objective(np.array([[0,0,1.15,.08]]));selfobj.targets=I
    ls,gs=selfobj.loss_grad(g);out['self_target']={'loss':ls,'gradient_norm':float(np.linalg.norm(gs))}
    out['passed']=bool(out['directional_derivative']['relative_error']<1e-6 and out['collapsed_identity']['gradient_relative_error']<1e-12 and all(x['parseval_relative_error']<1e-12 and x['inverse_relative_error']<1e-12 and x['blaze_peak']==x['expected_peak'] for x in out['discrete_checks']))
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);a=ap.parse_args();r=verify();txt=json.dumps(r,indent=2);print(txt)
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(txt+'\n')
    if not r['passed']:raise SystemExit(1)
