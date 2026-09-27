"""Independent reconstruction from manuscript equations, not recovered original code.

All target rasterization and RNG conventions are explicit here because the supplied
archive does not contain the original Python, targets, commands, or per-seed data.
Arrays use (..., slot, y, x); the two time slots are independent patterns.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter

METHODS = ('GS', 'Nominal', 'R-U', 'R-F', 'Collapsed')
DEFAULT = dict(n=128, wavelength_m=532e-9, pitch_m=8e-6, eta=0.86,
               target_power=0.60, gamma0=1.15, b0=0.08,
               background_weight=0.05, spill_weight=8.0,
               warmup=40, updates=250, robust_updates=50,
               learning_rate=0.035, eval_draws=3, eval_seed_offset=100000,
               seeds=[11,23,37,51,71,89], target_family='symbols')

def fft(u):
    return np.fft.fftshift(np.fft.fft2(u, axes=(-2,-1), norm='ortho'), axes=(-2,-1))

def adjoint(U):
    return np.fft.ifft2(np.fft.ifftshift(U, axes=(-2,-1)), axes=(-2,-1), norm='ortho')

def couple(g,b):
    return (1-b)*g + b/4*(np.roll(g,1,-1)+np.roll(g,-1,-1)+np.roll(g,1,-2)+np.roll(g,-1,-2))

def targets(n, family='symbols'):
    """Explicit NEW rasterization; it need not equal the unavailable original.

    Local pixel-center coordinates are 2*(index+.5)/m-1. Gaussian sigma
    is 0.6*m/32, reflect boundaries. Every target is normalized to unit sum.
    """
    if n % 8: raise ValueError('n must be divisible by eight')
    m=n//4; c=2*(np.arange(m)+.5)/m-1
    x,y=np.meshgrid(c,c); r=np.hypot(x,y)
    if family=='symbols':
        a=((np.abs(r-.59)<=.075)|((np.abs(x)<=.065)&(np.abs(y)<=.64))).astype(float)
        b=(((np.abs(r-.55)<=.07)&(y<0))|((np.abs(np.abs(x)-.55)<=.07)&(y>=0)&(y<=.57))|((np.abs(y-.58)<=.065)&(np.abs(x)<=.78))).astype(float)
    elif family=='lines':
        a=(((np.abs(x)<.055)|(np.abs(x-.35)<.055)|(np.abs(x+.35)<.055))&(np.abs(y)<.70)).astype(float)
        b=(((np.abs(x-y)<.08)|(np.abs(x+y)<.08))&(np.abs(x)<.67)&(np.abs(y)<.67)).astype(float)
    elif family=='texture':
        a=np.exp(-((x+.15)**2+(y-.12)**2)/.32)*(1+.60*np.cos(9*x)*np.cos(7*y))
        b=np.exp(-(x*x+y*y)/.42)*(1+.50*np.cos(11*x+5*y)+.30*np.sin(6*y))
    else: raise ValueError(f'Unknown target family {family}')
    local=np.stack([gaussian_filter(a,.6*m/32,mode='reflect'),gaussian_filter(b,.6*m/32,mode='reflect')])
    local/=local.sum(axis=(-2,-1),keepdims=True)
    full=np.zeros((2,n,n)); mask=np.zeros_like(full)
    for j,cx in enumerate((n//4,3*n//4)):
        sl=(slice(n//2-m//2,n//2+m//2),slice(cx-m//2,cx+m//2))
        full[(j,)+sl]=local[j]; mask[(j,)+sl]=1
    return local,full,mask

def aberration(n):
    c=2*(np.arange(n)+.5)/n-1; x,y=np.meshgrid(c,c)
    return .32*(x*x-y*y)+.18*x*y

def design_scenarios(config, registration_only=False):
    g,b=config['gamma0'],config['b0']
    s=np.array([[0,0,g,b],[-1,-1,1.07,.05],[-1,1,1.23,.05],[1,-1,1.07,.11],[1,1,1.23,.11]],float)
    if registration_only: s[:,2:]=[g,b]
    return s

def eval_scenarios(seed,r,config,calibration=True):
    # Radius-specific RNG streams; identical draws shared by every method.
    rng=np.random.default_rng(config['eval_seed_offset']+100*int(seed)+int(r))
    rows=[]
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            for draw in range(config['eval_draws']):
                gamma,b=(rng.uniform(1.07,1.23),rng.uniform(.05,.11))
                if not calibration: gamma,b=config['gamma0'],config['b0']
                rows.append([dx,dy,gamma,b])
    return np.array(rows,float)

class Problem:
    def __init__(self,config):
        self.config=dict(config); self.n=config['n']; self.a=aberration(self.n)
        self.local,self.target,self.mask=targets(self.n,config['target_family'])
        self.den=((config['target_power']*self.target)**2).sum(axis=(-2,-1))
    def forward(self,g,scenarios):
        gamma=scenarios[:,2,None,None,None]; b=scenarios[:,3,None,None,None]
        h=np.clip(couple(g[None,:,:,:],b),0,1)
        phase=2*np.pi*h**gamma+self.a
        u=np.sqrt(self.config['eta'])/self.n*np.exp(1j*phase)
        U=fft(u); I=np.abs(U)**2
        return h,u,U,I
    def objective(self,scenarios,spill_weight=None):
        if spill_weight is None: spill_weight=self.config['spill_weight']
        ws=[];ts=[]
        for dx,dy,_,_ in scenarios:
            M=np.roll(self.mask,(int(dy),int(dx)),axis=(-2,-1))
            ws.append(self.config['background_weight']+(1-self.config['background_weight'])*M+(spill_weight-self.config['background_weight'])*M[::-1])
            ts.append(self.config['target_power']*np.roll(self.target,(int(dy),int(dx)),axis=(-2,-1)))
        return Objective(self,scenarios,np.array(ws),np.array(ts))
    def metrics(self,g,scenarios,quantized=True):
        if quantized:g=np.round(255*g)/255
        rows=[]
        # Chunking bounds memory for N=256 and the r=3 sensitivity.
        for k,scenario in enumerate(scenarios):
            if k % 12 == 0:
                _,_,_,Is=self.forward(g,scenarios[k:k+12])
            local_k=k % 12
            dx,dy,gamma,b=scenario; dx=int(dx);dy=int(dy)
            for j in range(2):
                M=np.roll(self.mask[j],(dy,dx),axis=(-2,-1)).astype(bool)
                other=np.roll(self.mask[1-j],(dy,dx),axis=(-2,-1)).astype(bool)
                crop=Is[local_k,j][M].reshape(self.local[j].shape)
                E=float(crop.sum()); W=float(Is[local_k,j][other].sum())
                e=float(((crop/max(E,np.finfo(float).tiny)-self.local[j])**2).sum()/(self.local[j]**2).sum())
                B=float(crop[self.local[j]<=.01*self.local[j].max()].sum())
                rows.append(dict(scenario=k,slot=j+1,dx=dx,dy=dy,gamma=gamma,b=b,nmse=e,correct_power=E,wrong_power=W,background_power=B))
        return rows

class Objective:
    def __init__(self,problem,scenarios,weights,targets,constant=0.0):
        self.problem=problem;self.scenarios=scenarios;self.weights=weights;self.targets=targets;self.constant=float(constant)
    def loss_grad(self,g,gradient=True):
        p=self.problem;h,u,U,I=p.forward(g,self.scenarios)
        v=self.weights*(I-self.targets)/p.den[None,:,None,None]
        loss=.5*np.mean(np.sum(v*(I-self.targets),axis=(1,2,3)))+self.constant
        if not gradient:return float(loss)
        z=2*np.imag(u.conj()*adjoint(v*U))
        gamma=self.scenarios[:,2,None,None,None];b=self.scenarios[:,3,None,None,None]
        grad=np.mean(couple(2*np.pi*gamma*h**(gamma-1)*z,b),axis=0)
        return float(loss),grad
    def collapse(self):
        # Exact only when forward calibration is identical for every scenario.
        if not np.all(self.scenarios[:,2:]==self.scenarios[0,2:]):raise ValueError('Cannot collapse scenario-dependent calibration')
        A=self.weights.mean(axis=0);T=(self.weights*self.targets).mean(axis=0)/A
        C=.5*np.mean(np.sum(self.weights*(self.targets-T)**2/self.problem.den[None,:,None,None],axis=(1,2,3)))
        s=self.scenarios[:1].copy();s[:,:2]=0
        return Objective(self.problem,s,A[None],T[None],C)

def gs(start,target,iterations):
    g=start.copy();n=g.shape[-1];amp=1/n
    for _ in range(iterations):
        U=fft(amp*np.exp(2j*np.pi*g))
        u=adjoint(np.sqrt(target)*np.exp(1j*np.angle(U)))
        g=np.mod(np.angle(u),2*np.pi)/(2*np.pi)
    return g

def optimize(obj,start,updates,learning_rate=.035,record_every=5):
    g=start.copy();m=np.zeros_like(g);v=np.zeros_like(g);history=[]
    scenarios=len(obj.scenarios)
    for k in range(updates):
        loss,grad=obj.loss_grad(g)
        if k%record_every==0:history.append(dict(update=k,fft_pairs_per_slot=k*scenarios,loss=loss))
        t=k+1;m=.9*m+.1*grad;v=.999*v+.001*grad*grad
        lr=learning_rate*(.25+.75*(1-k/updates))
        g=np.clip(g-lr*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8),0,1)
    history.append(dict(update=updates,fft_pairs_per_slot=updates*scenarios,loss=obj.loss_grad(g,False)))
    return g,history
