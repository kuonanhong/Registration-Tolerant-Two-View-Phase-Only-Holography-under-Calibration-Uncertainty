#!/usr/bin/env python3
"""Audit the calibration-varying / collapsed-objective discrepancy.

Reads saved continuous commands; does not train or select an optimizer.  The
inequality is command-specific on the five design scenarios and concerns the
training loss, not held-out NMSE or physical robustness.  This diagnostic was
added during manuscript revision after the original reported experiment.

Example:
  python code/calibration_discrepancy.py --results results_reconstructed
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Problem, design_scenarios


METHODS = ('GS', 'Nominal', 'R-U', 'R-F', 'Collapsed')
COLORS = ('#657788', '#087e8b', '#b85042', '#b28a18', '#7d6aa5')


def discrepancy(problem, commands, scenarios):
    obj = problem.objective(scenarios)
    fixed_scenarios = scenarios.copy()
    fixed_scenarios[:, 2:] = [problem.config['gamma0'], problem.config['b0']]
    fixed = problem.objective(fixed_scenarios)
    collapse = fixed.collapse()
    I = problem.forward(commands, scenarios)[3]
    I0 = problem.forward(commands, fixed_scenarios[:1])[3]
    w = obj.weights/(2*len(scenarios)*problem.den[None, :, None, None])
    R = I0-obj.targets
    delta = I-I0
    L = float(np.sum(w*(R+delta)**2))
    L0 = float(np.sum(w*R**2))
    eps2 = float(np.sum(w*delta**2))
    epsilon = np.sqrt(eps2)
    B = collapse.constant
    cross = float(np.sum(w*R*delta))
    gap = abs(L-L0)
    bound = 2*np.sqrt(L0)*epsilon+eps2
    floor = max(np.sqrt(B)-epsilon, 0)**2
    identity_error = abs(L0-collapse.loss_grad(commands, False))
    expansion_error = abs(L-L0-2*cross-eps2)
    assert identity_error < 1e-10
    assert expansion_error < 1e-10
    assert gap <= bound+1e-12
    assert abs(np.sqrt(L)-np.sqrt(L0)) <= epsilon+1e-12
    assert L >= floor-1e-12
    return dict(coupled_loss=L, fixed_calibration_loss=L0,
                irreducible_variance=B, epsilon=epsilon,
                observed_absolute_gap=gap, absolute_gap_bound=bound,
                gap_over_bound=gap/bound if bound else 0.,
                perturbed_loss_lower_bound=floor,
                collapse_identity_error=identity_error,
                perturbation_expansion_error=expansion_error)


def make_plot(rows, path):
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'pdf.fonttype': 42,
                         'ps.fonttype': 42})
    w = np.linspace(-np.pi, np.pi, 301)
    wx, wy = np.meshgrid(w, w)
    H5 = (1+4*np.cos(wx)*np.cos(wy))/5
    H9 = (1+2*np.cos(wx))*(1+2*np.cos(wy))/9
    fig = plt.figure(figsize=(11.2, 3.65), layout='constrained')
    gs = fig.add_gridspec(1, 4, width_ratios=(1, 1, .045, 1.2))
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1, 3)]
    cbax = fig.add_subplot(gs[0, 2])
    for ax, H, title in zip(axes[:2], (H5, H9), ('(a) Five-shift average', '(b) Nine-shift average')):
        im = ax.imshow(H, origin='lower', extent=(-1, 1, -1, 1),
                       vmin=-1, vmax=1, cmap='RdBu_r', interpolation='nearest')
        ax.set(title=title, xlabel=r'$\omega_x / \pi$', ylabel=r'$\omega_y / \pi$')
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    fig.colorbar(im, cax=cbax, label='Signed Fourier multiplier')
    ax = axes[2]
    allx, ally = [], []
    for method, color in zip(METHODS, COLORS):
        rr = [r for r in rows if r['method'] == method]
        x = [r['absolute_gap_bound'] for r in rr]
        y = [max(r['observed_absolute_gap'], 1e-16) for r in rr]
        allx.extend(x); ally.extend(y)
        ax.scatter(x, y, s=25, color=color, label=method, alpha=.85,
                   edgecolor='white', linewidth=.3)
    lo = max(min(allx+ally)*.55, 1e-12)
    hi = max(allx+ally)*1.6
    ax.plot([lo, hi], [lo, hi], '--', color='#333333', linewidth=.8)
    ax.set(xscale='log', yscale='log', xlim=(lo, hi), ylim=(lo, hi),
           title='(c) Command-specific loss discrepancy',
           xlabel=r'Bound $2\sqrt{L_0}\,\varepsilon+\varepsilon^2$',
           ylabel=r'Observed $|L-L_0|$')
    ax.grid(alpha=.16, which='both')
    ax.legend(loc='upper left', fontsize=7, framealpha=.9)
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results_reconstructed'))
    args = parser.parse_args()
    cfg = json.loads((args.results/'config.json').read_text())
    p = Problem(cfg)
    scenarios = design_scenarios(cfg)
    rows = []
    for file in sorted(args.results.glob('commands_seed*.npz')):
        seed = int(re.search(r'seed(\d+)', file.stem).group(1))
        with np.load(file) as archive:
            for method in METHODS:
                row = discrepancy(p, archive[method], scenarios)
                rows.append(dict(seed=seed, method=method, **row))
    if not rows:
        raise FileNotFoundError('No saved command archives found; run run_study.py first.')
    out = args.results/'diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'calibration_discrepancy.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    method_means = {}
    for method in METHODS:
        rr = [r for r in rows if r['method'] == method]
        method_means[method] = {k: float(np.mean([r[k] for r in rr]))
                                for k in rows[0] if k not in ('seed', 'method')}
    summary = dict(
        provenance='revision diagnostic on independently reconstructed model; original raw commands unavailable',
        calibration='five coupled design scenarios versus fixed nominal calibration at same commands',
        quantization='continuous saved optimization commands; no 8-bit rounding',
        scope='training objective only; command-specific bounds, not global physical robustness certificates',
        source_script='code/calibration_discrepancy.py',
        command_records=len(rows), seeds=sorted(set(r['seed'] for r in rows)),
        max_collapse_identity_error=max(r['collapse_identity_error'] for r in rows),
        max_expansion_error=max(r['perturbation_expansion_error'] for r in rows),
        max_gap_over_bound=max(r['gap_over_bound'] for r in rows),
        H5_at_pi_pi=1.0, H9_at_pi_pi=1/9,
        method_means=method_means, all_bounds_pass=True,
    )
    (out/'calibration_discrepancy_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    make_plot(rows, out/'calibration_discrepancy.pdf')
    print(json.dumps({k: summary[k] for k in ('command_records', 'max_collapse_identity_error',
                                           'max_expansion_error', 'max_gap_over_bound', 'all_bounds_pass')}, indent=2))


if __name__ == '__main__':
    main()
