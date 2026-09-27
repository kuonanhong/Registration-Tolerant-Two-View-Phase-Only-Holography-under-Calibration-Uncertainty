#!/usr/bin/env python3
"""Independent algebra and derivative tests for the mathematical extension.

These are numerical consistency tests, not a rerun of the manuscript's six-seed
experiment.  All test inputs are synthetic and fixed by the seed below.  Only
NumPy is required.  Run from any directory with Python 3:
    python verify_math_extension.py --output math_extension_validation.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


SHIFTS = ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))
SCENARIOS = ((0, 0, 1.15, .08), (-1, -1, 1.07, .05),
             (-1, 1, 1.23, .05), (1, -1, 1.07, .11),
             (1, 1, 1.23, .11))


def shift(a, dx, dy):
    return np.roll(a, (dy, dx), axis=(-2, -1))


def coupling(g, b):
    return (1-b)*g + b/4*sum(shift(g, dx, dy) for dx, dy in
                             ((1, 0), (-1, 0), (0, 1), (0, -1)))


def forward(g, gamma, b, aberration, eta=.86):
    h = coupling(g, b)
    u = np.sqrt(eta)/g.shape[-1]*np.exp(1j*(2*np.pi*h**gamma+aberration))
    U = np.fft.fftshift(np.fft.fft2(u, norm='ortho'), axes=(-2, -1))
    return np.abs(U)**2, u, U, h


def inverse(U):
    return np.fft.ifft2(np.fft.ifftshift(U, axes=(-2, -1)), norm='ortho')


def test_problem(n=32, seed=20260927):
    rng = np.random.default_rng(seed)
    # Strictly interior commands avoid projection/clipping nondifferentiability.
    g = rng.uniform(.1, .9, (2, n, n))
    masks = np.zeros_like(g)
    m = n//4
    for j, cx in enumerate((n//4, 3*n//4)):
        masks[j, n//2-m//2:n//2+m//2, cx-m//2:cx+m//2] = 1
    targets = masks*rng.uniform(.1, 1, g.shape)
    targets /= targets.sum(axis=(-2, -1), keepdims=True)
    T0 = .60*targets
    d = (T0*T0).sum(axis=(-2, -1))
    x = (np.arange(n)+.5)*2/n-1
    xx, yy = np.meshgrid(x, x)
    aberration = .32*(xx*xx-yy*yy)+.18*xx*yy
    Ts, ws = [], []
    for dx, dy in SHIFTS:
        M = shift(masks, dx, dy)
        Ts.append(shift(T0, dx, dy))
        ws.append(.05+.95*M+7.95*M[::-1])
    return rng, g, aberration, np.array(Ts), np.array(ws), d


def losses_and_gradient(g, a, T, w, d):
    nsc = len(SCENARIOS)
    I0, *_ = forward(g, 1.15, .08, a)
    norm_weights = w/(2*nsc*d[None, :, None, None])
    intensities, gradient = [], np.zeros_like(g)
    for idx, (_, _, gamma, b) in enumerate(SCENARIOS):
        I, u, U, h = forward(g, gamma, b, a)
        intensities.append(I)
        v = w[idx]*(I-T[idx])/d[:, None, None]
        z = 2*np.imag(np.conj(u)*inverse(v*U))
        gradient += coupling(2*np.pi*gamma*h**(gamma-1)*z, b)/nsc
    intensities = np.array(intensities)
    R, delta = I0[None]-T, intensities-I0[None]
    L = float((norm_weights*(R+delta)**2).sum())
    L0 = float((norm_weights*R**2).sum())
    eps2 = float((norm_weights*delta**2).sum())
    cross = float((norm_weights*R*delta).sum())
    A = w.sum(axis=0)
    Tbar = (w*T).sum(axis=0)/A
    B = float((norm_weights*(T-Tbar[None])**2).sum())
    fit = float((A*(I0-Tbar)**2/(2*nsc*d[:, None, None])).sum())
    return dict(L=L, L0=L0, epsilon=np.sqrt(eps2), epsilon_squared=eps2,
                cross_inner_product=cross, collapsed_fit=fit,
                irreducible_variance=B), gradient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='math_extension_validation.json')
    args = ap.parse_args()
    rng, g, a, T, w, d = test_problem()
    vals, grad = losses_and_gradient(g, a, T, w, d)
    L, L0, eps, B = (vals[k] for k in ('L', 'L0', 'epsilon', 'irreducible_variance'))
    gap = abs(L-L0)
    gap_bound = 2*np.sqrt(L0)*eps+eps**2
    assert gap <= gap_bound+1e-12
    assert abs(np.sqrt(L)-np.sqrt(L0)) <= eps+1e-12
    assert L >= max(np.sqrt(B)-eps, 0)**2-1e-12
    identity_resid = abs(L0-(vals['collapsed_fit']+B))
    expansion_resid = abs(L-(L0+2*vals['cross_inner_product']+eps**2))
    direction = rng.normal(size=g.shape)
    direction /= np.linalg.norm(direction)
    step = 1e-5
    p = losses_and_gradient(g+step*direction, a, T, w, d)[0]['L']
    m = losses_and_gradient(g-step*direction, a, T, w, d)[0]['L']
    fd = (p-m)/(2*step)
    analytic = float((grad*direction).sum())
    relerr = abs(fd-analytic)/max(abs(fd), abs(analytic), 1e-14)
    assert relerr < 1e-6
    # Exhaustively verify the shift multiplier at every discrete Fourier mode.
    n = g.shape[-1]
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    max_spectral_error = 0.
    for kx in range(n):
        for ky in range(n):
            wx, wy = 2*np.pi*kx/n, 2*np.pi*ky/n
            mode = np.exp(1j*(wx*xx+wy*yy))
            avg = sum(shift(mode, dx, dy) for dx, dy in SHIFTS)/5
            H = (1+4*np.cos(wx)*np.cos(wy))/5
            max_spectral_error = max(max_spectral_error, float(np.max(np.abs(avg-H*mode))))
    assert max_spectral_error < 1e-12
    # Check many random command arrays, not only the derivative-test instance.
    max_bound_ratio, min_floor_margin = 0., float('inf')
    for _ in range(25):
        v, _ = losses_and_gradient(rng.uniform(0, 1, g.shape), a, T, w, d)
        bound = 2*np.sqrt(v['L0'])*v['epsilon']+v['epsilon_squared']
        max_bound_ratio = max(max_bound_ratio, abs(v['L']-v['L0'])/bound)
        floor = max(np.sqrt(v['irreducible_variance'])-v['epsilon'], 0)**2
        min_floor_margin = min(min_floor_margin, v['L']-floor)
        assert abs(v['L']-v['L0']) <= bound+1e-12
        assert v['L'] >= floor-1e-12
    out = {
        'kind': 'independent synthetic mathematical consistency tests; not manuscript reproduction',
        'test_seed': 20260927, 'test_grid': 32, 'numpy_version': np.__version__,
        'reference_command': vals,
        'objective_gap': gap, 'objective_gap_bound': float(gap_bound),
        'collapse_identity_absolute_error': identity_resid,
        'perturbation_expansion_absolute_error': expansion_resid,
        'gradient_directional_fd': fd, 'gradient_directional_analytic': analytic,
        'gradient_directional_relative_error': relerr,
        'all_1024_fourier_modes_max_error': max_spectral_error,
        'H5_at_zero_zero': 1., 'H5_at_pi_zero': -.6, 'H5_at_pi_pi': 1.,
        'additional_random_commands': 25,
        'max_objective_gap_over_bound': max_bound_ratio,
        'minimum_perturbed_floor_margin': min_floor_margin,
        'all_tests_pass': True,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
