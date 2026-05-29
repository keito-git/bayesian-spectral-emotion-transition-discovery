"""Directional spectral analysis: BSETD Stage 2 without symmetrization.

The base Stage 2 uses A_sym = (T + T^T) / 2, which discards directionality.
A directional alternative is the Chung directed-graph random-walk Laplacian:

    L_chung = I - (1/2)(P + Pi^{-1} P^T Pi)

where P is the row-stochastic transition matrix and Pi is the diagonal of
the Perron stationary distribution. This Laplacian is symmetric and admits
real eigenvalues while preserving directional information through the
weighting by Pi.

We provide both decompositions and compare which one is more informative
for separating inertia from contagion.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

import sys
ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from bsetd.stage2_spectral import (
    spectral_decompose, reconstruct_band, split_low_high
)


def stationary_distribution(P: np.ndarray, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Compute Perron stationary distribution of row-stochastic P via power method."""
    K = P.shape[0]
    pi = np.full(K, 1.0 / K)
    for _ in range(max_iter):
        new_pi = pi @ P
        new_pi = new_pi / new_pi.sum()
        if np.linalg.norm(new_pi - pi, ord=1) < tol:
            break
        pi = new_pi
    return pi


def chung_directed_laplacian(P: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Chung (2005) Laplacian for directed graphs."""
    pi = stationary_distribution(P)
    pi = np.maximum(pi, eps)
    Pi = np.diag(pi)
    Pi_inv = np.diag(1.0 / pi)
    L = np.eye(P.shape[0]) - 0.5 * (P + Pi_inv @ P.T @ Pi)
    L_sym = 0.5 * (L + L.T)  # numerical symmetrization (should be near-symmetric already)
    return L_sym, pi


def directional_decompose(P: np.ndarray, split_mode: str = "median") -> dict:
    L, pi = chung_directed_laplacian(P)
    eigvals, eigvecs = eigh(L)
    low, high = split_low_high(eigvals, mode=split_mode)
    A = 0.5 * (P + P.T)  # used only for band reconstruction comparison
    A_low = reconstruct_band(A, eigvecs, low)
    A_high = reconstruct_band(A, eigvecs, high)
    inertia = np.diag(A_low)
    contagion_mat = A_high - np.diag(np.diag(A_high))
    contagion = np.abs(contagion_mat).sum(axis=1)
    return {
        'pi': pi.tolist(),
        'eigvals': eigvals.tolist(),
        'inertia_index_directional': inertia.tolist(),
        'contagion_index_directional': contagion.tolist(),
    }


def main() -> None:
    s1 = np.load(ROOT / 'experiments' / 'stage1_emotionlines' / 'stage1_total.npz')
    P = s1['transition_post_mean']  # row-stochastic

    sym_res = spectral_decompose(P)
    dir_res = directional_decompose(P)

    out = {
        'symmetrized': {
            'inertia_index': sym_res.inertia_index.tolist(),
            'contagion_index': sym_res.contagion_index.tolist(),
            'eigvals': sym_res.eigvals.tolist(),
        },
        'directional_chung': dir_res,
    }
    sym_in = np.asarray(sym_res.inertia_index)
    dir_in = np.asarray(dir_res['inertia_index_directional'])
    rho_in = float(np.corrcoef(sym_in, dir_in)[0, 1])
    sym_co = np.asarray(sym_res.contagion_index)
    dir_co = np.asarray(dir_res['contagion_index_directional'])
    rho_co = float(np.corrcoef(sym_co, dir_co)[0, 1])
    out['rho_inertia_sym_vs_directional'] = rho_in
    out['rho_contagion_sym_vs_directional'] = rho_co

    out_path = ROOT / 'experiments' / 'stage2_emotionlines' / 'emotionlines' / 'directional_spectral.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({
        'sym_inertia_index': sym_res.inertia_index.round(3).tolist(),
        'dir_inertia_index': np.round(dir_res['inertia_index_directional'], 3).tolist(),
        'rho_inertia_sym_vs_directional': rho_in,
        'rho_contagion_sym_vs_directional': rho_co,
        'stationary_pi': np.round(np.asarray(dir_res['pi']), 3).tolist(),
    }, indent=2))


if __name__ == '__main__':
    main()
