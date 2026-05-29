"""BSETD Stage 2: symmetrized graph Laplacian spectral decomposition.

Given a K x K posterior-mean emotion transition matrix from Stage 1,
construct a directed weighted graph over the K emotion categories,
symmetrize it as A_sym = (A + A^T) / 2 (Chung 2005), and compute the
normalized Laplacian L = I - D^{-1/2} A_sym D^{-1/2}. Spectral
decomposition L = U Lambda U^T then yields K orthonormal modes
ordered by eigenvalue.

Following the Kuppens-Hatfield correspondence, we partition the modes
into a low-frequency band (lambda <= median(lambda), inertia-dominant)
and a high-frequency band (lambda > median, contagion/shift-dominant),
and reconstruct band-limited transition matrices:

    A^{(low)}  = P_lo  A_sym  P_lo   with  P_lo  = U_lo  U_lo^T
    A^{(high)} = P_hi  A_sym  P_hi   with  P_hi  = U_hi  U_hi^T

These two matrices feed the hero figure (Chord = low, Sankey = high).

References
----------
- Chung (2005). Laplacians and the Cheeger Inequality for Directed Graphs.
- Shuman et al. (2013). The Emerging Field of Signal Processing on Graphs.
- Meng et al. (AAAI 2025). GS-MCC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


@dataclass
class SpectralResult:
    """Stage 2 output for one input transition matrix."""
    A: np.ndarray  # (K, K) symmetrized adjacency
    L: np.ndarray  # (K, K) normalized Laplacian
    eigvals: np.ndarray  # (K,)
    eigvecs: np.ndarray  # (K, K), columns are eigenvectors
    low_mask: np.ndarray  # (K,) bool, True for low-frequency modes
    high_mask: np.ndarray
    A_low: np.ndarray  # (K, K) low-frequency reconstruction
    A_high: np.ndarray  # (K, K) high-frequency reconstruction
    inertia_index: np.ndarray  # (K,) per-emotion inertia score (diagonal of A_low)
    contagion_index: np.ndarray  # (K,) per-emotion contagion score (off-diagonal mass in A_high)
    metadata: dict = field(default_factory=dict)


def symmetrize_adjacency(A: np.ndarray) -> np.ndarray:
    """A_sym = (A + A^T) / 2. Removes loop self-bias via diagonal preservation."""
    return 0.5 * (A + A.T)


def normalized_laplacian(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L = I - D^{-1/2} A D^{-1/2}."""
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, eps))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    # Force exact symmetry (numerical hygiene)
    L = 0.5 * (L + L.T)
    return L


def split_low_high(eigvals: np.ndarray, mode: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """Return (low_mask, high_mask) over eigenvalue indices.

    mode = 'median' splits at the median eigenvalue.
    mode = 'half'  splits the K modes into the lowest floor(K/2) vs the rest.
    """
    K = eigvals.size
    if mode == "median":
        thr = float(np.median(eigvals))
        low = eigvals <= thr
        high = ~low
    elif mode == "half":
        low = np.zeros(K, dtype=bool)
        low[: K // 2] = True
        high = ~low
    else:
        raise ValueError(f"unknown split mode: {mode}")
    return low, high


def reconstruct_band(
    A_sym: np.ndarray, eigvecs: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Project A_sym onto the subspace spanned by selected eigenvectors."""
    U_b = eigvecs[:, mask]  # (K, k_b)
    P = U_b @ U_b.T  # (K, K)
    return P @ A_sym @ P


def spectral_decompose(
    A: np.ndarray, split_mode: str = "median"
) -> SpectralResult:
    """Run Stage 2 on a K x K (possibly directed) transition matrix."""
    A_sym = symmetrize_adjacency(A)
    L = normalized_laplacian(A_sym)
    eigvals, eigvecs = eigh(L)  # ascending
    low_mask, high_mask = split_low_high(eigvals, mode=split_mode)
    A_low = reconstruct_band(A_sym, eigvecs, low_mask)
    A_high = reconstruct_band(A_sym, eigvecs, high_mask)

    inertia_index = np.diag(A_low)  # diagonal mass after low-pass
    contagion_mat = A_high - np.diag(np.diag(A_high))
    contagion_index = np.abs(contagion_mat).sum(axis=1)

    return SpectralResult(
        A=A_sym,
        L=L,
        eigvals=eigvals,
        eigvecs=eigvecs,
        low_mask=low_mask,
        high_mask=high_mask,
        A_low=A_low,
        A_high=A_high,
        inertia_index=inertia_index,
        contagion_index=contagion_index,
        metadata={
            'split_mode': split_mode,
            'n_low': int(low_mask.sum()),
            'n_high': int(high_mask.sum()),
            'eigval_min': float(eigvals.min()),
            'eigval_max': float(eigvals.max()),
            'eigval_median': float(np.median(eigvals)),
        },
    )


def run_stage2_on_stage1_npz(
    stage1_npz: str | Path,
    out_dir: str | Path,
    split_mode: str = "median",
) -> dict:
    """End-to-end Stage 2 on a Stage 1 output .npz file."""
    stage1_npz = Path(stage1_npz)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(stage1_npz)
    A = data['transition_post_mean']
    res = spectral_decompose(A, split_mode=split_mode)
    np.savez(
        out_dir / f"{stage1_npz.stem}_stage2.npz",
        A_sym=res.A,
        L=res.L,
        eigvals=res.eigvals,
        eigvecs=res.eigvecs,
        low_mask=res.low_mask,
        high_mask=res.high_mask,
        A_low=res.A_low,
        A_high=res.A_high,
        inertia_index=res.inertia_index,
        contagion_index=res.contagion_index,
    )
    summary = {
        'source_npz': str(stage1_npz),
        **res.metadata,
        'inertia_index': res.inertia_index.tolist(),
        'contagion_index': res.contagion_index.tolist(),
    }
    import json
    with open(out_dir / f"{stage1_npz.stem}_stage2_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1-npz",
        default="experiments/stage1_emotionlines/stage1_total.npz",
    )
    parser.add_argument("--out", default="experiments/stage2_emotionlines/")
    parser.add_argument("--split-mode", default="median",
                        choices=["median", "half"])
    args = parser.parse_args()
    summary = run_stage2_on_stage1_npz(
        args.stage1_npz, args.out, split_mode=args.split_mode
    )
    print(json.dumps(summary, indent=2))
