"""BSETD Stage 1: Hierarchical Dirichlet-Multinomial empirical Bayes.

Estimates per-source-target K x K soft-label emotion transition matrices
under a Dirichlet prior whose concentration alpha is fit by empirical
Bayes (Minka 2000 fixed-point iteration). Provides:

    - soft-label transition count construction via outer products
      of consecutive utterance probability vectors
    - inertia (same-speaker) vs contagion (cross-speaker) decomposition
    - posterior credible intervals for each P(k | j) cell
    - BH-FDR rejection of the uniform-null hypothesis P(k|j) = 1/K

References
----------
- Minka, T. (2000). Estimating a Dirichlet distribution. MIT TR.
- Kruschke, J. (2018). Rejecting or Accepting Parameter Values in
  Bayesian Estimation. AMPPS.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the FDR. JRSS-B.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln
from scipy.stats import beta

EMOTIONS = ['neutral', 'joy', 'sadness', 'fear', 'anger', 'surprise', 'disgust']
K = 7


@dataclass
class TransitionCounts:
    """K x K transition counts plus optional inertia/contagion split."""
    total: np.ndarray  # (K, K), real-valued soft counts
    inertia: np.ndarray  # (K, K), same-speaker transitions
    contagion: np.ndarray  # (K, K), cross-speaker transitions
    n_transitions: int  # number of utterance pairs aggregated


@dataclass
class DirichletMultinomialFit:
    """Empirical-Bayes fit summary for one source row j of the transition matrix."""
    alpha: np.ndarray  # shape (K,), fitted Dirichlet concentration
    post_mean: np.ndarray  # shape (K,), E[P(k|j)] under posterior
    post_var: np.ndarray  # shape (K,), Var[P(k|j)]
    hdi_low: np.ndarray  # shape (K,)
    hdi_high: np.ndarray  # shape (K,)
    n_observations: float  # sum of row counts (real-valued for soft labels)
    n_iter: int = 0
    converged: bool = False


@dataclass
class Stage1Result:
    """Full BSETD Stage 1 output."""
    transition_post_mean: np.ndarray  # (K, K)
    transition_post_var: np.ndarray  # (K, K)
    hdi_low: np.ndarray  # (K, K)
    hdi_high: np.ndarray  # (K, K)
    alpha: np.ndarray  # (K, K) fitted Dirichlet alpha per-row
    p_value: np.ndarray  # (K, K) one-sided posterior tail prob versus uniform 1/K
    rejected_bh: np.ndarray  # (K, K) bool, BH-FDR rejection
    counts: TransitionCounts = field(default=None)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transition count construction (soft labels via outer product)
# ---------------------------------------------------------------------------


def soft_transition_counts(
    p_list: list[np.ndarray],
    speaker_ids: list[str] | None = None,
) -> TransitionCounts:
    """Aggregate K x K soft transition counts from one ordered utterance sequence.

    For each adjacent pair (p_t, p_{t+1}) we add p_t[j] * p_{t+1}[k] to
    cell (j, k). When speaker_ids is provided we additionally partition
    each pair into inertia (same speaker) or contagion (different speaker).
    """
    total = np.zeros((K, K))
    inertia = np.zeros((K, K))
    contagion = np.zeros((K, K))
    n_pairs = 0
    for i in range(len(p_list) - 1):
        pt = np.asarray(p_list[i], dtype=float)
        pt1 = np.asarray(p_list[i + 1], dtype=float)
        if pt.shape != (K,) or pt1.shape != (K,):
            continue
        outer = np.outer(pt, pt1)
        total += outer
        if speaker_ids is not None:
            if speaker_ids[i] == speaker_ids[i + 1]:
                inertia += outer
            else:
                contagion += outer
        n_pairs += 1
    return TransitionCounts(
        total=total, inertia=inertia, contagion=contagion, n_transitions=n_pairs
    )


def aggregate_corpus_transitions(
    df: pd.DataFrame,
    dialog_col: str = 'dialog_id',
    turn_col: str = 'turn_id',
    p_col: str = 'p_dist',
    speaker_col: str = 'speaker_id',
) -> TransitionCounts:
    """Aggregate soft transitions across all dialogs in df."""
    total = np.zeros((K, K))
    inertia = np.zeros((K, K))
    contagion = np.zeros((K, K))
    n_pairs = 0
    for _, sub in df.sort_values([dialog_col, turn_col]).groupby(dialog_col):
        p_list = list(sub[p_col].apply(np.asarray).to_numpy())
        speakers = (
            list(sub[speaker_col].to_numpy()) if speaker_col in sub.columns else None
        )
        tc = soft_transition_counts(p_list, speaker_ids=speakers)
        total += tc.total
        inertia += tc.inertia
        contagion += tc.contagion
        n_pairs += tc.n_transitions
    return TransitionCounts(
        total=total, inertia=inertia, contagion=contagion, n_transitions=n_pairs
    )


# ---------------------------------------------------------------------------
# Empirical Bayes Dirichlet concentration fit (Minka 2000 fixed point)
# ---------------------------------------------------------------------------


def _fit_dirichlet_row(
    counts_row: np.ndarray,
    alpha_init: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> DirichletMultinomialFit:
    """Fit a single Dirichlet posterior alpha + counts_row using EB.

    For a single source j, we treat the K-vector counts_row as the
    soft-count of transitions to each target k. The posterior is
    Dir(alpha + counts_row). We use the row-conditional empirical
    Bayes shortcut: alpha is jointly tuned across all rows via the
    higher-level fit_transition_matrix function below; here we
    take alpha as given and compute posterior summaries.
    """
    counts_row = np.asarray(counts_row, dtype=float)
    alpha = np.full(K, alpha_init, dtype=float)
    post = alpha + counts_row
    s = float(post.sum())
    post_mean = post / s
    post_var = post_mean * (1.0 - post_mean) / (s + 1.0)
    hdi_lo = beta.ppf(0.025, post, s - post)
    hdi_hi = beta.ppf(0.975, post, s - post)
    return DirichletMultinomialFit(
        alpha=alpha,
        post_mean=post_mean,
        post_var=post_var,
        hdi_low=hdi_lo,
        hdi_high=hdi_hi,
        n_observations=float(counts_row.sum()),
        n_iter=0,
        converged=True,
    )


def _eb_alpha_global(
    counts_matrix: np.ndarray,
    max_iter: int = 5000,
    tol: float = 1e-5,
) -> tuple[np.ndarray, int, bool]:
    """Joint empirical-Bayes fit of a single Dirichlet concentration vector
    alpha (length K) shared across all K source rows of counts_matrix.

    Uses Minka (2000) fixed-point iteration on the marginal log-likelihood
    of the Dirichlet-Multinomial:

        L(alpha) = sum_j [log Gamma(sum alpha) - log Gamma(sum alpha + N_j)
                          + sum_k (log Gamma(alpha_k + n_{j,k}) - log Gamma(alpha_k))]

    Each row j contributes one Dirichlet-Multinomial observation with
    its row-sum N_j and row counts n_{j,*}.
    """
    counts_matrix = np.asarray(counts_matrix, dtype=float)
    n_rows = counts_matrix.shape[0]
    row_sums = counts_matrix.sum(axis=1)  # (K,)

    # Initialize alpha by method-of-moments
    p_bar = counts_matrix.sum(axis=0) / max(counts_matrix.sum(), 1.0)
    p_bar = np.where(p_bar > 0, p_bar, 1.0 / K)
    alpha = np.maximum(p_bar, 1e-3)  # small floor

    converged = False
    for it in range(1, max_iter + 1):
        alpha_sum = alpha.sum()
        # Minka fixed-point update:
        # alpha_k <- alpha_k * (sum_j [digamma(n_{j,k} + alpha_k) - digamma(alpha_k)])
        #                   / (sum_j [digamma(N_j + alpha_sum) - digamma(alpha_sum)])
        numer = np.sum(
            digamma(counts_matrix + alpha[None, :]) - digamma(alpha)[None, :], axis=0
        )
        denom = np.sum(digamma(row_sums + alpha_sum) - digamma(alpha_sum))
        if denom <= 0:
            break
        new_alpha = alpha * numer / denom
        new_alpha = np.maximum(new_alpha, 1e-6)
        delta = np.max(np.abs(new_alpha - alpha))
        alpha = new_alpha
        if delta < tol:
            converged = True
            break
    return alpha, it, converged


def fit_transition_matrix(
    counts: TransitionCounts,
    alpha_init_scale: float = 1.0,
) -> Stage1Result:
    """Run BSETD Stage 1 on aggregated transition counts.

    Returns posterior means, variances, 95% HDIs, BH-FDR rejection,
    and the fitted Dirichlet alpha vector.
    """
    counts_matrix = counts.total.astype(float)

    alpha_vec, n_iter, converged = _eb_alpha_global(counts_matrix)
    alpha_vec = alpha_vec * alpha_init_scale  # scale knob (Ablation 3)

    alpha_full = np.tile(alpha_vec[None, :], (K, 1))  # (K, K)
    post = alpha_full + counts_matrix
    row_post_sum = post.sum(axis=1, keepdims=True)
    post_mean = post / np.maximum(row_post_sum, 1e-12)
    post_var = post_mean * (1.0 - post_mean) / (row_post_sum + 1.0)

    a = post
    b = row_post_sum - post
    hdi_lo = beta.ppf(0.025, a, np.maximum(b, 1e-12))
    hdi_hi = beta.ppf(0.975, a, np.maximum(b, 1e-12))

    # Null hypothesis: P(k|j) = P(k), i.e. the next-emotion distribution
    # is independent of the current emotion. The marginal P(k) is the
    # column-sum proportion of the observed (smoothed) count matrix.
    smoothed = counts_matrix + alpha_full
    col_sums = smoothed.sum(axis=0)
    grand = smoothed.sum()
    marginal = col_sums / np.maximum(grand, 1e-12)  # (K,)

    # Two-sided posterior tail probability against the marginal.
    # p_two[j, k] = 2 * min( P(P(k|j) <= marginal_k | data),
    #                        P(P(k|j) >  marginal_k | data) )
    lower = beta.cdf(marginal[None, :], a, np.maximum(b, 1e-12))
    p_two_sided = 2.0 * np.minimum(lower, 1.0 - lower)
    # Use np.errstate to avoid divide-by-zero warnings on extreme tails.
    p_two_sided = np.clip(p_two_sided, 1e-300, 1.0)

    # log p-values for numerical stability when the BH threshold itself is tiny.
    log_p = np.log(p_two_sided)

    # BH-FDR over off-diagonal cells (we test K*K - K hypotheses)
    mask_off = ~np.eye(K, dtype=bool)
    rejected = _bh_fdr_mask(p_two_sided, mask=mask_off, q=0.10)

    return Stage1Result(
        transition_post_mean=post_mean,
        transition_post_var=post_var,
        hdi_low=hdi_lo,
        hdi_high=hdi_hi,
        alpha=alpha_full,
        p_value=p_two_sided,
        rejected_bh=rejected,
        counts=counts,
        metadata={
            'alpha_global': alpha_vec.tolist(),
            'eb_iter': int(n_iter),
            'eb_converged': bool(converged),
            'n_transitions': int(counts.n_transitions),
            'K': K,
            'emotions': EMOTIONS,
            'marginal': marginal.tolist(),
            'log_p_min': float(log_p.min()),
            'log_p_max': float(log_p.max()),
            'null_hypothesis': 'P(k|j) = P(k) (independence of next-emotion from current)',
        },
    )


def _bh_fdr_mask(p: np.ndarray, mask: np.ndarray, q: float = 0.10) -> np.ndarray:
    """BH-FDR over entries selected by mask; returns boolean rejection mask of p's shape."""
    p_flat = p[mask]
    n = p_flat.size
    if n == 0:
        return np.zeros_like(p, dtype=bool)
    order = np.argsort(p_flat)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    thresholds = q * ranks / n
    rejected_flat = p_flat <= thresholds
    # Step-up: find largest k such that p_{(k)} <= q*k/n, reject all up to that k
    sorted_p = p_flat[order]
    sorted_thresh = q * np.arange(1, n + 1) / n
    accept_idx = np.where(sorted_p <= sorted_thresh)[0]
    if accept_idx.size > 0:
        max_k = accept_idx.max()
        sorted_rejected = np.zeros(n, dtype=bool)
        sorted_rejected[: max_k + 1] = True
        rejected_flat = np.zeros(n, dtype=bool)
        rejected_flat[order] = sorted_rejected
    else:
        rejected_flat = np.zeros(n, dtype=bool)
    out = np.zeros_like(p, dtype=bool)
    out[mask] = rejected_flat
    return out


# ---------------------------------------------------------------------------
# Inertia / contagion separated fits
# ---------------------------------------------------------------------------


def fit_inertia_contagion(counts: TransitionCounts) -> dict[str, Stage1Result]:
    """Fit Stage 1 separately on inertia (same-speaker) and contagion (cross-speaker)."""
    out = {}
    out['total'] = fit_transition_matrix(counts)
    inertia_counts = TransitionCounts(
        total=counts.inertia,
        inertia=counts.inertia,
        contagion=np.zeros_like(counts.inertia),
        n_transitions=int(counts.inertia.sum()),
    )
    contagion_counts = TransitionCounts(
        total=counts.contagion,
        inertia=np.zeros_like(counts.contagion),
        contagion=counts.contagion,
        n_transitions=int(counts.contagion.sum()),
    )
    if counts.inertia.sum() > 0:
        out['inertia'] = fit_transition_matrix(inertia_counts)
    if counts.contagion.sum() > 0:
        out['contagion'] = fit_transition_matrix(contagion_counts)
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_stage1_on_emotionlines(
    parquet_path: str | Path,
    out_dir: str | Path,
    bh_q: float = 0.10,
) -> dict:
    """End-to-end Stage 1 on EmotionLines v2 BSETD parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet_path)
    counts = aggregate_corpus_transitions(df)
    fits = fit_inertia_contagion(counts)
    summary = {}
    for key, res in fits.items():
        np.savez(
            out_dir / f'stage1_{key}.npz',
            transition_post_mean=res.transition_post_mean,
            transition_post_var=res.transition_post_var,
            hdi_low=res.hdi_low,
            hdi_high=res.hdi_high,
            alpha=res.alpha,
            p_value=res.p_value,
            rejected_bh=res.rejected_bh,
            counts_total=counts.total,
            counts_inertia=counts.inertia,
            counts_contagion=counts.contagion,
        )
        summary[key] = {
            'n_transitions': res.metadata['n_transitions'],
            'alpha_global': res.metadata['alpha_global'],
            'eb_converged': res.metadata['eb_converged'],
            'n_rejected_offdiag': int(res.rejected_bh.sum()),
            'top_edges': _top_edges(res, k=10),
        }
    import json
    json.dump(summary, open(out_dir / 'stage1_summary.json', 'w'), indent=2)
    return summary


def _top_edges(res: Stage1Result, k: int = 10) -> list[dict]:
    marginal = np.asarray(res.metadata.get('marginal', np.full(K, 1.0 / K)))
    mask = ~np.eye(K, dtype=bool)
    edges = []
    for j in range(K):
        for kk in range(K):
            if not mask[j, kk]:
                continue
            pm = float(res.transition_post_mean[j, kk])
            mk = float(marginal[kk])
            lift = pm / max(mk, 1e-12)
            edges.append({
                'src': EMOTIONS[j],
                'tgt': EMOTIONS[kk],
                'post_mean': pm,
                'marginal_tgt': mk,
                'lift': lift,
                'log2_lift': float(np.log2(max(lift, 1e-12))),
                'hdi': [float(res.hdi_low[j, kk]), float(res.hdi_high[j, kk])],
                'p_value': float(res.p_value[j, kk]),
                'rejected_bh': bool(res.rejected_bh[j, kk]),
            })
    # Rank by absolute log2 lift (largest deviation from independence)
    edges.sort(key=lambda e: abs(e['log2_lift']), reverse=True)
    return edges[:k]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default='data_processed/emotionlines_softlabels_v2_bsetd.parquet',
    )
    parser.add_argument('--out', default='experiments/stage1_emotionlines/')
    parser.add_argument('--bh-q', type=float, default=0.10)
    args = parser.parse_args()
    summary = run_stage1_on_emotionlines(args.input, args.out, bh_q=args.bh_q)
    import json
    print(json.dumps(summary, indent=2))
