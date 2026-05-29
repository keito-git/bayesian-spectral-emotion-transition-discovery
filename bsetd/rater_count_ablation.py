"""Rater-count sensitivity ablation on EmotionLines.

EmotionLines provides exactly five ratings per utterance. We subsample the
vote vector to R in {2, 3, 4, 5} raters per utterance (random subset of the
five votes, repeated over multiple seeds), rebuild the soft labels, and
re-run BSETD Stage 1. We report:

    - Pearson correlation of off-diagonal log2 lift versus the full-R=5 version
    - mean absolute change in self-loop posterior means
    - mean absolute change in Inertia-minus-Contagion differentials

The goal is to characterize how many raters are required to obtain the
structure BSETD identifies at the population level.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsetd.stage1_dirichlet import (
    EMOTIONS, K, TransitionCounts,
    soft_transition_counts, fit_transition_matrix, fit_inertia_contagion,
)

SEEDS = [0, 1, 2, 3, 4]
RATER_LEVELS = [2, 3, 4, 5]


def subsample_softlabel(votes: list[int], R: int, rng: np.random.Generator) -> np.ndarray:
    """votes is a list of 5 emotion-index votes (one per rater).
    Return a K-dim soft-label vector after subsampling R raters."""
    if R >= len(votes):
        sel = votes
    else:
        sel_idx = rng.choice(len(votes), R, replace=False)
        sel = [votes[i] for i in sel_idx]
    counts = np.zeros(K)
    for v in sel:
        counts[v] += 1
    return counts / counts.sum()


def expand_vote_counts(p_dist: np.ndarray, n_raters: int) -> list[int]:
    """Reverse the soft-label aggregation: recover the multiset of individual votes.
    Given a probability vector p and the number of raters R, return the
    list of length R of emotion indices, by rounding p * R.
    """
    counts = (np.asarray(p_dist) * n_raters).round().astype(int)
    diff = n_raters - counts.sum()
    if diff != 0:
        # Adjust the largest bin to compensate for rounding error
        idx = int(np.argmax(counts))
        counts[idx] += diff
    return [k for k, c in enumerate(counts) for _ in range(int(c))]


def aggregate_with_R(df: pd.DataFrame, R: int, seed: int) -> TransitionCounts:
    rng = np.random.default_rng(seed)
    total = np.zeros((K, K))
    inertia = np.zeros((K, K))
    contagion = np.zeros((K, K))
    n_pairs = 0
    for _, sub in df.sort_values(['dialog_id', 'turn_id']).groupby('dialog_id'):
        soft = []
        for p, n in zip(sub['p_dist'], sub['n_raters']):
            votes = expand_vote_counts(np.asarray(p), int(n))
            soft.append(subsample_softlabel(votes, R, rng))
        spk = list(sub['speaker_id'].to_numpy())
        tc = soft_transition_counts(soft, speaker_ids=spk)
        total += tc.total
        inertia += tc.inertia
        contagion += tc.contagion
        n_pairs += tc.n_transitions
    return TransitionCounts(total=total, inertia=inertia, contagion=contagion, n_transitions=n_pairs)


def compute_metrics(counts: TransitionCounts) -> dict:
    res = fit_transition_matrix(counts)
    T = res.transition_post_mean
    smoothed = counts.total + res.alpha
    marginal = smoothed.sum(axis=0) / smoothed.sum()
    log2_lift = np.log2(np.maximum(T / np.maximum(marginal[None, :], 1e-12), 1e-12))
    return {
        'T': T,
        'log2_lift': log2_lift,
        'self_loop': np.diag(T).copy(),
    }


def main() -> None:
    df = pd.read_parquet(ROOT / 'data_processed' / 'emotionlines_softlabels_v2_bsetd.parquet')

    # Reference: R=5 baseline
    ref = compute_metrics(aggregate_with_R(df, R=5, seed=0))
    mask = ~np.eye(K, dtype=bool)
    summary = {'R5_reference_self_loop': ref['self_loop'].tolist()}

    for R in RATER_LEVELS:
        rho_list = []
        mad_diag_list = []
        mad_lift_list = []
        for seed in SEEDS:
            m = compute_metrics(aggregate_with_R(df, R=R, seed=seed))
            rho = float(np.corrcoef(m['log2_lift'][mask], ref['log2_lift'][mask])[0, 1])
            mad_diag = float(np.abs(m['self_loop'] - ref['self_loop']).mean())
            mad_lift = float(np.abs(m['log2_lift'][mask] - ref['log2_lift'][mask]).mean())
            rho_list.append(rho)
            mad_diag_list.append(mad_diag)
            mad_lift_list.append(mad_lift)
        summary[f'R={R}'] = {
            'pearson_off_diag_log2_lift_mean': float(np.mean(rho_list)),
            'pearson_off_diag_log2_lift_std': float(np.std(rho_list)),
            'mean_abs_self_loop_change_mean': float(np.mean(mad_diag_list)),
            'mean_abs_off_diag_lift_change_mean': float(np.mean(mad_lift_list)),
            'n_seeds': len(SEEDS),
        }
        print(f'R={R}: Pearson(off-diag lift vs R=5) = {np.mean(rho_list):.4f} +/- {np.std(rho_list):.4f}')

    out_path = ROOT / 'experiments' / 'stage1_emotionlines' / 'rater_count_ablation.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
