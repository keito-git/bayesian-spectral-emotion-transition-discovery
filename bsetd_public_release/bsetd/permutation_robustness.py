"""Permutation null test + robustness ablation for BSETD.

Permutation test:
    Shuffle soft labels independently across utterances within each
    dialog, then re-compute the BSETD posterior. The lift effect size
    under the shuffled null should be substantially smaller than the
    real-data lift, since within-dialog turn order is randomized.

Robustness ablation:
    Re-run BSETD on subpopulations of EmotionLines:
        - long dialogs (length >= 15)
        - short dialogs (length < 15)
        - high-disagreement (mean entropy >= median)
        - low-disagreement (mean entropy < median)
        - multi-speaker (>= 3 unique speakers)
        - dyadic (<= 2 unique speakers)
    Compute the pairwise Pearson of the resulting log2-lift maps.
    Stable Pearson > 0.8 across slices supports robustness.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from bsetd.stage1_dirichlet import (
    EMOTIONS, K, TransitionCounts,
    soft_transition_counts, fit_transition_matrix,
)


def aggregate(df: pd.DataFrame, shuffle_within_dialog: bool = False,
               rng: np.random.Generator | None = None) -> TransitionCounts:
    total = np.zeros((K, K))
    inertia = np.zeros((K, K))
    contagion = np.zeros((K, K))
    n_pairs = 0
    for _, sub in df.sort_values(['dialog_id', 'turn_id']).groupby('dialog_id'):
        soft = list(sub['p_dist'].apply(np.asarray).to_numpy())
        spk = list(sub['speaker_id'].to_numpy())
        if shuffle_within_dialog and rng is not None and len(soft) > 1:
            order = rng.permutation(len(soft))
            soft = [soft[i] for i in order]
            spk = [spk[i] for i in order]
        tc = soft_transition_counts(soft, speaker_ids=spk)
        total += tc.total
        inertia += tc.inertia
        contagion += tc.contagion
        n_pairs += tc.n_transitions
    return TransitionCounts(total=total, inertia=inertia, contagion=contagion, n_transitions=n_pairs)


def log2_lift_of(counts: TransitionCounts) -> np.ndarray:
    res = fit_transition_matrix(counts)
    smoothed = counts.total + res.alpha
    marginal = smoothed.sum(axis=0) / smoothed.sum()
    return np.log2(np.maximum(res.transition_post_mean / np.maximum(marginal[None, :], 1e-12), 1e-12))


def permutation_test(df: pd.DataFrame, B: int = 100, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    real_counts = aggregate(df)
    real_lift = log2_lift_of(real_counts)
    mask = ~np.eye(K, dtype=bool)
    real_max = float(np.max(np.abs(real_lift[mask])))

    perm_max = np.zeros(B)
    for b in range(B):
        pc = aggregate(df, shuffle_within_dialog=True, rng=rng)
        plift = log2_lift_of(pc)
        perm_max[b] = float(np.max(np.abs(plift[mask])))
    p_value = float(np.mean(perm_max >= real_max))
    return {
        'real_max_abs_log2_lift': real_max,
        'permutation_max_abs_log2_lift_mean': float(perm_max.mean()),
        'permutation_max_abs_log2_lift_std': float(perm_max.std()),
        'p_value_max_statistic': p_value,
        'B': B,
    }


def robustness_slices(df: pd.DataFrame) -> dict:
    by_dialog = df.groupby('dialog_id').agg(
        length=('turn_id', 'size'),
        mean_entropy=('soft_entropy', 'mean'),
        n_speakers=('speaker_id', 'nunique'),
    )
    median_len = by_dialog['length'].median()
    median_ent = by_dialog['mean_entropy'].median()

    slices = {
        'long_dialogs': by_dialog.index[by_dialog['length'] >= median_len],
        'short_dialogs': by_dialog.index[by_dialog['length'] < median_len],
        'high_disagreement': by_dialog.index[by_dialog['mean_entropy'] >= median_ent],
        'low_disagreement': by_dialog.index[by_dialog['mean_entropy'] < median_ent],
        'multispeaker': by_dialog.index[by_dialog['n_speakers'] >= 3],
        'dyadic': by_dialog.index[by_dialog['n_speakers'] <= 2],
    }
    lifts = {}
    for name, idxs in slices.items():
        sub = df[df['dialog_id'].isin(idxs)]
        lifts[name] = {
            'n_dialogs': int(len(idxs)),
            'log2_lift': log2_lift_of(aggregate(sub)).tolist(),
        }
    mask = ~np.eye(K, dtype=bool)
    keys = list(lifts.keys())
    pear = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            la = np.asarray(lifts[a]['log2_lift'])[mask]
            lb = np.asarray(lifts[b]['log2_lift'])[mask]
            pear[f'{a}__vs__{b}'] = float(np.corrcoef(la, lb)[0, 1])
    return {'slices': lifts, 'pairwise_pearson': pear,
            'median_dialog_length': float(median_len),
            'median_dialog_entropy': float(median_ent)}


def main() -> None:
    df = pd.read_parquet(ROOT / 'data_processed' / 'emotionlines_softlabels_v2_bsetd.parquet')
    print('=== Permutation null test ===')
    perm = permutation_test(df, B=50, seed=42)
    print(json.dumps(perm, indent=2))
    print()
    print('=== Robustness slices ===')
    rob = robustness_slices(df)
    print('Slice sizes:')
    for name, d in rob['slices'].items():
        print(f'  {name:>18}: n_dialogs={d["n_dialogs"]}')
    print('Pairwise Pearson of log2 lift across slices:')
    for k, v in rob['pairwise_pearson'].items():
        print(f'  {k:>50}: {v:.3f}')
    out_path = ROOT / 'experiments' / 'stage1_emotionlines' / 'permutation_robustness.json'
    with open(out_path, 'w') as f:
        json.dump({'permutation': perm, 'robustness': rob}, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
