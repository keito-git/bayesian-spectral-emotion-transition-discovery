"""Dialog-level cluster bootstrap for BSETD main statistics.

We resample dialogs (with replacement) to compute percentile bootstrap
confidence intervals for the headline statistics:

    - per-cell log2 lift
    - per-emotion self-loop posterior mean
    - per-emotion Inertia minus Contagion differential
    - Stage 2 per-emotion Inertia and Contagion indices

Dialog-level resampling is the appropriate cluster-bootstrap target
because utterance pairs within a dialog are not independent.
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
from bsetd.stage2_spectral import spectral_decompose


def aggregate_from_dialog_list(dialog_records: list[dict]) -> TransitionCounts:
    total = np.zeros((K, K))
    inertia = np.zeros((K, K))
    contagion = np.zeros((K, K))
    n_pairs = 0
    for rec in dialog_records:
        tc = soft_transition_counts(rec['soft'], speaker_ids=rec['speakers'])
        total += tc.total
        inertia += tc.inertia
        contagion += tc.contagion
        n_pairs += tc.n_transitions
    return TransitionCounts(total=total, inertia=inertia, contagion=contagion, n_transitions=n_pairs)


def build_dialog_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for did, sub in df.sort_values(['dialog_id', 'turn_id']).groupby('dialog_id'):
        records.append({
            'dialog_id': did,
            'soft': list(sub['p_dist'].apply(np.asarray).to_numpy()),
            'speakers': list(sub['speaker_id'].to_numpy()),
        })
    return records


def compute_metrics(counts_total, counts_inertia, counts_contagion) -> dict:
    tot = TransitionCounts(total=counts_total, inertia=counts_inertia,
                            contagion=counts_contagion, n_transitions=int(counts_total.sum()))
    res = fit_transition_matrix(tot)
    T = res.transition_post_mean
    smoothed = counts_total + res.alpha
    marginal = smoothed.sum(axis=0) / smoothed.sum()
    log2_lift = np.log2(np.maximum(T / np.maximum(marginal[None, :], 1e-12), 1e-12))

    in_tot = TransitionCounts(total=counts_inertia, inertia=counts_inertia,
                               contagion=np.zeros_like(counts_inertia),
                               n_transitions=int(counts_inertia.sum()))
    co_tot = TransitionCounts(total=counts_contagion, inertia=np.zeros_like(counts_contagion),
                               contagion=counts_contagion,
                               n_transitions=int(counts_contagion.sum()))
    T_in = fit_transition_matrix(in_tot).transition_post_mean
    T_co = fit_transition_matrix(co_tot).transition_post_mean
    diff_self_loop = np.diag(T_in) - np.diag(T_co)

    s2 = spectral_decompose(T)
    return {
        'self_loop': np.diag(T).copy(),  # (K,)
        'log2_lift_offdiag': log2_lift,  # (K, K)
        'diff_self_loop_inertia_minus_contagion': diff_self_loop,  # (K,)
        'inertia_index': s2.inertia_index.copy(),
        'contagion_index': s2.contagion_index.copy(),
    }


def bootstrap(
    df: pd.DataFrame,
    B: int = 200,
    seed: int = 0,
) -> dict:
    records = build_dialog_records(df)
    n_dialogs = len(records)
    rng = np.random.default_rng(seed)

    point = compute_metrics(*_counts_from_records(records))
    rep_self_loop = np.zeros((B, K))
    rep_lift = np.zeros((B, K, K))
    rep_diff = np.zeros((B, K))
    rep_in = np.zeros((B, K))
    rep_co = np.zeros((B, K))
    for b in range(B):
        idx = rng.integers(0, n_dialogs, n_dialogs)
        sample = [records[i] for i in idx]
        counts = aggregate_from_dialog_list(sample)
        m = compute_metrics(counts.total, counts.inertia, counts.contagion)
        rep_self_loop[b] = m['self_loop']
        rep_lift[b] = m['log2_lift_offdiag']
        rep_diff[b] = m['diff_self_loop_inertia_minus_contagion']
        rep_in[b] = m['inertia_index']
        rep_co[b] = m['contagion_index']

    def ci(arr, lo=2.5, hi=97.5):
        return (np.percentile(arr, lo, axis=0).tolist(),
                np.percentile(arr, hi, axis=0).tolist())

    return {
        'point': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in point.items()},
        'self_loop_ci95': ci(rep_self_loop),
        'log2_lift_ci95_low': np.percentile(rep_lift, 2.5, axis=0).tolist(),
        'log2_lift_ci95_high': np.percentile(rep_lift, 97.5, axis=0).tolist(),
        'diff_self_loop_ci95': ci(rep_diff),
        'inertia_index_ci95': ci(rep_in),
        'contagion_index_ci95': ci(rep_co),
        'B': B,
        'n_dialogs': n_dialogs,
    }


def _counts_from_records(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tc = aggregate_from_dialog_list(records)
    return tc.total, tc.inertia, tc.contagion


def main() -> None:
    df = pd.read_parquet(ROOT / 'data_processed' / 'emotionlines_softlabels_v2_bsetd.parquet')
    out = bootstrap(df, B=1000, seed=42)
    out_path = ROOT / 'experiments' / 'stage1_emotionlines' / 'bootstrap_ci.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {out_path}')
    print()
    print('Self-loop point estimate and 95% CI:')
    for i, e in enumerate(EMOTIONS):
        lo = out['self_loop_ci95'][0][i]
        hi = out['self_loop_ci95'][1][i]
        pt = out['point']['self_loop'][i]
        print(f'  {e:>9}: {pt:.3f}  [{lo:.3f}, {hi:.3f}]')
    print()
    print('Inertia - Contagion diff point and 95% CI:')
    for i, e in enumerate(EMOTIONS):
        lo = out['diff_self_loop_ci95'][0][i]
        hi = out['diff_self_loop_ci95'][1][i]
        pt = out['point']['diff_self_loop_inertia_minus_contagion'][i]
        print(f'  {e:>9}: {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]')


if __name__ == '__main__':
    main()
