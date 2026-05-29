"""Run BSETD Stage 1 + Stage 2 on MELD soft labels for consistency check.

MELD ships with single hard labels but our project re-derived soft labels
via Dirichlet smoothing in `meld_softlabels_real.parquet` (column p_dist,
7-dim probability vectors over the same emotion ordering as EmotionLines:
neutral, joy, sadness, fear, anger, surprise, disgust).

We treat each Dialogue_ID as a dialog, sort by Utterance_ID, and aggregate
soft transitions exactly as on EmotionLines. Inertia/contagion separation
uses the Speaker column (string name) as the speaker id.

The point of this run is to verify that the BSETD Stage 1 transition
matrix on a different corpus (still Ekman 7) is broadly consistent with
the EmotionLines result. We report inter-corpus Pearson correlation of
the off-diagonal posterior means.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from bsetd.stage1_dirichlet import (
    EMOTIONS, K,
    TransitionCounts, soft_transition_counts, fit_inertia_contagion,
)
from bsetd.stage2_spectral import run_stage2_on_stage1_npz


def aggregate_meld(df: pd.DataFrame) -> TransitionCounts:
    total = np.zeros((K, K))
    inertia = np.zeros((K, K))
    contagion = np.zeros((K, K))
    n_pairs = 0
    df = df.sort_values(['Dialogue_ID', 'Utterance_ID'])
    for _, sub in df.groupby('Dialogue_ID'):
        p_list = list(sub['p_dist'].apply(np.asarray).to_numpy())
        speakers = list(sub['Speaker'].astype(str).to_numpy())
        tc = soft_transition_counts(p_list, speaker_ids=speakers)
        total += tc.total
        inertia += tc.inertia
        contagion += tc.contagion
        n_pairs += tc.n_transitions
    return TransitionCounts(
        total=total, inertia=inertia, contagion=contagion, n_transitions=n_pairs
    )


def main() -> None:
    out_dir = ROOT / 'experiments' / 'stage1_emotionlines' / 'meld'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / 'data_processed' / 'meld_softlabels_real.parquet')

    counts = aggregate_meld(df)
    fits = fit_inertia_contagion(counts)
    summary = {}
    for key, res in fits.items():
        npz_path = out_dir / f'stage1_{key}.npz'
        np.savez(
            npz_path,
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
            'n_rejected_offdiag': int(res.rejected_bh.sum()),
        }
        s2_dir = ROOT / 'experiments' / 'stage2_emotionlines' / 'meld'
        s2 = run_stage2_on_stage1_npz(npz_path, s2_dir)
        summary[key]['stage2_inertia_index'] = s2['inertia_index']
        summary[key]['stage2_contagion_index'] = s2['contagion_index']

    # Cross-corpus consistency vs EmotionLines
    el_npz = np.load(ROOT / 'experiments' / 'stage1_emotionlines' / 'stage1_total.npz')
    meld_npz = np.load(ROOT / 'experiments' / 'stage1_emotionlines' / 'meld' / 'stage1_total.npz')
    A_el = el_npz['transition_post_mean']
    A_meld = meld_npz['transition_post_mean']
    mask = ~np.eye(K, dtype=bool)
    pearson = float(np.corrcoef(A_el[mask], A_meld[mask])[0, 1])
    summary['cross_corpus'] = {
        'emotionlines_vs_meld_offdiag_pearson': pearson,
        'emotionlines_diag': np.diag(A_el).tolist(),
        'meld_diag': np.diag(A_meld).tolist(),
    }

    with open(out_dir / 'stage1_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
