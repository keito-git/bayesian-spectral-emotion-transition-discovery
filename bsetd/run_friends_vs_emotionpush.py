"""Domain-cross BSETD: Friends vs EmotionPush within EmotionLines.

EmotionLines combines two subsets:
    Friends     - TV-show scripted dialog (1000 dialogs)
    EmotionPush - Facebook Messenger chat (1000 dialogs)

Running BSETD Stage 1 separately on each subset and comparing the resulting
posterior transition matrices isolates a within-corpus domain shift: same
annotation pipeline (real five-rater), same Ekman 7, same preprocessing,
but different communication medium and register. If the BSETD-discovered
structure survives this shift, the corpus-domain confound between
EmotionLines and MELD/DailyDialog can be more confidently attributed to
the soft-label provenance rather than to domain alone.
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
    soft_transition_counts, fit_inertia_contagion,
)
from bsetd.stage2_spectral import run_stage2_on_stage1_npz


def aggregate_subset(df: pd.DataFrame) -> TransitionCounts:
    total = np.zeros((K, K))
    inertia = np.zeros((K, K))
    contagion = np.zeros((K, K))
    n_pairs = 0
    for _, sub in df.sort_values(['dialog_id', 'turn_id']).groupby('dialog_id'):
        soft = list(sub['p_dist'].apply(np.asarray).to_numpy())
        spk = list(sub['speaker_id'].to_numpy())
        tc = soft_transition_counts(soft, speaker_ids=spk)
        total += tc.total
        inertia += tc.inertia
        contagion += tc.contagion
        n_pairs += tc.n_transitions
    return TransitionCounts(
        total=total, inertia=inertia, contagion=contagion, n_transitions=n_pairs
    )


def main() -> None:
    df = pd.read_parquet(ROOT / 'data_processed' / 'emotionlines_softlabels_v2_bsetd.parquet')
    summary = {}
    for subset in ('friends', 'emotionpush'):
        sub_df = df[df['dataset_source'] == subset]
        out_dir = ROOT / 'experiments' / f'stage1_emotionlines_{subset}'
        out_dir.mkdir(parents=True, exist_ok=True)
        counts = aggregate_subset(sub_df)
        fits = fit_inertia_contagion(counts)
        sub_sum = {}
        for key, res in fits.items():
            npz = out_dir / f'stage1_{key}.npz'
            np.savez(
                npz,
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
            sub_sum[key] = {
                'n_transitions': res.metadata['n_transitions'],
                'n_rejected_offdiag': int(res.rejected_bh.sum()),
            }
        summary[subset] = sub_sum

    mask = ~np.eye(K, dtype=bool)
    A_f = np.load(ROOT / 'experiments' / 'stage1_emotionlines_friends' / 'stage1_total.npz')['transition_post_mean']
    A_e = np.load(ROOT / 'experiments' / 'stage1_emotionlines_emotionpush' / 'stage1_total.npz')['transition_post_mean']
    pear = float(np.corrcoef(A_f[mask], A_e[mask])[0, 1])
    summary['friends_vs_emotionpush_offdiag_pearson'] = pear
    summary['friends_diag'] = np.diag(A_f).tolist()
    summary['emotionpush_diag'] = np.diag(A_e).tolist()
    print(json.dumps(summary, indent=2))
    with open(ROOT / 'experiments' / 'stage1_emotionlines' / 'friends_vs_emotionpush_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
