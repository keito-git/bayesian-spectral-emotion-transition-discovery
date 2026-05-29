"""Dialog-level cluster bootstrap CI for the cross-corpus Pearson correlations.

Resamples dialogs (with replacement) within each corpus, refits BSETD Stage 1
under the resampled dialog set, and recomputes the pairwise off-diagonal
Pearson correlations. We report the percentile-bootstrap 95% CI of every
pairwise Pearson in the four-corpus table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsetd.stage1_dirichlet import (
    K, TransitionCounts,
    soft_transition_counts, fit_transition_matrix,
)


B = 200


def build_records(df: pd.DataFrame, p_col: str = 'p_dist',
                  dialog_col: str = 'dialog_id', turn_col: str = 'turn_id',
                  speaker_col: str = 'speaker_id') -> list[dict]:
    records = []
    for did, sub in df.sort_values([dialog_col, turn_col]).groupby(dialog_col):
        records.append({
            'soft': list(sub[p_col].apply(np.asarray).to_numpy()),
            'speakers': list(sub[speaker_col].to_numpy()),
        })
    return records


def aggregate(records: list[dict]) -> TransitionCounts:
    total = np.zeros((K, K)); inertia = np.zeros((K, K)); contagion = np.zeros((K, K))
    n_pairs = 0
    for r in records:
        tc = soft_transition_counts(r['soft'], speaker_ids=r['speakers'])
        total += tc.total; inertia += tc.inertia; contagion += tc.contagion
        n_pairs += tc.n_transitions
    return TransitionCounts(total=total, inertia=inertia, contagion=contagion, n_transitions=n_pairs)


def posterior_offdiag(records: list[dict]) -> np.ndarray:
    counts = aggregate(records)
    res = fit_transition_matrix(counts)
    mask = ~np.eye(K, dtype=bool)
    return res.transition_post_mean[mask]


def bootstrap_pearson(records_a: list[dict], records_b: list[dict],
                       B: int, seed: int) -> tuple[float, float, float]:
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 100000)
    n_a = len(records_a); n_b = len(records_b)
    point_a = posterior_offdiag(records_a)
    point_b = posterior_offdiag(records_b)
    point_rho = float(np.corrcoef(point_a, point_b)[0, 1])
    reps = []
    for b in range(B):
        ia = rng_a.integers(0, n_a, n_a)
        ib = rng_b.integers(0, n_b, n_b)
        va = posterior_offdiag([records_a[i] for i in ia])
        vb = posterior_offdiag([records_b[i] for i in ib])
        reps.append(float(np.corrcoef(va, vb)[0, 1]))
    lo, hi = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
    return point_rho, lo, hi


def main() -> None:
    print('Loading corpora...')
    el = pd.read_parquet(ROOT / 'data_processed' / 'emotionlines_softlabels_v2_bsetd.parquet')

    md = pd.read_parquet(ROOT / 'data_processed' / 'meld_softlabels_real.parquet')
    md_recs = []
    for did, sub in md.sort_values(['Dialogue_ID', 'Utterance_ID']).groupby('Dialogue_ID'):
        md_recs.append({
            'soft': list(sub['p_dist'].apply(np.asarray).to_numpy()),
            'speakers': list(sub['Speaker'].astype(str).to_numpy()),
        })

    from bsetd.dailydialog_llm_softlabel import _load_dailydialog_raw_text
    from bsetd.run_dailydialog_hardlabel import label_to_onehot
    dd_dialogs = _load_dailydialog_raw_text()
    dd_recs = []
    for dialog in dd_dialogs:
        soft = [label_to_onehot(u['hard_label_dd']) for u in dialog]
        spk = [u['speaker_id'] for u in dialog]
        dd_recs.append({'soft': soft, 'speakers': spk})

    m3 = pd.read_parquet(ROOT / 'data_processed' / 'm3ed_softlabels.parquet')
    m3_recs = build_records(m3)

    el_recs = build_records(el)
    corpora = {'EmotionLines': el_recs, 'MELD': md_recs,
               'DailyDialog': dd_recs, 'M3ED': m3_recs}
    print({k: len(v) for k, v in corpora.items()})

    results = {}
    keys = list(corpora.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            print(f'  {a} vs {b}...')
            rho, lo, hi = bootstrap_pearson(corpora[a], corpora[b], B=B, seed=42)
            results[f'{a}_vs_{b}'] = {
                'rho_point': rho,
                'ci95_low': lo,
                'ci95_high': hi,
            }
            print(f'    rho = {rho:.3f}  CI95 = [{lo:.3f}, {hi:.3f}]')

    out = ROOT / 'experiments' / 'stage1_emotionlines' / 'cross_corpus_bootstrap.json'
    json.dump({'B': B, 'pairs': results}, open(out, 'w'), indent=2)
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
