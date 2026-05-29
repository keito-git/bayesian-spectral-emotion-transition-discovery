"""Per-dialog hierarchical extension of BSETD.

The base BSETD (Stage 1) aggregates transition counts over the entire
corpus into a single posterior. A natural extension partitions the
corpus into per-dialog count matrices N_d and treats each dialog's
transition matrix T_d as a draw from a shared hyperprior:

    T_d   ~ Dirichlet(alpha * pi),      pi ~ Dirichlet(beta_0 * pi_0),
    N_{d,j,k} ~ Multinomial(N_{d,j,.}, T_{d,j}),

where pi is the corpus-level mean transition row and alpha is the
hierarchical concentration controlling per-dialog deviation.

We use a simplified empirical-Bayes form: pi is estimated as the
corpus posterior from the base BSETD, alpha is fixed at a series of
values, and per-dialog posteriors are computed as
    T_d | N_d ~ Dirichlet(alpha * pi + N_d).
This isolates "which dialogs deviate from the corpus-level structure"
and provides a posterior decomposition of variability.
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


def per_dialog_counts(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for did, sub in df.sort_values(['dialog_id', 'turn_id']).groupby('dialog_id'):
        soft = list(sub['p_dist'].apply(np.asarray).to_numpy())
        spk = list(sub['speaker_id'].to_numpy())
        tc = soft_transition_counts(soft, speaker_ids=spk)
        out[did] = tc.total
    return out


def hierarchical_posteriors(
    per_d: dict[str, np.ndarray],
    pi: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Return per-dialog posterior mean T_d under the simple EB hierarchy."""
    out = {}
    for did, N in per_d.items():
        T_d = np.zeros((K, K))
        for j in range(K):
            row_post = alpha * pi[j] + N[j]
            T_d[j] = row_post / max(row_post.sum(), 1e-12)
        out[did] = T_d
    return out


def main() -> None:
    df = pd.read_parquet(ROOT / 'data_processed' / 'emotionlines_softlabels_v2_bsetd.parquet')
    pi = np.load(ROOT / 'experiments' / 'stage1_emotionlines' / 'stage1_total.npz')['transition_post_mean']
    per_d = per_dialog_counts(df)

    summary = {'corpus_pi_diag': np.diag(pi).tolist(), 'n_dialogs': len(per_d)}
    for alpha in [1.0, 10.0, 100.0]:
        post = hierarchical_posteriors(per_d, pi, alpha=alpha)
        # Compute KL of each per-dialog posterior row vs pi row
        kls = []
        for did, T_d in post.items():
            kl = 0.0
            for j in range(K):
                p = np.clip(T_d[j], 1e-12, 1.0)
                q = np.clip(pi[j], 1e-12, 1.0)
                kl += float(np.sum(p * np.log(p / q)))
            kls.append(kl)
        kls = np.asarray(kls)
        summary[f'alpha_{alpha}'] = {
            'kl_per_dialog_mean': float(kls.mean()),
            'kl_per_dialog_median': float(np.median(kls)),
            'kl_per_dialog_p95': float(np.percentile(kls, 95)),
        }

    # Find top deviating dialogs at alpha=10
    post = hierarchical_posteriors(per_d, pi, alpha=10.0)
    kls_dict = {}
    for did, T_d in post.items():
        kl = 0.0
        for j in range(K):
            p = np.clip(T_d[j], 1e-12, 1.0)
            q = np.clip(pi[j], 1e-12, 1.0)
            kl += float(np.sum(p * np.log(p / q)))
        kls_dict[did] = kl
    top = sorted(kls_dict.items(), key=lambda x: -x[1])[:10]
    summary['top_deviating_dialogs_alpha10'] = [{'dialog_id': d, 'kl': k} for d, k in top]

    out_path = ROOT / 'experiments' / 'stage1_emotionlines' / 'per_dialog_extension.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))


if __name__ == '__main__':
    main()
