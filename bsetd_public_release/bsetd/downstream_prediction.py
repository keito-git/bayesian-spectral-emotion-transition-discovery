"""Downstream next-emotion prediction with and without BSETD edge prior.

To test whether the structural BSETD discovery has practical predictive
utility, we run two simple next-emotion classifiers on EmotionLines:

    (a) Baseline: logistic regression on a 5-utterance lag-emotion-histogram
        feature (35-dim: 7 emotions x 5 lags), predicting the hard emotion
        of the next utterance.
    (b) BSETD-prior: same classifier with the addition of the BSETD lift
        feature, defined as the row of the BSETD log2-lift matrix indexed
        by the previous utterance's modal emotion (7 extra dimensions).

We split EmotionLines 80/20 by dialog and report macro-F1 of both models.
The point is not to beat ERC SOTA; it is to show that the corpus-level
structure discovered by BSETD carries non-trivial information for
downstream prediction beyond the lag-histogram baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import sys
ROOT = Path(".")
sys.path.insert(0, str(ROOT))


EMOTIONS = ['neutral', 'joy', 'sadness', 'fear', 'anger', 'surprise', 'disgust']
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}
K = 7
LAGS = 5


def build_features(df: pd.DataFrame, log2_lift: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_base, X_lift, y) where X_base is the 5-lag histogram features and
    X_lift adds 7-dim BSETD lift row.

    Drops the first 5 utterances of each dialog (need 5 lags).
    """
    X_base_rows = []
    X_lift_rows = []
    y = []
    for _, sub in df.sort_values(['dialog_id', 'turn_id']).groupby('dialog_id'):
        labels = sub['hard_label'].map(EMO2IDX).to_numpy()
        for t in range(LAGS, len(labels)):
            hist = np.zeros((LAGS, K))
            for l in range(LAGS):
                hist[l, labels[t - 1 - l]] = 1.0
            feat = hist.flatten()
            prev_emotion = labels[t - 1]
            lift_row = log2_lift[prev_emotion]
            X_base_rows.append(feat)
            X_lift_rows.append(np.concatenate([feat, lift_row]))
            y.append(labels[t])
    return np.asarray(X_base_rows), np.asarray(X_lift_rows), np.asarray(y)


def main() -> None:
    df = pd.read_parquet(ROOT / 'data_processed' / 'emotionlines_softlabels_v2_bsetd.parquet')
    s1 = np.load(ROOT / 'experiments' / 'stage1_emotionlines' / 'stage1_total.npz')
    T = s1['transition_post_mean']
    smoothed = s1['counts_total'] + s1['alpha']
    marginal = smoothed.sum(axis=0) / smoothed.sum()
    log2_lift = np.log2(np.maximum(T / np.maximum(marginal[None, :], 1e-12), 1e-12))

    rng = np.random.default_rng(0)
    dialog_ids = df['dialog_id'].unique()
    rng.shuffle(dialog_ids)
    n_train = int(0.8 * len(dialog_ids))
    train_ids = set(dialog_ids[:n_train])
    test_ids = set(dialog_ids[n_train:])

    df_train = df[df['dialog_id'].isin(train_ids)]
    df_test = df[df['dialog_id'].isin(test_ids)]
    X_base_tr, X_lift_tr, y_tr = build_features(df_train, log2_lift)
    X_base_te, X_lift_te, y_te = build_features(df_test, log2_lift)

    print(f'n_train={len(y_tr)}, n_test={len(y_te)}')

    results = {}
    for name, X_tr, X_te in [
        ('baseline_lag5', X_base_tr, X_base_te),
        ('with_bsetd_lift', X_lift_tr, X_lift_te),
    ]:
        scaler = StandardScaler().fit(X_tr)
        Xtr = scaler.transform(X_tr)
        Xte = scaler.transform(X_te)
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced')
        clf.fit(Xtr, y_tr)
        pred = clf.predict(Xte)
        f1m = f1_score(y_te, pred, average='macro')
        f1w = f1_score(y_te, pred, average='weighted')
        print(f'  {name}: macro-F1={f1m:.4f}, weighted-F1={f1w:.4f}')
        results[name] = {'macro_f1': float(f1m), 'weighted_f1': float(f1w)}

    delta_macro = results['with_bsetd_lift']['macro_f1'] - results['baseline_lag5']['macro_f1']
    delta_weighted = results['with_bsetd_lift']['weighted_f1'] - results['baseline_lag5']['weighted_f1']
    results['delta_macro_f1'] = float(delta_macro)
    results['delta_weighted_f1'] = float(delta_weighted)

    out_path = ROOT / 'experiments' / 'stage1_emotionlines' / 'downstream_prediction.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\ndelta_macro_F1 = {delta_macro:+.4f}')
    print(f'delta_weighted_F1 = {delta_weighted:+.4f}')


if __name__ == '__main__':
    main()
