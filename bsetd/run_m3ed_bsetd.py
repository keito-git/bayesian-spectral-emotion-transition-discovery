"""Run BSETD Stage 1+2 on M3ED (Chinese multi-modal emotional dialogue, ACL 2022).

M3ED ships per-utterance annotations from 3 raters per utterance with the
emotion taxonomy {Happy, Surprise, Sad, Disgust, Anger, Fear, Neutral}.
We map this to our Ekman-7 canonical order {neutral, joy, sadness, fear,
anger, surprise, disgust}, treat the 3-rater votes as the soft label,
and run the full BSETD pipeline. This adds a cross-lingual validation
to the corpus suite alongside EmotionLines + MELD + DailyDialog.

Annotation JSON schema:
    annotation[show][episode]['Dialog'][utt_id] = {
        'StartTime', 'EndTime', 'Text', 'Speaker',
        'EmoAnnotation': {
            'EmoAnnotator1': '...', 'EmoAnnotator2': '...', 'EmoAnnotator3': '...',
            'final_mul_emo', 'final_main_emo',
        },
    }
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
    soft_transition_counts, fit_inertia_contagion,
)
from bsetd.stage2_spectral import run_stage2_on_stage1_npz


# M3ED uses capitalized emotion words; map to our canonical ordering
M3ED_TO_EKMAN = {
    'Neutral': 'neutral',
    'Happy': 'joy',
    'Sad': 'sadness',
    'Fear': 'fear',
    'Anger': 'anger',
    'Surprise': 'surprise',
    'Disgust': 'disgust',
}
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}


def parse_annotation(annot: dict) -> tuple[np.ndarray, int]:
    """Return soft label vector + number of valid raters."""
    votes = np.zeros(K)
    n = 0
    for key in ('EmoAnnotator1', 'EmoAnnotator2', 'EmoAnnotator3'):
        v = annot.get(key)
        if v is None:
            continue
        e = M3ED_TO_EKMAN.get(v)
        if e is None:
            continue
        votes[EMO2IDX[e]] += 1
        n += 1
    if n == 0:
        return votes, 0
    return votes / n, n


def build_m3ed_dataframe(path: Path) -> pd.DataFrame:
    data = json.load(open(path))
    rows = []
    for show_name, show in data.items():
        for ep_id, ep in show.items():
            dialog = ep.get('Dialog', {})
            for turn_id, utt in enumerate(sorted(dialog)):
                u = dialog[utt]
                emo_annot = u.get('EmoAnnotation', {})
                p, n_raters = parse_annotation(emo_annot)
                if n_raters == 0:
                    continue
                rows.append({
                    'dataset_source': 'm3ed',
                    'dialog_id': f'{show_name}::{ep_id}',
                    'turn_id': turn_id,
                    'speaker_id': u.get('Speaker', 'A'),
                    'utterance': u.get('Text', ''),
                    'hard_label': EMOTIONS[int(p.argmax())],
                    'p_dist': p.tolist(),
                    'n_raters': n_raters,
                })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> TransitionCounts:
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
    return TransitionCounts(total=total, inertia=inertia, contagion=contagion, n_transitions=n_pairs)


def main() -> None:
    out_dir = ROOT / 'experiments' / 'stage1_emotionlines' / 'm3ed'
    out_dir.mkdir(parents=True, exist_ok=True)
    annot_path = ROOT / 'data' / 'raw' / 'm3ed' / 'annotation.json'
    print('Building M3ED dataframe...')
    df = build_m3ed_dataframe(annot_path)
    print(f'  n_utterances={len(df)}, n_dialogs={df["dialog_id"].nunique()}, '
          f'n_speakers={df["speaker_id"].nunique()}')
    print(f'  label distribution: {df["hard_label"].value_counts().to_dict()}')
    df.to_parquet(ROOT / 'data_processed' / 'm3ed_softlabels.parquet')

    counts = aggregate(df)
    fits = fit_inertia_contagion(counts)
    summary = {}
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
        summary[key] = {
            'n_transitions': res.metadata['n_transitions'],
            'alpha_global': res.metadata['alpha_global'],
            'eb_converged': res.metadata['eb_converged'],
            'n_rejected_offdiag': int(res.rejected_bh.sum()),
        }
        s2 = run_stage2_on_stage1_npz(npz, ROOT / 'experiments' / 'stage2_emotionlines' / 'm3ed')
        summary[key]['stage2_inertia_index'] = s2['inertia_index']
        summary[key]['stage2_contagion_index'] = s2['contagion_index']

    mask = ~np.eye(K, dtype=bool)
    for ref in ('emotionlines', 'meld', 'dailydialog'):
        try:
            ref_npz = np.load(ROOT / 'experiments' / 'stage1_emotionlines' / ref / 'stage1_total.npz')
            m3 = np.load(out_dir / 'stage1_total.npz')
            pear = float(np.corrcoef(ref_npz['transition_post_mean'][mask],
                                      m3['transition_post_mean'][mask])[0, 1])
            summary[f'm3ed_vs_{ref}_pearson'] = pear
        except FileNotFoundError:
            pass
    summary['m3ed_diag'] = np.diag(m3['transition_post_mean']).tolist()

    json.dump(summary, open(out_dir / 'stage1_summary.json', 'w'), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if 'pearson' in k or 'diag' in k}, indent=2))


if __name__ == '__main__':
    main()
