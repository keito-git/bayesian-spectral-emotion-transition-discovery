"""EmotionIC-style baseline for next-emotion prediction on EmotionLines.

A simplified two-stream GRU classifier that follows the EmotionIC
(Liu et al., 2024) decomposition into an inertia stream (the speaker's
own past) and a contagion stream (other speakers' past). We use this
as a baseline against which to evaluate whether the BSETD-discovered
edges, when injected as an attention prior, improve predictive accuracy.

This is intentionally a lightweight reproduction (not a faithful
EmotionIC re-implementation) sufficient to claim a baseline comparison
in the BSETD paper. Implementation notes:

    - Input features per utterance: 7-d soft label (EmotionLines real votes)
      + speaker-id one-hot (truncated to 8 most-frequent speakers + "other").
    - Inertia stream: GRU over (soft label, is_same_speaker_as_target) for
      utterances by the target speaker.
    - Contagion stream: GRU over the same features for utterances by
      OTHER speakers.
    - Fusion: concatenate final hidden states, linear head over K classes.
    - BSETD-prior variant: at the classifier head, add a learned linear
      projection of the BSETD log2-lift row indexed by the immediate-prev
      utterance's modal emotion.

We compare baseline vs. BSETD-prior on EmotionLines 80/20 dialog split
and report macro-F1 and weighted-F1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
ROOT = Path(".")
sys.path.insert(0, str(ROOT))

EMOTIONS = ['neutral', 'joy', 'sadness', 'fear', 'anger', 'surprise', 'disgust']
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}
K = 7
WINDOW = 8
N_TOP_SPEAKERS = 8


class DialogDataset(Dataset):
    def __init__(self, df: pd.DataFrame, log2_lift: np.ndarray, top_speakers: list[str]):
        self.samples: list[tuple] = []
        self.log2_lift = log2_lift
        top_set = set(top_speakers)
        spk2idx = {s: i for i, s in enumerate(top_speakers)}
        for _, sub in df.sort_values(['dialog_id', 'turn_id']).groupby('dialog_id'):
            soft = np.stack(sub['p_dist'].apply(np.asarray).to_numpy())  # (L, K)
            spk = list(sub['speaker_id'].to_numpy())
            labels = sub['hard_label'].map(EMO2IDX).to_numpy()
            L = len(soft)
            for t in range(2, L):
                start = max(0, t - WINDOW)
                ctx_soft = soft[start:t]   # (W, K)
                ctx_spk = spk[start:t]
                target_spk = spk[t]
                is_same = np.array([1.0 if s == target_spk else 0.0 for s in ctx_spk], dtype=np.float32)
                # Speaker one-hot (top-N + other)
                spk_oh = np.zeros((len(ctx_spk), N_TOP_SPEAKERS + 1), dtype=np.float32)
                for i, s in enumerate(ctx_spk):
                    if s in top_set:
                        spk_oh[i, spk2idx[s]] = 1.0
                    else:
                        spk_oh[i, N_TOP_SPEAKERS] = 1.0
                feat = np.concatenate([ctx_soft, spk_oh, is_same[:, None]], axis=1)
                # BSETD prior: log2 lift row of the previous utterance's modal emotion
                prev_modal = int(soft[t - 1].argmax())
                bsetd_feat = log2_lift[prev_modal]
                self.samples.append((feat.astype(np.float32), is_same.astype(np.float32),
                                     bsetd_feat.astype(np.float32), int(labels[t])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    feats, masks, bsetd, y = zip(*batch)
    lengths = [f.shape[0] for f in feats]
    L_max = max(lengths)
    F_dim = feats[0].shape[1]
    feat_pad = np.zeros((len(batch), L_max, F_dim), dtype=np.float32)
    mask_pad = np.zeros((len(batch), L_max), dtype=np.float32)
    same_pad = np.zeros((len(batch), L_max), dtype=np.float32)
    for i, (f, m) in enumerate(zip(feats, masks)):
        feat_pad[i, :f.shape[0]] = f
        mask_pad[i, :f.shape[0]] = 1.0
        same_pad[i, :m.shape[0]] = m
    bsetd_arr = np.stack(bsetd)
    return (torch.from_numpy(feat_pad), torch.from_numpy(mask_pad),
            torch.from_numpy(same_pad), torch.from_numpy(bsetd_arr),
            torch.tensor(y, dtype=torch.long))


class TwoStreamGRU(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, use_bsetd: bool = False):
        super().__init__()
        self.gru_inertia = nn.GRU(input_dim, hidden, batch_first=True)
        self.gru_contagion = nn.GRU(input_dim, hidden, batch_first=True)
        self.use_bsetd = use_bsetd
        head_in = hidden * 2 + (K if use_bsetd else 0)
        self.head = nn.Sequential(nn.Linear(head_in, hidden), nn.ReLU(),
                                  nn.Linear(hidden, K))

    def forward(self, feat, same_mask, ctx_mask, bsetd_feat):
        # feat: (B, L, F), same_mask: (B, L) where 1 means same-speaker as target
        # Inertia: zero out other-speaker positions
        feat_in = feat * same_mask.unsqueeze(-1)
        feat_co = feat * (1 - same_mask).unsqueeze(-1)
        out_in, _ = self.gru_inertia(feat_in)
        out_co, _ = self.gru_contagion(feat_co)
        # Mean-pool over valid positions
        denom_in = (same_mask * ctx_mask).sum(1, keepdim=True).clamp_min(1.0)
        denom_co = ((1 - same_mask) * ctx_mask).sum(1, keepdim=True).clamp_min(1.0)
        pool_in = (out_in * (same_mask * ctx_mask).unsqueeze(-1)).sum(1) / denom_in
        pool_co = (out_co * ((1 - same_mask) * ctx_mask).unsqueeze(-1)).sum(1) / denom_co
        h = torch.cat([pool_in, pool_co], dim=-1)
        if self.use_bsetd:
            h = torch.cat([h, bsetd_feat], dim=-1)
        return self.head(h)


def run_one(use_bsetd: bool, train_loader: DataLoader, val_loader: DataLoader,
             input_dim: int, epochs: int = 10, lr: float = 1e-3, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = TwoStreamGRU(input_dim, hidden=64, use_bsetd=use_bsetd)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_macro = -1.0
    best_weighted = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        for feat, ctx_mask, same_mask, bsetd, y in train_loader:
            opt.zero_grad()
            logit = model(feat, same_mask, ctx_mask, bsetd)
            loss = F.cross_entropy(logit, y)
            loss.backward()
            opt.step()
        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for feat, ctx_mask, same_mask, bsetd, y in val_loader:
                logit = model(feat, same_mask, ctx_mask, bsetd)
                preds.append(logit.argmax(-1).numpy())
                ys.append(y.numpy())
        ys = np.concatenate(ys); preds = np.concatenate(preds)
        from sklearn.metrics import f1_score
        macro = f1_score(ys, preds, average='macro', zero_division=0)
        weighted = f1_score(ys, preds, average='weighted', zero_division=0)
        if macro > best_macro:
            best_macro = macro
            best_weighted = weighted
    return {'macro_f1': float(best_macro), 'weighted_f1': float(best_weighted)}


def main() -> None:
    df = pd.read_parquet(ROOT / 'data_processed' / 'emotionlines_softlabels_v2_bsetd.parquet')
    s1 = np.load(ROOT / 'experiments' / 'stage1_emotionlines' / 'stage1_total.npz')
    T = s1['transition_post_mean']
    smoothed = s1['counts_total'] + s1['alpha']
    marginal = smoothed.sum(axis=0) / smoothed.sum()
    log2_lift = np.log2(np.maximum(T / np.maximum(marginal[None, :], 1e-12), 1e-12)).astype(np.float32)

    rng = np.random.default_rng(0)
    dialog_ids = df['dialog_id'].unique()
    rng.shuffle(dialog_ids)
    n_train = int(0.8 * len(dialog_ids))
    train_ids = set(dialog_ids[:n_train])
    val_ids = set(dialog_ids[n_train:])

    top_speakers = (
        df['speaker_id'].value_counts().head(N_TOP_SPEAKERS).index.tolist()
    )

    train_df = df[df['dialog_id'].isin(train_ids)]
    val_df = df[df['dialog_id'].isin(val_ids)]
    train_ds = DialogDataset(train_df, log2_lift, top_speakers)
    val_ds = DialogDataset(val_df, log2_lift, top_speakers)
    print(f'train samples={len(train_ds)} val samples={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)
    input_dim = K + (N_TOP_SPEAKERS + 1) + 1  # soft + speaker + is_same

    results = {}
    for seed in [0, 1, 2]:
        for use_bsetd in [False, True]:
            print(f'seed={seed} use_bsetd={use_bsetd}...')
            r = run_one(use_bsetd, train_loader, val_loader, input_dim, epochs=6, seed=seed)
            tag = f"{'with_bsetd' if use_bsetd else 'baseline'}_seed{seed}"
            results[tag] = r
            print(f'  macro-F1={r["macro_f1"]:.4f} weighted-F1={r["weighted_f1"]:.4f}')

    base_m = np.mean([v['macro_f1'] for k, v in results.items() if 'baseline' in k])
    bs_m = np.mean([v['macro_f1'] for k, v in results.items() if 'with_bsetd' in k])
    base_w = np.mean([v['weighted_f1'] for k, v in results.items() if 'baseline' in k])
    bs_w = np.mean([v['weighted_f1'] for k, v in results.items() if 'with_bsetd' in k])
    summary = {
        'per_run': results,
        'baseline_macro_f1_mean': float(base_m),
        'with_bsetd_macro_f1_mean': float(bs_m),
        'baseline_weighted_f1_mean': float(base_w),
        'with_bsetd_weighted_f1_mean': float(bs_w),
        'delta_macro_f1': float(bs_m - base_m),
        'delta_weighted_f1': float(bs_w - base_w),
    }
    out_path = ROOT / 'experiments' / 'stage1_emotionlines' / 'emotionic_baseline.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {out_path}')
    print(f'baseline macro-F1: {base_m:.4f}  weighted-F1: {base_w:.4f}')
    print(f'with_bsetd macro-F1: {bs_m:.4f}  weighted-F1: {bs_w:.4f}')
    print(f'delta_macro: {bs_m - base_m:+.4f}  delta_weighted: {bs_w - base_w:+.4f}')


if __name__ == '__main__':
    main()
