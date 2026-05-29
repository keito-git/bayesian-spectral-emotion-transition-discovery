"""BERT/RoBERTa baseline for next-emotion prediction on EmotionLines.

For each utterance we extract a frozen contextual embedding from a
pretrained BERT-family encoder (default: distilbert-base-uncased to keep
the experiment tractable on CPU/MPS in under 30 minutes). We then train:

    (a) a two-layer MLP head on the encoder embedding + lag-emotion
        histogram, predicting the next utterance's hard emotion;
    (b) the same MLP augmented with the BSETD log2-lift row of the
        previous utterance's modal emotion.

We compare macro-F1 and weighted-F1 across 3 seeds. Encoder is frozen
to keep the comparison fair across seeds and to isolate the BSETD
contribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


EMOTIONS = ['neutral', 'joy', 'sadness', 'fear', 'anger', 'surprise', 'disgust']
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}
K = 7
LAGS = 5
MODEL_NAME = "distilbert-base-uncased"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def extract_embeddings(
    texts: list[str], tokenizer, encoder, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    encoder.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=64)
        enc = {k: v.to(device) for k, v in enc.items()}
        h = encoder(**enc).last_hidden_state
        # Mean pool over valid tokens
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out, axis=0)


class FeatDataset(Dataset):
    def __init__(self, emb: np.ndarray, lag_hist: np.ndarray, lift_feat: np.ndarray, y: np.ndarray):
        self.emb = emb.astype(np.float32)
        self.lag_hist = lag_hist.astype(np.float32)
        self.lift_feat = lift_feat.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.emb[idx], self.lag_hist[idx], self.lift_feat[idx], self.y[idx]


class Head(nn.Module):
    def __init__(self, emb_dim: int, use_bsetd: bool):
        super().__init__()
        self.use_bsetd = use_bsetd
        in_dim = emb_dim + LAGS * K + (K if use_bsetd else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, K),
        )

    def forward(self, emb, lag_hist, lift_feat):
        x = torch.cat([emb, lag_hist], dim=-1)
        if self.use_bsetd:
            x = torch.cat([x, lift_feat], dim=-1)
        return self.net(x)


def build_supervision(df: pd.DataFrame, log2_lift: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    texts = []
    lag_hist_list = []
    lift_list = []
    y_list = []
    for _, sub in df.sort_values(['dialog_id', 'turn_id']).groupby('dialog_id'):
        utts = sub['utterance'].tolist()
        labels = sub['hard_label'].map(EMO2IDX).to_numpy()
        for t in range(LAGS, len(labels)):
            texts.append(utts[t])
            hist = np.zeros((LAGS, K))
            for l in range(LAGS):
                hist[l, labels[t - 1 - l]] = 1.0
            lag_hist_list.append(hist.flatten())
            lift_list.append(log2_lift[labels[t - 1]])
            y_list.append(int(labels[t]))
    return (texts,
            np.asarray(lag_hist_list, dtype=np.float32),
            np.asarray(lift_list, dtype=np.float32),
            np.asarray(y_list, dtype=np.int64))


def train_eval(use_bsetd: bool, train_ds, val_ds, emb_dim, device, seed=0, epochs=10):
    torch.manual_seed(seed)
    model = Head(emb_dim, use_bsetd).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    best_macro = -1.0
    best_w = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        for emb, hist, lift, y in tr_loader:
            emb = emb.to(device); hist = hist.to(device); lift = lift.to(device); y = y.to(device)
            opt.zero_grad()
            logit = model(emb, hist, lift)
            loss = F.cross_entropy(logit, y)
            loss.backward(); opt.step()
        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for emb, hist, lift, y in va_loader:
                emb = emb.to(device); hist = hist.to(device); lift = lift.to(device)
                logit = model(emb, hist, lift)
                preds.append(logit.argmax(-1).cpu().numpy())
                ys.append(y.numpy())
        ys = np.concatenate(ys); preds = np.concatenate(preds)
        macro = f1_score(ys, preds, average='macro', zero_division=0)
        w = f1_score(ys, preds, average='weighted', zero_division=0)
        if macro > best_macro:
            best_macro = macro
            best_w = w
    return {'macro_f1': float(best_macro), 'weighted_f1': float(best_w)}


def main() -> None:
    df = pd.read_parquet(ROOT / 'data_processed' / 'emotionlines_softlabels_v2_bsetd.parquet')
    s1 = np.load(ROOT / 'experiments' / 'stage1_emotionlines' / 'stage1_total.npz')
    T = s1['transition_post_mean']
    smoothed = s1['counts_total'] + s1['alpha']
    marginal = smoothed.sum(axis=0) / smoothed.sum()
    log2_lift = np.log2(np.maximum(T / np.maximum(marginal[None, :], 1e-12), 1e-12)).astype(np.float32)

    device = pick_device()
    print(f'device={device}')
    print(f'Loading {MODEL_NAME}...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    rng = np.random.default_rng(0)
    dialog_ids = df['dialog_id'].unique()
    rng.shuffle(dialog_ids)
    n_train = int(0.8 * len(dialog_ids))
    train_ids = set(dialog_ids[:n_train])
    val_ids = set(dialog_ids[n_train:])
    train_df = df[df['dialog_id'].isin(train_ids)]
    val_df = df[df['dialog_id'].isin(val_ids)]

    print('Building supervision...')
    tr_texts, tr_hist, tr_lift, tr_y = build_supervision(train_df, log2_lift)
    va_texts, va_hist, va_lift, va_y = build_supervision(val_df, log2_lift)
    print(f'train n={len(tr_y)}, val n={len(va_y)}')

    print('Extracting embeddings...')
    tr_emb = extract_embeddings(tr_texts, tokenizer, encoder, device, batch_size=64)
    va_emb = extract_embeddings(va_texts, tokenizer, encoder, device, batch_size=64)
    print(f'tr_emb shape={tr_emb.shape}, va_emb shape={va_emb.shape}')

    tr_ds_base = FeatDataset(tr_emb, tr_hist, tr_lift, tr_y)
    va_ds_base = FeatDataset(va_emb, va_hist, va_lift, va_y)

    results = {}
    for seed in [0, 1, 2]:
        for use in [False, True]:
            tag = f"{'with_bsetd' if use else 'baseline'}_seed{seed}"
            r = train_eval(use, tr_ds_base, va_ds_base, tr_emb.shape[1], device, seed=seed, epochs=8)
            results[tag] = r
            print(f'  {tag}: macro-F1={r["macro_f1"]:.4f}, weighted-F1={r["weighted_f1"]:.4f}')

    base_m = np.mean([v['macro_f1'] for k, v in results.items() if 'baseline' in k])
    bs_m = np.mean([v['macro_f1'] for k, v in results.items() if 'with_bsetd' in k])
    base_w = np.mean([v['weighted_f1'] for k, v in results.items() if 'baseline' in k])
    bs_w = np.mean([v['weighted_f1'] for k, v in results.items() if 'with_bsetd' in k])
    summary = {
        'encoder': MODEL_NAME,
        'device': str(device),
        'per_run': results,
        'baseline_macro_f1_mean': float(base_m),
        'with_bsetd_macro_f1_mean': float(bs_m),
        'baseline_weighted_f1_mean': float(base_w),
        'with_bsetd_weighted_f1_mean': float(bs_w),
        'delta_macro_f1': float(bs_m - base_m),
        'delta_weighted_f1': float(bs_w - base_w),
    }
    out_path = ROOT / 'experiments' / 'stage1_emotionlines' / 'bert_baseline.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {out_path}')
    print(f'baseline (frozen {MODEL_NAME} + lag5): macro-F1={base_m:.4f}, weighted-F1={base_w:.4f}')
    print(f'with_bsetd: macro-F1={bs_m:.4f}, weighted-F1={bs_w:.4f}')
    print(f'delta_macro={bs_m - base_m:+.4f}, delta_weighted={bs_w - base_w:+.4f}')


if __name__ == '__main__':
    main()
