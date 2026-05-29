"""BSETD synthetic ablation: ground-truth recovery accuracy.

Generates synthetic dialog corpora from a known transition matrix and
soft-label generation process, runs Stage 1 (and optionally Stage 2),
and reports per-edge recovery metrics (precision, recall, F1) as a
function of:
    - true edge structure (chain / fork / star / cycle)
    - soft-label sharpness (annotator agreement rate p_acc and rater count R)
    - dialog count (N_dialogs)
    - mean dialog length (L)

The point is to show that BSETD recovers the ground-truth transition
structure under realistic dialog-scale soft-label regimes, including
when the soft labels are noisy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import sys
ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from bsetd.stage1_dirichlet import (
    EMOTIONS, K, TransitionCounts,
    soft_transition_counts, fit_transition_matrix,
)

OUT_DIR = ROOT / "experiments" / "bsetd_synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


TOPOLOGIES: dict[str, list[tuple[int, int]]] = {
    "chain":  [(0, 1), (1, 2), (2, 3), (3, 4)],
    "fork":   [(0, 1), (0, 2), (0, 3)],
    "star":   [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)],
    "cycle":  [(0, 1), (1, 2), (2, 3), (3, 0)],
}


def make_transition_matrix(
    edges: list[tuple[int, int]], on_value: float = 0.55, off_value: float = 0.05
) -> np.ndarray:
    """Build a K x K row-stochastic ground-truth transition matrix.

    Each source row j puts on_value on its preferred successor and
    spreads off_value uniformly over the remaining K-1 cells (subject
    to rowsum=1). Multiple edges out of the same source share on_value
    equally.
    """
    T = np.full((K, K), off_value)
    by_src: dict[int, list[int]] = {}
    for j, k in edges:
        by_src.setdefault(j, []).append(k)
    for j, ks in by_src.items():
        share = on_value / len(ks)
        for k in ks:
            T[j, k] = share
    # Renormalize each row
    T = T / T.sum(axis=1, keepdims=True)
    return T


def sample_dialog(
    T: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a single dialog as a sequence of integer emotion ids of given length.

    Starts from a uniform initial distribution, then walks the Markov chain
    defined by T.
    """
    n = T.shape[0]
    seq = np.empty(length, dtype=int)
    seq[0] = int(rng.integers(0, n))
    for t in range(1, length):
        seq[t] = int(rng.choice(n, p=T[seq[t - 1]]))
    return seq


def to_soft_label(
    e: int,
    R: int,
    p_acc: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a soft label by simulating R noisy annotators.

    Each annotator votes for the true emotion e with probability p_acc,
    otherwise uniformly over the remaining K-1 categories.
    """
    votes = np.zeros(K)
    for _ in range(R):
        if rng.random() < p_acc:
            votes[e] += 1
        else:
            wrong = int(rng.choice([k for k in range(K) if k != e]))
            votes[wrong] += 1
    return votes / R


def build_corpus(
    T: np.ndarray,
    n_dialogs: int,
    mean_length: int,
    R: int,
    p_acc: float,
    rng: np.random.Generator,
) -> tuple[list[list[np.ndarray]], list[list[int]]]:
    """Build a synthetic corpus of soft-label dialogs.

    Returns (dialogs, speakers) where each dialog is a list of K-dim soft labels
    and each speaker list alternates 0/1 within a dialog.
    """
    dialogs = []
    speakers = []
    for _ in range(n_dialogs):
        L = max(2, rng.poisson(mean_length))
        emo_seq = sample_dialog(T, L, rng)
        soft = [to_soft_label(int(e), R, p_acc, rng) for e in emo_seq]
        spk = [t % 2 for t in range(L)]
        dialogs.append(soft)
        speakers.append(spk)
    return dialogs, speakers


def aggregate_synthetic(
    dialogs: list[list[np.ndarray]],
    speakers: list[list[int]],
) -> TransitionCounts:
    total = np.zeros((K, K))
    inertia = np.zeros((K, K))
    contagion = np.zeros((K, K))
    n_pairs = 0
    for soft, spk in zip(dialogs, speakers):
        tc = soft_transition_counts(soft, speaker_ids=[str(s) for s in spk])
        total += tc.total
        inertia += tc.inertia
        contagion += tc.contagion
        n_pairs += tc.n_transitions
    return TransitionCounts(
        total=total, inertia=inertia, contagion=contagion, n_transitions=n_pairs
    )


def evaluate_recovery(
    T_hat: np.ndarray,
    rejected: np.ndarray,
    edges: list[tuple[int, int]],
) -> dict:
    """Compute precision/recall/F1 for the discovered edges vs. the ground truth set."""
    truth = set((j, k) for j, k in edges)
    predicted = {(j, k) for j in range(K) for k in range(K) if j != k and rejected[j, k]}
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    # Lift-ranked recovery on top-|truth| edges
    lift_rows = []
    mask = ~np.eye(K, dtype=bool)
    smoothed_marginal = T_hat.sum(axis=0) / K
    smoothed_marginal = np.maximum(smoothed_marginal, 1e-12)
    lift = T_hat / smoothed_marginal[None, :]
    log2_lift = np.log2(np.maximum(lift, 1e-12))
    abs_lift_score = np.where(mask, np.abs(log2_lift), -np.inf)
    flat = sorted(
        [(j, k, abs_lift_score[j, k]) for j in range(K) for k in range(K) if j != k],
        key=lambda x: -x[2],
    )
    top_K_edges = {(j, k) for j, k, _ in flat[: len(truth)]}
    top_K_recall = len(truth & top_K_edges) / max(len(truth), 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_bh": prec,
        "recall_bh": rec,
        "f1_bh": f1,
        "top_K_recall_by_lift": top_K_recall,
    }


@dataclass
class RunConfig:
    topology: str
    n_dialogs: int
    mean_length: int
    rater_count: int
    p_acc: float
    seed: int


def run_one(cfg: RunConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    edges = TOPOLOGIES[cfg.topology]
    T_true = make_transition_matrix(edges)
    dialogs, speakers = build_corpus(
        T_true, cfg.n_dialogs, cfg.mean_length, cfg.rater_count, cfg.p_acc, rng
    )
    counts = aggregate_synthetic(dialogs, speakers)
    result = fit_transition_matrix(counts)
    metrics = evaluate_recovery(result.transition_post_mean, result.rejected_bh, edges)
    return {
        "topology": cfg.topology,
        "n_dialogs": cfg.n_dialogs,
        "mean_length": cfg.mean_length,
        "R": cfg.rater_count,
        "p_acc": cfg.p_acc,
        "seed": cfg.seed,
        **metrics,
    }


def main() -> None:
    rows = []
    grid = []
    for topo in TOPOLOGIES:
        for n_dialogs in [200, 500]:
            for L in [10, 20]:
                for R in [5]:
                    for p_acc in [0.95, 0.75, 0.55]:
                        for seed in range(3):
                            grid.append(RunConfig(topo, n_dialogs, L, R, p_acc, seed))
    print(f"Running {len(grid)} configurations...")
    for cfg in grid:
        row = run_one(cfg)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "synthetic_ablation.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    summary = df.groupby(["topology", "n_dialogs", "mean_length", "p_acc"]).agg(
        f1_mean=("f1_bh", "mean"),
        f1_std=("f1_bh", "std"),
        topk_recall_mean=("top_K_recall_by_lift", "mean"),
    ).reset_index()
    out_summary = OUT_DIR / "synthetic_ablation_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
