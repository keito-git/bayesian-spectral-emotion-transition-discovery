"""Run BSETD Stage 1+2 on DailyDialog with one-hot soft labels (hard label baseline).

This serves two purposes: it adds DailyDialog as a third corpus to our cross-corpus
robustness analysis, and it gives the hard-label baseline against which the future
GPT-4o N=5 LLM-soft-label run will be compared. The expected pattern is that
hard-label transitions on DailyDialog show similar off-diagonal structure to the
multi-rater corpora (high Pearson with EmotionLines, lower than EL vs MELD),
quantifying how much structural information is recoverable from hard labels alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from bsetd.dailydialog_llm_softlabel import _load_dailydialog_raw_text, DD_HARD_LABEL
from bsetd.stage1_dirichlet import (
    EMOTIONS, K, TransitionCounts,
    soft_transition_counts, fit_inertia_contagion,
)
from bsetd.stage2_spectral import run_stage2_on_stage1_npz


def label_to_onehot(label: str) -> np.ndarray:
    """Convert string emotion label to one-hot Ekman-7 vector in our ordering."""
    idx_map = {e: i for i, e in enumerate(EMOTIONS)}
    v = np.zeros(K)
    if label in idx_map:
        v[idx_map[label]] = 1.0
    else:
        v[0] = 1.0  # default to neutral
    return v


def aggregate_dailydialog(dialogs: list[list[dict]]) -> TransitionCounts:
    total = np.zeros((K, K))
    inertia = np.zeros((K, K))
    contagion = np.zeros((K, K))
    n_pairs = 0
    for dialog in dialogs:
        soft = [label_to_onehot(u['hard_label_dd']) for u in dialog]
        speakers = [u['speaker_id'] for u in dialog]
        tc = soft_transition_counts(soft, speaker_ids=speakers)
        total += tc.total
        inertia += tc.inertia
        contagion += tc.contagion
        n_pairs += tc.n_transitions
    return TransitionCounts(
        total=total, inertia=inertia, contagion=contagion, n_transitions=n_pairs
    )


def main() -> None:
    out_dir = ROOT / 'experiments' / 'stage1_emotionlines' / 'dailydialog'
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading DailyDialog...")
    dialogs = _load_dailydialog_raw_text()
    print(f"  n_dialogs={len(dialogs)}, n_utts={sum(len(d) for d in dialogs)}")

    counts = aggregate_dailydialog(dialogs)
    fits = fit_inertia_contagion(counts)
    summary = {}
    for key, res in fits.items():
        npz_path = out_dir / f"stage1_{key}.npz"
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
            "n_transitions": res.metadata["n_transitions"],
            "alpha_global": res.metadata["alpha_global"],
            "n_rejected_offdiag": int(res.rejected_bh.sum()),
        }
        s2_dir = ROOT / "experiments" / "stage2_dailydialog"
        s2 = run_stage2_on_stage1_npz(npz_path, s2_dir)
        summary[key]["stage2_inertia_index"] = s2["inertia_index"]
        summary[key]["stage2_contagion_index"] = s2["contagion_index"]

    # Cross-corpus consistency vs EmotionLines and MELD
    el = np.load(ROOT / "experiments" / "stage1_emotionlines" / "stage1_total.npz")
    md = np.load(ROOT / "experiments" / "stage1_meld" / "stage1_total.npz")
    dd = np.load(out_dir / "stage1_total.npz")
    mask = ~np.eye(K, dtype=bool)
    pear_el = float(np.corrcoef(el["transition_post_mean"][mask],
                                 dd["transition_post_mean"][mask])[0, 1])
    pear_md = float(np.corrcoef(md["transition_post_mean"][mask],
                                 dd["transition_post_mean"][mask])[0, 1])
    summary["cross_corpus"] = {
        "dailydialog_vs_emotionlines_pearson": pear_el,
        "dailydialog_vs_meld_pearson": pear_md,
        "dailydialog_diag": np.diag(dd["transition_post_mean"]).tolist(),
    }

    with open(out_dir / "stage1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["cross_corpus"], indent=2))


if __name__ == "__main__":
    main()
