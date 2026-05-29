"""Download and aggregate the 12 DailyDialog GPT batch outputs into a single
soft-label parquet, then run BSETD Stage 1+2 with the LLM-generated soft labels.

Waits until every batch reaches a terminal state (completed / failed /
cancelled / expired) before doing anything, then downloads each batch's
output JSONL, concatenates them, parses N=5 votes per utterance into
Ekman-7 soft labels, and writes
data/processed/dailydialog_softlabels_llm.parquet.

Finally runs BSETD Stage 1+2 on the LLM-soft labels and appends the
five-corpus pairwise Pearson correlations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsetd.dailydialog_llm_softlabel import (
    _read_openai_key, BATCH_DIR, load_dailydialog,
    aggregate_votes, EMOTIONS, K,
)
from bsetd.stage1_dirichlet import (
    soft_transition_counts, fit_inertia_contagion, TransitionCounts,
)
from bsetd.stage2_spectral import run_stage2_on_stage1_npz


def wait_all_terminal(batch_ids: list[str], poll_sec: int = 60, max_wait_sec: int = 86400) -> dict[str, dict]:
    from openai import OpenAI
    client = OpenAI(api_key=_read_openai_key())
    start = time.time()
    while True:
        infos = {bid: client.batches.retrieve(bid).model_dump() for bid in batch_ids}
        terminal = {'completed', 'failed', 'cancelled', 'expired'}
        statuses = [b['status'] for b in infos.values()]
        n_done = sum(1 for s in statuses if s in terminal)
        elapsed = int(time.time() - start)
        print(f'[t+{elapsed}s] terminal: {n_done}/{len(batch_ids)}  states: '
              + ', '.join(f"{s}={statuses.count(s)}" for s in sorted(set(statuses))))
        if n_done == len(batch_ids):
            return infos
        if time.time() - start > max_wait_sec:
            raise TimeoutError(f"Batches did not finish in {max_wait_sec}s")
        time.sleep(poll_sec)


def download_outputs(infos: dict[str, dict], out_jsonl: Path) -> int:
    from openai import OpenAI
    client = OpenAI(api_key=_read_openai_key())
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with out_jsonl.open("w") as fout:
        for bid, b in infos.items():
            if b['status'] != 'completed':
                print(f'  SKIP {bid[:30]}: status={b["status"]}')
                continue
            of = b.get('output_file_id')
            if not of:
                continue
            text = client.files.content(of).text
            fout.write(text)
            if not text.endswith('\n'):
                fout.write('\n')
            n_lines += sum(1 for _ in text.splitlines())
    return n_lines


def main() -> None:
    ids_file = BATCH_DIR / 'batch_ids.json'
    if not ids_file.exists():
        raise FileNotFoundError(ids_file)
    batch_ids = json.load(open(ids_file))['batch_ids']
    print(f'Waiting for {len(batch_ids)} batches...')
    infos = wait_all_terminal(batch_ids, poll_sec=60)

    results_path = BATCH_DIR / 'results_all.jsonl'
    n_lines = download_outputs(infos, results_path)
    print(f'Downloaded {n_lines} response lines to {results_path}')

    print('Loading DailyDialog dialogs to map custom_id -> utterance...')
    dialogs = load_dailydialog()
    df = aggregate_votes(results_path, dialogs, n_samples=5)
    out_parquet = ROOT / 'data_processed' / 'dailydialog_softlabels_llm.parquet'
    df.to_parquet(out_parquet)
    print(f'Wrote {out_parquet} ({len(df)} utterances)')

    counts_total = np.zeros((K, K))
    counts_in = np.zeros((K, K))
    counts_co = np.zeros((K, K))
    n_pairs = 0
    for _, sub in df.sort_values(['dialog_id', 'turn_id']).groupby('dialog_id'):
        soft = list(sub['p_dist'].apply(np.asarray).to_numpy())
        spk = list(sub['speaker_id'].to_numpy())
        tc = soft_transition_counts(soft, speaker_ids=spk)
        counts_total += tc.total
        counts_in += tc.inertia
        counts_co += tc.contagion
        n_pairs += tc.n_transitions
    counts = TransitionCounts(total=counts_total, inertia=counts_in,
                                contagion=counts_co, n_transitions=n_pairs)
    fits = fit_inertia_contagion(counts)
    out_dir = ROOT / 'experiments' / 'stage1_emotionlines' / 'dailydialog_llm'
    out_dir.mkdir(parents=True, exist_ok=True)
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
            'eb_converged': res.metadata['eb_converged'],
            'n_rejected_offdiag': int(res.rejected_bh.sum()),
        }
        s2 = run_stage2_on_stage1_npz(npz, ROOT / 'experiments' / 'stage2_emotionlines' / 'dailydialog_llm')
        summary[key]['stage2_inertia_index'] = s2['inertia_index']
        summary[key]['stage2_contagion_index'] = s2['contagion_index']

    mask = ~np.eye(K, dtype=bool)
    A_llm = np.load(out_dir / 'stage1_total.npz')['transition_post_mean']
    for ref in ('emotionlines', 'meld', 'dailydialog', 'm3ed'):
        try:
            A_ref = np.load(ROOT / 'experiments' / 'stage1_emotionlines' / ref / 'stage1_total.npz')['transition_post_mean']
            pear = float(np.corrcoef(A_llm[mask], A_ref[mask])[0, 1])
            summary[f'dailydialog_llm_vs_{ref}_pearson'] = pear
        except FileNotFoundError:
            pass
    summary['llm_diag'] = np.diag(A_llm).tolist()
    json.dump(summary, open(out_dir / 'stage1_summary.json', 'w'), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if 'pearson' in k or 'diag' in k}, indent=2))


if __name__ == '__main__':
    main()
