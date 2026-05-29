"""DailyDialog LLM soft-label pipeline via the OpenAI Batch API.

Generates virtual multi-annotator Ekman-7 soft labels for each utterance
in DailyDialog by sampling an LLM N=5 times with temperature 1.0 and
aggregating the votes into a probability vector, following the AnnoLLM
framing of LLMs as virtual annotators.

This module downloads DailyDialog, builds OpenAI Batch API request files,
submits the batch, polls until complete, parses N=5 votes per utterance,
and writes a parquet file in the BSETD schema.

The OPENAI_API_KEY environment variable is required. Execution is gated
behind --execute; the default run only writes the request file and prints
a cost estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

EMOTIONS = ['neutral', 'joy', 'sadness', 'fear', 'anger', 'surprise', 'disgust']
K = 7
ROOT = Path(".")
RAW_DIR = ROOT / "data" / "raw" / "dailydialog"
OUT_DIR = ROOT / "data_processed"
BATCH_DIR = ROOT / "data" / "raw" / "dailydialog_batches"

# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------


def _read_openai_key() -> str:
    """Read an OpenAI API key from the OPENAI_API_KEY environment variable.
    Never log the key.
    """
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]
    raise RuntimeError(
        "OPENAI_API_KEY not found in environment. Export it before executing the batch."
    )


# ---------------------------------------------------------------------------
# DailyDialog loader
# ---------------------------------------------------------------------------


DD_HARD_LABEL = {0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear',
                 4: 'joy', 5: 'sadness', 6: 'surprise'}


def load_dailydialog() -> list[list[dict]]:
    """Load DailyDialog dialogues from the HF datasets parquet mirror
    or, if unavailable, from raw text files under ``RAW_DIR``.

    Each utterance dict contains: utterance, hard_label_dd, speaker_id,
    dialog_id, turn_id.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("benjamin-paine/daily_dialog")
        dialogs = []
        for split in ['train', 'validation', 'test']:
            if split not in ds:
                continue
            for d_idx, ex in enumerate(ds[split]):
                utts = ex['dialog']
                emos = ex['emotion']
                dialog_id = f"dd_{split[:2]}_{d_idx:05d}"
                row = []
                for t, (u, e) in enumerate(zip(utts, emos)):
                    row.append({
                        'dialog_id': dialog_id,
                        'turn_id': t,
                        'speaker_id': f"{dialog_id}_spk_{t % 2}",
                        'utterance': u.strip(),
                        'hard_label_dd': DD_HARD_LABEL.get(int(e), 'neutral'),
                    })
                dialogs.append(row)
        return dialogs
    except Exception:
        pass
    return _load_dailydialog_raw_text()


def _load_dailydialog_raw_text() -> list[list[dict]]:
    """Load from raw text files distributed with the DailyDialog release.

    Expected layout: RAW_DIR / {train,validation,test} / {dialogues_<split>.txt,
    dialogues_emotion_<split>.txt}. The roskoN/dailydialog HF mirror produces
    this layout after unzipping.
    """
    text_lines: list[str] = []
    emo_lines: list[str] = []
    splits_found = []
    for split in ("train", "validation", "test"):
        sub = RAW_DIR / split
        t_file = sub / f"dialogues_{split}.txt"
        e_file = sub / f"dialogues_emotion_{split}.txt"
        if t_file.exists() and e_file.exists():
            text_lines.extend(t_file.read_text(encoding="utf-8").splitlines())
            emo_lines.extend(e_file.read_text(encoding="utf-8").splitlines())
            splits_found.append(split)
    if not text_lines:
        raise FileNotFoundError(
            f"DailyDialog raw text files not found under {RAW_DIR}. "
            "Download train.zip / validation.zip / test.zip from "
            "https://huggingface.co/datasets/roskoN/dailydialog and "
            "unzip them under data/raw/dailydialog/."
        )
    dialogs = []
    for d_idx, (tline, eline) in enumerate(zip(text_lines, emo_lines)):
        utts = [u.strip() for u in tline.split("__eou__") if u.strip()]
        emos = [int(e) for e in eline.split() if e.strip()]
        n = min(len(utts), len(emos))
        if n < 2:
            continue
        dialog_id = f"dd_{d_idx:05d}"
        row = [
            {
                'dialog_id': dialog_id,
                'turn_id': t,
                'speaker_id': f"{dialog_id}_spk_{t % 2}",
                'utterance': utts[t],
                'hard_label_dd': DD_HARD_LABEL.get(emos[t], 'neutral'),
            }
            for t in range(n)
        ]
        dialogs.append(row)
    return dialogs


# ---------------------------------------------------------------------------
# Batch request building
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are one of five independent annotators labeling the emotion of "
    "single utterances drawn from natural English dialogues. "
    "Choose exactly one Ekman emotion category from this set: "
    "neutral, joy, sadness, fear, anger, surprise, disgust. "
    "Your judgment is your own and may differ from other annotators. "
    "Respond with the single category word and nothing else."
)


def _build_user_prompt(utterance: str, context_before: list[str]) -> str:
    ctx = "\n".join(f"- {c}" for c in context_before[-3:]) if context_before else "(none)"
    return (
        f"Recent dialog context:\n{ctx}\n\n"
        f"Target utterance: {utterance}\n\n"
        f"Your single-word emotion label:"
    )


def build_batch_requests(
    dialogs: list[list[dict]],
    n_samples: int = 5,
    model: str = "gpt-5.4-mini",
) -> Iterable[dict]:
    """Yield one Batch API request line per (utterance, sample) pair."""
    for dialog in dialogs:
        context_before: list[str] = []
        for utt in dialog:
            request_key_base = f"{utt['dialog_id']}::{utt['turn_id']}"
            user_msg = _build_user_prompt(utt['utterance'], context_before)
            for s in range(n_samples):
                custom_id = f"{request_key_base}::sample{s}"
                yield {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 1.0,
                        "max_completion_tokens": 10,
                        "n": 1,
                    },
                }
            context_before.append(utt['utterance'])


def write_batch_file(dialogs: list[list[dict]], out_path: Path, n_samples: int = 5) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for req in build_batch_requests(dialogs, n_samples=n_samples):
            f.write(json.dumps(req) + "\n")
            n += 1
    return n


def estimate_cost(n_requests: int, model: str = "gpt-5.4-mini") -> dict:
    """Rough USD cost estimate using batch-tier prices."""
    approx_input_tokens_per_req = 95
    approx_output_tokens_per_req = 3
    in_tok = n_requests * approx_input_tokens_per_req
    out_tok = n_requests * approx_output_tokens_per_req
    in_usd = in_tok / 1_000_000 * 0.20
    out_usd = out_tok / 1_000_000 * 0.80
    return {
        "n_requests": n_requests,
        "approx_input_tokens": in_tok,
        "approx_output_tokens": out_tok,
        "estimated_cost_usd_batch": round(in_usd + out_usd, 2),
    }


# ---------------------------------------------------------------------------
# Batch submission / polling / parsing
# ---------------------------------------------------------------------------


def submit_batch(batch_file: Path) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=_read_openai_key())
    upload = client.files.create(file=batch_file.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"project": "bsetd-dailydialog"},
    )
    return batch.id


def poll_batch(batch_id: str, poll_sec: int = 60) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=_read_openai_key())
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in {"completed", "failed", "cancelled", "expired"}:
            return batch.model_dump()
        time.sleep(poll_sec)


def parse_batch_output(output_file_id: str, out_jsonl: Path) -> int:
    from openai import OpenAI
    client = OpenAI(api_key=_read_openai_key())
    content = client.files.content(output_file_id).text
    out_jsonl.write_text(content)
    return sum(1 for _ in content.splitlines())


def aggregate_votes(output_jsonl: Path, dialogs: list[list[dict]],
                     n_samples: int = 5) -> "pandas.DataFrame":
    import pandas as pd
    label_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
    votes: dict[str, list[str]] = {}
    for line in output_jsonl.read_text().splitlines():
        obj = json.loads(line)
        cid = obj["custom_id"]
        parts = cid.split("::")
        key = "::".join(parts[:2])
        choice = obj.get("response", {}).get("body", {}).get("choices", [])
        if not choice:
            continue
        text = choice[0]["message"]["content"].strip().lower()
        matched = None
        for emo in EMOTIONS:
            if emo in text:
                matched = emo
                break
        if matched is None:
            matched = 'neutral'
        votes.setdefault(key, []).append(matched)
    rows = []
    for dialog in dialogs:
        for utt in dialog:
            key = f"{utt['dialog_id']}::{utt['turn_id']}"
            sampled = votes.get(key, [])
            counts = [0] * K
            for lab in sampled:
                counts[label_to_idx[lab]] += 1
            total = sum(counts)
            if total == 0:
                continue
            p = [c / total for c in counts]
            rows.append({
                'dataset_source': 'dailydialog_llm',
                'dialog_id': utt['dialog_id'],
                'turn_id': utt['turn_id'],
                'speaker_id': utt['speaker_id'],
                'utterance': utt['utterance'],
                'hard_label_dd': utt['hard_label_dd'],
                'vote_counts': counts,
                'n_raters': total,
                'p_dist': p,
                'hard_label': EMOTIONS[counts.index(max(counts))],
                'has_disagreement': max(counts) < total,
                'is_non_neutral': max(counts) <= total / 2,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--out", type=Path, default=BATCH_DIR / "request_n5.jsonl")
    parser.add_argument("--execute", action="store_true",
                        help="Actually submit the batch (otherwise dry-run + cost).")
    parser.add_argument("--submit-only", action="store_true",
                        help="Submit and return batch_id without polling (24h completion).")
    parser.add_argument("--results", type=Path, default=BATCH_DIR / "results_n5.jsonl",
                        help="Where to save the downloaded batch output.")
    parser.add_argument("--parquet", type=Path,
                        default=OUT_DIR / "dailydialog_softlabels_llm.parquet")
    args = parser.parse_args()

    print("Loading DailyDialog...")
    dialogs = load_dailydialog()
    n_dialogs = len(dialogs)
    n_utts = sum(len(d) for d in dialogs)
    print(f"  dialogs={n_dialogs}, utterances={n_utts}")

    n_req = write_batch_file(dialogs, args.out, n_samples=args.n_samples)
    cost = estimate_cost(n_req)
    print(f"Wrote batch file: {args.out} ({n_req} requests)")
    print(json.dumps(cost, indent=2))

    if not args.execute:
        print("\nDry-run only. Re-run with --execute after reviewing the cost.")
        return

    print("\nSubmitting batch to OpenAI...")
    batch_id = submit_batch(args.out)
    print(f"batch_id={batch_id}")
    (BATCH_DIR / "batch_id.txt").write_text(batch_id + "\n")
    print(f"Saved batch_id to {BATCH_DIR / 'batch_id.txt'}")
    if args.submit_only:
        print("--submit-only set, exiting before polling. Run polling separately.")
        return
    batch_state = poll_batch(batch_id)
    print(json.dumps(batch_state, indent=2))
    if batch_state["status"] != "completed":
        print(f"Batch ended in non-completed state: {batch_state['status']}")
        sys.exit(1)
    output_file_id = batch_state["output_file_id"]
    n_out = parse_batch_output(output_file_id, args.results)
    print(f"Downloaded {n_out} response lines to {args.results}")

    df = aggregate_votes(args.results, dialogs, n_samples=args.n_samples)
    df.to_parquet(args.parquet)
    print(f"Wrote {args.parquet} ({len(df)} rows)")


if __name__ == "__main__":
    main()
