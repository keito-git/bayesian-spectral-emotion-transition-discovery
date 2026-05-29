"""Submit the DailyDialog GPT-5.4-mini N=5 batch in chunks under the OpenAI 50K limit.

The full request file has ~515k requests, which exceeds the OpenAI Batch API
50,000-request per-batch cap. We split it into ~50k-line chunks, submit each
as an independent batch, persist the batch IDs, and aggregate results once
all batches complete.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bsetd.dailydialog_llm_softlabel import (
    _read_openai_key, BATCH_DIR, load_dailydialog,
    build_batch_requests, EMOTIONS, K,
)

CHUNK_SIZE = 45_000  # safe margin under the 50,000 cap


def split_jsonl(in_path: Path, out_dir: Path, chunk_size: int = CHUNK_SIZE) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    chunk_idx = 0
    written = 0
    fh = None
    with in_path.open("r") as src:
        for line in src:
            if fh is None:
                out_path = out_dir / f"chunk_{chunk_idx:03d}.jsonl"
                fh = out_path.open("w")
                paths.append(out_path)
            fh.write(line)
            written += 1
            if written >= chunk_size:
                fh.close()
                fh = None
                chunk_idx += 1
                written = 0
    if fh is not None:
        fh.close()
    return paths


def submit_all(chunk_paths: list[Path]) -> list[str]:
    from openai import OpenAI
    client = OpenAI(api_key=_read_openai_key())
    batch_ids: list[str] = []
    for p in chunk_paths:
        upload = client.files.create(file=p.open("rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"project": "bsetd-dailydialog", "chunk": p.name},
        )
        batch_ids.append(batch.id)
        print(f"  {p.name} -> batch_id={batch.id}")
    return batch_ids


def main() -> None:
    src_path = BATCH_DIR / "request_n5.jsonl"
    if not src_path.exists():
        raise FileNotFoundError(
            f"{src_path} not found. Run dailydialog_llm_softlabel.py first to produce it."
        )
    out_dir = BATCH_DIR / "chunks"
    paths = split_jsonl(src_path, out_dir, chunk_size=CHUNK_SIZE)
    print(f"Split into {len(paths)} chunks of <= {CHUNK_SIZE} requests each")
    for p in paths:
        n = sum(1 for _ in p.open())
        print(f"  {p.name}: {n} requests")

    print("\nSubmitting chunks to OpenAI...")
    batch_ids = submit_all(paths)
    out_ids = BATCH_DIR / "batch_ids.json"
    json.dump({"batch_ids": batch_ids, "chunk_count": len(paths)}, open(out_ids, "w"), indent=2)
    print(f"\nWrote {out_ids}")


if __name__ == "__main__":
    main()
