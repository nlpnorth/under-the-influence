"""
Download and preprocess English Wikipedia into the chunk_XX/train_full.txt
layout expected by the goldfish training scripts and the BEAR-facts corpus
filtering pipeline (one sentence per line, plain text).

Run this LOCALLY (the cluster has unreliable internet access), then rsync the
resulting data/wikipedia/ directory to $WIKI_ROOT on the server (see
config/env.sh), e.g.

    rsync -avP data/wikipedia/ <user>@<host>:<WIKI_ROOT>/

Source: the "wikimedia/wikipedia" HuggingFace dataset (pre-extracted, cleaned
plain text — no wiki markup/XML-dump parsing needed). Streamed rather than
fully cached locally, since we only need the derived sentence files, not the
~20GB raw dataset cache.

Output: data/wikipedia/chunk_00/train_full.txt ... chunk_<N-1>/train_full.txt
  — one sentence per line, articles round-robined across chunks so each chunk
  is a representative, similarly-sized sample of the full corpus.

Usage
-----
    python data/scripts/setup_wikipedia_data.py
    python data/scripts/setup_wikipedia_data.py --num-chunks 20 --num-workers 8
    python data/scripts/setup_wikipedia_data.py --max-articles 100000  # smoke test

Prerequisites
-------------
    pip install datasets nltk
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "wikipedia"


def load_hf_token() -> str | None:
    """Load HF_TOKEN from the environment or the repo .env file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        token = os.environ.get("HF_TOKEN")
        if token:
            return token
    except ImportError:
        pass

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "HF_TOKEN":
            token = value.strip().strip("\"'")
            if token:
                os.environ["HF_TOKEN"] = token
                return token
    return None

HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.en"

MIN_SENTENCE_CHARS = 20
_HEADING_RE = re.compile(r"^=+.*=+$")


def _ensure_nltk_punkt() -> None:
    import nltk

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def _article_to_sentences(text: str) -> list[str]:
    """Split one Wikipedia article's plain text into clean, one-per-line sentences."""
    from nltk.tokenize import sent_tokenize

    sentences: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph or _HEADING_RE.match(paragraph):
            continue
        for sent in sent_tokenize(paragraph):
            sent = " ".join(sent.split())  # collapse internal whitespace
            if len(sent) >= MIN_SENTENCE_CHARS:
                sentences.append(sent)
    return sentences


def _init_worker() -> None:
    _ensure_nltk_punkt()


def _outputs_complete(output_dir: Path, num_chunks: int) -> bool:
    chunk_files = (output_dir / f"chunk_{i:02d}" / "train_full.txt" for i in range(num_chunks))
    return all(f.exists() and f.stat().st_size > 0 for f in chunk_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-chunks", type=int, default=20)
    parser.add_argument(
        "--num-workers", type=int, default=max(1, multiprocessing.cpu_count() - 1),
        help="Parallel workers for sentence segmentation (CPU-bound).",
    )
    parser.add_argument(
        "--max-articles", type=int, default=None,
        help="Stop after this many articles (for a quick smoke test).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if chunk_XX/train_full.txt files already exist.",
    )
    args = parser.parse_args()

    if not args.force and _outputs_complete(args.output_dir, args.num_chunks):
        print(f"All {args.num_chunks} chunks already exist under {args.output_dir} "
              "— skipping. Pass --force to rebuild.")
        return

    _ensure_nltk_punkt()

    from datasets import load_dataset

    token = load_hf_token()
    if token:
        print("Using HF_TOKEN from environment/.env for authenticated download.")
    else:
        print("Warning: HF_TOKEN not found in environment or .env; continuing unauthenticated.")

    print(f"Streaming {HF_DATASET} ({HF_CONFIG}) ...")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True, token=token)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writers = []
    for i in range(args.num_chunks):
        chunk_dir = args.output_dir / f"chunk_{i:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        writers.append(open(chunk_dir / "train_full.txt", "w", encoding="utf-8"))

    def texts():
        for n, row in enumerate(ds):
            if args.max_articles is not None and n >= args.max_articles:
                return
            yield row["text"]

    n_articles = n_sentences = n_words = 0
    start = time.time()
    try:
        with multiprocessing.Pool(args.num_workers, initializer=_init_worker) as pool:
            for sentences in pool.imap(_article_to_sentences, texts(), chunksize=64):
                writer = writers[n_articles % args.num_chunks]
                for sent in sentences:
                    writer.write(sent + "\n")
                    n_sentences += 1
                    n_words += len(sent.split())
                n_articles += 1
                if n_articles % 50_000 == 0:
                    elapsed = time.time() - start
                    print(f"  {n_articles:,} articles | {n_sentences:,} sentences | "
                          f"{n_words / 1e9:.2f}B words | {elapsed:.0f}s elapsed "
                          f"({n_articles / elapsed:.0f} articles/s)", flush=True)
    finally:
        for w in writers:
            w.close()

    print(f"\nDone: {n_articles:,} articles -> {n_sentences:,} sentences, "
          f"~{n_words / 1e9:.2f}B words across {args.num_chunks} chunks "
          f"in {args.output_dir}")
    print("(word count is a rough proxy for token count — actual BPE token count "
          "will differ slightly)")


if __name__ == "__main__":
    main()
