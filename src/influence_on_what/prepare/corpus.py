#!/usr/bin/env python3
"""Deduplicate the parsed corpus and cut it into chunks.

This is the first step of the pipeline.  It turns the released CoNLL-U parse of
Common Corpus into the ``chunk_XX.conllu`` files that ``run/filter_linguistic.sh``
reads.  Everything upstream of it — sentence segmentation and dependency parsing
with MaChAmp — is NOT reproduced here: the parsed corpus is the released
artifact, and it is this module's input.

WHAT IT DOES
  One streaming pass over the input.  A sentence is kept the first time its
  token forms are seen and dropped every time after; kept sentences are written
  to chunk_00.conllu, chunk_01.conllu, …, each closed once it holds
  --tokens-per-chunk tokens.  The pass stops at --target-tokens.

WHY THE HASH IS WHAT IT IS
  md5 over the space-joined FORM column of the plain-integer-id tokens — the
  identical rule used by the original dedup script (clean_data.py in
  github.com/arzuburcuguven/parsed_data), so the corpus this produces is
  deduplicated by the same criterion as the one the paper's models were trained
  on.  Multiword-range rows (7-8) and empty nodes (7.1) are excluded from both
  the hash and the token count, exactly as they were there.

  Only the leading 8 bytes of each digest are stored.  At the ~333M unique
  sentences of the full 10B-token corpus, a Python set of hashes costs ~20 GB of
  interpreter overhead; the open-addressed uint64 table used instead costs 8.6 GB
  at --hash-bits 30, and saves and reloads as a plain .npy — which is what makes
  --resume affordable.  Collision probability at that scale is ~3e-3, and a
  collision drops one sentence from a corpus of hundreds of millions.

CHUNK SIZE
  The default --tokens-per-chunk 65_700_000 is one budget unit, so that a budget
  in config/budgets.sh lands on a chunk boundary.  Deriving it takes two steps,
  and skipping the second costs 10% of every budget:

    1. The 68M-subword model was trained on 331,365,837 bytes of sentence text,
       which at a measured 5.6044 bytes per CoNLL-U token is 59,126,015 tokens.
    2. That is the corpus AFTER run/filter_linguistic.sh's 90/10 split.  What
       this script writes is the corpus BEFORE it, so the chunk has to be
       59,126,015 / 0.90 = 65,695,572 tokens — rounded to 65,700,000 — for the
       split to leave one budget unit behind.

  Sizing a chunk to a rounder number is a trap: at 54M the 68M budget no longer
  fits in one chunk and selects two, overshooting by 84%.  Unlike the corpus
  this project first ran on, which was cut in two passes at two different sizes
  (2.0M sentences for chunk_00–09, 3.25M for chunk_10–55), every chunk here
  holds the same token count.

WHAT IT WRITES  (under --output-dir)
  chunk_XX.conllu    the deduplicated corpus, one file per chunk
  manifest.json      per-chunk sentence and token counts, the inputs consumed,
                     and the resume checkpoint
  seen_hashes.npy    the hash table, rewritten every --checkpoint-every chunks
                     so an interrupted run can be resumed

RESUMING
  --resume restarts from the last checkpoint: chunks written after it are
  discarded and recut, because their sentences are already in the hash table and
  would otherwise be dropped as duplicates.  Resuming still re-reads the input
  from the start (a .gz cannot be seeked), skipping blocks until the checkpoint
  offset — cheap relative to the parse, but not free.

Usage
-----
    python -m influence_on_what.prepare.corpus \\
        --input /path/to/cc-en-10b.conllu.gz \\
        --output-dir "$COMMON_CORPUS_CHUNKS"

    # smoke test: two tiny chunks, nothing else
    python -m influence_on_what.prepare.corpus --input sample.conllu \\
        --output-dir /tmp/chunks --tokens-per-chunk 1000 --target-tokens 2000
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[prep %(levelname)s %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Comments that describe the DOCUMENT rather than the sentence.  MaChAmp emits
# them once, before the document's first sentence, and they carry the per-
# document licence and collection fields that make the released corpus
# redistributable — so they must survive a first sentence that is dropped as a
# duplicate.  See ChunkWriter.note_document.
DOC_COMMENT_PREFIXES = ("# newdoc", "# meta::")


# ── Reading ──────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def open_conllu(path: Path):
    """Open a CoNLL-U file, transparently decompressing .gz.

    Decompression is handed to an external pigz or gzip when one is on PATH: the
    released corpus is ~90 GB compressed, and the gzip module decompresses it
    several times slower than a separate process does on another core.
    """
    if path.suffix != ".gz":
        with path.open("r", encoding="utf-8") as fh:
            yield fh
        return

    # pigz first, gzip second.  Not zcat: on macOS it only understands .Z and
    # fails on a .gz, which would look like an empty corpus rather than an error.
    exe = shutil.which("pigz") or shutil.which("gzip")
    if exe is None:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield fh
        return

    proc = subprocess.Popen([exe, "-dc", str(path)], stdout=subprocess.PIPE)
    try:
        yield io.TextIOWrapper(proc.stdout, encoding="utf-8")
    finally:
        # The reader usually stops early (target reached), leaving the writer
        # blocked on a full pipe; kill it rather than wait for an EOF that is
        # never coming.  A decompressor that has already exited was not
        # interrupted by us, so its status is worth checking — a silent failure
        # here would read as a corpus with no sentences in it.
        finished = proc.poll() is not None
        if not finished:
            proc.kill()
        proc.stdout.close()
        proc.wait()
        if finished and proc.returncode != 0:
            raise RuntimeError(
                f"{exe} failed on {path} (exit {proc.returncode})"
            )


def iter_blocks(fh):
    """Yield one (doc_lines, sent_block, forms) triple per CoNLL-U sentence.

    ``sent_block`` is the verbatim text of the block minus its document-level
    comments, so what is written out is what the parser wrote — no re-
    serialisation.  ``doc_lines`` is non-empty only for a document's first
    sentence.  ``forms`` holds the FORM column of the plain-integer-id tokens,
    which is both what gets hashed and what gets counted as tokens.
    """
    doc_lines: list[str] = []
    sent_lines: list[str] = []
    forms: list[str] = []

    for line in fh:
        if not line.strip():
            if sent_lines or doc_lines:
                yield doc_lines, "".join(sent_lines), forms
                doc_lines, sent_lines, forms = [], [], []
            continue

        if line[0] == "#":
            if line.startswith(DOC_COMMENT_PREFIXES):
                doc_lines.append(line)
            else:
                sent_lines.append(line)
            continue

        sent_lines.append(line)
        fields = line.split("\t")
        # Plain integer ids only: "7-8" (multiword range) and "7.1" (empty
        # node) are not tokens, and clean_data.py excluded them too.
        if fields[0].isdigit():
            forms.append(fields[1])

    if sent_lines or doc_lines:
        yield doc_lines, "".join(sent_lines), forms


def sentence_key(forms: list[str]) -> int:
    """The dedup key: leading 8 bytes of md5 over the space-joined forms.

    0 is reserved by the hash table as "empty slot", so a digest of 0 is mapped
    to 1 — a one-in-2^64 nudge that costs nothing and removes a special case.
    """
    digest = hashlib.md5(" ".join(forms).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") or 1


# ── The hash table ───────────────────────────────────────────────────────────


class SeenHashes:
    """Open-addressed uint64 set, sized once and never grown.

    Growing would mean holding two tables at ~9 GB each, so the capacity is
    fixed by --hash-bits and exceeding the load factor is an error with an
    actionable message rather than a silent slowdown.
    """

    MAX_LOAD = 0.7

    def __init__(self, bits: int, table: np.ndarray | None = None, size: int = 0):
        if table is None:
            table = np.zeros(1 << bits, dtype=np.uint64)
        self.bits = bits
        self.table = table
        self.mask = table.size - 1
        self.size = size
        self.limit = int(table.size * self.MAX_LOAD)

    def add(self, key: int) -> bool:
        """Insert a key; return True if it was not already present."""
        table, mask = self.table, self.mask
        i = key & mask
        while True:
            cur = int(table[i])
            if cur == 0:
                if self.size >= self.limit:
                    raise RuntimeError(
                        f"hash table full at {self.size:,} entries "
                        f"({self.MAX_LOAD:.0%} of 2^{self.bits}). "
                        f"Re-run with --hash-bits {self.bits + 1}."
                    )
                table[i] = key
                self.size += 1
                return True
            if cur == key:
                return False
            i = (i + 1) & mask

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        # Written through a file object: np.save given a path appends its own
        # .npy, which would leave the temporary file under a name replace()
        # cannot find.
        with tmp.open("wb") as fh:
            np.save(fh, self.table)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, size: int) -> SeenHashes:
        table = np.load(path)
        return cls(int(table.size).bit_length() - 1, table=table, size=size)


# ── The pass ─────────────────────────────────────────────────────────────────


class ChunkWriter:
    """Writes kept sentences into chunk_XX.conllu files of a fixed token size."""

    def __init__(self, out_dir: Path, tokens_per_chunk: int, first_index: int = 0):
        self.out_dir = out_dir
        self.tokens_per_chunk = tokens_per_chunk
        self.index = first_index
        self.fh = None
        self.chunk_tokens = 0
        self.chunk_sentences = 0
        self.chunks: list[dict] = []
        # The document we are currently inside, and whether its header has been
        # written into the chunk that is open now.  Both are needed so that a
        # chunk boundary in the middle of a document repeats the header, leaving
        # every chunk readable on its own.
        self.current_doc: list[str] = []
        self.doc_written = False

    def _open(self) -> None:
        path = self.out_dir / f"chunk_{self.index:02d}.conllu"
        self.fh = path.open("w", encoding="utf-8")
        self.chunk_tokens = 0
        self.chunk_sentences = 0
        self.doc_written = False

    def note_document(self, doc_lines: list[str]) -> None:
        """Record the document whose header just went past.

        Called for every block carrying one, including blocks skipped on resume
        and sentences dropped as duplicates, so the header is still available
        for whichever of the document's sentences is written first.
        """
        self.current_doc = doc_lines
        self.doc_written = False

    def write(self, sent_block: str, n_tokens: int) -> None:
        if self.fh is None:
            self._open()
        if self.current_doc and not self.doc_written:
            self.fh.write("".join(self.current_doc))
            self.doc_written = True
        self.fh.write(sent_block)
        self.fh.write("\n")
        self.chunk_tokens += n_tokens
        self.chunk_sentences += 1

    def full(self) -> bool:
        return self.fh is not None and self.chunk_tokens >= self.tokens_per_chunk

    def close_chunk(self) -> None:
        if self.fh is None:
            return
        self.fh.close()
        self.fh = None
        self.chunks.append(
            {
                "name": f"chunk_{self.index:02d}.conllu",
                "sentences": self.chunk_sentences,
                "tokens": self.chunk_tokens,
            }
        )
        self.index += 1


def _write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    inputs: list[Path],
    writer: ChunkWriter,
    seen: SeenHashes,
    sentences_seen: int,
    checkpoint: dict,
) -> None:
    manifest = {
        "inputs": [
            {"path": str(p), "bytes": p.stat().st_size if p.exists() else None}
            for p in inputs
        ],
        "tokens_per_chunk": args.tokens_per_chunk,
        "target_tokens": args.target_tokens,
        "hash_bits": seen.bits,
        "chunks": writer.chunks,
        "totals": {
            "chunks": len(writer.chunks),
            "sentences_seen": sentences_seen,
            "sentences_written": sum(c["sentences"] for c in writer.chunks),
            "tokens_written": sum(c["tokens"] for c in writer.chunks),
            "unique_hashes": seen.size,
        },
        "checkpoint": checkpoint,
    }
    tmp = out_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_dir / "manifest.json")


def _restore(out_dir: Path) -> tuple[ChunkWriter | None, SeenHashes | None, dict]:
    """Load the last checkpoint, discarding chunks written after it."""
    manifest_path = out_dir / "manifest.json"
    hashes_path = out_dir / "seen_hashes.npy"
    if not manifest_path.exists() or not hashes_path.exists():
        raise SystemExit(
            f"--resume: no checkpoint in {out_dir} "
            "(need both manifest.json and seen_hashes.npy)"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cp = manifest["checkpoint"]
    kept = manifest["chunks"][: cp["chunks_done"]]

    for stale in manifest["chunks"][cp["chunks_done"] :]:
        path = out_dir / stale["name"]
        if path.exists():
            logger.info("Discarding post-checkpoint chunk %s", stale["name"])
            path.unlink()

    seen = SeenHashes.load(hashes_path, size=cp["unique_hashes"])
    writer = ChunkWriter(out_dir, manifest["tokens_per_chunk"], first_index=len(kept))
    writer.chunks = kept
    logger.info(
        "Resuming after chunk %d: %s sentences seen, %s unique",
        len(kept) - 1,
        f"{cp['sentences_seen']:,}",
        f"{seen.size:,}",
    )
    return writer, seen, cp


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = [Path(p) for p in args.input]
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"input not found: {path}")

    if args.resume:
        writer, seen, cp = _restore(out_dir)
        start_input, skip_blocks = cp["input_index"], cp["blocks_consumed"]
        sentences_seen = cp["sentences_seen"]
    else:
        writer = ChunkWriter(out_dir, args.tokens_per_chunk)
        seen = SeenHashes(args.hash_bits)
        start_input, skip_blocks, sentences_seen = 0, 0, 0

    tokens_written = sum(c["tokens"] for c in writer.chunks)
    hashes_path = out_dir / "seen_hashes.npy"
    chunks_at_last_save = len(writer.chunks)
    started = time.time()

    def checkpoint(input_index: int, blocks_consumed: int, force: bool = False) -> None:
        """Persist the hash table and manifest at a chunk boundary.

        The table is ~9 GB, so it is written every --checkpoint-every chunks
        rather than every chunk; the manifest records the offset that goes with
        the table that is actually on disk.
        """
        nonlocal chunks_at_last_save
        due = len(writer.chunks) - chunks_at_last_save >= args.checkpoint_every
        if not (due or force):
            return
        seen.save(hashes_path)
        chunks_at_last_save = len(writer.chunks)
        _write_manifest(
            out_dir,
            args,
            inputs,
            writer,
            seen,
            sentences_seen,
            {
                "chunks_done": len(writer.chunks),
                "input_index": input_index,
                "blocks_consumed": blocks_consumed,
                "sentences_seen": sentences_seen,
                "unique_hashes": seen.size,
            },
        )
        logger.info("Checkpoint: %d chunks, %s tokens", len(writer.chunks),
                    f"{tokens_written:,}")

    done = False
    input_index, blocks = start_input, skip_blocks
    for input_index in range(start_input, len(inputs)):
        if done:
            break
        path = inputs[input_index]
        blocks = 0
        skip = skip_blocks if input_index == start_input else 0
        logger.info("Reading %s%s", path, f" (skipping {skip:,} blocks)" if skip else "")

        with open_conllu(path) as fh:
            for doc_lines, sent_block, forms in iter_blocks(fh):
                blocks += 1
                if doc_lines:
                    writer.note_document(doc_lines)
                if blocks <= skip or not forms:
                    continue
                sentences_seen += 1

                if seen.add(sentence_key(forms)):
                    writer.write(sent_block, len(forms))
                    tokens_written += len(forms)

                    if writer.full():
                        writer.close_chunk()
                        checkpoint(input_index, blocks)
                        if tokens_written >= args.target_tokens:
                            logger.info("Reached target of %s tokens",
                                        f"{args.target_tokens:,}")
                            done = True
                            break

                if sentences_seen % 1_000_000 == 0:
                    rate = sentences_seen / max(time.time() - started, 1e-9)
                    logger.info(
                        "seen %s | written %s | tokens %s | %s chunks | %.0f sent/s",
                        f"{sentences_seen:,}",
                        f"{seen.size:,}",
                        f"{tokens_written:,}",
                        len(writer.chunks),
                        rate,
                    )

    # A partial final chunk is kept: it is a valid chunk, just a short one, and
    # dropping it would silently lose up to --tokens-per-chunk tokens.
    writer.close_chunk()
    # The final checkpoint records where the pass actually stopped, not the end
    # of the inputs: a run that stopped on --target-tokens can be resumed with a
    # larger target and pick up from there.
    checkpoint(input_index, blocks, force=True)

    logger.info(
        "Done: %d chunks, %s sentences (%s unique), %s tokens → %s",
        len(writer.chunks),
        f"{sentences_seen:,}",
        f"{seen.size:,}",
        f"{tokens_written:,}",
        out_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Parsed CoNLL-U file(s), .conllu or .conllu.gz, read in the order given.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Where chunk_XX.conllu, manifest.json and seen_hashes.npy are written.",
    )
    parser.add_argument(
        "--tokens-per-chunk", type=int, default=65_700_000,
        help="CoNLL-U tokens per chunk (default: 65.7M, one 68M budget unit "
             "before the filter's 90/10 split).",
    )
    parser.add_argument(
        "--target-tokens", type=int, default=5_387_400_000,
        help="Stop after this many tokens (default: 5.3874B = 82 chunks, the "
             "5.6B-subword budget measured before the 90/10 split).",
    )
    parser.add_argument(
        "--hash-bits", type=int, default=30,
        help="Hash table capacity, 2^N slots (default: 30 → 8.6 GB, holds 750M).",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Save the hash table every N chunks (default: 10).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue from the checkpoint in --output-dir.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
