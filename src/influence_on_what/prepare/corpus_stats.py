#!/usr/bin/env python3
"""Provenance and size statistics for a parsed CoNLL-U corpus.

Common Corpus attaches licensing and provenance metadata to every document, and
the parse keeps it: each document opens with ``# newdoc id`` and a block of
``# meta::`` comments.  This module streams a corpus and counts them, producing
the JSON behind the appendix's corpus figure and table.

WHY IT EXISTS TWICE OVER
  It is meant to be run on BOTH corpora and the two outputs compared:

    full    the released cc-en-10b.conllu.gz — every document that was parsed
    subset  the chunk_XX.conllu that run/prepare_corpus.sh kept — the documents
            that survive deduplication and the token budget, i.e. what the
            models are actually trained on

  Deduplication is not metadata-blind: a collection full of boilerplate loses
  more of itself than one of prose, so the two distributions can differ and the
  difference is the thing worth reporting.  (The filter's later 90/10 split IS
  blind — an independent Bernoulli draw that never looks at a document — so it
  needs no third pass.)

WHAT IS COUNTED
  Documents, for every metadata field: one document contributes one count, so
  every distribution here is over documents, not sentences and not tokens.
  Sentences and tokens are counted separately, as corpus_totals.

  A token is a row whose ID is a plain integer.  Multiword ranges (``7-8``) and
  empty nodes (``7.1``) are not tokens and are excluded, the same rule
  prepare/corpus.py deduplicates and chunks by, so the counts here and the
  counts in manifest.json mean the same thing.

  A document is counted once even if its header is seen more than once.  This
  is not hypothetical: prepare/corpus.py repeats the ``# newdoc id`` block in
  every chunk a document spans, deliberately, so that each chunk can be read on
  its own — and scanning the whole chunk set would otherwise count each of those
  documents twice.  The cost is holding the document ids in a set; at a few
  million documents that is a few hundred MB, which is why the job asks for
  32 GB.  A corpus with no ``newdoc id`` at all cannot be deduplicated this way
  and is counted as it comes.

WHAT IT WRITES
  A JSON object with one entry per field — ``collection``, ``license``,
  ``open_type``, ``date``, ``creator`` — plus ``corpus_totals``.  The schema is
  the one the corpus_provenance notebook reads.

  ``creator`` is summarised rather than dumped: it is long-tailed to the point
  of being unplottable (95% of values occur once), so only the head and the
  concentration profile are kept.

Usage
-----
    python -m influence_on_what.prepare.corpus_stats \\
        --input /path/to/cc-en-10b.conllu.gz \\
        --output corpus_stats_full_corpus.json --label full

    python -m influence_on_what.prepare.corpus_stats \\
        --input "$COMMON_CORPUS_CHUNKS"/chunk_*.conllu \\
        --output corpus_stats_subset.json --label subset
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

from influence_on_what.prepare.corpus import open_conllu

logging.basicConfig(
    level=logging.INFO,
    format="[stats %(levelname)s %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# The metadata fields the appendix reports.  Anything else Common Corpus
# attaches (identifier, language, language_type) is skipped: identifier is
# unique per document and the language fields are constant across an
# English-only corpus, so neither is a distribution.
FIELDS = ("collection", "license", "open_type", "date", "creator")

# "2011", "2011-03", "1978-04-12" — take the leading year and nothing else.
YEAR_RE = re.compile(r"(\d{4})")

# How many of the most frequent creators to keep.
CREATOR_TOP_N = 25


class CorpusStats:
    """Accumulates per-document metadata counts and per-corpus size totals."""

    def __init__(self) -> None:
        self.counters = {field: Counter() for field in FIELDS}
        self.documents = 0
        self.sentences = 0
        self.tokens = 0
        self.date_unparsed = 0
        # Document ids already counted, and how many headers were skipped for
        # naming one of them.  Reported rather than silently absorbed: a
        # non-zero count over a single input file would mean the corpus itself
        # repeats a document, which is a different fact from a chunk boundary.
        self.seen_docs: set[str] = set()
        self.documents_repeated = 0

    def add_document(self, meta: dict[str, str], doc_id: str | None = None) -> None:
        if doc_id is not None:
            if doc_id in self.seen_docs:
                self.documents_repeated += 1
                return
            self.seen_docs.add(doc_id)
        self.documents += 1
        for field in FIELDS:
            if field not in meta:
                continue
            # An absent field and an empty one are different facts: a document
            # with `meta::creator = ` was attributed to nobody, one without the
            # line was never asked.  Only the first is counted, under "".
            self.counters[field][meta[field]] += 1

    def summary(self, n_inputs: int) -> dict:
        out: dict[str, object] = {}

        for field in ("collection", "license", "open_type"):
            counts = self.counters[field]
            out[field] = {
                "total": sum(counts.values()),
                "n_unique": len(counts),
                "counts": dict(counts.most_common()),
            }

        out["date"] = self._date_summary()
        out["creator"] = self._creator_summary()
        out["corpus_totals"] = {
            "n_chunks": n_inputs,
            "documents": self.documents,
            "documents_repeated": self.documents_repeated,
            "sentences": self.sentences,
            "words": self.tokens,
            "avg_sent_len": self.tokens / self.sentences if self.sentences else 0.0,
        }
        return out

    def _date_summary(self) -> dict:
        """Dates folded to a year.  Anything with no 4-digit run is unparsed."""
        years: Counter[int] = Counter()
        for value, n in self.counters["date"].items():
            match = YEAR_RE.search(value)
            if match:
                years[int(match.group(1))] += n
            else:
                self.date_unparsed += n
        return {
            "total": sum(years.values()),
            "unparsed": self.date_unparsed,
            # Sorted by year, and str keys because JSON has no integer keys —
            # the notebook casts them back.
            "year_counts": {str(y): years[y] for y in sorted(years)},
        }

    def _creator_summary(self) -> dict:
        """Head plus concentration profile; the full tail is not worth keeping."""
        counts = self.counters["creator"]
        total = sum(counts.values())
        ranked = counts.most_common()

        def share(n: int) -> float:
            return sum(c for _, c in ranked[:n]) / total if total else 0.0

        singletons = sum(1 for _, c in ranked if c == 1)
        return {
            "total": total,
            "n_unique": len(counts),
            "empty": counts.get("", 0),
            "share_top10": share(10),
            "share_top100": share(100),
            "share_top1000": share(1000),
            "singleton_frac": singletons / len(counts) if counts else 0.0,
            "top": [[name, n] for name, n in ranked[:CREATOR_TOP_N]],
        }


def scan(path: Path, stats: CorpusStats, started: float) -> None:
    """One streaming pass, counting comments and token rows.

    Deliberately NOT built on prepare/corpus.py's iter_blocks: that assembles
    the text and the FORM column of every sentence, which is most of the work
    and none of it is needed here.  This loop touches the first tab-separated
    field of a token row and nothing else.
    """
    meta: dict[str, str] = {}
    # doc_id belongs to the metadata block currently being accumulated;
    # next_doc_id holds an id announced by `# newdoc id` but not yet claimed by
    # a block.  Two variables rather than one because a corpus may have
    # documents WITHOUT a newdoc id (the pre-release parser output does), and
    # with a single variable those would silently inherit the id of the
    # document before them and be discarded as repeats.
    doc_id: str | None = None
    next_doc_id: str | None = None
    pending_doc = False       # a metadata block has been read, not yet filed
    sentence_open = False     # the current block has produced a token

    with open_conllu(path) as fh:
        for line in fh:
            if line.startswith("#"):
                key, sep, value = line[1:].partition("=")
                if not sep:
                    continue
                key = key.strip()
                if key.startswith("meta::"):
                    # The first meta:: line after a document was filed opens the
                    # next document's block, so the previous document's fields
                    # have to go.  Without this a document that simply omits
                    # `date` would silently inherit the date of the one before
                    # it — and `date` really is optional (56 of 64 documents
                    # carry it in a sample of the release).
                    if not pending_doc:
                        meta = {}
                        doc_id, next_doc_id = next_doc_id, None
                    meta[key[6:]] = value.strip()
                    pending_doc = True
                elif key == "newdoc id":
                    # A new document begins here.  File whatever the previous
                    # meta:: block described — under the PREVIOUS id, which is
                    # why the flush comes before doc_id is reassigned — then
                    # clear both.  This only fires for a document that had a
                    # header and no sentences; the usual case is filed when the
                    # document's first sentence starts, below.
                    if pending_doc:
                        stats.add_document(meta, doc_id)
                        pending_doc = False
                    next_doc_id = value.strip()
                    meta = {}
                continue

            if not line.strip():
                sentence_open = False
                continue

            # A token row.  The first field is the ID; plain integers only.
            if line.partition("\t")[0].isdigit():
                stats.tokens += 1
                if not sentence_open:
                    stats.sentences += 1
                    sentence_open = True

            if pending_doc and sentence_open:
                # The metadata block is complete once the document's first
                # sentence starts, so file it here.  This covers corpora with
                # no `newdoc id` line at all, where the meta:: block is the
                # only document delimiter there is.
                stats.add_document(meta, doc_id)
                pending_doc = False
                if stats.documents % 100_000 == 0:
                    rate = stats.sentences / max(time.time() - started, 1e-9)
                    logger.info(
                        "%s docs | %s sentences | %s tokens | %.0f sent/s",
                        f"{stats.documents:,}",
                        f"{stats.sentences:,}",
                        f"{stats.tokens:,}",
                        rate,
                    )

    if pending_doc:
        stats.add_document(meta, doc_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Parsed CoNLL-U file(s), .conllu or .conllu.gz.",
    )
    parser.add_argument("--output", required=True, help="Where to write the JSON.")
    parser.add_argument(
        "--label", default="",
        help="Recorded in the JSON as `label`, to tell the two passes apart "
             "(e.g. 'full' vs 'subset').",
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input]
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"input not found: {path}")

    stats = CorpusStats()
    started = time.time()
    for path in inputs:
        logger.info("Reading %s", path)
        scan(path, stats, started)

    summary = stats.summary(len(inputs))
    summary["label"] = args.label
    summary["inputs"] = [str(p) for p in inputs]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    totals = summary["corpus_totals"]
    logger.info(
        "Done: %s documents, %s sentences, %s tokens (%.2f tokens/sentence) → %s",
        f"{totals['documents']:,}",
        f"{totals['sentences']:,}",
        f"{totals['words']:,}",
        totals["avg_sent_len"],
        out,
    )


if __name__ == "__main__":
    main()
