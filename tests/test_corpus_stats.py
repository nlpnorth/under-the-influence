"""Tests for the corpus provenance statistics pass.

These statistics are what the appendix reports about the training data, so the
failure mode that matters is a count that is quietly wrong rather than a crash:

  1. one document contributes exactly one count per field, and a document that
     omits a field does not inherit the previous document's value — the corpus
     really does omit `meta::date` on some documents, and a corpus with no
     `newdoc id` at all (the pre-release parser output) has nothing but the
     meta:: block to separate one document from the next
  2. tokens are counted by the same rule prepare/corpus.py chunks by, so that
     these totals and manifest.json mean the same thing
  3. dates fold to a year, and a value with no year in it is reported as
     unparsed rather than dropped
  4. the reader agrees with the conllu library on how many sentences there are
  5. a document whose header is repeated — which prepare/corpus.py does on
     purpose at every chunk boundary, so each chunk stays readable alone — is
     counted once, not once per chunk it spans

Run with:  python tests/test_corpus_stats.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from conllu import parse_incr  # noqa: E402

from influence_on_what.prepare.corpus_stats import CorpusStats, scan  # noqa: E402

SAMPLE = REPO / "sample_data/supplementary_data_samples/parsed_corpus/parser_output_sample.conllu"


def _sentence(sent_id: str, rows: str) -> str:
    return f"# sent_id = {sent_id}\n{rows}\n"


# Two documents.  The first carries a date, the second does NOT — if the reader
# leaks state, the second inherits 1899 and both the date total and the year
# histogram are wrong.  The second also has no `newdoc id`, so the meta:: block
# is the only thing marking where it begins.
TWO_DOCS = (
    "# newdoc id = doc/1\n"
    "# meta::collection = USPTO\n"
    "# meta::license = Public Domain\n"
    "# meta::open_type = Open Government\n"
    "# meta::date = 1899-04-02\n"
    + _sentence("doc/1/0", "1\tA\ta\tDET\tDT\t_\t2\tdet\t_\t_\n2\tcat\tcat\tNOUN\tNN\t_\t0\troot\t_\t_")
    + "\n"
    + _sentence("doc/1/1", "1\tIt\tit\tPRON\tPRP\t_\t0\troot\t_\t_")
    + "\n"
    "# meta::collection = Wikipedia\n"
    "# meta::license = CC-By-SA\n"
    "# meta::open_type = Open Web\n"
    + _sentence("doc/2/0", "1\tDogs\tdog\tNOUN\tNNS\t_\t0\troot\t_\t_")
    + "\n"
)

# A multiword range (1-2) and an empty node (3.1): CoNLL-U rows that are not
# tokens.  prepare/corpus.py excludes both from its token count and so must
# this, or a chunk's manifest and these statistics disagree about its size.
NON_TOKEN_ROWS = (
    "# meta::collection = SEC\n"
    "# meta::date = sometime\n"
    "# sent_id = doc/3/0\n"
    "1-2\tHes\t_\t_\t_\t_\t_\t_\t_\t_\n"
    "1\tHe\the\tPRON\tPRP\t_\t2\tnsubj\t_\t_\n"
    "2\tis\tbe\tAUX\tVBZ\t_\t0\troot\t_\t_\n"
    "3\there\there\tADV\tRB\t_\t2\tadvmod\t_\t_\n"
    "3.1\t_\t_\t_\t_\t_\t_\t_\t_\t_\n"
    "\n"
)


# The same document, its header repeated as prepare/corpus.py repeats it when a
# document straddles a chunk boundary.  Both halves carry real sentences, so the
# sentences must both count and the document must not.
SPANNING_DOC = (
    "# newdoc id = doc/9\n"
    "# meta::collection = USPTO\n"
    + _sentence("doc/9/0", "1\tOne\tone\tNUM\tCD\t_\t0\troot\t_\t_")
    + "\n"
    "# newdoc id = doc/9\n"
    "# meta::collection = USPTO\n"
    + _sentence("doc/9/1", "1\tTwo\ttwo\tNUM\tCD\t_\t0\troot\t_\t_")
    + "\n"
)


def _stats_for(text: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "corpus.conllu"
        path.write_text(text, encoding="utf-8")
        stats = CorpusStats()
        scan(path, stats, 0.0)
        return stats.summary(1)


def check_documents_are_counted_once_and_do_not_leak() -> list[str]:
    """(1) Two documents, one field each, and no inheritance across the pair."""
    out = _stats_for(TWO_DOCS)
    fails = []
    if out["corpus_totals"]["documents"] != 2:
        fails.append(f"documents = {out['corpus_totals']['documents']}, want 2")
    if out["collection"]["counts"] != {"USPTO": 1, "Wikipedia": 1}:
        fails.append(f"collection = {out['collection']['counts']}")
    if out["open_type"]["counts"] != {"Open Government": 1, "Open Web": 1}:
        fails.append(f"open_type = {out['open_type']['counts']}")
    # The second document has no date at all; only the first may be counted.
    if out["date"]["total"] != 1 or out["date"]["year_counts"] != {"1899": 1}:
        fails.append(f"date = {out['date']}, want exactly one document at 1899")
    return fails


def check_non_token_rows_are_excluded() -> list[str]:
    """(2) Multiword ranges and empty nodes count as neither tokens nor rows."""
    out = _stats_for(NON_TOKEN_ROWS)
    fails = []
    if out["corpus_totals"]["words"] != 3:
        fails.append(f"tokens = {out['corpus_totals']['words']}, want 3 (He/is/here)")
    if out["corpus_totals"]["sentences"] != 1:
        fails.append(f"sentences = {out['corpus_totals']['sentences']}, want 1")
    return fails


def check_unparseable_dates_are_reported() -> list[str]:
    """(3) 'sometime' has no year, so it is unparsed rather than silently gone."""
    out = _stats_for(NON_TOKEN_ROWS)
    fails = []
    if out["date"]["unparsed"] != 1:
        fails.append(f"unparsed = {out['date']['unparsed']}, want 1")
    if out["date"]["total"] != 0:
        fails.append(f"date total = {out['date']['total']}, want 0")
    return fails


def check_repeated_document_header_counts_once() -> list[str]:
    """(5) A document split across chunks is one document with two sentences."""
    out = _stats_for(SPANNING_DOC)
    totals = out["corpus_totals"]
    fails = []
    if totals["documents"] != 1:
        fails.append(f"documents = {totals['documents']}, want 1")
    if totals["documents_repeated"] != 1:
        fails.append(f"documents_repeated = {totals['documents_repeated']}, want 1")
    if totals["sentences"] != 2:
        fails.append(f"sentences = {totals['sentences']}, want 2")
    if out["collection"]["counts"] != {"USPTO": 1}:
        fails.append(f"collection = {out['collection']['counts']}, want one USPTO")
    return fails


def check_sentence_count_agrees_with_conllu() -> list[str]:
    """(4) Same number of sentences as the library, over real parser output."""
    stats = CorpusStats()
    scan(SAMPLE, stats, 0.0)
    with SAMPLE.open(encoding="utf-8") as fh:
        expected = sum(1 for _ in parse_incr(fh))
    if stats.sentences != expected:
        return [f"sentences = {stats.sentences}, conllu library sees {expected}"]
    return []


def check_cli_writes_the_notebook_schema() -> list[str]:
    """(5) The JSON on disk has the keys corpus_provenance.ipynb reads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        out_path = tmp / "stats.json"
        proc = subprocess.run(
            [sys.executable, "-m", "influence_on_what.prepare.corpus_stats",
             "--input", str(SAMPLE), "--output", str(out_path), "--label", "test"],
            cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")},
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return [f"exit {proc.returncode}: {proc.stderr.strip().splitlines()[-1:]}"]

        out = json.loads(out_path.read_text())
        wanted = {"collection", "license", "open_type", "date", "creator", "corpus_totals"}
        missing = wanted - set(out)
        if missing:
            return [f"missing top-level keys: {sorted(missing)}"]
        totals = out["corpus_totals"]
        for key in ("n_chunks", "documents", "sentences", "words", "avg_sent_len"):
            if key not in totals:
                return [f"corpus_totals missing {key}"]
        for key in ("total", "n_unique", "empty", "singleton_frac",
                    "share_top10", "share_top100", "share_top1000", "top"):
            if key not in out["creator"]:
                return [f"creator missing {key}"]
    return []


CHECKS = [
    ("documents counted once, no field leaks", check_documents_are_counted_once_and_do_not_leak),
    ("multiword rows and empty nodes excluded", check_non_token_rows_are_excluded),
    ("unparseable dates are reported", check_unparseable_dates_are_reported),
    ("repeated document header counts once", check_repeated_document_header_counts_once),
    ("sentence count agrees with conllu", check_sentence_count_agrees_with_conllu),
    ("JSON matches the notebook schema", check_cli_writes_the_notebook_schema),
]


def main() -> int:
    failures = 0
    for label, check in CHECKS:
        problems = check()
        if problems:
            failures += len(problems)
            for problem in problems:
                print(f"  FAIL  {label:40s} {problem}")
        else:
            print(f"  ok    {label}")

    print(f"\nFAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
