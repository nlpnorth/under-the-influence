"""Tests for Step 0 — deduplicating and chunking the released parse.

This step decides which sentences the whole pipeline ever sees, so the things
worth checking are the ones that would silently change the corpus rather than
crash:

  1. the dedup key is the rule the original corpus was built with
  2. the hand-written CoNLL-U reader sees the same sentences as the conllu
     library, which the reference implementation used
  3. a repeated sentence is dropped, and its first occurrence is the one kept
  4. chunks close on the token boundary and the manifest describes the files
     that are actually on disk
  5. document headers — the per-document licence and collection fields —
     survive a dropped first sentence and a chunk boundary
  6. resuming from a checkpoint reproduces an uninterrupted run exactly
  7. filtering a .conllu chunk gives the same result as filtering the pickle it
     replaces, which is what run/filter_linguistic.sh relies on when it prefers
     the .conllu

Run with:  python tests/test_corpus_prepare.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "vendor" / "corpus_filtering" / "src"))

from conllu import parse, parse_incr  # noqa: E402

from influence_on_what.prepare.corpus import (  # noqa: E402
    SeenHashes,
    iter_blocks,
    sentence_key,
)

SAMPLE = REPO / "sample_data/supplementary_data_samples/parsed_corpus/parser_output_sample.conllu"

# A sentence carrying both of the row types that are NOT tokens: a multiword
# range (1-2) and an empty node (4.1).  Both must stay out of the hash and out
# of the token count, as they did in the original dedup script.
FIXTURE = """\
# newdoc id = doc/1
# meta::license = CC-By
# sent_id = doc/1/0
# text = Hes a dog.
1-2\tHes\t_\t_\t_\t_\t_\t_\t_\t_
1\tHe\the\tPRON\tPRP\t_\t3\tnsubj\t_\t_
2\tis\tbe\tAUX\tVBZ\t_\t3\tcop\t_\t_
3\tdog\tdog\tNOUN\tNN\t_\t0\troot\t_\t_
4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_
4.1\t_\t_\t_\t_\t_\t_\t_\t_\t_

"""


def _run_prepare(out_dir: Path, inputs: list[Path], **flags) -> dict:
    """Invoke the module the way run/prepare_corpus.sh does, return its manifest."""
    cmd = [
        sys.executable, "-m", "influence_on_what.prepare.corpus",
        "--input", *[str(p) for p in inputs],
        "--output-dir", str(out_dir),
        "--hash-bits", "12",
    ]
    for key, value in flags.items():
        flag = "--" + key.replace("_", "-")
        cmd += [flag] if value is True else [flag, str(value)]
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"}
    subprocess.run(cmd, check=True, capture_output=True, env=env)
    return json.loads((out_dir / "manifest.json").read_text())


def check_dedup_key_matches_the_original_rule() -> list[str]:
    """(1) md5 over the space-joined FORM column of the plain-integer-id tokens."""
    sent = parse(FIXTURE)[0]
    forms = [t["form"] for t in sent if isinstance(t["id"], int)]
    expected = hashlib.md5(" ".join(forms).encode("utf-8")).hexdigest()

    fails = []
    if forms != ["He", "is", "dog", "."]:
        fails.append(f"reference forms are wrong: {forms}")
    if sentence_key(forms) != int(expected[:16], 16):
        fails.append("sentence_key is not the leading 8 bytes of the md5")

    # And the reader must extract that same token list from the raw text.
    _, _, mine = next(iter_blocks(FIXTURE.splitlines(keepends=True)))
    if mine != forms:
        fails.append(f"reader extracted {mine}, expected {forms}")
    return fails


def check_reader_agrees_with_the_conllu_library() -> list[str]:
    """(2) Same sentences, same tokens, same order as conllu.parse_incr."""
    with SAMPLE.open(encoding="utf-8") as f:
        ref = [
            [t["form"] for t in s if isinstance(t["id"], int)] for s in parse_incr(f)
        ]
    with SAMPLE.open(encoding="utf-8") as f:
        mine = [forms for _, _, forms in iter_blocks(f) if forms]

    if mine != ref:
        return [f"reader saw {len(mine)} sentences, conllu saw {len(ref)}"]
    return []


def check_duplicates_are_dropped() -> list[str]:
    """(3) A repeated sentence is written once, on its first occurrence."""
    fails = []
    seen = SeenHashes(bits=8)
    key = sentence_key(["a", "b"])
    if not seen.add(key):
        fails.append("first insertion reported the key as already present")
    if seen.add(key):
        fails.append("second insertion of the same key was not reported as duplicate")
    if seen.size != 1:
        fails.append(f"table holds {seen.size} entries, expected 1")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "dup.conllu").write_text(FIXTURE * 3, encoding="utf-8")
        manifest = _run_prepare(tmp / "out", [tmp / "dup.conllu"], tokens_per_chunk=10)
        totals = manifest["totals"]
        if (totals["sentences_seen"], totals["sentences_written"]) != (3, 1):
            fails.append(
                f"saw {totals['sentences_seen']} / wrote {totals['sentences_written']}, "
                "expected 3 / 1"
            )
    return fails


def check_chunking_and_manifest() -> list[str]:
    """(4) Chunks close at the token boundary; the manifest matches the files."""
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        manifest = _run_prepare(out, [SAMPLE], tokens_per_chunk=800)

        for chunk in manifest["chunks"][:-1]:
            if chunk["tokens"] < 800:
                fails.append(f"{chunk['name']} closed early at {chunk['tokens']} tokens")

        for chunk in manifest["chunks"]:
            with (out / chunk["name"]).open(encoding="utf-8") as f:
                sents = list(parse_incr(f))
            tokens = sum(
                sum(1 for t in s if isinstance(t["id"], int)) for s in sents
            )
            if (len(sents), tokens) != (chunk["sentences"], chunk["tokens"]):
                fails.append(
                    f"{chunk['name']} holds {len(sents)}/{tokens}, "
                    f"manifest says {chunk['sentences']}/{chunk['tokens']}"
                )

        written = sum(c["sentences"] for c in manifest["chunks"])
        if written != manifest["totals"]["sentences_written"]:
            fails.append("manifest totals disagree with its own chunk list")
    return fails


def check_document_headers_survive() -> list[str]:
    """(5) The licence/collection header reaches every chunk of its document."""
    fails = []
    # Three sentences of one document; the first is a duplicate of nothing, but
    # the chunk size forces a boundary between them.
    doc = FIXTURE
    second = FIXTURE.replace("# newdoc id = doc/1\n", "").replace(
        "# meta::license = CC-By\n", ""
    ).replace("Hes a dog.", "Hes a cat.").replace("\tdog\tdog\t", "\tcat\tcat\t")
    third = second.replace("a cat.", "a fox.").replace("\tcat\tcat\t", "\tfox\tfox\t")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.conllu").write_text(doc + second + third, encoding="utf-8")
        out = tmp / "out"
        manifest = _run_prepare(out, [tmp / "in.conllu"], tokens_per_chunk=1)

        if len(manifest["chunks"]) < 2:
            fails.append("expected the document to span more than one chunk")
        for chunk in manifest["chunks"]:
            text = (out / chunk["name"]).read_text(encoding="utf-8")
            if "# meta::license = CC-By" not in text:
                fails.append(f"{chunk['name']} lost its document licence header")
    return fails


def check_resume_reproduces_an_uninterrupted_run() -> list[str]:
    """(6) Stopping at a checkpoint and resuming gives byte-identical chunks."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref, res = tmp / "ref", tmp / "res"
        _run_prepare(ref, [SAMPLE], tokens_per_chunk=800)
        # Stop after two chunks, then continue with the full target.
        _run_prepare(res, [SAMPLE], tokens_per_chunk=800, target_tokens=1600,
                     checkpoint_every=1)
        _run_prepare(res, [SAMPLE], tokens_per_chunk=800, resume=True)

        ref_chunks = sorted(ref.glob("chunk_*.conllu"))
        res_chunks = sorted(res.glob("chunk_*.conllu"))
        if len(ref_chunks) != len(res_chunks):
            return [f"resumed run wrote {len(res_chunks)} chunks, reference wrote "
                    f"{len(ref_chunks)}"]
        return [
            f"{a.name} differs between the resumed and the uninterrupted run"
            for a, b in zip(ref_chunks, res_chunks)
            if a.read_bytes() != b.read_bytes()
        ]


def check_conllu_and_pickle_filter_identically() -> list[str]:
    """(7) The filter output does not depend on which of the two it was fed."""
    script = REPO / "vendor/corpus_filtering/scripts/run_all_filters.py"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        chunks = tmp / "chunks"
        _run_prepare(chunks, [SAMPLE], tokens_per_chunk=800)
        chunk = chunks / "chunk_00.conllu"

        with chunk.open(encoding="utf-8") as f:
            sentences = list(parse_incr(f))
        pkl = tmp / "chunk_00.pkl"
        pkl.write_bytes(pickle.dumps(sentences))

        for src, dest in ((chunk, tmp / "from_conllu"), (pkl, tmp / "from_pickle")):
            subprocess.run(
                [sys.executable, str(script), "--input", str(src), "--output", str(dest)],
                check=True, capture_output=True,
            )

        fails = []
        for produced in sorted((tmp / "from_conllu").rglob("*.txt")):
            other = tmp / "from_pickle" / produced.relative_to(tmp / "from_conllu")
            if not other.exists():
                fails.append(f"{produced.name} missing from the pickle run")
            elif produced.read_bytes() != other.read_bytes():
                fails.append(f"{produced.relative_to(tmp / 'from_conllu')} differs")
        return fails


CHECKS = [
    ("dedup key matches the original rule", check_dedup_key_matches_the_original_rule),
    ("reader agrees with the conllu library", check_reader_agrees_with_the_conllu_library),
    ("duplicate sentences are dropped", check_duplicates_are_dropped),
    ("chunks and manifest agree", check_chunking_and_manifest),
    ("document headers survive", check_document_headers_survive),
    ("resume reproduces an uninterrupted run", check_resume_reproduces_an_uninterrupted_run),
    (".conllu and .pkl filter identically", check_conllu_and_pickle_filter_identically),
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
