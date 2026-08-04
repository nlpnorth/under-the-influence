"""Smoke test for the four linguistic filters of Step 1.

Each filter decides, per sentence, whether it belongs to D_X (matched) or
D_base (clean).  That decision defines the retrieval ground truth for every
Precision@k number in the paper, so it is worth checking that the filters
still fire on a hand-built example of each phenomenon and stay silent on a
neutral sentence.

Run with:  python tests/test_filters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "corpus_filtering" / "src"))

from conllu import parse  # noqa: E402

from corpus_filtering.filters.base import (  # noqa: E402
    BindingReflexive,
    ExistentialThereQuantifierFilter,
    InterrogativeWhModifierFilter,
    NukeNPI,
)

# CoNLL-U fixtures: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
NEUTRAL = """\
1\tThe\tthe\tDET\t_\t_\t2\tdet\t_\t_
2\tdog\tdog\tNOUN\t_\t_\t3\tnsubj\t_\t_
3\tsleeps\tsleep\tVERB\t_\t_\t0\troot\t_\t_
4\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_
"""

EXISTENTIAL_THERE = """\
1\tThere\tthere\tPRON\t_\t_\t2\texpl\t_\t_
2\tare\tbe\tVERB\t_\t_\t0\troot\t_\t_
3\tfew\tfew\tDET\t_\t_\t4\tdet\t_\t_
4\tcars\tcar\tNOUN\t_\t_\t2\tnsubj\t_\t_
5\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_
"""

BINDING_REFLEXIVE = """\
1\tThe\tthe\tDET\t_\t_\t2\tdet\t_\t_
2\tboy\tboy\tNOUN\t_\t_\t3\tnsubj\t_\t_
3\thurt\thurt\tVERB\t_\t_\t0\troot\t_\t_
4\thimself\thimself\tPRON\t_\tReflex=Yes\t3\tobj\t_\t_
5\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_
"""

WH_ISLAND = """\
1\tWhich\twhich\tDET\t_\tPronType=Int\t2\tdet\t_\t_
2\tgirls\tgirl\tNOUN\t_\t_\t3\tnsubj\t_\t_
3\tran\trun\tVERB\t_\t_\t0\troot\t_\t_
4\t?\t?\tPUNCT\t_\t_\t3\tpunct\t_\t_
"""

NPI = """\
1\tI\tI\tPRON\t_\t_\t4\tnsubj\t_\t_
2\tam\tbe\tAUX\t_\t_\t4\taux\t_\t_
3\tnot\tnot\tPART\t_\t_\t4\tadvmod\t_\t_
4\tplaying\tplay\tVERB\t_\t_\t0\troot\t_\t_
5\tanymore\tanymore\tADV\t_\t_\t4\tadvmod\t_\t_
6\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\t_
"""

CASES = [
    ("existential_there",   ExistentialThereQuantifierFilter(), EXISTENTIAL_THERE),
    ("binding_reflexives",  BindingReflexive(),                 BINDING_REFLEXIVE),
    ("wh_islands",          InterrogativeWhModifierFilter(),    WH_ISLAND),
    ("npi",                 NukeNPI(),                          NPI),
]


def main() -> int:
    failures = 0
    neutral_sent = parse(NEUTRAL)[0]

    for label, filt, conllu in CASES:
        sent = parse(conllu)[0]

        # The positive example must land in D_X …
        if filt._exclude_sent(sent):
            print(f"  ok    {label:20s} matches its D_X example (filter '{filt.name}')")
        else:
            print(f"  FAIL  {label:20s} did NOT match its D_X example")
            failures += 1

        # … and the neutral sentence must stay in D_base.
        if filt._exclude_sent(neutral_sent):
            print(f"  FAIL  {label:20s} wrongly matched the neutral sentence")
            failures += 1
        else:
            print(f"  ok    {label:20s} leaves the neutral sentence in D_base")

    print(f"\nFAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
