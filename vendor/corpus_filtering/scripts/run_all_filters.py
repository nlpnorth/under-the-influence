import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.corpus_filtering.filters.base import (
    BindingReflexive,
    ExistentialThereQuantifierFilter,
    InterrogativeWhModifierFilter,
    NukeNPI,
)
from src.corpus_filtering.pipeline import run_filters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Path to input (pickle, conllu file, or UD dir)"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    filters = [
        ExistentialThereQuantifierFilter(),
        BindingReflexive(),
        InterrogativeWhModifierFilter(),
        NukeNPI(),
    ]
    run_filters(filters, args.input, output_dir=args.output)
