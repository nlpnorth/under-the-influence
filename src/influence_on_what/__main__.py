"""Step-3 attribution pipeline.

Three subcommands, normally invoked by the scripts in ``run/`` rather than
by hand:

  prepare-data  Build the training dataset (D_base ∪ D_X) for attribution.
                Only a fallback: ``run/attribute_full.sh`` prepares its data
                with ``prepare/linguistic.py`` or ``prepare/facts.py``, which
                sample D_base at the corpus ratio and attach the per-fact or
                per-phenomenon metadata that ``analyze`` needs.

  attribute     Score every training example by the influence contrast
                ΔI(z) = I(s⁺, z) − I(s⁻, z) for each held-out query pair.
                Writes delta_I_stats.npz, top_inspect.parquet, prec_at_k.json.

  analyze       Aggregate ΔI across seeds: mean/median with CIs over D_X and
                over the D_base sample, the one-sided Wilcoxon test, and
                Prec@k. Writes results_<run_id>.json.

Every run is wrapped in a CodeCarbon tracker, so the energy and carbon cost of
each step is measured rather than estimated after the fact.

Model training is NOT part of this package — it uses the Goldfish recipe
directly, driven by ``run/train_full.sh`` and ``run/train_no_x.sh``.
Step-2 verification lives in ``probe_models.py``, driven by ``run/evaluate.sh``.

Usage
-----
  python -m influence_on_what attribute \\
      --config config/attribution_linguistic.yaml \\
      --name my_experiment --method ekfac
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import simple_parsing
from codecarbon import OfflineEmissionsTracker
from dotenv import load_dotenv

from . import analyze as analyze_mod
from . import attribute as attribute_mod
from . import data as data_mod
from .config import ExperimentCfg

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[influence_on_what %(levelname)s %(asctime)s - %(name)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

if "HF_TOKEN" not in os.environ:
    logging.info(
        "HF_TOKEN not set. Only needed for gated HuggingFace assets; "
        "the pipeline runs without it."
    )


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------


def _load_config_file(argv: list[str]) -> tuple[ExperimentCfg | None, list[str]]:
    if "--config" not in argv:
        return None, argv

    idx = argv.index("--config")
    config_path = Path(argv[idx + 1])
    argv = argv[:idx] + argv[idx + 2 :]

    cfg = ExperimentCfg.loads_yaml(config_path.read_text())
    return cfg, argv


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    raw = sys.argv[1:]

    sub_pos = next((i for i, a in enumerate(raw) if not a.startswith("-")), None)
    if sub_pos is not None:
        subcmd = raw[sub_pos]
        rest = raw[:sub_pos] + raw[sub_pos + 1 :]
        base_cfg, rest = _load_config_file(rest)
        sys.argv = [sys.argv[0], subcmd] + rest
    else:
        base_cfg = None
        sys.argv = [sys.argv[0]] + raw

    parser = simple_parsing.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd, help_text in [
        ("prepare-data", "Build the attribution training dataset (fallback)"),
        ("attribute", "Step 3 — influence attribution for each seed"),
        ("analyze", "Step 3 — aggregate ΔI, compute Prec@k and statistics"),
    ]:
        sub = subparsers.add_parser(cmd, help=help_text)
        sub.add_arguments(ExperimentCfg, dest="exp", default=base_cfg)

        if cmd == "attribute":
            sub.add_argument(
                "--seed",
                type=int,
                default=None,
                help="Run for a single seed only (default: all seeds in config).",
            )

    ns = parser.parse_args()
    exp: ExperimentCfg = ns.exp

    if ns.command in ("attribute", "analyze"):
        experiment_name = f"{ns.command}_{exp.attribution.method}"
    else:
        experiment_name = ns.command

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # One CSV per invocation, stamped with the start time.  A fixed filename
    # would let a re-run of the same step overwrite an earlier measurement, and
    # the true cost of the work is the sum over ALL runs — including the ones
    # that failed and had to be resubmitted.
    emissions_dir = exp.artifacts_dir / "codecarbon" / experiment_name
    emissions_dir.mkdir(parents=True, exist_ok=True)

    tracker = OfflineEmissionsTracker(
        output_dir=str(emissions_dir),
        output_file=f"emissions_{timestamp}.csv",
        save_to_api=False,
        country_iso_code="DNK",
        log_level=logging.WARNING,
    )
    tracker.start()

    try:
        if ns.command == "prepare-data":
            data_mod.prepare_and_save(exp)

        elif ns.command == "attribute":
            seeds = [ns.seed] if ns.seed is not None else exp.seeds
            for seed in seeds:
                attribute_mod.run(exp, seed)

        elif ns.command == "analyze":
            analyze_mod.run(exp, run_id=timestamp)
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
