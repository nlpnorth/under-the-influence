"""Regenerate benchmark plots as PDF from archived CSV data.

Reads from docs/benchmarks/archive/ and CLI benchmark run directories,
then writes all plots to a single output directory.

Usage:
    python scripts/regenerate_plots.py
    python scripts/regenerate_plots.py --output_dir docs/benchmarks/pdf --format pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

ARCHIVE_DIR = Path("docs/benchmarks/archive")

CLI_BENCH_ROOTS = {
    1: Path("/projects/a6a/public/lucia/cli_bench_1gpu"),
    4: Path("/projects/a6a/public/lucia/cli_bench_4gpu"),
}

# -- projection comparison styles --
METHOD_STYLES = {
    "bergson": ("#2ca02c", "o"),
    "dattri": ("#d62728", "^"),
}


# ---------------------------------------------------------------------------
# CLI benchmark data loading
# ---------------------------------------------------------------------------


def _load_cli_records(run_root: Path, num_gpus: int) -> pd.DataFrame:
    """Load CLI benchmark JSON records from a run root directory."""
    rows = []
    if not run_root.exists():
        print(f"CLI bench root not found: {run_root}")
        return pd.DataFrame()

    for json_path in sorted(run_root.rglob("benchmark_cli.json")):
        with open(json_path) as f:
            r = json.load(f)
        if r.get("status") != "success" or r.get("total_runtime_seconds") is None:
            continue
        rows.append(
            {
                "model_key": r["model_key"],
                "model_name": r.get("model_name"),
                "model_params": r.get("params"),
                "train_tokens": r["train_tokens"],
                "eval_tokens": r.get("eval_tokens"),
                "dataset": r.get("dataset"),
                "batch_size": r.get("batch_size"),
                "build_seconds": r.get("build_seconds"),
                "reduce_seconds": r.get("reduce_seconds"),
                "score_seconds": r.get("score_seconds"),
                "total_runtime_seconds": r["total_runtime_seconds"],
                "run_path": r.get("run_path"),
                "num_gpus": r.get("num_gpus", num_gpus),
                "hardware": r.get("hardware"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        # Keep only the latest run per (model, train_tokens)
        df = df.sort_values("run_path").drop_duplicates(
            subset=["model_key", "train_tokens"],
            keep="last",
        )
    print(f"Loaded {len(df)} CLI runs from {run_root}")
    return df


def _extract_gpu_name(hardware: str) -> str:
    """Extract GPU name from hardware string like
    'nid010546 (4x NVIDIA GH200 120GB)'."""
    if "(" in hardware and ")" in hardware:
        inner = hardware.split("(")[1].split(")")[0]
        parts = inner.split(" ", 1)
        if len(parts) == 2 and parts[0].endswith("x"):
            return parts[1]
        return inner
    return hardware


# ---------------------------------------------------------------------------
# Projection comparison
# ---------------------------------------------------------------------------


def _plot_projection_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    if df.empty:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    for model_key in sorted(df["model_key"].unique()):
        for method in ["bergson", "dattri"]:
            subset = df[(df["method"] == method) & (df["model_key"] == model_key)]
            if subset.empty:
                continue
            subset = subset.sort_values("train_tokens")
            color, marker = METHOD_STYLES[method]
            ax.plot(
                subset["train_tokens"],
                subset["runtime_seconds"],
                marker=marker,
                color=color,
                label=f"{method} ({model_key})",
                linewidth=1.5,
                markersize=6,
                alpha=0.8,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tokens", fontsize=11)
    ax.set_ylabel("Runtime (seconds)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=7, ncol=1)


def plot_projection_comparison(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    fixed_df = df[df["batch_strategy"] == "fixed"]
    optimal_df = df[df["batch_strategy"] == "optimal"]

    panels = [
        ("With Projection, Fixed Batch", fixed_df[fixed_df["projection"] == "with"]),
        (
            "Without Projection, Fixed Batch",
            fixed_df[fixed_df["projection"] == "without"],
        ),
        (
            "With Projection, Optimal Batch",
            optimal_df[optimal_df["projection"] == "with"],
        ),
        (
            "Without Projection, Optimal Batch",
            optimal_df[optimal_df["projection"] == "without"],
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

    for ax, (title, panel_df) in zip(axes, panels):
        _plot_projection_panel(ax, panel_df, title)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved projection comparison to {output_path}")


# ---------------------------------------------------------------------------
# Build vs score panel (shared by programmatic + CLI)
# ---------------------------------------------------------------------------


def _plot_build_score(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    build_marker: str = "^",
    build_linestyle: str = "--",
) -> None:
    for model_key in sorted(df["model_key"].unique()):
        subset = df[df["model_key"] == model_key].sort_values("train_tokens")
        if subset["build_seconds"].notna().any():
            ax.plot(
                subset["train_tokens"],
                subset["build_seconds"],
                marker=build_marker,
                label=f"{model_key} (build)",
                linewidth=2,
                linestyle=build_linestyle,
            )
        if subset["score_seconds"].notna().any():
            ax.plot(
                subset["train_tokens"],
                subset["score_seconds"],
                marker="D",
                label=f"{model_key} (score)",
                linewidth=2,
                linestyle=":",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tokens", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8, ncol=2)


# ---------------------------------------------------------------------------
# Programmatic benchmark
# ---------------------------------------------------------------------------


def plot_programmatic_benchmark(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    if df.empty:
        print("No data to plot")
        return

    _, ax = plt.subplots(figsize=(7, 6))
    _plot_build_score(ax, df, "")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved programmatic benchmark to {output_path}")


# ---------------------------------------------------------------------------
# CLI benchmark
# ---------------------------------------------------------------------------


def plot_cli_benchmark(
    gpu_data: dict[int, pd.DataFrame],
    output_path: Path,
    archive_dir: Path | None = None,
) -> None:
    """Create a combined 1xN CLI benchmark plot, one subplot per GPU count."""
    panels: list[tuple[pd.DataFrame, str]] = []
    hw_names: set[str] = set()

    for num_gpus in sorted(gpu_data):
        df = gpu_data[num_gpus]
        hw_col = df["hardware"].dropna()
        hw = "unknown"
        if not hw_col.empty:
            hw = _extract_gpu_name(hw_col.iloc[0])
        if hw != "unknown":
            hw_names.add(hw)

        # Save CSV to archive if requested
        if archive_dir is not None:
            label = hw.replace(" ", "_")
            csv_path = archive_dir / f"cli_benchmark_{num_gpus}x_{label}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"Saved {num_gpus}-GPU CSV to {csv_path}")

        title = f"{num_gpus} GPU"
        panels.append((df, title))

    if not panels:
        print("No CLI benchmark data to plot")
        return

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, (df, title) in zip(axes, panels):
        _plot_build_score(ax, df, title, build_marker="s", build_linestyle="-")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved CLI benchmark to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate benchmark plots as PDF from archived CSVs "
        "and CLI benchmark run directories.",
    )
    parser.add_argument(
        "--output_dir",
        default="docs/benchmarks/pdf",
        help="Directory to write plots into",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["png", "pdf", "svg"],
    )
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    # Projection comparison
    proj_csv = ARCHIVE_DIR / "projection_comparison_4x_NVIDIA_GH200_120GB.csv"
    if proj_csv.exists():
        plot_projection_comparison(
            proj_csv,
            out / f"projection_comparison_4x_NVIDIA_GH200_120GB.{fmt}",
        )
    else:
        print(f"Skipping projection comparison: {proj_csv} not found")

    # Programmatic benchmark
    prog_csv = ARCHIVE_DIR / "programmatic_benchmark_1x_NVIDIA_GH200_120GB.csv"
    if prog_csv.exists():
        plot_programmatic_benchmark(
            prog_csv,
            out / f"programmatic_benchmark_1x_NVIDIA_GH200_120GB.{fmt}",
        )
    else:
        print(f"Skipping programmatic benchmark: {prog_csv} not found")

    # CLI benchmark
    gpu_data: dict[int, pd.DataFrame] = {}
    hw_names: set[str] = set()
    for num_gpus, root in sorted(CLI_BENCH_ROOTS.items()):
        df = _load_cli_records(root, num_gpus)
        if not df.empty:
            gpu_data[num_gpus] = df
            hw_col = df["hardware"].dropna()
            if not hw_col.empty:
                hw_names.add(_extract_gpu_name(hw_col.iloc[0]))

    if gpu_data:
        file_hw = (
            "_".join(sorted(hw_names)).replace(" ", "_") if hw_names else "unknown"
        )
        plot_cli_benchmark(
            gpu_data,
            out / f"cli_benchmark_{file_hw}.{fmt}",
            archive_dir=ARCHIVE_DIR,
        )
    else:
        print("Skipping CLI benchmark: no data found")


if __name__ == "__main__":
    main()
