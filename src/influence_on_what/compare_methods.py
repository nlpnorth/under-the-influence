"""
Compare attribution methods pairwise: score correlations and Prec@k.

For each method pair computes:
  - Spearman / Pearson r on mean-ΔI-per-D_X (marginalised over pairs)
  - Mean per-pair Spearman ρ (and std) across individual query pairs
  - Jaccard overlap of top-k D_X examples (by mean ΔI) for several k
  - ΔI distribution summary (mean, std, p5, p95)
  - Side-by-side Prec@k table

Usage:
  python compare_methods.py --artifacts-dir /path/to/artifacts --name exp_name [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# method_dir name → short display label
METHODS = {
    "vanilla":            "vanilla",
    "trackstar":          "trackstar",
    "ekfac_full_precond": "ekfac-sketched",
    "kfac_full_precond":  "kfac-sketched",
    "kfac_effic_precond": "kfac-efficient",
    "ekfac":              "ekfac-query-batch",
    "kfac":               "kfac-query-batch",
}

_FMT4 = lambda x: f"{x:.4f}"
_FMT6 = lambda x: f"{x:.6f}"


def _ensure_extracted(npz_path: Path, raw_dir: Path) -> Path:
    """Unzip dx.npy/base.npy/n_pairs.npy etc. from *npz_path* into *raw_dir*
    via the system ``unzip`` (cached — skips if already extracted).

    numpy's own npz reader decompresses through Python's zipfile module,
    which is impractically slow for multi-GB, barely-compressible arrays
    (e.g. the Wikipedia facts panel, ~56GB dx.npy). Unzipping once to local
    scratch and then mmap-loading the raw .npy avoids that bottleneck and
    avoids holding the whole array in RAM at once.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not (raw_dir / "dx.npy").exists():
        subprocess.run(
            ["unzip", "-o", "-q", str(npz_path), "-d", str(raw_dir)], check=True
        )
    return raw_dir


def _load_scores(method_dir: Path, extract_dir: Path | None = None):
    """Load (mean_dI, n_pairs, per_pair_dI) or return None if missing.

    dx layout in npz: (n_pairs * n_dx,) with stride n_dx per pair.
    per_pair_dI shape: (n_pairs, n_dx).

    If *extract_dir* is given, dx/base are unzipped to raw .npy under it and
    memory-mapped instead of being fully decompressed by np.load — see
    _ensure_extracted. mean_dI (a full-array reduction over a bad access
    pattern for mmap'd data) is left as None in that mode; callers must skip
    the sections that need it.
    """
    f = method_dir / "delta_I_stats.npz"
    if not f.exists():
        return None
    if extract_dir is not None:
        raw_dir = _ensure_extracted(f, extract_dir / method_dir.name)
        dx = np.load(raw_dir / "dx.npy", mmap_mode="r")
        n_pairs = int(np.load(raw_dir / "n_pairs.npy"))
        n_dx = dx.size // n_pairs
        per_pair = dx.reshape(n_pairs, n_dx)
        return None, n_pairs, per_pair
    data = np.load(str(f))
    dx = data["dx"].astype(np.float32)
    n_pairs = int(data["n_pairs"])
    n_dx = dx.size // n_pairs
    per_pair = dx.reshape(n_pairs, n_dx)   # (n_pairs, n_dx)
    mean_dI = per_pair.mean(axis=0)        # (n_dx,)
    return mean_dI, n_pairs, per_pair


def _load_prec(method_dir: Path) -> dict | None:
    f = method_dir / "prec_at_k.json"
    if not f.exists():
        return None
    with open(f) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--k-overlap", nargs="+", type=int, default=[10, 50, 100, 500],
        help="k values for top-k Jaccard overlap",
    )
    ap.add_argument(
        "--extract-dir", default=None,
        help="Scratch dir to unzip huge npz files into and mmap-load from "
             "(needed for panels whose delta_I_stats.npz is many GB, e.g. "
             "Wikipedia facts). Skips the mean-ΔI-based sections (ΔI "
             "distribution, mean-ΔI Spearman/Pearson, Jaccard overlap) since "
             "those need a full-array column reduction that's impractically "
             "slow over mmap'd data — only the per-pair correlations and "
             "Prec@k table are produced.",
    )
    args = ap.parse_args()
    extract_dir = Path(args.extract_dir) if args.extract_dir else None

    base = Path(args.artifacts_dir) / args.name
    if args.seed is not None:
        seed = args.seed
    else:
        seed_dirs = sorted(base.glob("seed_*"))
        if not seed_dirs:
            raise FileNotFoundError(f"No seed_* dirs under {base}")
        seed = int(seed_dirs[0].name.split("_")[1])

    attr_dir = base / f"seed_{seed}" / "attribution"
    out_dir  = base / f"seed_{seed}" / "method_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    loaded: dict[str, tuple] = {}
    for mdir_name, label in METHODS.items():
        result = _load_scores(attr_dir / mdir_name, extract_dir=extract_dir)
        if result is not None:
            loaded[label] = result
        else:
            print(f"[skip] {label} — delta_I_stats.npz not found at {attr_dir / mdir_name}")

    if len(loaded) < 2:
        print("Need at least 2 methods to compare.")
        return

    # Sanity check: all n_dx must match (same D_X set) — use per_pair's own
    # shape rather than mean_dI, which is None in mmap (--extract-dir) mode.
    n_dxs = {lbl: v[2].shape[1] for lbl, v in loaded.items()}
    if len(set(n_dxs.values())) > 1:
        print(f"WARNING: n_dx mismatch: {n_dxs} — truncating to min")
        min_ndx = min(n_dxs.values())
        loaded = {
            lbl: (v[0][:min_ndx] if v[0] is not None else None, v[1], v[2][:, :min_ndx])
            for lbl, v in loaded.items()
        }

    labels = list(loaded.keys())

    # ── 1. ΔI distribution summary (skipped in mmap mode — needs mean_dI) ────
    if extract_dir is None:
        dist_rows = []
        for lbl, (mean_dI, n_pairs, _) in loaded.items():
            dist_rows.append({
                "method":  lbl,
                "n_pairs": n_pairs,
                "mean_ΔI": float(mean_dI.mean()),
                "std_ΔI":  float(mean_dI.std()),
                "p5_ΔI":   float(np.percentile(mean_dI, 5)),
                "p95_ΔI":  float(np.percentile(mean_dI, 95)),
            })
        dist_df = pd.DataFrame(dist_rows).set_index("method")
        print("\n=== ΔI distribution (mean-ΔI per D_X, marginalised over pairs) ===")
        print(dist_df.to_string(float_format=_FMT6))
        dist_df.to_csv(out_dir / "dI_distribution.csv")
    else:
        print("\n[skip] ΔI distribution — mmap mode (--extract-dir), no mean_dI")

    # ── 2. Pairwise correlations ──────────────────────────────────────────────
    n = len(labels)
    sp_mat = pd.DataFrame(np.eye(n), index=labels, columns=labels)
    pe_mat = pd.DataFrame(np.eye(n), index=labels, columns=labels)
    pp_rows = []

    for a, b in combinations(labels, 2):
        if extract_dir is None:
            mean_a = loaded[a][0]
            mean_b = loaded[b][0]
            sp, _ = spearmanr(mean_a, mean_b)
            pe, _ = pearsonr(mean_a, mean_b)
            sp_mat.loc[a, b] = sp_mat.loc[b, a] = float(sp)
            pe_mat.loc[a, b] = pe_mat.loc[b, a] = float(pe)

        # Per-pair Spearman/Pearson: correlate each pair column independently
        n_shared = min(loaded[a][1], loaded[b][1])
        pp_a = loaded[a][2][:n_shared]
        pp_b = loaded[b][2][:n_shared]
        pair_rhos = [float(spearmanr(pp_a[j], pp_b[j])[0]) for j in range(n_shared)]
        pair_rs = [float(pearsonr(pp_a[j], pp_b[j])[0]) for j in range(n_shared)]
        pp_rows.append({
            "pair":          f"{a} vs {b}",
            "mean_spearman": float(np.mean(pair_rhos)),
            "std_spearman":  float(np.std(pair_rhos)),
            "min_spearman":  float(np.min(pair_rhos)),
            "max_spearman":  float(np.max(pair_rhos)),
            "mean_pearson":  float(np.mean(pair_rs)),
            "std_pearson":   float(np.std(pair_rs)),
            "min_pearson":   float(np.min(pair_rs)),
            "max_pearson":   float(np.max(pair_rs)),
        })

    if extract_dir is None:
        print("\n=== Spearman ρ — mean ΔI per D_X example ===")
        print(sp_mat.to_string(float_format=_FMT4))
        print("\n=== Pearson r  — mean ΔI per D_X example ===")
        print(pe_mat.to_string(float_format=_FMT4))
        sp_mat.to_csv(out_dir / "spearman_mean_dI.csv")
        pe_mat.to_csv(out_dir / "pearson_mean_dI.csv")

    pp_df = pd.DataFrame(pp_rows).set_index("pair")
    print("\n=== Mean per-pair Spearman ρ / Pearson r (averaged across query pairs) ===")
    print(pp_df.to_string(float_format=_FMT4))
    pp_df.to_csv(out_dir / "per_pair_correlations.csv")

    # ── 3. Top-k Jaccard overlap on mean ΔI (skipped in mmap mode) ──────────
    if extract_dir is None:
        print("\n=== Top-k Jaccard overlap (top D_X by mean ΔI) ===")
        n_dx = next(iter(loaded.values()))[0].size
        for k in sorted(args.k_overlap):
            k_eff = min(k, n_dx)
            if k_eff < k:
                print(f"\n  k={k} — clamped to {k_eff} (only {n_dx} D_X examples)")
            jac = pd.DataFrame(np.eye(n), index=labels, columns=labels)
            for a, b in combinations(labels, 2):
                top_a = set(np.argpartition(loaded[a][0], -k_eff)[-k_eff:].tolist())
                top_b = set(np.argpartition(loaded[b][0], -k_eff)[-k_eff:].tolist())
                j = len(top_a & top_b) / len(top_a | top_b)
                jac.loc[a, b] = jac.loc[b, a] = j
            print(f"\n  k={k_eff}")
            print(jac.to_string(float_format=_FMT4))
            jac.to_csv(out_dir / f"jaccard_top{k_eff}.csv")
    else:
        print("\n[skip] Jaccard overlap — mmap mode (--extract-dir), no mean_dI")

    # ── 4. Prec@k table ───────────────────────────────────────────────────────
    prec_rows = []
    for mdir_name, label in METHODS.items():
        p = _load_prec(attr_dir / mdir_name)
        if p is None:
            continue
        row: dict = {"method": label}
        row.update({f"P@{k}":      v for k, v in p.get("prec_at_k",     {}).items()})
        row.update({f"P@{k}(s+)":  v for k, v in p.get("prec_at_k_good", {}).items()})
        row.update({f"P@{k}(s-)":  v for k, v in p.get("prec_at_k_bad",  {}).items()})
        prec_rows.append(row)
    if prec_rows:
        prec_df = pd.DataFrame(prec_rows).set_index("method")
        print("\n=== Prec@k ===")
        print(prec_df.to_string(float_format=_FMT6))
        prec_df.to_csv(out_dir / "prec_at_k.csv")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
