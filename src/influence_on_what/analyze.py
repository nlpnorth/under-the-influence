"""Step 2 analysis: aggregate ΔI scores, compute statistics, save results.

Loads ``delta_I_stats.npz`` and ``prec_at_k.json`` from all available seeds
of the full-condition model and computes the two-part evaluation:

**Presence of effect** (z ∈ D_X):
    • Mean, median, std of ΔI(z) with 95 % CIs (normal approximation).
    • One-sided Wilcoxon signed-rank test  H₀: ΔI(z) ≤ 0.
    • Prec@k — averaged across seeds from pre-computed prec_at_k.json files.

**Specificity** (z ∈ D_base subsample — sampled in the attribute step):
    • Mean, median, std of ΔI(z) with 95 % CIs.
    • Ratio  mean_base / mean_DX  (should be ≪ 1 for high specificity).

Verdict:
    PASS    — Wilcoxon p < 0.05  AND  |mean_base / mean_DX| < 0.1
    PARTIAL — one of the two criteria met
    FAIL    — neither criterion met

Results saved to ``artifacts/<name>/results/step2/<method>/results_<run_id>.json``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import pandas as pd
from scipy import stats

from .config import ExperimentCfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stats loader
# ---------------------------------------------------------------------------


def _load_stats_seed(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict, int]:
    """Load a seed's ``delta_I_stats.npz``.

    Returns
    -------
    dx_arr      : float32[ N_X ]    — ΔI for D_X examples
    base_arr    : float32[ N_base ] — ΔI for D_base subsample
    score_state : Welford state for raw ℐ(s, z) scores (n=0 for global-ΔI methods)
    n_pairs     : number of query pairs
    """
    data = np.load(str(path))
    dx_arr = data["dx"].astype(np.float32)
    base_arr = data["base"].astype(np.float32)
    score_state = {
        "n": int(data["score_n"]),
        "mean": float(data["score_mean"]),
        "M2": float(data["score_M2"]),
    }
    n_pairs = int(data["n_pairs"])
    return dx_arr, base_arr, score_state, n_pairs


def _welford_init() -> dict:
    return {"n": 0, "mean": 0.0, "M2": 0.0}


def _welford_update(state: dict, x: np.ndarray) -> None:
    """Chan parallel-update of running ``(n, mean, M2)`` with batch ``x``."""
    n_b = int(x.size)
    if n_b == 0:
        return
    mean_b = float(x.mean())
    var_b = float(x.var(ddof=0)) if n_b > 1 else 0.0
    M2_b = var_b * n_b
    n_a = state["n"]
    if n_a == 0:
        state["n"] = n_b
        state["mean"] = mean_b
        state["M2"] = M2_b
        return
    n = n_a + n_b
    delta = mean_b - state["mean"]
    state["mean"] = state["mean"] + delta * n_b / n
    state["M2"] = state["M2"] + M2_b + delta * delta * n_a * n_b / n
    state["n"] = n


def _welford_combine(a: dict, b: dict) -> None:
    """Merge Welford state ``b`` into ``a`` in place."""
    n_b = b["n"]
    if n_b == 0:
        return
    if a["n"] == 0:
        a["n"] = n_b
        a["mean"] = b["mean"]
        a["M2"] = b["M2"]
        return
    n_a = a["n"]
    n = n_a + n_b
    delta = b["mean"] - a["mean"]
    a["mean"] = a["mean"] + delta * n_b / n
    a["M2"] = a["M2"] + b["M2"] + delta * delta * n_a * n_b / n
    a["n"] = n


def _welford_std(state: dict, ddof: int = 1) -> float:
    n = state["n"]
    if n <= ddof:
        return 0.0
    return float(np.sqrt(state["M2"] / (n - ddof)))


def _build_global_array(per_seed: dict[int, np.ndarray]) -> np.ndarray:
    """Concatenate per-seed arrays into a single buffer, popping each source
    so peak memory stays close to ``total + max_seed`` instead of ``2 × total``."""
    total = sum(int(a.size) for a in per_seed.values())
    if total == 0:
        per_seed.clear()
        return np.zeros(0, dtype=np.float32)
    out = np.empty(total, dtype=np.float32)
    offset = 0
    for seed in sorted(per_seed.keys()):
        arr = per_seed.pop(seed)
        n = int(arr.size)
        out[offset : offset + n] = arr
        offset += n
        del arr
    return out


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _ci95(values: np.ndarray) -> tuple[float, float]:
    """95 % confidence interval via normal approximation."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    se = values.std(ddof=1) / np.sqrt(n)
    margin = 1.96 * se
    mean = values.mean()
    return float(mean - margin), float(mean + margin)


def _summarise_delta_I(values: np.ndarray, score_std: float | None = None) -> dict:
    """Return descriptive statistics for a 1-D array of ΔI values.

    If *score_std* is provided (std of all raw ℐ(s, z) scores), the dict also
    includes ``norm_mean`` and ``norm_median`` — ΔI divided by that global std.
    This puts different attribution methods on a comparable scale while keeping
    the original (unnormalized) statistics intact.
    """
    ci_lo, ci_hi = _ci95(values)
    result = {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        "ci95_lower": ci_lo,
        "ci95_upper": ci_hi,
    }
    if score_std is not None and score_std > 0:
        result["norm_mean"] = result["mean"] / score_std
        result["norm_median"] = result["median"] / score_std
    return result


def _summarise_named_values(
    values: list[tuple[int, float]], *, value_name: str = "value"
) -> dict:
    """Summarise seed-level values while preserving the individual inputs."""
    arr = np.asarray([v for _, v in values], dtype=np.float32)
    if len(arr) == 0:
        mean = float("nan")
        std = float("nan")
    else:
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")

    return {
        "mean": mean,
        "std": std,
        "n": int(len(values)),
        "individual_values": [
            {"seed": int(seed), value_name: float(value)} for seed, value in values
        ],
    }


def _seed_delta_I_summaries(
    per_seed: dict[int, np.ndarray], score_std: float | None
) -> list[dict]:
    """Per-seed ΔI summaries from a ``{seed: float32 array}`` mapping."""
    summaries = [
        {
            "seed": int(seed),
            "delta_I": _summarise_delta_I(per_seed[seed], score_std=score_std),
        }
        for seed in sorted(per_seed)
    ]
    return summaries


def _aggregate_seed_delta_I(seed_summaries: list[dict]) -> dict:
    """Aggregate seed-level ΔI summaries and retain each seed's value."""
    aggregates = {
        "mean": _summarise_named_values(
            [(s["seed"], s["delta_I"]["mean"]) for s in seed_summaries],
            value_name="delta_I_mean",
        ),
        "median": _summarise_named_values(
            [(s["seed"], s["delta_I"]["median"]) for s in seed_summaries],
            value_name="delta_I_median",
        ),
    }

    if seed_summaries and all("norm_mean" in s["delta_I"] for s in seed_summaries):
        aggregates["norm_mean"] = _summarise_named_values(
            [(s["seed"], s["delta_I"]["norm_mean"]) for s in seed_summaries],
            value_name="norm_delta_I_mean",
        )
        aggregates["norm_median"] = _summarise_named_values(
            [(s["seed"], s["delta_I"]["norm_median"]) for s in seed_summaries],
            value_name="norm_delta_I_median",
        )

    return aggregates


def _wilcoxon_one_sided(values: np.ndarray) -> dict:
    """One-sided Wilcoxon signed-rank test  H₀: median ≤ 0  (H₁: median > 0)."""
    if len(values) < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "passed": False}
    result = stats.wilcoxon(values, alternative="greater")
    p = float(result.pvalue)
    return {
        "statistic": float(result.statistic),
        "p_value": p,
        "passed": p < 0.05,
    }


def _average_prec_at_k(prec_payloads: list[dict], key: str = "prec_at_k") -> dict:
    """Average Prec@k across seeds.

    Each payload has the shape produced by attribute.py:
      {"seed": int, "mode": str, "prec_at_k": {"10": float, "20": float, …}}

    Returns a dict with the averaged values plus metadata.
    """
    if not prec_payloads:
        return {}

    all_k = sorted({k for p in prec_payloads for k in p.get(key, {})})
    if not all_k:
        return {}
    by_k = {
        k: [
            (int(p["seed"]), float(p[key][k]))
            for p in prec_payloads
            if k in p.get(key, {})
        ]
        for k in all_k
    }
    by_k_summary = {
        k: _summarise_named_values(values, value_name="precision")
        for k, values in by_k.items()
    }
    averaged = {k: v["mean"] for k, v in by_k_summary.items()}
    std = {k: v["std"] for k, v in by_k_summary.items()}
    individual_values = {k: v["individual_values"] for k, v in by_k_summary.items()}
    per_seed = sorted(
        [
            {
                "seed": int(p["seed"]),
                "mode": p.get("mode"),
                "values": {k: float(v) for k, v in p.get(key, {}).items()},
            }
            for p in prec_payloads
        ],
        key=lambda item: item["seed"],
    )
    modes = list({p.get("mode") for p in prec_payloads})
    return {
        "values": averaged,
        "std": std,
        "individual_values": individual_values,
        "summary": by_k_summary,
        "per_seed": per_seed,
        "n_seeds": len(prec_payloads),
        "mode": modes[0] if len(modes) == 1 else modes,
    }


def _resolve_prec_pool_size(
    mode: str | None, n_dx: int, exp: ExperimentCfg
) -> int | None:
    """Size of the ranking pool a Prec@k payload's ``mode`` was computed over.

    ``"subset_base_N"``  → pool = n_dx + N (D_X ∪ N sampled D_base rows).
    ``"full_corpus"``    → pool = n_dx + D_base actually scored, i.e. capped by
        ``scoring_base_size`` if that subsamples D_base before scoring,
        otherwise the full ``train_data_path()`` row count.
    """
    if not mode or n_dx <= 0:
        return None
    if mode.startswith("subset_base_"):
        try:
            n_sub = int(mode.rsplit("_", 1)[-1])
        except ValueError:
            return None
        return n_dx + n_sub
    if mode == "full_corpus":
        train_path = exp.train_data_path()
        if not train_path.exists():
            return None
        n_base_full = len(hf_datasets.load_from_disk(str(train_path))) - n_dx
        scoring_base_size = exp.attribution.scoring_base_size
        if scoring_base_size > 0:
            return n_dx + min(scoring_base_size, n_base_full)
        return n_dx + n_base_full
    return None


def _lift_at_k(prec_result: dict, n_dx: int, pool_size: int | None) -> dict:
    """Lift@k = Prec@k / baseline_rate, where baseline_rate = n_dx / pool_size."""
    if not prec_result or not pool_size or n_dx <= 0:
        return {}
    baseline = n_dx / pool_size
    return {k: v / baseline for k, v in prec_result.get("values", {}).items()}


def _seed_ratio_summaries(
    per_seed_dx: dict[int, np.ndarray], per_seed_base: dict[int, np.ndarray]
) -> dict:
    values: list[tuple[int, float]] = []
    for seed in sorted(per_seed_dx):
        dx_vals = per_seed_dx[seed]
        base_vals = per_seed_base.get(seed, np.zeros(0, dtype=np.float32))
        mean_dx = float(np.mean(dx_vals)) if dx_vals.size > 0 else float("nan")
        mean_base = float(np.mean(base_vals)) if base_vals.size > 0 else float("nan")
        if mean_dx != 0 and not np.isnan(mean_dx):
            ratio = abs(mean_base / mean_dx)
        else:
            ratio = float("nan")
        values.append((int(seed), ratio))

    return _summarise_named_values(values, value_name="mean_ratio_base_to_dx")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _run_core(
    exp: ExperimentCfg,
    run_id: str,
    *,
    stats_path_fn,
    prec_path_fn,
    suffix: str,
    split: str,
    required: bool,
) -> None:
    """Aggregate ΔI scores across seeds, compute statistics, save JSON report.

    *stats_path_fn*/*prec_path_fn* select which variant to aggregate (combined,
    learned, or not_learned — see ExperimentCfg.delta_I_stats_path and its
    _learned/_not_learned siblings). *required*=False means "this split may
    legitimately not exist yet" (e.g. Step 1 evaluate hasn't run, or the
    method is Family B which has no per-pair split) — log and return instead
    of raising.
    """
    # ── Stream subset parquets seed-by-seed (D_X + D_base sample) ──────────
    per_seed_dx: dict[int, np.ndarray] = {}
    per_seed_base: dict[int, np.ndarray] = {}
    score_state = _welford_init()
    delta_state = _welford_init()
    has_scores_any = False
    prec_payloads: list[dict] = []
    pair_universe: set[int] = set()

    for seed in exp.seeds:
        stats_p = stats_path_fn(seed)
        prec_p = prec_path_fn(seed)

        if stats_p.exists():
            dx_arr, base_arr, seed_score_state, seed_n_pairs = _load_stats_seed(stats_p)
            per_seed_dx[int(seed)] = dx_arr
            per_seed_base[int(seed)] = base_arr
            if seed_score_state["n"] > 0:
                has_scores_any = True
                _welford_combine(score_state, seed_score_state)
            _welford_update(delta_state, dx_arr)
            _welford_update(delta_state, base_arr)
            pair_universe.add(seed_n_pairs)
        else:
            logger.warning(
                "[seed=%d][%s] stats file not found at %s — skipping.",
                seed,
                split,
                stats_p,
            )

        if prec_p.exists():
            with open(prec_p) as fh:
                prec_payloads.append(json.load(fh))
        else:
            logger.warning(
                "[seed=%d][%s] prec_at_k file not found at %s — skipping.",
                seed,
                split,
                prec_p,
            )

    if not per_seed_dx:
        msg = f"No {split} delta_I_stats.npz files found."
        if required:
            raise RuntimeError(f"{msg} Run 'attribute' first.")
        logger.info("%s Skipping the %s results report.", msg, split)
        return

    n_pairs = max(pair_universe) if pair_universe else 0
    n_dx_total = sum(int(a.size) for a in per_seed_dx.values())
    n_base_total = sum(int(a.size) for a in per_seed_base.values())

    logger.info(
        "Loaded stats: %d seeds, %d pairs, %d D_X entries, %d D_base entries",
        len(per_seed_dx),
        n_pairs,
        n_dx_total,
        n_base_total,
    )

    # ── Normalization denominator: std over all raw ℐ(s, z) ─────────────────
    # Pools score_good and score_bad across all rows so that normalized ΔI
    # values are comparable across attribution methods with different scales.
    # For global-ΔI methods (EK-FAC with aggregate query) the individual score
    # columns are zero-filled; fall back to std(ΔI) in that case.
    score_std: float | None = None
    if has_scores_any:
        s = _welford_std(score_state, ddof=1)
        if s > 0:
            score_std = s
    if score_std is None:
        s = _welford_std(delta_state, ddof=1)
        if s > 0:
            score_std = s

    # ── Per-seed summaries (computed before consuming per-seed arrays) ─────
    dx_by_seed = _seed_delta_I_summaries(per_seed_dx, score_std)
    base_by_seed = _seed_delta_I_summaries(per_seed_base, score_std)
    ratio_summaries = _seed_ratio_summaries(per_seed_dx, per_seed_base)

    # ── Build global vectors (consumes per-seed dicts to limit peak RAM) ───
    dx_vals = _build_global_array(per_seed_dx)
    base_vals = _build_global_array(per_seed_base)

    # ── Presence of effect: z ∈ D_X ────────────────────────────────────────
    presence = {
        "n_pairs": n_pairs,
        "n_dx_total": int(dx_vals.size),
        "delta_I": _summarise_delta_I(dx_vals, score_std=score_std),
        "delta_I_by_seed": dx_by_seed,
        "delta_I_across_seeds": _aggregate_seed_delta_I(dx_by_seed),
        "wilcoxon": _wilcoxon_one_sided(dx_vals),
        "precision_at_k": _average_prec_at_k(prec_payloads),
        "precision_at_k_good": _average_prec_at_k(prec_payloads, key="prec_at_k_good"),
        "precision_at_k_bad": _average_prec_at_k(prec_payloads, key="prec_at_k_bad"),
    }

    # ── Lift@k = Prec@k / baseline_rate, baseline_rate = n_dx / ranking_pool_size ──
    # delta_I_stats arrays are flattened (n_dx_examples * n_pairs) for per-pair
    # methods, so recover the actual D_X example count before sizing the pool.
    n_dx_seed = dx_by_seed[0]["delta_I"]["n"] if dx_by_seed else 0
    if n_pairs > 0:
        n_dx_seed //= n_pairs
    prec_mode = presence["precision_at_k"].get("mode")
    if isinstance(prec_mode, list):
        prec_mode = prec_mode[0] if prec_mode else None
    pool_size = _resolve_prec_pool_size(prec_mode, n_dx_seed, exp)
    presence["prec_pool"] = {
        "mode": prec_mode,
        "n_dx": n_dx_seed,
        "n_base": (pool_size - n_dx_seed) if pool_size else None,
        "pool_size": pool_size,
    }
    presence["lift_at_k"] = _lift_at_k(presence["precision_at_k"], n_dx_seed, pool_size)
    presence["lift_at_k_good"] = _lift_at_k(
        presence["precision_at_k_good"], n_dx_seed, pool_size
    )
    presence["lift_at_k_bad"] = _lift_at_k(
        presence["precision_at_k_bad"], n_dx_seed, pool_size
    )

    # ── Specificity: D_base subsample (already sampled in attribute step) ──
    mean_dx = presence["delta_I"]["mean"]
    mean_base = float(np.mean(base_vals)) if base_vals.size > 0 else float("nan")
    if mean_dx != 0 and not np.isnan(mean_dx):
        ratio = abs(mean_base / mean_dx)
    else:
        ratio = float("nan")

    specificity = {
        "n_base_rows": int(base_vals.size),
        "n_pairs": n_pairs,
        "delta_I": _summarise_delta_I(base_vals, score_std=score_std),
        "delta_I_by_seed": base_by_seed,
        "delta_I_across_seeds": _aggregate_seed_delta_I(base_by_seed),
        "mean_ratio_base_to_dx": ratio,
        "mean_ratio_base_to_dx_across_seeds": ratio_summaries,
    }

    # ── Verdict ─────────────────────────────────────────────────────────────
    wilcoxon_passed = presence["wilcoxon"]["passed"]
    specificity_passed = (not np.isnan(ratio)) and ratio < 0.1

    if wilcoxon_passed and specificity_passed:
        verdict = "PASS"
    elif wilcoxon_passed or specificity_passed:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    # ── Experiment metadata ─────────────────────────────────────────────────
    active_seeds = [int(s) for s in exp.seeds if stats_path_fn(s).exists()]
    config = exp.to_dict()
    experiment_meta = {
        "config": config,
        "checkpoints": {str(s): str(exp.model_dir(s)) for s in active_seeds},
    }

    # ── Load step1 results for each condition ───────────────────────────────
    # Each step1 summary file covers a single condition (full or base_only),
    # so collect all summary*.json files, group by their `by_condition` key,
    # and pick the exact-run_id match if present, else the most recent.
    step1_dir = exp.results_dir() / "step1"
    step1_summaries: dict[str, dict] = {}
    if step1_dir.exists():
        by_cond: dict[str, list[tuple]] = {}
        for path in sorted(step1_dir.glob("summary*.json")):
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            for cond in data.get("by_condition", {}):
                by_cond.setdefault(cond, []).append((path, data))
        exact_name = f"summary_{run_id}.json"
        for cond, entries in by_cond.items():
            chosen = next(
                (e for e in entries if e[0].name == exact_name),
                entries[-1],
            )
            path, data = chosen
            data["_source"] = path.name
            step1_summaries[cond] = data

    # ── Assemble and save ───────────────────────────────────────────────────
    summary = {
        "verdict": verdict,
        "split": split,
        "seeds": active_seeds,
        "condition": exp.data.condition,
        "experiment": experiment_meta,
        "step1": step1_summaries or None,
        "score_std": score_std,
        "presence": presence,
        "specificity": specificity,
    }

    out_path = (
        exp.results_dir()
        / "step2"
        / exp.attribution.method
        / f"results_{run_id}{suffix}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    prec_str = "  ".join(
        f"Prec@{k}={v:.3f}"
        for k, v in (presence["precision_at_k"].get("values") or {}).items()
    )
    lift_str = "  ".join(
        f"Lift@{k}={v:.3f}" for k, v in presence["lift_at_k"].items()
    )
    if lift_str:
        pool = presence["prec_pool"]
        prec_str = (
            f"{prec_str}\n  {lift_str}"
            f"  (pool: n_dx={pool['n_dx']} n_base={pool['n_base']} mode={pool['mode']})"
        )
    norm_str = ""
    if "norm_mean" in presence["delta_I"]:
        norm_str = (
            f"  Normalized ΔI:    mean={presence['delta_I']['norm_mean']:.10f}  "
            f"median={presence['delta_I']['norm_median']:.10f}  "
            f"(σ_scores={score_std:.10f})\n"
        )
    logger.info(
        "\n── Step 2 verdict: %s ──\n"
        "  Presence (D_X):   mean ΔI=%.10f  median=%.10f  "
        "Wilcoxon p=%.10f  %s\n"
        "%s"
        "  Specificity (D_base n=%d):  "
        "mean ΔI=%.10f  ratio=%.10f\n"
        "  Saved → %s",
        verdict,
        presence["delta_I"]["mean"],
        presence["delta_I"]["median"],
        presence["wilcoxon"]["p_value"],
        prec_str,
        norm_str,
        int(base_vals.size),
        specificity["delta_I"]["mean"],
        ratio,
        out_path,
    )

    if verdict == "FAIL":
        logger.warning(
            "Step 2 verdict is FAIL — the injected D_X examples are not "
            "significantly ranked above base examples. Check that (a) the "
            "model learned the property (Step 1 PASS) and (b) the gradient "
            "index is consistent between query and training."
        )


def run(exp: ExperimentCfg, run_id: str | None = None) -> None:
    """Aggregate ΔI scores across seeds and save the Step 2 results report.

    Produces the combined report (all query pairs — unchanged filename, for
    backward compatibility) plus, when the learned/not-learned split was
    computed in 'attribute' (Family A methods + bm25, from probe_models.py
    --pairs-output labels — see attribute.py's _load_pair_labels), separate
    reports restricted to pairs the model has/hasn't learned, and — when a
    no-X model's labels were also supplied — further restricted to pairs
    where the ablation worked (learned_specific) or didn't (learned_confounded).
    Use _learned_specific as the headline result when available (it isolates
    learning attributable to X); combined and the others are kept for context.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # For BEAR experiments (queries have a `fact_key` column), run_bear_per_source
    # is the sole output — it captures all per-fact and pooled statistics and
    # replaces the generic _run_core JSON.  _run_core is still used for
    # non-BEAR (linguistic) experiments.
    queries_path = exp.queries_path()
    _is_bear = False
    if queries_path.exists():
        import datasets as _hf
        _qds = _hf.load_from_disk(str(queries_path))
        _is_bear = "fact_key" in _qds.column_names
        del _qds

    if _is_bear:
        run_bear_per_source(exp, run_id=run_id)
    else:
        _run_core(
            exp, run_id,
            stats_path_fn=exp.delta_I_stats_path,
            prec_path_fn=exp.prec_at_k_path,
            suffix="",
            split="combined",
            required=True,
        )

        for label in ("learned", "not_learned", "learned_specific", "learned_confounded", "not_learned_no_x"):
            stats_path_fn = getattr(exp, f"delta_I_stats_{label}_path")
            if not any(stats_path_fn(seed).exists() for seed in exp.seeds):
                continue
            _run_core(
                exp, run_id,
                stats_path_fn=stats_path_fn,
                prec_path_fn=getattr(exp, f"prec_at_k_{label}_path"),
                suffix=f"_{label}",
                split=label,
                required=False,
            )


# ---------------------------------------------------------------------------
# Per-source BEAR attribution analysis — helpers
# ---------------------------------------------------------------------------

# Atomic filtered-source categories used for the exact ΔI comparison (all_x is
# their union; base is the pure-base contrast).  The _not_filtered sources are
# deliberately excluded from ΔI — that fact-level detail lives in the inspect
# parquets — but are retained for the top-k precision splits below.
_DELTA_SRCS = ("cooccur", "subj_occur", "obj_occur")
_PREC_CATS = (
    "cooccur",           # X: both subject and object co-occur (explicit filter)
    "subj_occur",        # X: subject-occurrence filter (subject above threshold)
    "subj_not_filtered", # base: subject appears in text but was NOT filtered
    "obj_occur",         # X: object-occurrence filter (object above threshold)
    "obj_not_filtered",  # base: object appears in text but was NOT filtered
    "all_x",             # derived: everything fact-relevant (filtered ∪ not_filtered)
)


def _posthoc_sources(text: str, subject: str, obj: str, *, is_fact_x: bool) -> set[str]:
    """Infer source tags from text via case-insensitive substring match.

    For fact-specific X entries: determines whether the sentence was removed
    because of the subject, the object, or both co-occurring.
    For non-X (or globally-X-but-not-this-fact) entries: finds subject/object
    mentions that survived because the entity count was below the filter threshold.
    """
    t = text.lower()
    s = subject.lower()
    o = obj.lower()
    has_s = bool(s) and s in t
    has_o = bool(o) and o in t
    if is_fact_x:
        if has_s and has_o:
            return {"cooccur"}
        if has_s:
            return {"subj_occur"}
        if has_o:
            return {"obj_occur"}
        return set()  # X via alias match — cannot classify without alias lookup
    # effective base (globally-base or X for another fact)
    srcs: set[str] = set()
    if has_s:
        srcs.add("subj_not_filtered")
    if has_o:
        srcs.add("obj_not_filtered")
    return srcs


def _expand_sources(raw: set[str]) -> set[str]:
    """Materialise derived aggregate categories from atomic source tags."""
    exp = set(raw)
    if exp & {"subj_occur", "obj_occur"}:
        exp.add("occur")
    if exp & {"cooccur", "subj_occur", "obj_occur"}:
        exp.add("all_filtered")
    if exp & {"subj_not_filtered", "obj_not_filtered"}:
        exp.add("not_filtered")
    if exp & {"all_filtered", "not_filtered"}:
        exp.add("all_x")
    return exp


def _empty_delta() -> dict:
    nan = float("nan")
    return {"n": 0, "mean": nan, "median": nan, "std": nan,
            "ci95_lower": nan, "ci95_upper": nan,
            "norm_mean": nan, "norm_median": nan}


def _delta_full(arr: np.ndarray, score_std: float | None) -> dict:
    """n/mean/median/std/CI + normalized mean/median for a raw ΔI array."""
    if arr.size == 0:
        return _empty_delta()
    d = _summarise_delta_I(
        arr, score_std=score_std if (score_std and score_std > 0) else None
    )
    d.setdefault("norm_mean", float("nan"))
    d.setdefault("norm_median", float("nan"))
    return d


def _ratio(num: float, den: float) -> float:
    if np.isnan(num) or np.isnan(den) or den == 0:
        return float("nan")
    return float(num / den)


def _pool_delta_npz(npz, fk_list: list[str]) -> dict:
    """Pool exact per-source ΔI stats over the facts in *fk_list* (a verdict split).

    Reads the raw per-source ΔI arrays written by attribute.py's
    ``_write_per_fact_src_delta_I`` (delta_I_per_source.npz), restricts to the
    requested facts, and returns a single delta_I dict: atomic sources, their
    union (all_x), base, and per-source/base mean & median ratios.  Uses ALL D_X
    items — no top-k selection bias.
    """
    fact_keys = [str(fk) for fk in npz["fact_keys"]]
    score_std = float(npz["score_std"])
    wanted = set(fk_list)
    want = np.array(
        [i for i, fk in enumerate(fact_keys) if fk in wanted], dtype=np.int64
    )

    def _sel(name: str) -> np.ndarray:
        vals = npz[f"{name}_vals"]
        if vals.size == 0 or want.size == 0:
            return np.empty(0, dtype=np.float32)
        return vals[np.isin(npz[f"{name}_fact_ids"], want)]

    src_arrs = {src: _sel(src) for src in _DELTA_SRCS}
    base_arr = _sel("base")
    all_x_arr = (
        np.concatenate(list(src_arrs.values()))
        if any(a.size for a in src_arrs.values())
        else np.empty(0, dtype=np.float32)
    )

    out: dict[str, object] = {"score_std": score_std}
    for src in _DELTA_SRCS:
        out[src] = _delta_full(src_arrs[src], score_std)
    out["all_x"] = _delta_full(all_x_arr, score_std)
    out["base"] = _delta_full(base_arr, score_std)

    base_mean: float = out["base"]["mean"]      # type: ignore[index]
    base_median: float = out["base"]["median"]  # type: ignore[index]
    for src in (*_DELTA_SRCS, "all_x"):
        s: dict = out[src]  # type: ignore[assignment]
        out[f"ratio_{src}_to_base"] = {
            "mean": _ratio(s["mean"], base_mean),
            "median": _ratio(s["median"], base_median),
        }
    return out


def _pool_prec(
    fact_side_flags: dict[str, dict[str, dict[str, list[bool]]]],
    fk_list: list[str],
    side: str,
    k_values: list[int],
) -> dict:
    """Micro-pool top-k precision per category over the facts in *fk_list*.

    prec@k[cat] = (Σ_facts #top-k items in cat) / (Σ_facts #top-k items).
    """
    out: dict[str, dict[str, float]] = {}
    for cat in _PREC_CATS:
        prec_k: dict[str, float] = {}
        for k in k_values:
            num = 0.0
            den = 0
            for fk in fk_list:
                flags = fact_side_flags.get(fk, {}).get(side, {}).get(cat)
                if not flags:
                    continue
                kk = min(k, len(flags))
                num += float(sum(flags[:kk]))
                den += kk
            prec_k[str(k)] = num / den if den > 0 else float("nan")
        out[cat] = prec_k
    return out


# ---------------------------------------------------------------------------
# Per-source BEAR attribution analysis
# ---------------------------------------------------------------------------


def run_bear_per_source(exp: ExperimentCfg, run_id: str | None = None) -> None:
    """Per-verdict-split BEAR facts attribution report (mirrors the blimp splits).

    Emits one JSON per method and verdict split, using the blimp file naming:
      results_<id>.json                     all facts
      results_<id>_learned.json             full model learned (Forgotten ∪ Retained)
      results_<id>_learned_specific.json    Forgotten — filtered model lost the fact
      results_<id>_learned_confounded.json  Retained — filtered model kept the fact
      results_<id>_not_learned.json         full model never learned the fact

    Verdicts are computed upstream in prepare/facts.py (a fact is
    learned when the full model ranks the correct object first on ALL templates;
    Forgotten/Retained split on whether the filtered model still knows it).

    Each file reports, per seed:
      delta_I — exact pooled ΔI over ALL D_X items of the split's facts (from the
        per-source .npz written by attribute.py, NOT the top-k parquets), split
        into cooccur / subj_occur / obj_occur / all_x / base with n/mean/median/
        std/CI/norm and per-source-to-base mean & median ratios.
      prec_at_k_{contrast,good,bad} — top-k precision from the inspect parquets,
        split into cooccur / subj_occur / subj_not_filtered / obj_occur /
        obj_not_filtered / all_x (base omitted — it is the complement of all_x).

    The per-fact detail is intentionally dropped here; it lives in the parquets.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    queries_path = exp.queries_path()
    if not queries_path.exists():
        return
    queries_ds = hf_datasets.load_from_disk(str(queries_path))
    if "fact_key" not in queries_ds.column_names:
        return

    # ── Training dataset (optional — authoritative source tags for X entries) ──
    idx_to_facts: dict[int, set[str]] | None = None
    idx_to_sources: dict[int, set[str]] | None = None
    train_path = exp.train_data_path()
    if train_path.exists():
        train_ds = hf_datasets.load_from_disk(str(train_path))
        if "fact_ids" in train_ds.column_names and "source" in train_ds.column_names:
            idx_to_facts = {}
            idx_to_sources = {}
            for i, (raw_facts, raw_src) in enumerate(
                zip(train_ds["fact_ids"], train_ds["source"])
            ):
                if raw_facts:
                    idx_to_facts[i] = set(raw_facts.split(","))
                if raw_src:
                    srcs = set(raw_src.split(","))
                    if "occur" in srcs and "subj_occur" not in srcs and "obj_occur" not in srcs:
                        srcs.discard("occur")
                        srcs.add("subj_occur")
                    idx_to_sources[i] = srcs
        else:
            logger.info(
                "run_bear_per_source: training dataset missing fact_ids/source columns "
                "— using text-based source inference for X entries."
            )

    # ── Queries → fact mapping + verdicts ──────────────────────────────────────
    fact_keys: list[str] = queries_ds["fact_key"]
    verdicts_src: list[str] = (
        queries_ds["verdict"] if "verdict" in queries_ds.column_names
        else ["unknown"] * len(fact_keys)
    )
    fact_to_pairs: dict[str, list[int]] = {}
    verdict_per_fact: dict[str, str] = {}
    for j, fk in enumerate(fact_keys):
        fact_to_pairs.setdefault(fk, []).append(j)
        verdict_per_fact.setdefault(fk, verdicts_src[j])

    has_neg_idx = "negative_idx" in queries_ds.column_names
    if has_neg_idx:
        neg_idx_arr = np.array(queries_ds["negative_idx"])
        good_only_pairs: dict[str, list[int]] = {}
        for j, fk in enumerate(fact_keys):
            if neg_idx_arr[j] == 0:
                good_only_pairs.setdefault(fk, []).append(j)
    else:
        good_only_pairs = fact_to_pairs

    unique_facts = sorted(fact_to_pairs.keys())
    k_values = exp.attribution.k_precisions
    method = exp.attribution.method
    n_top = exp.attribution.inspect_top_k

    # Verdict splits.  "" is the bare all-facts file; the rest mirror blimp naming.
    def _by_verdict(*keep: str) -> list[str]:
        return [fk for fk in unique_facts if verdict_per_fact.get(fk) in keep]

    splits: dict[str, list[str]] = {
        "":                   unique_facts,
        "learned":            _by_verdict("Forgotten", "Retained"),
        "learned_specific":   _by_verdict("Forgotten"),
        "learned_confounded": _by_verdict("Retained"),
        "not_learned":        _by_verdict("not_learned"),
    }

    per_split_seeds: dict[str, list[dict]] = {name: [] for name in splits}

    for seed in exp.seeds:
        inspect_paths = {
            "contrast": exp.top_inspect_path(seed),
            "good":     exp.top_inspect_good_path(seed),
            "bad":      exp.top_inspect_bad_path(seed),
        }
        missing = [k for k, p in inspect_paths.items() if not p.exists()]
        if missing:
            logger.warning(
                "[seed=%d] run_bear_per_source: missing top_inspect parquets (%s), skipping seed.",
                seed, missing,
            )
            continue
        dfs: dict[str, pd.DataFrame] = {
            side: pd.read_parquet(str(p)) for side, p in inspect_paths.items()
        }
        score_col_by_side = {"contrast": "delta_I", "good": "score_good", "bad": "score_bad"}
        sides = list(dfs.keys())

        # Exact pooled ΔI source — raw per-source arrays written by attribute.py.
        npz_path = exp.delta_I_per_source_path(seed).with_suffix(".npz")
        npz = np.load(str(npz_path)) if npz_path.exists() else None
        if npz is None:
            logger.warning(
                "[seed=%d] run_bear_per_source: %s not found — delta_I omitted "
                "(re-run 'attribute' to regenerate).",
                seed, npz_path.name,
            )

        # ── Per-fact ordered top-k category flags per side ──────────────────────
        fact_side_flags: dict[str, dict[str, dict[str, list[bool]]]] = {}
        fact_subjects: dict[str, str] = {}
        fact_objects: dict[str, str] = {}

        for fk in unique_facts:
            pair_indices = set(fact_to_pairs[fk])
            fact_side_flags[fk] = {}

            for side, df in dfs.items():
                active_pairs = (
                    good_only_pairs.get(fk, set()) if side == "good" else pair_indices
                )
                sub = df[df["pair_idx"].isin(active_pairs)]
                if sub.empty:
                    fact_side_flags[fk][side] = {c: [] for c in _PREC_CATS}
                    continue

                score_col = score_col_by_side[side]
                if fk not in fact_subjects and "subject" in sub.columns:
                    fact_subjects[fk] = str(sub["subject"].iloc[0])
                if fk not in fact_objects and "correct_object" in sub.columns:
                    fact_objects[fk] = str(sub["correct_object"].iloc[0])
                subject = fact_subjects.get(fk, "")
                obj = fact_objects.get(fk, "")

                # Effective source set per train_idx (fact-specific X classification).
                ti_info = (
                    sub.drop_duplicates("train_idx")
                    .set_index("train_idx")[["category", "text"]]
                )
                ti_eff_srcs: dict[int, set[str]] = {}
                for ti_val, row in ti_info.iterrows():
                    ti = int(ti_val)
                    cat = str(row.get("category", "base"))
                    text = str(row.get("text", ""))

                    if cat in ("cooccur", "subj_occur", "obj_occur"):
                        raw: set[str] = {cat}
                    elif cat == "X-not-filtered":
                        raw = {"subj_not_filtered", "obj_not_filtered"}
                    elif cat == "subj_not_filtered":
                        raw = {"subj_not_filtered"}
                    elif cat == "obj_not_filtered":
                        raw = {"obj_not_filtered"}
                    elif cat == "base":
                        raw = set()
                    else:
                        if idx_to_facts is not None:
                            is_fact_x = fk in idx_to_facts.get(ti, set())
                        else:
                            is_fact_x = bool(
                                _posthoc_sources(text, subject, obj, is_fact_x=True)
                            )
                        if is_fact_x:
                            if idx_to_sources is not None:
                                raw = set(idx_to_sources.get(ti, set()))
                                if not raw:
                                    raw = _posthoc_sources(text, subject, obj, is_fact_x=True)
                            else:
                                raw = _posthoc_sources(text, subject, obj, is_fact_x=True)
                        else:
                            raw = _posthoc_sources(text, subject, obj, is_fact_x=False)

                    exp_srcs = _expand_sources(raw)
                    if not exp_srcs:
                        exp_srcs = {"base"}
                    ti_eff_srcs[ti] = exp_srcs

                mean_scores = sub.groupby("train_idx")[score_col].mean()
                top_tis = (
                    mean_scores.nsmallest(n_top).index.tolist()
                    if side == "bad"
                    else mean_scores.nlargest(n_top).index.tolist()
                )

                cat_flags: dict[str, list[bool]] = {c: [] for c in _PREC_CATS}
                for ti in top_tis:
                    srcs = ti_eff_srcs.get(ti, {"base"})
                    for c in _PREC_CATS:
                        cat_flags[c].append(c in srcs)
                fact_side_flags[fk][side] = cat_flags

        # ── Assemble per split for this seed ────────────────────────────────────
        for name, fk_list in splits.items():
            entry: dict[str, object] = {"seed": int(seed), "n_facts": len(fk_list)}
            if npz is not None:
                entry["delta_I"] = _pool_delta_npz(npz, fk_list)
            for side in sides:
                entry[f"prec_at_k_{side}"] = _pool_prec(
                    fact_side_flags, fk_list, side, k_values
                )
            per_split_seeds[name].append(entry)

        logger.info(
            "[seed=%d] run_bear_per_source: %d facts; splits=%s",
            seed, len(unique_facts), {n: len(f) for n, f in splits.items()},
        )

    if not any(per_split_seeds.values()):
        logger.warning("run_bear_per_source: no seeds produced results, skipping save.")
        return

    out_dir = exp.results_dir() / "step2" / method
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, seeds_list in per_split_seeds.items():
        suffix = f"_{name}" if name else ""
        out_path = out_dir / f"results_{run_id}{suffix}.json"
        with open(out_path, "w") as fh:
            json.dump(
                {
                    "run_id": run_id,
                    "method": method,
                    "condition": exp.data.condition,
                    "split": name or "all",
                    "n_facts": len(splits[name]),
                    "k_precisions": k_values,
                    "seeds": seeds_list,
                },
                fh,
                indent=2,
            )
        logger.info(
            "run_bear_per_source: saved %s (%d facts) → %s",
            name or "all", len(splits[name]), out_path,
        )
