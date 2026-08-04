"""Step 2 attribution: directional influence scores via bergson.

Supported methods (``attribution.method`` in config):

  vanilla    — projected gradient cosine similarity.  Optionally normalised
               by Adam's exp_avg_sq when ``use_optimizer_state=True``.
               Per-pair ΔI(z, j) = I(s⁺_j, z) − I(s⁻_j, z).

  trackstar  — TrackStar (Chang et al., 2024): same projection as vanilla but
               with a curvature-aware mixed preconditioner built from both the
               training-data and query-data gradient distributions.
               Per-pair ΔI.

  ekfac      — EK-FAC preconditioned influence (full parameter space, no
               random projection).  The Hessian is fitted once on D_train.
               Per-pair ΔI by default (ekfac_aggregate_query=False): builds a
               2m-entry query index, applies H⁻¹ per sentence, scores once →
               (N, 2m) memmap, same granularity as vanilla/trackstar.
               Global ΔI only when ekfac_aggregate_query=True (cheaper: two
               1-entry mean-gradient indices instead of 2m full gradients).

Unit normalization (``attribution.unit_norm``) is threaded through every
method so the resulting ΔI values are on comparable scales:
  - vanilla, trackstar, ekfac (sketched & non-sketched) pass unit_normalize
    to bergson's PreprocessConfig at query-build and scoring time.
  - bm25  L2-normalizes each query column post-hoc before ΔI.

Output files per seed (under ``attribution_method_dir(seed)``):
    delta_I_stats.npz      — ΔI arrays (D_X + D_base sample) + Welford score state
    top_inspect.parquet    — top-_INSPECT_TOP_K examples (D_X or D_base) with text, ranked over the same pool as Prec@k
    prec_at_k.json         — pre-computed Prec@k results

Family A (vanilla, trackstar, ekfac w/ ekfac_sketch=True) additionally write:
    query_index/           — bergson gradient index (2m per-sentence)
    scores_raw/            — (N, 2m) bergson score memmap
  ekfac (sketched only) also writes:
    hessian/               — Kronecker-factor Hessian approximation

Family B (ekfac w/ ekfac_sketch=False) writes:
    query_{plus,minus}_agg/   — aggregated (mean) query gradient (aggregate mode)
    hessian/                  — Kronecker-factor Hessian approximation
    query_{plus,minus}_ivhp/  — H⁻¹-transformed query gradients
    scores_{plus,minus}/      — (N, 1) bergson score memmaps

Family B (ekfac, aggregate query) writes:
"""

from __future__ import annotations

import gc
import json
import logging
import re
import shutil
from copy import deepcopy
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from bergson.build import build
from bergson.config import DataConfig, IndexConfig, PreprocessConfig, ScoreConfig
from bergson.score.score import score_dataset, score_dataset_streaming
from bergson.score.score_writer import ScoreWriter

from .config import LEARNED_THRESHOLD, ExperimentCfg

logger = logging.getLogger(__name__)

_STATS_SUBSAMPLE_SEED = 0  # D_base sample for Wilcoxon / mean stats
_PREC_SUBSAMPLE_SEED = 1  # D_base sample for Prec@k (subset mode)
_SCORING_SUBSET_SEED = 2  # D_base sample for the scoring-pass subset


# ═══════════════════════════════════════════════════════════════════════════
# Post-hoc unit-normalization (for methods that don't use bergson's
# PreprocessConfig.unit_normalize — currently bm25)
# ═══════════════════════════════════════════════════════════════════════════


def _l2_normalize(x: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """Scale *x* so that the L2 norm along *axis* is 1 (zero-vectors left as-is)."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.where(norm < eps, 1.0, norm)
    return (x / norm).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Sidecar lookups for qualitative inspection
# ═══════════════════════════════════════════════════════════════════════════
#
# train_lookup.parquet : (train_idx, category, text)        — ~|D_train| rows
# query_lookup.parquet : (pair_idx, feature, text_good, text_bad) — ~m rows
#
# top_inspect.parquet (per seed, in attribution_method_dir) contains the
# top-_INSPECT_TOP_K training examples by mean ΔI, with text and category
# (X / base) inlined — ranked over the same candidate pool as Prec@k, so a
# base example shows up whenever it actually outranks D_X.

_INSPECT_TOP_K = 50

_TRAIN_LOOKUP_SCHEMA = pa.schema(
    [
        ("train_idx", pa.int32()),
        ("category", pa.dictionary(pa.int8(), pa.string())),
        ("text", pa.string()),
    ]
)

_CAT_DICT_VALUES = pa.array(["X", "base"], type=pa.string())


def _write_lookups(
    exp, train_ds: datasets.Dataset, queries_ds: datasets.Dataset
) -> None:
    """Write small sidecar lookups for manual top-k inspection. Idempotent.

    Train lookup is written in row-chunks via pa.ParquetWriter so peak RAM is
    bounded by the chunk size, not the full corpus (matters at 500k+ rows).
    Query lookup is tiny (~1k rows); no streaming needed.
    """
    train_lp = exp.shared_dir() / "train_lookup.parquet"
    query_lp = exp.shared_dir() / "query_lookup.parquet"

    if not train_lp.exists():
        train_lp.parent.mkdir(parents=True, exist_ok=True)
        n = len(train_ds)
        chunk = 10_000
        train_arrow = train_ds.with_format("arrow")
        with pq.ParquetWriter(
            str(train_lp), _TRAIN_LOOKUP_SCHEMA, compression="zstd"
        ) as w:
            for i0 in range(0, n, chunk):
                i1 = min(i0 + chunk, n)
                sub: pa.Table = train_arrow[
                    i0:i1
                ]  # Arrow-native slice, no Python materialisation
                cat_col = sub.column("category")
                codes_np = np.where(
                    cat_col.to_numpy(zero_copy_only=False) == "X", 0, 1
                ).astype(np.int8)
                tbl = pa.Table.from_arrays(
                    [
                        pa.array(np.arange(i0, i1, dtype=np.int32)),
                        pa.DictionaryArray.from_arrays(
                            pa.array(codes_np), _CAT_DICT_VALUES
                        ),
                        sub.column("text"),
                    ],
                    schema=_TRAIN_LOOKUP_SCHEMA,
                )
                w.write_table(tbl)
        logger.info("Wrote train lookup → %s (%d rows, chunked)", train_lp, n)

    if not query_lp.exists():
        query_lp.parent.mkdir(parents=True, exist_ok=True)
        cols = queries_ds.column_names
        feats = (
            [str(f) for f in queries_ds["feature"]]
            if "feature" in cols
            else [""] * len(queries_ds)
        )
        pd.DataFrame(
            {
                "pair_idx": np.arange(len(queries_ds), dtype=np.int32),
                "feature": feats,
                "text_good": [str(t) for t in queries_ds["text_good"]],
                "text_bad": [str(t) for t in queries_ds["text_bad"]],
            }
        ).to_parquet(str(query_lp), index=False, compression="zstd")
        logger.info("Wrote query lookup → %s (%d rows)", query_lp, len(queries_ds))


def _write_delta_I_stats(
    path: Path,
    dx: np.ndarray,
    base: np.ndarray,
    score_n: int,
    score_mean: float,
    score_M2: float,
    n_pairs: int,
) -> None:
    """Save ΔI arrays + Welford score state to a compressed npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        dx=dx.astype(np.float32),
        base=base.astype(np.float32),
        score_n=np.int64(score_n),
        score_mean=np.float64(score_mean),
        score_M2=np.float64(score_M2),
        n_pairs=np.int64(n_pairs),
    )


def _write_top_inspect(
    path: Path,
    pool_indices: np.ndarray | None,
    delta_I_all: np.ndarray,
    cats_arr: np.ndarray,
    train_ds: datasets.Dataset,
    top_k: int = _INSPECT_TOP_K,
) -> None:
    """Write top-k training examples (D_X or D_base) by mean ΔI, ranked over
    the same candidate pool as Prec@k, with text and category inlined."""
    pool = pool_indices if pool_indices is not None else np.arange(len(delta_I_all))
    pool_dI = delta_I_all[pool]
    n = min(top_k, len(pool))
    top_local = np.argpartition(pool_dI, -n)[-n:]
    top_local = top_local[np.argsort(-pool_dI[top_local])]
    top_global = pool[top_local]
    texts = [str(train_ds[int(i)]["text"]) for i in top_global]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "train_idx": top_global.astype(np.int32),
            "category": cats_arr[top_global],
            "delta_I": pool_dI[top_local].astype(np.float32),
            "text": texts,
        }
    ).to_parquet(str(path), index=False, compression="zstd")


def _write_per_pair_top_inspect(
    path: Path,
    mmap: np.memmap,
    pool_indices: np.ndarray | None,
    cats_arr: np.ndarray,
    num_pairs: int,
    train_ds: datasets.Dataset,
    queries_ds: datasets.Dataset,
    side: str = "contrast",
    top_k: int = _INSPECT_TOP_K,
    labels: "PairLabels | None" = None,
    fact_lk: "_FactLookups | None" = None,
) -> None:
    """Write top-k examples (D_X or D_base) per query pair, ranked over the
    same candidate pool as Prec@k.

    side='contrast' — top-k by ΔI = I(s⁺) − I(s⁻), descending.
    side='good'     — top-k by I(s⁺), descending.
    side='bad'      — top-k by I(s⁻), ascending (most negative first).

    Output columns: pair_idx, feature, rank, train_idx, category, <score_col>,
    text, text_good, text_bad [, log_prob_good_full, log_prob_bad_full,
    log_prob_delta_full, softmax_prob_full, full_model_learned_X,
    log_prob_good_no_x, log_prob_bad_no_x, log_prob_delta_no_x,
    softmax_prob_no_x, no_model_learned_X] — the bracketed columns are
    included whenever *labels* is provided (the no-X columns only when a no-X
    model's labels were also available). *_learned_X is softmax_prob_good >
    LEARNED_THRESHOLD (config.py).
    """
    cols = queries_ds.column_names
    features = queries_ds["feature"] if "feature" in cols else [""] * num_pairs

    score_col = {"contrast": "delta_I", "good": "score_good", "bad": "score_bad"}[side]

    pool_sub = mmap[pool_indices] if pool_indices is not None else mmap

    # For good-only, deduplicate: text_good is identical across all negatives for the
    # same (fact, template) group.  Only emit one block per unique text_good so the
    # parquet isn't bloated with identical score columns.
    cols = queries_ds.column_names
    if side == "good" and "negative_idx" in cols:
        neg_idx_arr = np.array(queries_ds["negative_idx"])
        pairs_iter = [j for j in range(num_pairs) if neg_idx_arr[j] == 0]
    else:
        pairs_iter = list(range(num_pairs))

    pair_idxs, train_idxs, ranks_out, feats_out, scores_out = [], [], [], [], []

    for j in pairs_iter:
        sg = np.asarray(pool_sub[f"score_{2 * j}"], dtype=np.float32)
        sb = np.asarray(pool_sub[f"score_{2 * j + 1}"], dtype=np.float32)

        if side == "contrast":
            scores = sg - sb
        elif side == "good":
            scores = sg
        else:
            scores = sb

        n = min(top_k, len(scores))
        if side == "bad":
            local = np.argpartition(scores, n)[:n]
            local = local[np.argsort(scores[local])]
        else:
            local = np.argpartition(scores, -n)[-n:]
            local = local[np.argsort(-scores[local])]

        feat = str(features[j])
        for rank, li in enumerate(local):
            global_idx = int(pool_indices[li]) if pool_indices is not None else int(li)
            pair_idxs.append(j)
            train_idxs.append(global_idx)
            ranks_out.append(rank)
            feats_out.append(feat)
            scores_out.append(float(scores[li]))

    texts = [str(train_ds[ti]["text"]) for ti in train_idxs]
    train_idx_arr = np.array(train_idxs, dtype=np.int32)
    pair_idx_arr = np.array(pair_idxs, dtype=np.int32)
    if fact_lk is not None:
        cats_final = [
            _fact_category(ti, pi, txt, fact_lk)
            for ti, pi, txt in zip(train_idxs, pair_idxs, texts)
        ]
    else:
        cats_final = cats_arr[train_idx_arr].tolist()
    columns: dict[str, object] = {
        "pair_idx": pair_idx_arr,
        "feature": feats_out,
        "rank": np.array(ranks_out, dtype=np.int16),
        "train_idx": train_idx_arr,
        "category": cats_final,
        score_col: np.array(scores_out, dtype=np.float32),
        "text": texts,
    }
    _inject_query_label_columns(columns, pair_idx_arr, queries_ds, labels, side=side)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_parquet(str(path), index=False, compression="zstd")


# ═══════════════════════════════════════════════════════════════════════════
# Fact-specific parquet category helpers
# ═══════════════════════════════════════════════════════════════════════════


class _FactLookups:
    """Precomputed per-row lookups for fact-specific parquet category labels."""

    __slots__ = (
        "idx_to_facts",
        "idx_to_sources",
        "pair_to_fact",
        "pair_to_subject",
        "pair_to_object",
    )

    def __init__(
        self,
        idx_to_facts: dict,
        idx_to_sources: dict,
        pair_to_fact: list,
        pair_to_subject: list,
        pair_to_object: list,
    ) -> None:
        self.idx_to_facts = idx_to_facts
        self.idx_to_sources = idx_to_sources
        self.pair_to_fact = pair_to_fact
        self.pair_to_subject = pair_to_subject
        self.pair_to_object = pair_to_object


def _build_fact_lookups(
    train_ds: datasets.Dataset, queries_ds: datasets.Dataset
) -> "_FactLookups | None":
    """Build fact-specific category lookups. Returns None if required columns absent."""
    tcols = train_ds.column_names
    qcols = queries_ds.column_names
    if "fact_ids" not in tcols or "fact_key" not in qcols:
        return None

    idx_to_facts: dict[int, frozenset] = {}
    idx_to_sources: dict[int, frozenset] = {}
    for i, raw in enumerate(train_ds["fact_ids"]):
        if raw:
            idx_to_facts[i] = frozenset(raw.split(","))
    if "source" in tcols:
        for i, raw in enumerate(train_ds["source"]):
            if raw:
                srcs = set(raw.split(","))
                if (
                    "occur" in srcs
                    and "subj_occur" not in srcs
                    and "obj_occur" not in srcs
                ):
                    srcs.discard("occur")
                    srcs.add("subj_occur")
                idx_to_sources[i] = frozenset(srcs)

    return _FactLookups(
        idx_to_facts=idx_to_facts,
        idx_to_sources=idx_to_sources,
        pair_to_fact=queries_ds["fact_key"],
        pair_to_subject=queries_ds["subject"] if "subject" in qcols else [],
        pair_to_object=queries_ds["correct_object"]
        if "correct_object" in qcols
        else [],
    )


def _fact_category(ti: int, pi: int, text: str, lk: _FactLookups) -> str:
    """Fact-specific category label for one (train_idx, pair_idx) parquet row."""
    fk = lk.pair_to_fact[pi] if pi < len(lk.pair_to_fact) else ""
    is_fact_x = bool(fk) and fk in lk.idx_to_facts.get(ti, frozenset())
    if is_fact_x:
        srcs = lk.idx_to_sources.get(ti, frozenset())
        if "cooccur" in srcs:
            return "cooccur"
        if "subj_occur" in srcs:
            return "subj_occur"
        if "obj_occur" in srcs:
            return "obj_occur"
        return "X"
    t = text.lower()
    subj = str(lk.pair_to_subject[pi]).lower() if pi < len(lk.pair_to_subject) else ""
    obj = str(lk.pair_to_object[pi]).lower() if pi < len(lk.pair_to_object) else ""
    # Word-boundary match, not a raw substring check — otherwise a short/
    # ambiguous name like "Sint" (Dutch for "Saint") falsely matches inside
    # unrelated words like "Sintang" (an Indonesian regency).
    has_s = bool(subj) and re.search(r"(?<!\w)" + re.escape(subj) + r"(?!\w)", t) is not None
    has_o = bool(obj) and re.search(r"(?<!\w)" + re.escape(obj) + r"(?!\w)", t) is not None
    if has_s and has_o:
        return "X-not-filtered"
    if has_s:
        return "subj_not_filtered"
    if has_o:
        return "obj_not_filtered"
    return "base"


# ═══════════════════════════════════════════════════════════════════════════
# Config fingerprinting — stale-artifact detection
# ═══════════════════════════════════════════════════════════════════════════

_FP_FILENAME = "_config.json"


def _up_to_date(directory: Path, cfg: dict) -> bool:
    """True iff *directory* exists and its _config.json matches *cfg*."""
    fp = directory / _FP_FILENAME
    if not directory.exists() or not fp.exists():
        return False
    try:
        with open(fp) as f:
            return json.load(f) == cfg
    except Exception:
        return False


def _mark(directory: Path, cfg: dict) -> None:
    """Stamp an artifact directory with its config fingerprint."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / _FP_FILENAME, "w") as f:
        json.dump(cfg, f, sort_keys=True, indent=2)


def _up_to_date_file(path: Path, cfg: dict) -> bool:
    """True iff *path* exists and its sidecar .meta.json matches *cfg*."""
    sidecar = path.with_suffix(".meta.json")
    if not path.exists() or not sidecar.exists():
        return False
    try:
        with open(sidecar) as f:
            return json.load(f) == cfg
    except Exception:
        return False


def _mark_file(path: Path, cfg: dict) -> None:
    """Stamp a file artifact with a sidecar .meta.json fingerprint."""
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump(cfg, f, sort_keys=True, indent=2)


def _maybe_delete_stale(directory: Path, cfg: dict, label: str) -> None:
    """If *directory* exists but its fingerprint doesn't match *cfg*, delete it."""
    if directory.exists() and not _up_to_date(directory, cfg):
        logger.warning(
            "Config changed for %s — removing stale artifact: %s", label, directory
        )
        shutil.rmtree(directory)


def _method_cfg(exp: ExperimentCfg) -> dict:
    """Config dict covering all params that affect the *final* scores for the
    current method. Used as the post-processing fingerprint and for non-ekfac
    methods' per-stage fingerprints. EK-FAC has per-stage fingerprints via the
    _ekfac_*_cfg helpers below — this dict matches the deepest (scoring) stage
    so that post-processing invalidates correctly on any ekfac-param change."""
    a = exp.attribution
    cfg: dict = {"method": a.method, "scoring_base_size": a.scoring_base_size}
    if a.method == "bm25":
        pass  # no tunable hyper-parameters beyond the training corpus
    elif a.method in ("vanilla", "trackstar"):
        cfg.update(
            {
                "projection_dim": a.projection_dim,
                "unit_norm": a.unit_norm,
                "use_optimizer_state": a.use_optimizer_state,
            }
        )
        if a.method == "trackstar":
            cfg["trackstar_downweight_components"] = a.trackstar_downweight_components
    elif a.method == "ekfac":
        cfg.update(
            {
                "ekfac_method": a.ekfac_method,
                "ekfac_lambda_damp": a.ekfac_lambda_damp,
                "ekfac_aggregate_query": a.ekfac_aggregate_query,
                "ekfac_filter_modules": a.ekfac_filter_modules,
                "ekfac_query_low_rank": a.ekfac_query_low_rank,
                "ekfac_sketch": a.ekfac_sketch,
                "unit_norm": a.unit_norm,
            }
        )
        if a.ekfac_sketch:
            cfg["projection_dim"] = a.projection_dim
    return cfg


def _ekfac_query_grad_cfg(exp: ExperimentCfg) -> dict:
    """Fingerprint for the EK-FAC query gradient index (Step 1)."""
    a = exp.attribution
    return {
        "method": a.method,
        "ekfac_filter_modules": a.ekfac_filter_modules,
        "ekfac_aggregate_query": a.ekfac_aggregate_query,
        "unit_norm": a.unit_norm,
    }


def _ekfac_hessian_cfg(exp: ExperimentCfg) -> dict:
    """Fingerprint for the EK-FAC Hessian factors fit on D_train.

    ekfac_method controls ev_correction, which changes what bergson writes to
    hessian_dir (eigval_{activation,gradient}_sharded for plain kfac vs.
    eigenvalue_correction_sharded for ekfac) — it must be included so the
    kfac and ekfac variants don't share a stale cache."""
    a = exp.attribution
    return {
        "method": a.method,
        "scoring_base_size": a.scoring_base_size,
        "ekfac_filter_modules": a.ekfac_filter_modules,
        "ekfac_method": a.ekfac_method,
    }


def _ekfac_ivhp_cfg(exp: ExperimentCfg) -> dict:
    """Fingerprint for the H⁻¹-transformed query gradient (Step 3)."""
    a = exp.attribution
    return {
        **_ekfac_hessian_cfg(exp),
        "ekfac_aggregate_query": a.ekfac_aggregate_query,
        "ekfac_lambda_damp": a.ekfac_lambda_damp,
        "unit_norm": a.unit_norm,
    }


def _ekfac_scores_cfg(exp: ExperimentCfg) -> dict:
    """Fingerprint for EK-FAC scoring of D_train vs H⁻¹ query (Step 4)."""
    return {
        **_ekfac_ivhp_cfg(exp),
        "ekfac_query_low_rank": exp.attribution.ekfac_query_low_rank,
    }


def _ekfac_sketch_query_cfg(exp: ExperimentCfg) -> dict:
    """Fingerprint for the sketched EK-FAC query gradient index."""
    a = exp.attribution
    return {
        "method": a.method,
        "ekfac_sketch": True,
        "ekfac_modified_projections": a.ekfac_modified_projections,
        "ekfac_filter_modules": a.ekfac_filter_modules,
        "ekfac_lambda_damp": a.ekfac_lambda_damp,
        "ekfac_method": a.ekfac_method,
        "projection_dim": a.projection_dim,
        "unit_norm": a.unit_norm,
    }


def _ekfac_sketch_scores_cfg(exp: ExperimentCfg) -> dict:
    """Fingerprint for sketched EK-FAC scoring."""
    return {
        **_ekfac_sketch_query_cfg(exp),
        "scoring_base_size": exp.attribution.scoring_base_size,
    }


def _postproc_cfg(exp: ExperimentCfg, seed: int) -> dict:
    """Config dict for post-processing artifacts (delta_I_stats, prec_at_k).

    Always includes unit_norm so that toggling it invalidates the post-processed
    outputs for bm25 (where unit_norm is applied as a post-hoc score
    normalization, not via an upstream bergson PreprocessConfig).

    Includes the mtimes of the full/no-X probe_models.py --pairs-output
    parquets (attribution.full_pairs_parquet / .no_x_pairs_parquet), if
    configured, so that (a) outputs computed before the learned/not-learned
    split existed are invalidated once (they're missing the per-pair label
    columns and the split-variant files), and (b) re-running check_pretrained
    with fresh labels correctly invalidates the split."""
    a = exp.attribution

    def _mtime(path: str | None) -> float | None:
        if not path:
            return None
        p = Path(path)
        return p.stat().st_mtime if p.exists() else None

    return {
        **_method_cfg(exp),
        "stats_base_size": a.stats_base_size,
        "prec_base_size": a.prec_base_size,
        "k_precisions": a.k_precisions,
        "unit_norm": a.unit_norm,
        "full_pairs_parquet_mtime": _mtime(a.full_pairs_parquet),
        "no_x_pairs_parquet_mtime": _mtime(a.no_x_pairs_parquet),
    }


def _train_subset_cfg(exp: ExperimentCfg) -> dict:
    """Config dict for the attribution training subset."""
    return {
        "condition": exp.data.condition,
        "scoring_base_size": exp.attribution.scoring_base_size,
        "sentence_level": exp.attribution.sentence_level,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Scoring-pass training dataset (full corpus or subsampled)
# ═══════════════════════════════════════════════════════════════════════════


def _ensure_train_data_path(exp: ExperimentCfg) -> Path:
    """Return the configured training dataset path, preparing it if missing."""
    train_path = exp.train_data_path()
    if train_path.exists():
        return train_path

    logger.warning(
        "Training dataset not found at %s; running prepare-data for this config.",
        train_path,
    )
    from . import data as data_mod

    data_mod.prepare_and_save(exp)
    if not train_path.exists():
        raise FileNotFoundError(
            "Training dataset is still missing after prepare-data: "
            f"{train_path}. Check data.data_dir={exp.data.data_dir!s} and rerun "
            "`python -m influence_on_what prepare-data` with the same config/overrides."
        )
    return train_path


def _build_sentence_level_data(exp: ExperimentCfg) -> Path:
    """Build a sentence-level HuggingFace dataset from the raw training .txt files.

    Each line in data_training/{base,X}/*.txt becomes one dataset row.  The
    resulting dataset has the same ``text`` / ``category`` schema as the packed
    training dataset, but each entry is a single sentence instead of a 512-token
    chunk — so the gradient index built from it has one entry per sentence.

    scoring_base_size caps the number of D_base sentences included (always
    includes all D_X sentences); 0 = include all sentences.
    """
    out = exp.attribution_sentence_data_path()
    cfg = _train_subset_cfg(exp)
    if _up_to_date(out, cfg):
        logger.info("Sentence-level attribution data already at %s, skipping.", out)
        return out
    _maybe_delete_stale(out, cfg, "sentence-level attribution data")

    data_dir = Path(exp.data.data_dir)
    texts: list[str] = []
    categories: list[str] = []

    for cat, subdir in [("X", "X"), ("base", "base")]:
        cat_dir = data_dir / "data_training" / subdir
        if not cat_dir.exists():
            logger.warning("Directory not found, skipping: %s", cat_dir)
            continue
        for txt_file in sorted(cat_dir.glob("*.txt")):
            for line in txt_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    texts.append(line)
                    categories.append(cat)

    full_ds = datasets.Dataset.from_dict({"text": texts, "category": categories})

    if exp.attribution.scoring_base_size > 0:
        cats_np = np.array(full_ds["category"])
        dx_idx = np.where(cats_np == "X")[0]
        base_idx = np.where(cats_np == "base")[0]
        rng = np.random.default_rng(_SCORING_SUBSET_SEED)
        n_base = min(exp.attribution.scoring_base_size, len(base_idx))
        sampled_base = np.sort(rng.choice(base_idx, n_base, replace=False))
        all_idx = np.sort(np.concatenate([dx_idx, sampled_base]))
        full_ds = full_ds.select(all_idx.tolist())

    out.parent.mkdir(parents=True, exist_ok=True)
    full_ds.save_to_disk(str(out))
    _mark(out, cfg)
    logger.info(
        "Saved sentence-level attribution data: %d sentences → %s", len(full_ds), out
    )
    return out


def _get_scoring_train_path(exp: ExperimentCfg) -> Path:
    """Return the training-data path to use for all scoring/Hessian calls.

    When ``sentence_level=True`` returns a sentence-level dataset built from the
    raw .txt files (one entry per sentence).  scoring_base_size still applies as
    a cap on D_base sentence count.

    When ``sentence_level=False`` and ``scoring_base_size == 0`` this is just
    ``train_data_path()`` (full packed corpus, current behaviour).

    When ``sentence_level=False`` and ``scoring_base_size > 0`` a subsampled
    dataset (D_X + a deterministic D_base sample) is built once in ``shared_dir``
    and reused across all seeds.
    """
    if exp.attribution.sentence_level:
        return _build_sentence_level_data(exp)

    if exp.attribution.scoring_base_size == 0:
        return _ensure_train_data_path(exp)

    out = exp.attribution_train_subset_path()
    cfg = _train_subset_cfg(exp)
    if _up_to_date(out, cfg):
        logger.info(
            "Attribution train subset already exists at %s, skipping build.", out
        )
        return out
    _maybe_delete_stale(out, cfg, "attribution train subset")

    logger.info(
        "Building attribution train subset (scoring_base_size=%d) → %s",
        exp.attribution.scoring_base_size,
        out,
    )
    full_ds = datasets.load_from_disk(str(_ensure_train_data_path(exp)))
    cats = np.array(full_ds["category"])
    dx_indices = np.where(cats == "X")[0]
    base_indices = np.where(cats == "base")[0]

    rng = np.random.default_rng(_SCORING_SUBSET_SEED)
    n_base = min(exp.attribution.scoring_base_size, len(base_indices))
    sampled_base = np.sort(rng.choice(base_indices, n_base, replace=False))
    all_indices = np.sort(np.concatenate([dx_indices, sampled_base]))

    subset_ds = full_ds.select(all_indices.tolist())
    out.parent.mkdir(parents=True, exist_ok=True)
    subset_ds.save_to_disk(str(out))
    _mark(out, cfg)
    logger.info(
        "Saved attribution train subset: %d D_X + %d D_base = %d total → %s",
        len(dx_indices),
        n_base,
        len(subset_ds),
        out,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Shared query-text preparation
# ═══════════════════════════════════════════════════════════════════════════


def _prepare_query_texts(exp: ExperimentCfg) -> Path:
    """Interleaved s⁺/s⁻ dataset (index 2j → s⁺_j, 2j+1 → s⁻_j).
    Shared across seeds; used by vanilla, trackstar, and ekfac (sketched).
    """
    out = exp.query_texts_path()
    if out.exists():
        logger.info("Query texts already prepared at %s, skipping.", out)
        return out

    queries = datasets.load_from_disk(str(exp.queries_path()))
    texts, pair_idxs, is_good_flags, features = [], [], [], []
    for j, ex in enumerate(queries):
        for text, flag in [(ex["text_good"], True), (ex["text_bad"], False)]:
            texts.append(str(text))
            pair_idxs.append(j)
            is_good_flags.append(flag)
            features.append(str(ex.get("feature", "")))

    ds = datasets.Dataset.from_dict(
        {
            "text": texts,
            "pair_idx": pair_idxs,
            "is_good": is_good_flags,
            "feature": features,
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    logger.info("Saved %d query sentences (%d pairs) → %s", len(ds), len(queries), out)
    return out


def _prepare_split_query_texts(exp: ExperimentCfg) -> tuple[Path, Path]:
    """Separate s⁺ and s⁻ datasets; ekfac only."""
    plus_path = exp.query_plus_texts_path()
    minus_path = exp.query_minus_texts_path()

    if plus_path.exists() and minus_path.exists():
        return plus_path, minus_path

    queries = datasets.load_from_disk(str(exp.queries_path()))
    for path, key in [(plus_path, "text_good"), (minus_path, "text_bad")]:
        ds = datasets.Dataset.from_dict(
            {
                "text": [str(ex[key]) for ex in queries],
                "pair_idx": list(range(len(queries))),
                "feature": [str(ex.get("feature", "")) for ex in queries],
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(path))

    logger.info(
        "Saved s⁺ and s⁻ query datasets (%d pairs each) → %s / %s",
        len(queries),
        plus_path,
        minus_path,
    )
    return plus_path, minus_path


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# Streaming accumulator — avoids writing a full (N, 2m) scores file to disk
# ═══════════════════════════════════════════════════════════════════════════


class StreamingAccumulatorWriter(ScoreWriter):
    """Accumulates per-pair top-k scores and full D_X/base-sample scores.

    Replaces MemmapSequenceScoreWriter for Family A methods (vanilla,
    trackstar, ekfac-sketch). Instead of an N×2m file on disk, keeps:
      - dx_scores   : (n_dx, 2*num_pairs) float32 — all D_X scores
      - base_scores : (n_base_sub, 2*num_pairs) float32 — base sample scores
      - _topk_*     : (max_k, num_pairs) per-metric top-k candidates

    Peak extra RAM ≈ n_dx × 2m × 4 B  (e.g. ~5 GB for NPI with 7 k pairs).
    """

    def __init__(
        self,
        num_train: int,
        num_queries: int,
        dx_indices: np.ndarray,
        base_sub_indices: np.ndarray,
        is_dx_full: np.ndarray,
        max_k: int,
    ):
        self.num_queries = num_queries
        self.num_pairs = num_queries // 2
        self.max_k = min(max_k, num_train)

        n_dx = len(dx_indices)
        n_base = len(base_sub_indices)

        # Fast O(1) lookup: training index → local row in dx_scores / base_scores
        self._dx_local = np.full(num_train, -1, dtype=np.int32)
        self._dx_local[dx_indices] = np.arange(n_dx, dtype=np.int32)
        self._base_local = np.full(num_train, -1, dtype=np.int32)
        self._base_local[base_sub_indices] = np.arange(n_base, dtype=np.int32)
        self._is_dx_full = is_dx_full.astype(np.int8)

        self.dx_scores = np.empty((n_dx, num_queries), dtype=np.float32)
        self.base_scores = np.empty((n_base, num_queries), dtype=np.float32)

        NEG_INF = np.finfo(np.float32).min
        mk, np_ = self.max_k, self.num_pairs
        self._topk_c = np.full((mk, np_), NEG_INF, dtype=np.float32)
        self._topk_g = np.full((mk, np_), NEG_INF, dtype=np.float32)
        self._topk_b = np.full((mk, np_), NEG_INF, dtype=np.float32)
        self._topk_isdx_c = np.zeros((mk, np_), dtype=np.int8)
        self._topk_isdx_g = np.zeros((mk, np_), dtype=np.int8)
        self._topk_isdx_b = np.zeros((mk, np_), dtype=np.int8)
        self._topk_idx_c = np.full((mk, np_), -1, dtype=np.int32)
        self._topk_idx_g = np.full((mk, np_), -1, dtype=np.int32)
        self._topk_idx_b = np.full((mk, np_), -1, dtype=np.int32)

    def __call__(self, indices: list[int], scores: torch.Tensor):
        batch = scores.to(dtype=torch.float32).cpu().numpy()  # (B, 2*num_pairs)
        idx_arr = np.asarray(indices, dtype=np.int32)

        # Store full scores for D_X and base subsample rows
        dx_local = self._dx_local[idx_arr]
        mask_dx = dx_local >= 0
        if mask_dx.any():
            self.dx_scores[dx_local[mask_dx]] = batch[mask_dx]

        base_local = self._base_local[idx_arr]
        mask_base = base_local >= 0
        if mask_base.any():
            self.base_scores[base_local[mask_base]] = batch[mask_base]

        # Update per-pair top-k for Prec@k / top_inspect (full-corpus, all 3 metrics)
        sg = batch[:, 0::2]  # (B, num_pairs)
        sb = batch[:, 1::2]
        isdx = self._is_dx_full[idx_arr][
            :, np.newaxis
        ]  # (B, 1) broadcasts to (B, num_pairs)
        idx_col = idx_arr[:, np.newaxis]

        for scores_b, top_s, top_i, top_x in (
            (sg - sb, self._topk_c, self._topk_isdx_c, self._topk_idx_c),
            (sg, self._topk_g, self._topk_isdx_g, self._topk_idx_g),
            (-sb, self._topk_b, self._topk_isdx_b, self._topk_idx_b),
        ):
            combined = np.concatenate([top_s, scores_b], axis=0)  # (max_k+B, num_pairs)
            comb_i = np.concatenate(
                [top_i, np.broadcast_to(isdx, scores_b.shape)], axis=0
            )
            comb_x = np.concatenate(
                [top_x, np.broadcast_to(idx_col, scores_b.shape)], axis=0
            )
            keep = np.argpartition(combined, -self.max_k, axis=0)[-self.max_k :]
            top_s[:] = np.take_along_axis(combined, keep, axis=0)
            top_i[:] = np.take_along_axis(comb_i, keep, axis=0)
            top_x[:] = np.take_along_axis(comb_x, keep, axis=0)

    def flush(self):
        pass

    @property
    def scores(self):
        raise AttributeError("StreamingAccumulatorWriter has no flat scores array")


# Family A helpers — vanilla & trackstar (per-pair ΔI via bergson memmap)
# ═══════════════════════════════════════════════════════════════════════════


def _open_score_mmap(scores_raw_dir: Path) -> tuple[np.memmap, int, int]:
    """Open bergson score memmap without loading data into RAM.

    Returns (mmap, num_train, num_pairs).
    """
    info_path = scores_raw_dir / "info.json"
    with open(info_path) as fh:
        info = json.load(fh)

    dtype_info = info["dtype"]
    struct_dtype = np.dtype(
        {
            "names": dtype_info["names"],
            "formats": [np.dtype(f) for f in dtype_info["formats"]],
            "offsets": dtype_info["offsets"],
            "itemsize": dtype_info["itemsize"],
        }
    )
    mmap = np.memmap(
        str(scores_raw_dir / "scores.bin"),
        dtype=struct_dtype,
        mode="r",
        shape=(info["num_items"],),
    )
    return mmap, info["num_items"], info["num_scores"] // 2


def _compute_family_a_stats(
    mmap: np.memmap,
    dx_indices: np.ndarray,
    base_sub: np.ndarray,
    pair_js: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Compute ΔI arrays and Welford score state from the memmap, restricted to
    the query pairs in *pair_js* (pass ``np.arange(num_pairs)`` for all pairs).

    Returns (dx_dI, base_dI, score_n, score_mean, score_M2).
    dx_dI / base_dI have shape (N * len(pair_js),).
    """
    n_dx = len(dx_indices)
    n_base = len(base_sub)

    dx_sub = mmap[dx_indices]
    base_sub_data = mmap[base_sub]

    dx_dI = np.empty(n_dx * len(pair_js), dtype=np.float32)
    base_dI = np.empty(n_base * len(pair_js), dtype=np.float32)
    score_chunks: list[np.ndarray] = []

    for pos, j in enumerate(pair_js):
        sg_dx = np.asarray(dx_sub[f"score_{2 * j}"], dtype=np.float32)
        sb_dx = np.asarray(dx_sub[f"score_{2 * j + 1}"], dtype=np.float32)
        dx_dI[pos * n_dx : (pos + 1) * n_dx] = sg_dx - sb_dx

        sg_base = np.asarray(base_sub_data[f"score_{2 * j}"], dtype=np.float32)
        sb_base = np.asarray(base_sub_data[f"score_{2 * j + 1}"], dtype=np.float32)
        base_dI[pos * n_base : (pos + 1) * n_base] = sg_base - sb_base

        score_chunks.extend([sg_dx, sb_dx, sg_base, sb_base])

    all_scores = np.concatenate(score_chunks)
    score_n = int(all_scores.size)
    score_mean = float(all_scores.mean())
    score_M2 = float(all_scores.var(ddof=0)) * score_n

    return dx_dI, base_dI, score_n, score_mean, score_M2


def _family_a_stats_from_score_arrays(
    sg_dx: np.ndarray,
    sb_dx: np.ndarray,
    sg_base: np.ndarray,
    sb_base: np.ndarray,
    pair_js: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Same computation as _compute_family_a_stats, restricted to *pair_js*,
    but from already-materialized (n_dx/n_base, num_pairs) score arrays — used
    by the streaming code path (StreamingAccumulatorWriter), which never
    builds an (N, 2m) memmap."""
    dx_dI = (sg_dx[:, pair_js] - sb_dx[:, pair_js]).T.reshape(-1).astype(np.float32)
    base_dI = (
        (sg_base[:, pair_js] - sb_base[:, pair_js]).T.reshape(-1).astype(np.float32)
    )
    all_s = np.concatenate(
        [
            sg_dx[:, pair_js].reshape(-1),
            sb_dx[:, pair_js].reshape(-1),
            sg_base[:, pair_js].reshape(-1),
            sb_base[:, pair_js].reshape(-1),
        ]
    )
    score_n = int(all_s.size)
    score_mean = float(all_s.mean())
    score_M2 = float(all_s.var(ddof=0)) * score_n
    return dx_dI, base_dI, score_n, score_mean, score_M2


def _compute_prec_at_k(
    mmap: np.memmap,
    num_train: int,
    pair_js: np.ndarray,
    all_categories: list[str],
    k_values: list[int],
    prec_indices: np.ndarray | None,
    side: str = "contrast",
) -> dict[str, float]:
    """Prec@k macro-averaged across the query pairs in *pair_js* (pass
    ``np.arange(num_pairs)`` for all pairs).

    side='contrast' (default) — top-k by ΔI = I(s⁺) − I(s⁻).
    side='good'               — top-k by I(s⁺) only.
    side='bad'                — top-k by most negative I(s⁻) (i.e. −I(s⁻)).

    prec_indices is None  → stream columns per pair (~2×N×4 B peak RAM).
    prec_indices provided → load subset rows once, iterate in memory.
    """

    def _score(data: np.memmap, j: int) -> np.ndarray:
        sg = np.asarray(data[f"score_{2 * j}"], dtype=np.float32)
        if side == "good":
            return sg
        sb = np.asarray(data[f"score_{2 * j + 1}"], dtype=np.float32)
        if side == "bad":
            return -sb
        return sg - sb

    if prec_indices is not None:
        is_dx = np.array(
            [1 if all_categories[int(i)] == "X" else 0 for i in prec_indices],
            dtype=np.int8,
        )
        subset = mmap[prec_indices]
        per_k: dict[str, list[float]] = {str(k): [] for k in k_values}
        for j in pair_js:
            scores = _score(subset, j)
            for k in k_values:
                actual_k = min(k, len(scores))
                top = np.argpartition(scores, -actual_k)[-actual_k:]
                per_k[str(k)].append(float(is_dx[top].sum()) / actual_k)
    else:
        is_dx_full = np.array(
            [1 if c == "X" else 0 for c in all_categories], dtype=np.int8
        )
        per_k = {str(k): [] for k in k_values}
        for j in pair_js:
            scores = _score(mmap, j)
            for k in k_values:
                actual_k = min(k, num_train)
                top = np.argpartition(scores, -actual_k)[-actual_k:]
                per_k[str(k)].append(float(is_dx_full[top].sum()) / actual_k)

    return {k: float(np.mean(v)) for k, v in per_k.items()}


def _prec_pool_indices(
    dx_indices: np.ndarray,
    base_indices: np.ndarray,
    prec_base_size: int,
) -> tuple[np.ndarray | None, str]:
    """Candidate pool for Prec@k / top_inspect ranking — same pool for both.

    Returns (pool_indices, mode). pool_indices is None when the pool is the
    entire scoring dataset (no subset to materialize).
    """
    if prec_base_size > 0:
        rng_p = np.random.default_rng(_PREC_SUBSAMPLE_SEED)
        n_p = min(prec_base_size, len(base_indices))
        prec_base = rng_p.choice(base_indices, n_p, replace=False)
        pool = np.sort(np.concatenate([dx_indices, prec_base]))
        return pool, f"subset_base_{n_p}"
    return None, "full_corpus"


# ═══════════════════════════════════════════════════════════════════════════
# Learned / not-learned query-pair split (Family A + bm25 only — see
# _uses_scored_memmap / _attribute_bm25; Family B's aggregated query gradient
# has no per-pair granularity left to split by the time it's scored).
#
# Labels come from probe_models.py's --pairs-output parquet (softmax_prob
# of the good sentence), for both the full model (attribution.full_pairs_parquet)
# and, optionally, the matching no-X/ablated model (attribution.no_x_pairs_parquet).
# "Learned" is recomputed here from softmax_prob_good against
# LEARNED_THRESHOLD (config.py) rather than trusting the parquet's learned_x
# column, so the threshold can't drift out of sync with whatever value
# probe_models.py used when the parquet was generated.
# ═══════════════════════════════════════════════════════════════════════════

_LEARNED_VIEW_LABELS = (
    "learned",
    "not_learned",
    "learned_specific",
    "learned_confounded",
    "not_learned_no_x",
)


class PairLabels:
    """Per-pair_idx (0..num_pairs-1) learned-status arrays, aligned to queries_ds."""

    def __init__(
        self,
        logp_good_full: np.ndarray,
        logp_bad_full: np.ndarray,
        softmax_prob_full: np.ndarray,
        learned_full: np.ndarray,
        logp_good_no_x: np.ndarray | None,
        logp_bad_no_x: np.ndarray | None,
        softmax_prob_no_x: np.ndarray | None,
        learned_no_x: np.ndarray | None,
    ):
        self.logp_good_full = logp_good_full
        self.logp_bad_full = logp_bad_full
        self.softmax_prob_full = softmax_prob_full
        self.learned_full = learned_full
        self.logp_good_no_x = logp_good_no_x
        self.logp_bad_no_x = logp_bad_no_x
        self.softmax_prob_no_x = softmax_prob_no_x
        self.learned_no_x = learned_no_x

    def view_groups(self) -> dict[str, np.ndarray]:
        """Named pair-index arrays for the learned/not_learned (+ no-X) split views."""
        groups = {
            "learned": np.where(self.learned_full)[0],
            "not_learned": np.where(~self.learned_full)[0],
        }
        if self.learned_no_x is not None:
            groups["learned_specific"] = np.where(
                self.learned_full & ~self.learned_no_x
            )[0]
            groups["learned_confounded"] = np.where(
                self.learned_full & self.learned_no_x
            )[0]
            groups["not_learned_no_x"] = np.where(~self.learned_no_x)[0]
        return groups

    def path_for(self, exp: ExperimentCfg, seed: int, label: str) -> tuple[Path, Path]:
        """(stats_path, prec_path) for one view label, e.g. 'learned_specific'."""
        return (
            getattr(exp, f"delta_I_stats_{label}_path")(seed),
            getattr(exp, f"prec_at_k_{label}_path")(seed),
        )


def _align_pairs_parquet(
    path: str, queries_ds: datasets.Dataset, what: str
) -> pd.DataFrame | None:
    """Load a probe_models.py --pairs-output parquet and align its rows to
    *queries_ds*, matched by exact (text_good, text_bad) content.

    Content-based matching (rather than row-count + pair_idx-order) is robust
    to queries_ds being a reordered/subsampled subset of what probe_models.py
    scored — e.g. prepare/linguistic.py's --max-queries truncation —
    since it doesn't require the two pipelines to visit pairs in the same order
    or count.

    Returns None (with a logged warning) if the file is missing or any
    queries_ds row has no matching (text_good, text_bad) pair in the parquet —
    callers then skip the affected split.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("%s pairs parquet not found at %s — skipping.", what, p)
        return None

    df = pd.read_parquet(
        str(p),
        columns=[
            "text_good",
            "text_bad",
            "logp_good",
            "logp_bad",
            "softmax_prob_good",
            "learned_x",
        ],
    )
    lookup = {
        (g, b): i for i, (g, b) in enumerate(zip(df["text_good"], df["text_bad"]))
    }
    idx = [
        lookup.get((g, b))
        for g, b in zip(queries_ds["text_good"], queries_ds["text_bad"])
    ]
    missing = sum(1 for i in idx if i is None)
    if missing:
        logger.warning(
            "%s pairs parquet %s: %d/%d queries have no matching (text_good, "
            "text_bad) row — skipping.",
            what,
            p,
            missing,
            len(idx),
        )
        return None
    return df.iloc[idx].reset_index(drop=True)


def _load_pair_labels(
    exp: ExperimentCfg, queries_ds: datasets.Dataset
) -> PairLabels | None:
    """Build per-pair learned-status labels from attribution.full_pairs_parquet
    (required) and attribution.no_x_pairs_parquet (optional). Returns None when
    full_pairs_parquet isn't configured — callers then skip the split entirely
    and only produce the combined ('all pairs') outputs, as before."""
    full_path = exp.attribution.full_pairs_parquet
    if not full_path:
        logger.info(
            "attribution.full_pairs_parquet not set — skipping learned/not-learned "
            "split (point it at a probe_models.py --pairs-output parquet to enable it)."
        )
        return None

    full_df = _align_pairs_parquet(full_path, queries_ds, "full-model")
    if full_df is None:
        return None

    logp_good_no_x = logp_bad_no_x = softmax_prob_no_x = learned_no_x = None
    if exp.attribution.no_x_pairs_parquet:
        no_x_df = _align_pairs_parquet(
            exp.attribution.no_x_pairs_parquet, queries_ds, "no-X-model"
        )
        if no_x_df is not None:
            logp_good_no_x = no_x_df["logp_good"].to_numpy(dtype=np.float32)
            logp_bad_no_x = no_x_df["logp_bad"].to_numpy(dtype=np.float32)
            softmax_prob_no_x = no_x_df["softmax_prob_good"].to_numpy(dtype=np.float32)
            learned_no_x = softmax_prob_no_x > LEARNED_THRESHOLD

    full_logp_good = full_df["logp_good"].to_numpy(dtype=np.float32)
    full_logp_bad = full_df["logp_bad"].to_numpy(dtype=np.float32)
    full_softmax_prob = full_df["softmax_prob_good"].to_numpy(dtype=np.float32)
    labels = PairLabels(
        logp_good_full=full_logp_good,
        logp_bad_full=full_logp_bad,
        softmax_prob_full=full_softmax_prob,
        learned_full=full_softmax_prob > LEARNED_THRESHOLD,
        logp_good_no_x=logp_good_no_x,
        logp_bad_no_x=logp_bad_no_x,
        softmax_prob_no_x=softmax_prob_no_x,
        learned_no_x=learned_no_x,
    )
    n = len(labels.learned_full)
    msg = "[learned/not-learned split] %d/%d pairs learned by the full model (%.1f%%)"
    args = [int(labels.learned_full.sum()), n, 100 * labels.learned_full.mean()]
    if learned_no_x is not None:
        msg += "; of those, %d/%d also learned by the no-X model (confounded)"
        n_learned = int(labels.learned_full.sum())
        n_confounded = int((labels.learned_full & learned_no_x).sum())
        args += [n_confounded, n_learned]
    logger.info(msg, *args)
    return labels


def _inject_query_label_columns(
    columns: dict[str, object],
    pair_idx_arr: np.ndarray,
    queries_ds: datasets.Dataset,
    labels: PairLabels | None,
    side: str = "contrast",
) -> None:
    """Add query text and metadata columns to a top_inspect column dict in place."""
    qcols = queries_ds.column_names
    if side != "bad" and "text_good" in qcols:
        columns["text_good"] = np.array(queries_ds["text_good"])[pair_idx_arr]
    if side != "good" and "text_bad" in qcols:
        columns["text_bad"] = np.array(queries_ds["text_bad"])[pair_idx_arr]
    # Pass through per-pair metadata columns added by prepare/facts.py.
    # negative_idx is dropped for good-only (always 0 there, no meaning).
    skip = {"negative_idx"} if side == "good" else set()
    for _col in (
        "subject",
        "correct_object",
        "template_idx",
        "negative_idx",
        "full_fact_learned",
        "full_p_correct",
        "full_entropy_norm",
        "full_n_correct_templates",
        "filtered_fact_learned",
        "filtered_p_correct",
        "filtered_entropy_norm",
        "filtered_n_correct_templates",
        "verdict",
        # Added by prepare/facts.py's build_cascade_bear_queries:
        # group labels which cascade bucket a row belongs to
        # (learned_specific/not_learned_both); p_good/p_bad are the full
        # model's probabilities for whichever text ended up in
        # text_good/text_bad for that row (see that function's docstring).
        "group",
        "p_good",
        "p_bad",
    ):
        if _col in qcols and _col not in skip:
            columns[_col] = np.array(queries_ds[_col])[pair_idx_arr]
    if labels is None:
        return
    lp_g = labels.logp_good_full[pair_idx_arr]
    lp_b = labels.logp_bad_full[pair_idx_arr]
    columns["log_prob_good_full"] = lp_g
    columns["log_prob_bad_full"] = lp_b
    columns["log_prob_delta_full"] = lp_g - lp_b
    columns["softmax_prob_full"] = labels.softmax_prob_full[pair_idx_arr]
    columns["full_model_learned_X"] = labels.learned_full[pair_idx_arr]
    if labels.learned_no_x is not None:
        lp_g_nx = labels.logp_good_no_x[pair_idx_arr]
        lp_b_nx = labels.logp_bad_no_x[pair_idx_arr]
        columns["log_prob_good_no_x"] = lp_g_nx
        columns["log_prob_bad_no_x"] = lp_b_nx
        columns["log_prob_delta_no_x"] = lp_g_nx - lp_b_nx
        columns["softmax_prob_no_x"] = labels.softmax_prob_no_x[pair_idx_arr]
        columns["no_model_learned_X"] = labels.learned_no_x[pair_idx_arr]


def _load_facts_country_metadata(
    data_dir: Path,
    train_ds: datasets.Dataset,
    num_pairs: int,
) -> tuple[list[str | None], list[str]] | None:
    """Return (train_countries, query_countries) for facts experiments.

    train_countries[i] is the country slug for training example i (None for base rows).
    query_countries[j] is the country slug for query pair j.
    Returns None if the sidecar files are absent or row counts don't match.
    """
    countries_file = (
        data_dir / "data_test" / "X" / "good" / "capital_facts.countries.txt"
    )
    matched_dir = data_dir / "data_training" / "X" / "matched_by_country"
    if not countries_file.exists() or not matched_dir.exists():
        return None
    query_countries = [
        ln.strip()
        for ln in countries_file.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if len(query_countries) != num_pairs:
        logger.warning(
            "Country sidecar has %d entries but num_pairs=%d; skipping exact Prec@k",
            len(query_countries),
            num_pairs,
        )
        return None
    text_to_country: dict[str, str] = {}
    for cfile in sorted(matched_dir.glob("*.txt")):
        slug = cfile.stem
        for line in cfile.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                text_to_country[line] = slug
    train_countries: list[str | None] = [
        text_to_country.get(row["text"]) for row in train_ds
    ]
    return train_countries, query_countries


def _compute_prec_at_k_exact(
    mmap: np.memmap,
    num_train: int,
    num_pairs: int,
    train_countries: list[str | None],
    query_countries: list[str],
    k_values: list[int],
    prec_indices: np.ndarray | None,
) -> dict[str, float]:
    """Per-query exact Prec@k macro-averaged across pairs.

    For query j, relevant = training examples whose country matches query_countries[j].
    Negatives are both D_base and D_X examples from other countries.
    """
    per_k: dict[str, list[float]] = {str(k): [] for k in k_values}
    tc_arr = np.array(train_countries, dtype=object)

    if prec_indices is not None:
        sub_tc = tc_arr[prec_indices]
        subset = mmap[prec_indices]
        for j in range(num_pairs):
            is_exact = (sub_tc == query_countries[j]).astype(np.int8)
            dI = np.asarray(subset[f"score_{2 * j}"], dtype=np.float32) - np.asarray(
                subset[f"score_{2 * j + 1}"], dtype=np.float32
            )
            for k in k_values:
                actual_k = min(k, len(dI))
                top = np.argpartition(dI, -actual_k)[-actual_k:]
                per_k[str(k)].append(float(is_exact[top].sum()) / actual_k)
    else:
        for j in range(num_pairs):
            is_exact = (tc_arr == query_countries[j]).astype(np.int8)
            dI = np.asarray(mmap[f"score_{2 * j}"], dtype=np.float32) - np.asarray(
                mmap[f"score_{2 * j + 1}"], dtype=np.float32
            )
            for k in k_values:
                actual_k = min(k, num_train)
                top = np.argpartition(dI, -actual_k)[-actual_k:]
                per_k[str(k)].append(float(is_exact[top].sum()) / actual_k)

    return {k: float(np.mean(v)) for k, v in per_k.items()}


def _write_family_a_stats_and_prec(
    exp: ExperimentCfg,
    seed: int,
    mmap: np.memmap,
    num_train: int,
    dx_indices: np.ndarray,
    base_indices: np.ndarray,
    cats_arr: np.ndarray,
    pair_js: np.ndarray,
    stats_out: Path,
    prec_out: Path,
    pp_cfg: dict,
    label: str,
) -> None:
    """Compute+write delta_I_stats/prec_at_k for one pair subset (combined,
    learned, or not_learned) from a Family-A (N, 2m) score memmap."""
    if not _up_to_date_file(stats_out, pp_cfg):
        n_avail = len(base_indices)
        n_sub = (
            exp.attribution.stats_base_size
            if exp.attribution.stats_base_size > 0
            else n_avail
        )
        n_sub = min(n_sub, n_avail)
        if n_sub < n_avail:
            rng = np.random.default_rng(_STATS_SUBSAMPLE_SEED)
            base_sub = np.sort(rng.choice(base_indices, n_sub, replace=False))
        else:
            base_sub = base_indices

        dx_dI, base_dI, score_n, score_mean, score_M2 = _compute_family_a_stats(
            mmap, dx_indices, base_sub, pair_js
        )
        _write_delta_I_stats(
            stats_out, dx_dI, base_dI, score_n, score_mean, score_M2, len(pair_js)
        )
        _mark_file(stats_out, pp_cfg)
        logger.info(
            "[seed=%d][%s] Stats → %s (%d D_X × %d pairs, %d base × %d pairs)",
            seed,
            label,
            stats_out,
            len(dx_indices),
            len(pair_js),
            len(base_sub),
            len(pair_js),
        )

    if _up_to_date_file(prec_out, pp_cfg):
        return
    prec_indices, mode = _prec_pool_indices(
        dx_indices, base_indices, exp.attribution.prec_base_size
    )
    cats_list = list(cats_arr)
    prec = _compute_prec_at_k(
        mmap,
        num_train,
        pair_js,
        cats_list,
        exp.attribution.k_precisions,
        prec_indices,
        side="contrast",
    )
    prec_good = _compute_prec_at_k(
        mmap,
        num_train,
        pair_js,
        cats_list,
        exp.attribution.k_precisions,
        prec_indices,
        side="good",
    )
    prec_bad = _compute_prec_at_k(
        mmap,
        num_train,
        pair_js,
        cats_list,
        exp.attribution.k_precisions,
        prec_indices,
        side="bad",
    )
    result: dict = {
        "seed": seed,
        "condition": exp.data.condition,
        "mode": mode,
        "prec_at_k": prec,
        "prec_at_k_good": prec_good,
        "prec_at_k_bad": prec_bad,
    }
    prec_out.parent.mkdir(parents=True, exist_ok=True)
    with open(prec_out, "w") as fh:
        json.dump(result, fh, indent=2)
    _mark_file(prec_out, pp_cfg)
    logger.info(
        "[seed=%d][%s] Prec@k (%s)\n  contrast:  %s\n  good (s⁺): %s\n  bad  (s⁻): %s",
        seed,
        label,
        mode,
        "  ".join(f"P@{k}={v:.7f}" for k, v in prec.items()),
        "  ".join(f"P@{k}={v:.7f}" for k, v in prec_good.items()),
        "  ".join(f"P@{k}={v:.7f}" for k, v in prec_bad.items()),
    )


def _process_scored_memmap(
    exp: ExperimentCfg,
    seed: int,
    train_ds: datasets.Dataset,
    queries_ds: datasets.Dataset,
) -> None:
    """Shared post-processing for Family A (vanilla, trackstar)."""
    stats_out = exp.delta_I_stats_path(seed)
    inspect_out = exp.top_inspect_path(seed)
    prec_out = exp.prec_at_k_path(seed)

    mmap, num_train, num_pairs = _open_score_mmap(exp.scores_raw_dir(seed))
    cats_arr = np.array(train_ds["category"])
    dx_indices = np.where(cats_arr == "X")[0]
    base_indices = np.where(cats_arr == "base")[0]

    logger.info(
        "[seed=%d] D_X=%d  D_base=%d  pairs=%d",
        seed,
        len(dx_indices),
        len(base_indices),
        num_pairs,
    )

    _write_lookups(exp, train_ds, queries_ds)

    pp_cfg = _postproc_cfg(exp, seed)
    labels = _load_pair_labels(exp, queries_ds)
    fact_lk = _build_fact_lookups(train_ds, queries_ds)

    inspect_good_out = exp.top_inspect_good_path(seed)
    inspect_bad_out = exp.top_inspect_bad_path(seed)

    needs_stats = not _up_to_date_file(stats_out, pp_cfg)
    needs_inspect = not _up_to_date_file(inspect_out, pp_cfg)
    needs_inspect_good = not _up_to_date_file(inspect_good_out, pp_cfg)
    needs_inspect_bad = not _up_to_date_file(inspect_bad_out, pp_cfg)
    if needs_stats:
        n_avail = len(base_indices)
        n_sub = (
            exp.attribution.stats_base_size
            if exp.attribution.stats_base_size > 0
            else n_avail
        )
        n_sub = min(n_sub, n_avail)
        if n_sub < n_avail:
            rng = np.random.default_rng(_STATS_SUBSAMPLE_SEED)
            base_sub = np.sort(rng.choice(base_indices, n_sub, replace=False))
        else:
            base_sub = base_indices

        dx_dI, base_dI, score_n, score_mean, score_M2 = _compute_family_a_stats(
            mmap, dx_indices, base_sub, np.arange(num_pairs)
        )
        _write_delta_I_stats(
            stats_out, dx_dI, base_dI, score_n, score_mean, score_M2, num_pairs
        )
        _mark_file(stats_out, pp_cfg)
        logger.info(
            "[seed=%d] Stats → %s (%d D_X × %d pairs, %d base × %d pairs)",
            seed,
            stats_out,
            len(dx_indices),
            num_pairs,
            len(base_sub),
            num_pairs,
        )

    if needs_inspect or needs_inspect_good or needs_inspect_bad:
        pool_indices, _ = _prec_pool_indices(
            dx_indices, base_indices, exp.attribution.prec_base_size
        )

        if needs_inspect:
            _write_per_pair_top_inspect(
                inspect_out,
                mmap,
                pool_indices,
                cats_arr,
                num_pairs,
                train_ds,
                queries_ds,
                side="contrast",
                top_k=exp.attribution.inspect_top_k,
                labels=labels,
                fact_lk=fact_lk,
            )
            _mark_file(inspect_out, pp_cfg)
            logger.info(
                "[seed=%d] Top-%d/pair inspect (contrast) → %s",
                seed,
                exp.attribution.inspect_top_k,
                inspect_out,
            )

        if needs_inspect_good:
            _write_per_pair_top_inspect(
                inspect_good_out,
                mmap,
                pool_indices,
                cats_arr,
                num_pairs,
                train_ds,
                queries_ds,
                side="good",
                top_k=exp.attribution.inspect_top_k,
                labels=labels,
                fact_lk=fact_lk,
            )
            _mark_file(inspect_good_out, pp_cfg)
            logger.info(
                "[seed=%d] Top-%d/pair inspect (s⁺) → %s",
                seed,
                exp.attribution.inspect_top_k,
                inspect_good_out,
            )

        if needs_inspect_bad:
            _write_per_pair_top_inspect(
                inspect_bad_out,
                mmap,
                pool_indices,
                cats_arr,
                num_pairs,
                train_ds,
                queries_ds,
                side="bad",
                top_k=exp.attribution.inspect_top_k,
                labels=labels,
                fact_lk=fact_lk,
            )
            _mark_file(inspect_bad_out, pp_cfg)
            logger.info(
                "[seed=%d] Top-%d/pair inspect (s⁻ most negative) → %s",
                seed,
                exp.attribution.inspect_top_k,
                inspect_bad_out,
            )

    if not _up_to_date_file(prec_out, pp_cfg):
        prec_indices, mode = _prec_pool_indices(
            dx_indices, base_indices, exp.attribution.prec_base_size
        )

        cats_list = list(cats_arr)
        prec = _compute_prec_at_k(
            mmap,
            num_train,
            np.arange(num_pairs),
            cats_list,
            exp.attribution.k_precisions,
            prec_indices,
            side="contrast",
        )
        prec_good = _compute_prec_at_k(
            mmap,
            num_train,
            np.arange(num_pairs),
            cats_list,
            exp.attribution.k_precisions,
            prec_indices,
            side="good",
        )
        prec_bad = _compute_prec_at_k(
            mmap,
            num_train,
            np.arange(num_pairs),
            cats_list,
            exp.attribution.k_precisions,
            prec_indices,
            side="bad",
        )
        result: dict = {
            "seed": seed,
            "condition": exp.data.condition,
            "mode": mode,
            "prec_at_k": prec,
            "prec_at_k_good": prec_good,
            "prec_at_k_bad": prec_bad,
        }

        if exp.data.experiment_type == "facts":
            country_meta = _load_facts_country_metadata(
                Path(exp.data.data_dir), train_ds, num_pairs
            )
            if country_meta is not None:
                train_countries, query_countries = country_meta
                prec_exact = _compute_prec_at_k_exact(
                    mmap,
                    num_train,
                    num_pairs,
                    train_countries,
                    query_countries,
                    exp.attribution.k_precisions,
                    prec_indices,
                )
                result["prec_at_k_exact"] = prec_exact
                logger.info(
                    "[seed=%d] Exact Prec@k:     %s",
                    seed,
                    "  ".join(f"P@{k}={v:.4f}" for k, v in prec_exact.items()),
                )

        prec_out.parent.mkdir(parents=True, exist_ok=True)
        with open(prec_out, "w") as fh:
            json.dump(result, fh, indent=2)
        _mark_file(prec_out, pp_cfg)
        logger.info(
            "[seed=%d] Prec@k (%s)\n  contrast:  %s\n  good (s⁺): %s\n  bad  (s⁻): %s",
            seed,
            mode,
            "  ".join(f"P@{k}={v:.7f}" for k, v in prec.items()),
            "  ".join(f"P@{k}={v:.7f}" for k, v in prec_good.items()),
            "  ".join(f"P@{k}={v:.7f}" for k, v in prec_bad.items()),
        )

    if labels is not None:
        for label, pair_js in labels.view_groups().items():
            if len(pair_js) == 0:
                logger.info(
                    "[seed=%d][%s] 0 pairs in this split — skipping.", seed, label
                )
                continue
            stats_split, prec_split = labels.path_for(exp, seed, label)
            _write_family_a_stats_and_prec(
                exp,
                seed,
                mmap,
                num_train,
                dx_indices,
                base_indices,
                cats_arr,
                pair_js,
                stats_split,
                prec_split,
                pp_cfg,
                label,
            )


def _stream_score_and_process(
    exp: ExperimentCfg,
    seed: int,
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    train_ds: datasets.Dataset,
    queries_ds: datasets.Dataset,
) -> None:
    """Score training data and compute all Family-A outputs in one streaming pass.

    Replaces _score_training_data + _process_scored_memmap for vanilla,
    trackstar, and ekfac-sketch. Never writes a (N, 2m) scores file to disk;
    instead keeps per-pair top-k candidates and full D_X/base-sample scores
    in memory (~5 GB for NPI with 7k pairs vs ~200 GB on disk).
    """
    stats_out = exp.delta_I_stats_path(seed)
    inspect_out = exp.top_inspect_path(seed)
    inspect_good_out = exp.top_inspect_good_path(seed)
    inspect_bad_out = exp.top_inspect_bad_path(seed)
    prec_out = exp.prec_at_k_path(seed)
    pp_cfg = _postproc_cfg(exp, seed)

    num_pairs = len(queries_ds)
    labels = _load_pair_labels(exp, queries_ds)
    view_groups = labels.view_groups() if labels is not None else {}

    outputs_to_check = [
        stats_out,
        inspect_out,
        inspect_good_out,
        inspect_bad_out,
        prec_out,
    ]
    for label, pair_js in view_groups.items():
        if len(pair_js) > 0:
            outputs_to_check += list(labels.path_for(exp, seed, label))
    if "fact_key" in queries_ds.column_names:
        outputs_to_check.append(exp.delta_I_per_source_path(seed))
    if all(_up_to_date_file(p, pp_cfg) for p in outputs_to_check):
        logger.info("[seed=%d] All streaming outputs up to date, skipping.", seed)
        return

    _write_lookups(exp, train_ds, queries_ds)

    cats_arr = np.array(train_ds["category"])
    dx_indices = np.where(cats_arr == "X")[0]
    base_indices = np.where(cats_arr == "base")[0]
    fact_lk = _build_fact_lookups(train_ds, queries_ds)

    logger.info(
        "[seed=%d] D_X=%d  D_base=%d  pairs=%d",
        seed,
        len(dx_indices),
        len(base_indices),
        num_pairs,
    )

    n_avail = len(base_indices)
    n_sub = (
        exp.attribution.stats_base_size
        if exp.attribution.stats_base_size > 0
        else n_avail
    )
    n_sub = min(n_sub, n_avail)
    if n_sub < n_avail:
        rng = np.random.default_rng(_STATS_SUBSAMPLE_SEED)
        base_sub = np.sort(rng.choice(base_indices, n_sub, replace=False))
    else:
        base_sub = base_indices

    is_dx_full = (cats_arr == "X").astype(np.int8)
    max_k = max(max(exp.attribution.k_precisions), exp.attribution.inspect_top_k)
    writer = StreamingAccumulatorWriter(
        num_train=len(train_ds),
        num_queries=2 * num_pairs,
        dx_indices=dx_indices,
        base_sub_indices=base_sub,
        is_dx_full=is_dx_full,
        max_k=max_k,
    )

    score_dataset_streaming(index_cfg, score_cfg, preprocess_cfg, writer)

    sg_dx = writer.dx_scores[:, 0::2]  # (n_dx, num_pairs)
    sb_dx = writer.dx_scores[:, 1::2]
    sg_base = writer.base_scores[:, 0::2]
    sb_base = writer.base_scores[:, 1::2]

    # ── delta_I_stats ────────────────────────────────────────────────────
    if not _up_to_date_file(stats_out, pp_cfg):
        dx_dI = (sg_dx - sb_dx).T.reshape(-1).astype(np.float32)
        base_dI = (sg_base - sb_base).T.reshape(-1).astype(np.float32)
        all_s = np.concatenate(
            [
                sg_dx.reshape(-1),
                sb_dx.reshape(-1),
                sg_base.reshape(-1),
                sb_base.reshape(-1),
            ]
        )
        score_n = int(all_s.size)
        score_mean = float(all_s.mean())
        score_M2 = float(all_s.var(ddof=0)) * score_n
        _write_delta_I_stats(
            stats_out, dx_dI, base_dI, score_n, score_mean, score_M2, num_pairs
        )
        _mark_file(stats_out, pp_cfg)
        logger.info(
            "[seed=%d] Stats → %s (%d D_X × %d pairs, %d base × %d pairs)",
            seed,
            stats_out,
            len(dx_indices),
            num_pairs,
            len(base_sub),
            num_pairs,
        )

    # ── Per-fact per-source delta_I (BEAR facts only) ────────────────────
    if fact_lk is not None:
        _write_per_fact_src_delta_I(
            exp,
            seed,
            sg_dx,
            sb_dx,
            sg_base,
            sb_base,
            dx_indices,
            fact_lk,
            queries_ds,
            pp_cfg,
        )

    # ── top_inspect (contrast, good, bad) — ranked over the full streamed
    # pool (D_X ∪ D_base), same candidates the streaming Prec@k uses below.
    cols = queries_ds.column_names
    features = queries_ds["feature"] if "feature" in cols else [""] * num_pairs
    score_col_name = {"contrast": "delta_I", "good": "score_good", "bad": "score_bad"}

    for path, side, top_scores, top_isdx, top_idx, sign in [
        (
            inspect_out,
            "contrast",
            writer._topk_c,
            writer._topk_isdx_c,
            writer._topk_idx_c,
            1.0,
        ),
        (
            inspect_good_out,
            "good",
            writer._topk_g,
            writer._topk_isdx_g,
            writer._topk_idx_g,
            1.0,
        ),
        (
            inspect_bad_out,
            "bad",
            writer._topk_b,
            writer._topk_isdx_b,
            writer._topk_idx_b,
            -1.0,
        ),
    ]:
        if _up_to_date_file(path, pp_cfg):
            continue
        k = min(exp.attribution.inspect_top_k, top_scores.shape[0])
        pair_idxs, train_idxs, ranks_out, feats_out, scores_out = [], [], [], [], []
        global_cats_fallback = []
        for j in range(num_pairs):
            sc = top_scores[:, j]
            order = np.argsort(-sc)[:k]
            feat = str(features[j])
            for rank, li in enumerate(order):
                pair_idxs.append(j)
                train_idxs.append(int(top_idx[li, j]))
                ranks_out.append(rank)
                feats_out.append(feat)
                scores_out.append(float(sign * sc[li]))
                global_cats_fallback.append("X" if top_isdx[li, j] else "base")
        texts = [str(train_ds[ti]["text"]) for ti in train_idxs]
        if fact_lk is not None:
            cats_final = [
                _fact_category(ti, pi, txt, fact_lk)
                for ti, pi, txt in zip(train_idxs, pair_idxs, texts)
            ]
        else:
            cats_final = global_cats_fallback
        pair_idx_arr = np.array(pair_idxs, dtype=np.int32)
        columns: dict[str, object] = {
            "pair_idx": pair_idx_arr,
            "feature": feats_out,
            "rank": np.array(ranks_out, dtype=np.int16),
            "train_idx": np.array(train_idxs, dtype=np.int32),
            "category": cats_final,
            score_col_name[side]: np.array(scores_out, dtype=np.float32),
            "text": texts,
        }
        _inject_query_label_columns(
            columns, pair_idx_arr, queries_ds, labels, side=side
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns).to_parquet(str(path), index=False, compression="zstd")
        _mark_file(path, pp_cfg)
        logger.info(
            "[seed=%d] Top-%d/pair inspect (%s) → %s",
            seed,
            exp.attribution.inspect_top_k,
            side,
            path,
        )

    # ── Prec@k (full corpus) ─────────────────────────────────────────────
    if not _up_to_date_file(prec_out, pp_cfg):
        num_train = len(train_ds)
        k_values = exp.attribution.k_precisions

        def _prec_from_topk(
            top_scores: np.ndarray,
            top_isdx: np.ndarray,
            pair_js: np.ndarray | None = None,
        ) -> dict[str, float]:
            if pair_js is not None:
                top_scores = top_scores[:, pair_js]
                top_isdx = top_isdx[:, pair_js]
            topk_size = top_scores.shape[0]
            result = {}
            for k in k_values:
                actual_k = min(k, num_train)
                if actual_k >= topk_size:
                    prec_per_pair = top_isdx.sum(axis=0) / actual_k
                else:
                    sub_idx = np.argpartition(top_scores, -actual_k, axis=0)[-actual_k:]
                    sub_isdx = np.take_along_axis(top_isdx, sub_idx, axis=0)
                    prec_per_pair = sub_isdx.sum(axis=0) / actual_k
                result[str(k)] = float(np.mean(prec_per_pair))
            return result

        prec = _prec_from_topk(writer._topk_c, writer._topk_isdx_c)
        prec_good = _prec_from_topk(writer._topk_g, writer._topk_isdx_g)
        prec_bad = _prec_from_topk(writer._topk_b, writer._topk_isdx_b)

        prec_out.parent.mkdir(parents=True, exist_ok=True)
        with open(prec_out, "w") as fh:
            json.dump(
                {
                    "seed": seed,
                    "condition": exp.data.condition,
                    "mode": "full_corpus",
                    "prec_at_k": prec,
                    "prec_at_k_good": prec_good,
                    "prec_at_k_bad": prec_bad,
                },
                fh,
                indent=2,
            )
        _mark_file(prec_out, pp_cfg)
        logger.info(
            "[seed=%d] Prec@k (full_corpus)\n  contrast:  %s\n  good (s⁺): %s\n  bad  (s⁻): %s",
            seed,
            "  ".join(f"P@{k}={v:.7f}" for k, v in prec.items()),
            "  ".join(f"P@{k}={v:.7f}" for k, v in prec_good.items()),
            "  ".join(f"P@{k}={v:.7f}" for k, v in prec_bad.items()),
        )

    # ── Learned / not-learned (+ no-X) variants (stats + Prec@k only —
    # top_inspect above already carries the per-pair label columns) ────────
    if labels is not None:
        num_train = len(train_ds)
        k_values = exp.attribution.k_precisions

        def _prec_from_topk_subset(
            top_scores: np.ndarray, top_isdx: np.ndarray, pair_js: np.ndarray
        ) -> dict[str, float]:
            ts = top_scores[:, pair_js]
            ti = top_isdx[:, pair_js]
            topk_size = ts.shape[0]
            result = {}
            for k in k_values:
                actual_k = min(k, num_train)
                if actual_k >= topk_size:
                    prec_per_pair = ti.sum(axis=0) / actual_k
                else:
                    sub_idx = np.argpartition(ts, -actual_k, axis=0)[-actual_k:]
                    sub_isdx = np.take_along_axis(ti, sub_idx, axis=0)
                    prec_per_pair = sub_isdx.sum(axis=0) / actual_k
                result[str(k)] = float(np.mean(prec_per_pair))
            return result

        for label, pair_js in view_groups.items():
            split_stats_out, split_prec_out = labels.path_for(exp, seed, label)
            if len(pair_js) == 0:
                logger.info(
                    "[seed=%d][%s] 0 pairs in this split — skipping.", seed, label
                )
                continue

            if not _up_to_date_file(split_stats_out, pp_cfg):
                dx_dI, base_dI, score_n, score_mean, score_M2 = (
                    _family_a_stats_from_score_arrays(
                        sg_dx, sb_dx, sg_base, sb_base, pair_js
                    )
                )
                _write_delta_I_stats(
                    split_stats_out,
                    dx_dI,
                    base_dI,
                    score_n,
                    score_mean,
                    score_M2,
                    len(pair_js),
                )
                _mark_file(split_stats_out, pp_cfg)
                logger.info(
                    "[seed=%d][%s] Stats → %s (%d D_X × %d pairs, %d base × %d pairs)",
                    seed,
                    label,
                    split_stats_out,
                    len(dx_indices),
                    len(pair_js),
                    len(base_sub),
                    len(pair_js),
                )

            if not _up_to_date_file(split_prec_out, pp_cfg):
                prec = _prec_from_topk_subset(
                    writer._topk_c, writer._topk_isdx_c, pair_js
                )
                prec_good = _prec_from_topk_subset(
                    writer._topk_g, writer._topk_isdx_g, pair_js
                )
                prec_bad = _prec_from_topk_subset(
                    writer._topk_b, writer._topk_isdx_b, pair_js
                )
                split_prec_out.parent.mkdir(parents=True, exist_ok=True)
                with open(split_prec_out, "w") as fh:
                    json.dump(
                        {
                            "seed": seed,
                            "condition": exp.data.condition,
                            "mode": "full_corpus",
                            "prec_at_k": prec,
                            "prec_at_k_good": prec_good,
                            "prec_at_k_bad": prec_bad,
                        },
                        fh,
                        indent=2,
                    )
                _mark_file(split_prec_out, pp_cfg)
                logger.info(
                    "[seed=%d][%s] Prec@k (full_corpus)\n  contrast:  %s\n  good (s⁺): %s\n  bad  (s⁻): %s",
                    seed,
                    label,
                    "  ".join(f"P@{k}={v:.7f}" for k, v in prec.items()),
                    "  ".join(f"P@{k}={v:.7f}" for k, v in prec_good.items()),
                    "  ".join(f"P@{k}={v:.7f}" for k, v in prec_bad.items()),
                )


# ═══════════════════════════════════════════════════════════════════════════
# Family B helpers — ekfac (global ΔI)
# ═══════════════════════════════════════════════════════════════════════════


def _load_single_score_col(scores_dir: Path) -> np.ndarray:
    """Load a single-column bergson score memmap as a (N,) float32 array."""
    with open(scores_dir / "info.json") as fh:
        info = json.load(fh)
    dtype_info = info["dtype"]
    struct_dtype = np.dtype(
        {
            "names": dtype_info["names"],
            "formats": [np.dtype(f) for f in dtype_info["formats"]],
            "offsets": dtype_info["offsets"],
            "itemsize": dtype_info["itemsize"],
        }
    )
    mmap = np.memmap(
        str(scores_dir / "scores.bin"),
        dtype=struct_dtype,
        mode="r",
        shape=(info["num_items"],),
    )
    score_fields = [n for n in dtype_info["names"] if n.startswith("score_")]
    if len(score_fields) != 1:
        raise RuntimeError(
            f"Expected 1 score column in {scores_dir}, found {len(score_fields)}"
        )
    return np.asarray(mmap[score_fields[0]], dtype=np.float32)


def _process_global_delta_I(
    exp: ExperimentCfg,
    seed: int,
    delta_I_all: np.ndarray,
    train_ds: datasets.Dataset,
) -> None:
    """Post-processing for Family B (ekfac): global ΔI(z) per training example."""
    stats_out = exp.delta_I_stats_path(seed)
    inspect_out = exp.top_inspect_path(seed)
    prec_out = exp.prec_at_k_path(seed)

    # Sidecar lookups (load queries on-demand only if missing).
    train_lp = exp.shared_dir() / "train_lookup.parquet"
    query_lp = exp.shared_dir() / "query_lookup.parquet"
    if not train_lp.exists() or not query_lp.exists():
        queries_ds = datasets.load_from_disk(str(exp.queries_path()))
        _write_lookups(exp, train_ds, queries_ds)

    cats_arr = np.array(train_ds["category"])
    dx_indices = np.where(cats_arr == "X")[0]
    base_indices = np.where(cats_arr == "base")[0]

    logger.info(
        "[seed=%d] Global ΔI: D_X=%d  D_base=%d",
        seed,
        len(dx_indices),
        len(base_indices),
    )

    pp_cfg = _postproc_cfg(exp, seed)

    needs_stats = not _up_to_date_file(stats_out, pp_cfg)
    needs_inspect = not _up_to_date_file(inspect_out, pp_cfg)
    if needs_stats:
        n_avail = len(base_indices)
        n_sub = (
            exp.attribution.stats_base_size
            if exp.attribution.stats_base_size > 0
            else n_avail
        )
        n_sub = min(n_sub, n_avail)
        if n_sub < n_avail:
            rng = np.random.default_rng(_STATS_SUBSAMPLE_SEED)
            base_sub = np.sort(rng.choice(base_indices, n_sub, replace=False))
        else:
            base_sub = base_indices

        dx_dI = delta_I_all[dx_indices].astype(np.float32)
        base_dI = delta_I_all[base_sub].astype(np.float32)

        _write_delta_I_stats(stats_out, dx_dI, base_dI, 0, 0.0, 0.0, 1)
        _mark_file(stats_out, pp_cfg)
        logger.info(
            "[seed=%d] Global ΔI stats → %s (%d D_X, %d base)",
            seed,
            stats_out,
            len(dx_dI),
            len(base_dI),
        )

    if needs_inspect:
        pool_indices, _ = _prec_pool_indices(
            dx_indices, base_indices, exp.attribution.prec_base_size
        )
        _write_top_inspect(
            inspect_out,
            pool_indices,
            delta_I_all,
            cats_arr,
            train_ds,
            top_k=exp.attribution.inspect_top_k,
        )
        _mark_file(inspect_out, pp_cfg)
        logger.info(
            "[seed=%d] Top-%d inspect → %s",
            seed,
            exp.attribution.inspect_top_k,
            inspect_out,
        )

    if not _up_to_date_file(prec_out, pp_cfg):
        prec_indices, mode = _prec_pool_indices(
            dx_indices, base_indices, exp.attribution.prec_base_size
        )
        if prec_indices is not None:
            sub_dI = delta_I_all[prec_indices]
            sub_cats = cats_arr[prec_indices]
        else:
            sub_dI = delta_I_all
            sub_cats = cats_arr

        is_dx = (sub_cats == "X").astype(np.int8)
        prec = {}
        for k in exp.attribution.k_precisions:
            actual_k = min(k, len(sub_dI))
            top = np.argpartition(sub_dI, -actual_k)[-actual_k:]
            prec[str(k)] = float(is_dx[top].sum()) / actual_k

        prec_out.parent.mkdir(parents=True, exist_ok=True)
        with open(prec_out, "w") as fh:
            json.dump(
                {
                    "seed": seed,
                    "condition": exp.data.condition,
                    "mode": mode,
                    "prec_at_k": prec,
                },
                fh,
                indent=2,
            )
        _mark_file(prec_out, pp_cfg)
        logger.info(
            "[seed=%d] Global Prec@k (%s): %s",
            seed,
            mode,
            "  ".join(f"Prec@{k}={v:.3f}" for k, v in prec.items()),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Method: vanilla
# ═══════════════════════════════════════════════════════════════════════════


def _build_query_index(exp: ExperimentCfg, seed: int, query_texts_path: Path) -> None:
    """Build per-sentence query gradient index (vanilla, reused by trackstar step 4)."""
    out = exp.query_grad_dir(seed)
    cfg = _method_cfg(exp)
    if _up_to_date(out, cfg):
        logger.info("[seed=%d] Query index already exists, skipping.", seed)
        return
    _maybe_delete_stale(out, cfg, "query index")

    partial = Path(str(out) + ".part")
    if partial.exists():
        shutil.rmtree(partial)

    logger.info("[seed=%d] Building query gradient index → %s", seed, out)
    build(
        IndexConfig(
            run_path=str(out),
            model=str(exp.model_dir(seed)),
            token_batch_size=exp.attribution.token_batch_size or exp.model.max_length,
            max_batch_size=exp.attribution.max_batch_size or None,
            projection_dim=exp.attribution.projection_dim,
            data=DataConfig(dataset=str(query_texts_path), truncation=True),
            optimizer_state_path=str(exp.model_dir(seed))
            if exp.attribution.use_optimizer_state
            else "",
            skip_preconditioners=True,
        ),
        PreprocessConfig(unit_normalize=exp.attribution.unit_norm, aggregation="none"),
    )
    _mark(out, cfg)


def _score_training_data(exp: ExperimentCfg, seed: int, train_path: Path) -> None:
    """Score training examples against the query index (vanilla)."""
    out = exp.scores_raw_dir(seed)
    cfg = _method_cfg(exp)
    if _up_to_date(out, cfg):
        logger.info("[seed=%d] Raw scores already exist, skipping.", seed)
        return
    _maybe_delete_stale(out, cfg, "raw scores")

    partial = Path(str(out) + ".part")
    if partial.exists():
        shutil.rmtree(partial)

    logger.info("[seed=%d] Scoring training data → %s", seed, out)
    score_dataset(
        IndexConfig(
            run_path=str(out),
            model=str(exp.model_dir(seed)),
            token_batch_size=exp.attribution.token_batch_size or exp.model.max_length,
            projection_dim=exp.attribution.projection_dim,
            data=DataConfig(dataset=str(train_path), truncation=True),
            optimizer_state_path=str(exp.model_dir(seed))
            if exp.attribution.use_optimizer_state
            else "",
        ),
        ScoreConfig(query_path=str(exp.query_grad_dir(seed)), score="individual"),
        PreprocessConfig(unit_normalize=exp.attribution.unit_norm),
    )
    _mark(out, cfg)


def _attribute_vanilla(exp: ExperimentCfg, seed: int) -> None:
    train_path = _get_scoring_train_path(exp)
    query_texts_path = _prepare_query_texts(exp)
    _build_query_index(exp, seed, query_texts_path)

    train_ds = datasets.load_from_disk(str(train_path))
    queries_ds = datasets.load_from_disk(str(exp.queries_path()))
    _stream_score_and_process(
        exp,
        seed,
        IndexConfig(
            run_path="",
            model=str(exp.model_dir(seed)),
            token_batch_size=exp.attribution.token_batch_size or exp.model.max_length,
            max_batch_size=exp.attribution.max_batch_size or None,
            projection_dim=exp.attribution.projection_dim,
            data=DataConfig(dataset=str(train_path), truncation=True),
            optimizer_state_path=str(exp.model_dir(seed))
            if exp.attribution.use_optimizer_state
            else "",
        ),
        ScoreConfig(query_path=str(exp.query_grad_dir(seed)), score="individual"),
        PreprocessConfig(unit_normalize=exp.attribution.unit_norm),
        train_ds,
        queries_ds,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Method: trackstar
# ═══════════════════════════════════════════════════════════════════════════


def _attribute_trackstar(exp: ExperimentCfg, seed: int) -> None:
    """5-step TrackStar pipeline with per-sentence query gradients.

    Steps 1–4 run the model to build preconditioners and the query index.
    Step 5 scores training data on-the-fly against the query index, applying
    the mixed preconditioner during gradient collection (no train index stored).

    With aggregation='none' each of the 2m query sentences retains its
    individual gradient and produces a separate score column in the (N, 2m)
    output matrix.
    """
    from bergson.process_grads import mix_preconditioners

    train_path = _get_scoring_train_path(exp)
    query_texts_path = _prepare_query_texts(exp)
    value_precond_path = str(exp.ts_value_precond_dir(seed))
    query_precond_path = str(exp.ts_query_precond_dir(seed))
    mixed_precond_path = str(exp.ts_mixed_precond_dir(seed))
    query_index_path = str(exp.query_grad_dir(seed))

    base_icfg = IndexConfig(
        run_path="",  # overridden per step
        model=str(exp.model_dir(seed)),
        token_batch_size=exp.attribution.token_batch_size or exp.model.max_length,
        projection_dim=exp.attribution.projection_dim,
        data=DataConfig(dataset=str(train_path), truncation=True),
    )

    ts_cfg = _method_cfg(exp)

    # Step 1 — Value preconditioner (from D_train, index skipped)
    if not _up_to_date(Path(value_precond_path), ts_cfg):
        _maybe_delete_stale(Path(value_precond_path), ts_cfg, "value preconditioner")
        logger.info("[seed=%d][trackstar] Step 1/6: value preconditioner", seed)
        cfg = deepcopy(base_icfg)
        cfg.run_path = value_precond_path
        cfg.skip_index = True
        cfg.skip_preconditioners = False
        build(cfg, PreprocessConfig())
        _mark(Path(value_precond_path), ts_cfg)

    # Step 2 — Query preconditioner (from query texts, index skipped)
    if not _up_to_date(Path(query_precond_path), ts_cfg):
        _maybe_delete_stale(Path(query_precond_path), ts_cfg, "query preconditioner")
        logger.info("[seed=%d][trackstar] Step 2/6: query preconditioner", seed)
        cfg = deepcopy(base_icfg)
        cfg.run_path = query_precond_path
        cfg.data = DataConfig(dataset=str(query_texts_path), truncation=True)
        cfg.skip_index = True
        cfg.skip_preconditioners = False
        build(cfg, PreprocessConfig())
        _mark(Path(query_precond_path), ts_cfg)

    # Step 3 — Mix preconditioners
    if not _up_to_date(Path(mixed_precond_path), ts_cfg):
        _maybe_delete_stale(Path(mixed_precond_path), ts_cfg, "mixed preconditioner")
        logger.info("[seed=%d][trackstar] Step 3/6: mixing preconditioners", seed)
        mix_preconditioners(
            query_path=query_precond_path,
            index_path=value_precond_path,
            output_path=mixed_precond_path,
            target_downweight_components=exp.attribution.trackstar_downweight_components,
        )
        _mark(Path(mixed_precond_path), ts_cfg)

    # Step 4 — Build per-sentence query gradient index
    # aggregation='none': mixed preconditioner applied at score time (step 6)
    if not _up_to_date(Path(query_index_path), ts_cfg):
        _maybe_delete_stale(Path(query_index_path), ts_cfg, "query gradient index")
        logger.info(
            "[seed=%d][trackstar] Step 4/6: building query gradient index", seed
        )
        cfg = deepcopy(base_icfg)
        cfg.run_path = query_index_path
        cfg.data = DataConfig(dataset=str(query_texts_path), truncation=True)
        cfg.processor_path = query_precond_path
        cfg.skip_preconditioners = True
        build(
            cfg,
            PreprocessConfig(
                aggregation="none",
                unit_normalize=exp.attribution.unit_norm,
                preconditioner_path=mixed_precond_path,
            ),
        )
        _mark(Path(query_index_path), ts_cfg)

    # Step 5 — Score training data on-the-fly and collect all outputs.
    logger.info("[seed=%d][trackstar] Step 5/5: scoring training data on-the-fly", seed)
    train_ds = datasets.load_from_disk(str(train_path))
    queries_ds = datasets.load_from_disk(str(exp.queries_path()))
    _stream_score_and_process(
        exp,
        seed,
        IndexConfig(
            run_path="",
            model=str(exp.model_dir(seed)),
            token_batch_size=(
                exp.attribution.trackstar_score_token_batch_size
                or exp.attribution.token_batch_size
                or exp.model.max_length
            ),
            projection_dim=exp.attribution.projection_dim,
            data=DataConfig(dataset=str(train_path), truncation=True),
        ),
        ScoreConfig(query_path=query_index_path, score="individual"),
        PreprocessConfig(
            unit_normalize=exp.attribution.unit_norm,
            preconditioner_path=mixed_precond_path,
        ),
        train_ds,
        queries_ds,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Method: ekfac
# ═══════════════════════════════════════════════════════════════════════════


def _attribute_ekfac_sketched(exp: ExperimentCfg, seed: int) -> None:
    """Sketched EK-FAC/K-FAC pipeline (Family A).

    Two sub-paths controlled by ``ekfac_modified_projections``:

    modified_projections=False (default, any method):
        1. Fit Hessian on D_train                            → hessian_dir
        2. Build whitened query index (H^{-1/2} + random R) → query_grad_dir
        3. Score training data on-the-fly (same whitener)

    modified_projections=True (P-SIFT, kfac only):
        1. Fit K-FAC on D_train (no eigenvalue correction)   → hessian_dir
        2. Build M = R · cov^{-1/2} per side                → hessian_dir/projection_*
        3. Build query index with M applied at collection    → query_grad_dir
        4. Score training data on-the-fly with M at collection
    """
    from bergson.config import HessianConfig
    from bergson.distributed import launch_distributed_run
    from bergson.hessians.apply_hessian import EkfacConfig, build_projections_worker
    from bergson.hessians.hessian_approximations import approximate_hessians

    if exp.attribution.ekfac_aggregate_query:
        raise ValueError("ekfac_sketch requires ekfac_aggregate_query=False.")
    if exp.attribution.projection_dim <= 0:
        raise ValueError("ekfac_sketch requires projection_dim > 0.")

    train_path = _get_scoring_train_path(exp)
    query_texts_path = _prepare_query_texts(exp)
    hessian_dir = exp.ekfac_hessian_dir(seed)
    query_index_path = exp.query_grad_dir(seed)

    hessian_cfg = _ekfac_hessian_cfg(exp)
    sketch_q_cfg = _ekfac_sketch_query_cfg(exp)

    # "kfac" and "ekfac" both use bergson's kfac method; "ekfac" adds ev_correction.
    hessian_method = (
        "kfac"
        if exp.attribution.ekfac_method in ("ekfac", "kfac")
        else exp.attribution.ekfac_method
    )
    ev_correction = exp.attribution.ekfac_method == "ekfac"

    base_icfg = IndexConfig(
        run_path="",  # overridden per step
        model=str(exp.model_dir(seed)),
        token_batch_size=exp.attribution.ekfac_token_batch_size or exp.model.max_length,
        projection_dim=exp.attribution.projection_dim,
        data=DataConfig(dataset=str(train_path), truncation=True),
        skip_preconditioners=True,
        filter_modules=exp.attribution.ekfac_filter_modules,
    )

    # ── Step 1: Fit Hessian on D_train ────────────────────────────────────
    if not _up_to_date(hessian_dir, hessian_cfg):
        _maybe_delete_stale(hessian_dir, hessian_cfg, "ekfac Hessian")
        logger.info(
            "[seed=%d][ekfac-sketch] Fitting %s Hessian on D_train …",
            seed,
            exp.attribution.ekfac_method,
        )
        cfg = deepcopy(base_icfg)
        cfg.run_path = str(hessian_dir)
        cfg.projection_dim = 0  # Hessian factors are in full param space
        cfg.skip_preconditioners = False
        approximate_hessians(
            cfg, HessianConfig(method=hessian_method, ev_correction=ev_correction)
        )
        _mark(hessian_dir, hessian_cfg)
        gc.collect()

    hessian_method_path = str(hessian_dir / hessian_method)

    # An EK-FAC run must find the eigenvalue correction it was supposed to fit.
    # The shared hessian_dir is reused across methods and pre-copied from NFS,
    # where the last writer is usually a plain K-FAC run that leaves
    # eigenvalue_correction_sharded/ empty — EkfacWhitener would then silently
    # fall back to the uncorrected eigenvalues and produce K-FAC scores labelled
    # as EK-FAC. Fail here (seconds) rather than after a multi-hour scoring pass.
    if ev_correction and not exp.attribution.ekfac_modified_projections:
        corr = Path(hessian_method_path) / "eigenvalue_correction_sharded"
        if not list(corr.glob("shard_*.safetensors")):
            raise RuntimeError(
                f"EK-FAC requested but no eigenvalue correction shards under {corr}. "
                "The Hessian at this path is plain K-FAC (probably reused from the "
                "NFS cache); delete it and refit with ev_correction=True."
            )

    if exp.attribution.ekfac_modified_projections:
        # ── P-SIFT path ───────────────────────────────────────────────────
        from bergson.config import DistributedConfig

        projections_path = Path(hessian_method_path) / "projection_left_sharded"
        projections_right_path = Path(hessian_method_path) / "projection_right_sharded"
        proj_cfg = _ekfac_sketch_query_cfg(exp)

        # ── Step 2: Build M = R · cov^{-1/2} per side ────────────────────
        # Fingerprinted (not just existence-checked): M depends on
        # ekfac_lambda_damp/projection_dim, which can change while
        # ekfac_method stays the same — a bare .exists() check would
        # silently keep stale M built with the old damp/projection_dim.
        if not _up_to_date(projections_path, proj_cfg):
            _maybe_delete_stale(projections_path, proj_cfg, "P-SIFT projections")
            if projections_right_path.exists():
                shutil.rmtree(projections_right_path)
            logger.info(
                "[seed=%d][ekfac-sketch/P-SIFT] Step 2: building M = R · cov^{-1/2}",
                seed,
            )
            launch_distributed_run(
                "build_projections",
                build_projections_worker,
                [
                    EkfacConfig(
                        hessian_method_path=hessian_method_path,
                        gradient_path="",  # unused in build_projections_worker
                        run_path="",  # unused in build_projections_worker
                        lambda_damp_factor=exp.attribution.ekfac_lambda_damp,
                        projection_dim=exp.attribution.projection_dim,
                    )
                ],
                DistributedConfig(),
            )
            _mark(projections_path, proj_cfg)

        # ── Step 3: Query index with M applied at collection ──────────────
        if not _up_to_date(query_index_path, proj_cfg):
            _maybe_delete_stale(query_index_path, proj_cfg, "P-SIFT query index")
            logger.info(
                "[seed=%d][ekfac-sketch/P-SIFT] Step 3: building query index with M",
                seed,
            )
            cfg = deepcopy(base_icfg)
            cfg.run_path = str(query_index_path)
            cfg.data = DataConfig(dataset=str(query_texts_path), truncation=True)
            cfg.kfac_projection_path = hessian_method_path
            build(
                cfg,
                PreprocessConfig(
                    unit_normalize=exp.attribution.unit_norm, aggregation="none"
                ),
            )
            _mark(query_index_path, proj_cfg)
            gc.collect()

        # ── Step 4: Score training data on-the-fly with M at collection ───
        logger.info(
            "[seed=%d][ekfac-sketch/P-SIFT] Step 4: scoring training data on-the-fly",
            seed,
        )
        score_cfg = deepcopy(base_icfg)
        score_cfg.run_path = ""
        score_cfg.token_batch_size = (
            exp.attribution.ekfac_score_token_batch_size
            or exp.attribution.ekfac_token_batch_size
            or exp.model.max_length
        )
        score_cfg.kfac_projection_path = hessian_method_path
        train_ds = datasets.load_from_disk(str(train_path))
        queries_ds = datasets.load_from_disk(str(exp.queries_path()))
        _stream_score_and_process(
            exp,
            seed,
            score_cfg,
            ScoreConfig(query_path=str(query_index_path), score="individual"),
            PreprocessConfig(unit_normalize=exp.attribution.unit_norm),
            train_ds,
            queries_ds,
        )

    else:
        # ── Legacy whitener path (query batching) ─────────────────────────
        # ── Step 2: Whitened query gradient index ─────────────────────────
        if not _up_to_date(query_index_path, sketch_q_cfg):
            _maybe_delete_stale(query_index_path, sketch_q_cfg, "sketched query index")
            logger.info(
                "[seed=%d][ekfac-sketch] Step 2: building whitened query index", seed
            )
            cfg = deepcopy(base_icfg)
            cfg.run_path = str(query_index_path)
            cfg.data = DataConfig(dataset=str(query_texts_path), truncation=True)
            cfg.ekfac_whitener_path = hessian_method_path
            cfg.ekfac_whitener_damp = exp.attribution.ekfac_lambda_damp
            build(
                cfg,
                PreprocessConfig(
                    unit_normalize=exp.attribution.unit_norm, aggregation="none"
                ),
            )
            _mark(query_index_path, sketch_q_cfg)
            gc.collect()

        # ── Step 3+4: Score on-the-fly and compute all outputs ────────────
        logger.info(
            "[seed=%d][ekfac-sketch] Step 3/4: scoring training data on-the-fly", seed
        )
        score_cfg = deepcopy(base_icfg)
        score_cfg.run_path = ""
        score_cfg.token_batch_size = (
            exp.attribution.ekfac_score_token_batch_size
            or exp.attribution.ekfac_token_batch_size
            or exp.model.max_length
        )
        score_cfg.ekfac_whitener_path = hessian_method_path
        score_cfg.ekfac_whitener_damp = exp.attribution.ekfac_lambda_damp
        train_ds = datasets.load_from_disk(str(train_path))
        queries_ds = datasets.load_from_disk(str(exp.queries_path()))
        _stream_score_and_process(
            exp,
            seed,
            score_cfg,
            ScoreConfig(query_path=str(query_index_path), score="individual"),
            PreprocessConfig(unit_normalize=exp.attribution.unit_norm),
            train_ds,
            queries_ds,
        )


def _attribute_ekfac(exp: ExperimentCfg, seed: int) -> None:
    """EK-FAC pipeline.

    Per-pair mode (ekfac_aggregate_query=False, default):
        Builds a 2m-entry query index (aggregation='none'), applies H⁻¹ to
        each sentence gradient individually, then scores with score='individual'.
        Output: (N, 2m) score memmap → per-pair ΔI, same as vanilla/trackstar.
        Disk cost: 2m × full-param gradient size (~100 GB for m=100 on GPT-2).

    Aggregated mode (ekfac_aggregate_query=True):
        Collapses s⁺ and s⁻ each to one mean gradient, applies H⁻¹, scores
        twice, subtracts.  Output: global ΔI(z) only (pair_idx=0 for all rows).
        Disk cost: 2 × full-param gradient size (~1 GB on GPT-2).

    The Hessian is fitted once on D_train in both modes.
    """
    from bergson.config import DistributedConfig, HessianConfig
    from bergson.distributed import launch_distributed_run
    from bergson.hessians.apply_hessian import EkfacConfig, apply_worker
    from bergson.hessians.hessian_approximations import approximate_hessians

    if exp.attribution.ekfac_sketch:
        _attribute_ekfac_sketched(exp, seed)
        return

    aggregate = exp.attribution.ekfac_aggregate_query
    train_path = _get_scoring_train_path(exp)
    qgrad_cfg = _ekfac_query_grad_cfg(exp)
    hessian_cfg = _ekfac_hessian_cfg(exp)
    ivhp_cfg = _ekfac_ivhp_cfg(exp)
    scores_cfg = _ekfac_scores_cfg(exp)

    # Shared base IndexConfig (no random projection — EK-FAC needs full param space)
    #
    # Only Step 1 (the query-gradient index) is memory-critical here: it holds
    # full-parameter gradients per query sentence. The Hessian fit and the
    # scoring pass over D_train have the same memory profile as their sketched
    # counterparts and must not inherit Step 1's tiny batch — doing so made the
    # Hessian fit ~25x slower than the identical sketched fit. Each stage
    # therefore picks its own batch size below.
    base_icfg = IndexConfig(
        run_path="",
        model=str(exp.model_dir(seed)),
        token_batch_size=exp.attribution.ekfac_token_batch_size or exp.model.max_length,
        projection_dim=0,
        data=DataConfig(dataset=str(train_path), truncation=True),
        skip_preconditioners=True,
        filter_modules=exp.attribution.ekfac_filter_modules,
    )
    query_token_batch_size = (
        exp.attribution.ekfac_query_token_batch_size
        or exp.attribution.ekfac_token_batch_size
        or exp.model.max_length
    )
    score_token_batch_size = (
        exp.attribution.ekfac_score_token_batch_size
        or exp.attribution.ekfac_token_batch_size
        or exp.model.max_length
    )

    # ── Step 1: Build query gradient index ───────────────────────────────
    if aggregate:
        # Two 1-entry indices (mean per sign)
        plus_texts, minus_texts = _prepare_split_query_texts(exp)
        for sign, texts_path in [("plus", plus_texts), ("minus", minus_texts)]:
            out = exp.ekfac_query_agg_dir(seed, sign)
            if not _up_to_date(out, qgrad_cfg):
                _maybe_delete_stale(
                    out, qgrad_cfg, f"ekfac aggregated {sign} query gradient"
                )
                logger.info(
                    "[seed=%d][ekfac] Building aggregated %s query gradient", seed, sign
                )
                cfg = deepcopy(base_icfg)
                cfg.run_path = str(out)
                cfg.token_batch_size = query_token_batch_size
                cfg.data = DataConfig(dataset=str(texts_path), truncation=True)
                build(
                    cfg,
                    PreprocessConfig(
                        aggregation="mean",
                        unit_normalize=exp.attribution.unit_norm,
                    ),
                )
                _mark(out, qgrad_cfg)
    else:
        # Single 2m-entry index (one gradient per sentence)
        query_texts_path = _prepare_query_texts(exp)
        out = exp.query_grad_dir(seed)
        if not _up_to_date(out, qgrad_cfg):
            _maybe_delete_stale(
                out, qgrad_cfg, "ekfac per-sentence query gradient index"
            )
            logger.info(
                "[seed=%d][ekfac] Building per-sentence query gradient index (%d sentences)",
                seed,
                0,
            )  # count logged by build()
            cfg = deepcopy(base_icfg)
            cfg.run_path = str(out)
            cfg.token_batch_size = query_token_batch_size
            cfg.data = DataConfig(dataset=str(query_texts_path), truncation=True)
            build(
                cfg,
                PreprocessConfig(
                    aggregation="none",
                    unit_normalize=exp.attribution.unit_norm,
                ),
            )
            _mark(out, qgrad_cfg)

    hessian_method = (
        "kfac"
        if exp.attribution.ekfac_method == "ekfac"
        else exp.attribution.ekfac_method
    )

    # ── Step 2: Fit Hessian factors on D_train (once, shared by both modes) ──
    hessian_dir = exp.ekfac_hessian_dir(seed)
    if not _up_to_date(hessian_dir, hessian_cfg):
        _maybe_delete_stale(hessian_dir, hessian_cfg, "ekfac Hessian")
        logger.info(
            "[seed=%d][ekfac] Fitting %s Hessian on D_train …",
            seed,
            exp.attribution.ekfac_method,
        )
        cfg = deepcopy(base_icfg)
        cfg.run_path = str(hessian_dir)
        cfg.skip_preconditioners = False
        approximate_hessians(
            cfg,
            HessianConfig(
                method=hessian_method,
                ev_correction=exp.attribution.ekfac_method == "ekfac",
            ),
        )
        _mark(hessian_dir, hessian_cfg)

    hessian_method_path = str(hessian_dir / hessian_method)

    # ── Step 3: Apply H⁻¹ to query gradient(s) ───────────────────────────
    if aggregate:
        for sign in ("plus", "minus"):
            ivhp_dir = exp.ekfac_query_ivhp_dir(seed, sign)
            if not _up_to_date(ivhp_dir, ivhp_cfg):
                _maybe_delete_stale(ivhp_dir, ivhp_cfg, f"ekfac ivhp {sign}")
                logger.info("[seed=%d][ekfac] Applying H⁻¹ to %s query", seed, sign)
                launch_distributed_run(
                    "apply_hessian",
                    apply_worker,
                    [
                        EkfacConfig(
                            hessian_method_path=hessian_method_path,
                            gradient_path=str(exp.ekfac_query_agg_dir(seed, sign)),
                            run_path=str(ivhp_dir),
                            lambda_damp_factor=exp.attribution.ekfac_lambda_damp,
                            query_chunk_size=exp.attribution.ekfac_query_chunk_size,
                        )
                    ],
                    DistributedConfig(),
                )
                _mark(ivhp_dir, ivhp_cfg)
    else:
        ivhp_dir = exp.ekfac_ivhp_dir(seed)
        if not _up_to_date(ivhp_dir, ivhp_cfg):
            _maybe_delete_stale(ivhp_dir, ivhp_cfg, "ekfac per-sentence ivhp")
            logger.info(
                "[seed=%d][ekfac] Applying H⁻¹ to per-sentence query index", seed
            )
            launch_distributed_run(
                "apply_hessian",
                apply_worker,
                [
                    EkfacConfig(
                        hessian_method_path=hessian_method_path,
                        gradient_path=str(exp.query_grad_dir(seed)),
                        run_path=str(ivhp_dir),
                        lambda_damp_factor=exp.attribution.ekfac_lambda_damp,
                        query_chunk_size=exp.attribution.ekfac_query_chunk_size,
                    )
                ],
                DistributedConfig(),
            )
            _mark(ivhp_dir, ivhp_cfg)

    # ── Step 4: Score D_train against H⁻¹-transformed query ──────────────
    if aggregate:
        for sign in ("plus", "minus"):
            scores_dir = exp.ekfac_scores_dir(seed, sign)
            if not _up_to_date(scores_dir, scores_cfg):
                _maybe_delete_stale(scores_dir, scores_cfg, f"ekfac scores {sign}")
                logger.info("[seed=%d][ekfac] Scoring D_train vs %s query", seed, sign)
                cfg = deepcopy(base_icfg)
                cfg.run_path = str(scores_dir)
                cfg.token_batch_size = score_token_batch_size
                score_dataset(
                    cfg,
                    ScoreConfig(
                        query_path=str(exp.ekfac_query_ivhp_dir(seed, sign)),
                        score="individual",
                    ),
                    PreprocessConfig(unit_normalize=exp.attribution.unit_norm),
                )
                _mark(scores_dir, scores_cfg)
    else:
        scores_raw = exp.scores_raw_dir(seed)
        if not _up_to_date(scores_raw, scores_cfg):
            _maybe_delete_stale(scores_raw, scores_cfg, "ekfac raw scores")
            logger.info(
                "[seed=%d][ekfac] Scoring D_train vs per-sentence H⁻¹ query index", seed
            )
            cfg = deepcopy(base_icfg)
            cfg.run_path = str(scores_raw)
            cfg.token_batch_size = score_token_batch_size
            score_dataset(
                cfg,
                ScoreConfig(
                    query_path=str(exp.ekfac_ivhp_dir(seed)),
                    score="individual",
                    query_chunk_size=exp.attribution.ekfac_query_chunk_size,
                    query_low_rank=exp.attribution.ekfac_query_low_rank,
                ),
                PreprocessConfig(unit_normalize=exp.attribution.unit_norm),
            )
            _mark(scores_raw, scores_cfg)

    # ── Post-processing ───────────────────────────────────────────────────
    train_ds = datasets.load_from_disk(str(train_path))

    if aggregate:
        scores_plus = _load_single_score_col(exp.ekfac_scores_dir(seed, "plus"))
        scores_minus = _load_single_score_col(exp.ekfac_scores_dir(seed, "minus"))
        _process_global_delta_I(
            exp, seed, (scores_plus - scores_minus).astype(np.float32), train_ds
        )
    else:
        queries_ds = datasets.load_from_disk(str(exp.queries_path()))
        _process_scored_memmap(exp, seed, train_ds, queries_ds)


# ═══════════════════════════════════════════════════════════════════════════
# Per-fact per-source delta_I (BEAR facts) — unbiased, over all D_X items
# ═══════════════════════════════════════════════════════════════════════════

_ATOMIC_SRC_CATS = ("cooccur", "subj_occur", "obj_occur")


def _src_stats(vals: np.ndarray) -> dict:
    n = int(vals.size)
    if n == 0:
        nan = float("nan")
        return {
            "n": 0,
            "mean": nan,
            "median": nan,
            "std": nan,
            "ci95_lower": nan,
            "ci95_upper": nan,
        }
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n > 1 else float("nan")
    se = std / np.sqrt(n) if n > 1 and not np.isnan(std) else float("nan")
    margin = 1.96 * se if not np.isnan(se) else float("nan")
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(vals)),
        "std": std,
        "ci95_lower": float(mean - margin) if not np.isnan(margin) else float("nan"),
        "ci95_upper": float(mean + margin) if not np.isnan(margin) else float("nan"),
    }


def _write_per_fact_src_delta_I(
    exp: ExperimentCfg,
    seed: int,
    sg_dx: np.ndarray,
    sb_dx: np.ndarray,
    sg_base: np.ndarray,
    sb_base: np.ndarray,
    dx_indices: np.ndarray,
    fact_lk: "_FactLookups",
    queries_ds: datasets.Dataset,
    pp_cfg: dict,
) -> None:
    """Compute per-fact per-source ΔI over ALL D_X items (not just top-k) and save.

    sg_dx / sb_dx : (n_dx, num_pairs) — already unit-normalized if applicable.
    sg_base / sb_base : (n_base_sub, num_pairs) — same normalization.

    For each fact, for each atomic source (cooccur, subj_occur, obj_occur), computes
    ΔI = (sg − sb) over all items of that source for that fact, across all of the
    fact's query pair columns.  base stats use the full base subsample restricted to
    the same pair columns.
    """
    out = exp.delta_I_per_source_path(seed)
    if _up_to_date_file(out, pp_cfg):
        return

    fact_pair_cols: dict[str, list[int]] = {}
    for j, fk in enumerate(fact_lk.pair_to_fact):
        if fk:
            fact_pair_cols.setdefault(fk, []).append(j)

    if not fact_pair_cols:
        return

    # Local row index in sg_dx for each D_X item, grouped by (fact_key, source)
    dx_rows_by_fact_src: dict[str, dict[str, list[int]]] = {}
    for local_i, ti in enumerate(map(int, dx_indices)):
        for fk in fact_lk.idx_to_facts.get(ti, frozenset()):
            if fk not in fact_pair_cols:
                continue
            d = dx_rows_by_fact_src.setdefault(fk, {})
            for src in fact_lk.idx_to_sources.get(ti, frozenset()):
                d.setdefault(src, []).append(local_i)

    # Global std of all raw ℐ(s, z) scores — normalization denominator so that
    # normalized ΔI values are comparable across attribution methods.  Matches the
    # score_std the analyze step derives from delta_I_stats.npz.
    all_scores = np.concatenate(
        [
            sg_dx.reshape(-1),
            sb_dx.reshape(-1),
            sg_base.reshape(-1),
            sb_base.reshape(-1),
        ]
    )
    score_std = float(all_scores.std(ddof=1)) if all_scores.size > 1 else float("nan")

    # Raw ΔI values per atomic source (+ base), retained so the analyze step can
    # pool exact statistics (incl. median) over arbitrary fact subsets (verdict
    # splits) without the top-k selection bias of the inspect parquets.
    fact_key_list: list[str] = sorted(fact_pair_cols)
    raw_by_src: dict[str, list[np.ndarray]] = {src: [] for src in _ATOMIC_SRC_CATS}
    fid_by_src: dict[str, list[np.ndarray]] = {src: [] for src in _ATOMIC_SRC_CATS}
    base_raw: list[np.ndarray] = []
    base_fid: list[np.ndarray] = []

    per_fact: list[dict] = []
    for fi, fk in enumerate(fact_key_list):
        pair_js = np.array(fact_pair_cols[fk], dtype=np.int32)
        entry: dict = {"fact_key": fk, "n_pairs": int(len(pair_js))}
        fk_rows = dx_rows_by_fact_src.get(fk, {})
        for src in _ATOMIC_SRC_CATS:
            rows = fk_rows.get(src, [])
            if rows:
                rows_arr = np.array(rows, dtype=np.int32)
                dI = (sg_dx[rows_arr][:, pair_js] - sb_dx[rows_arr][:, pair_js]).ravel()
            else:
                dI = np.empty(0, dtype=np.float32)
            entry[src] = _src_stats(dI)
            if dI.size:
                raw_by_src[src].append(dI.astype(np.float32))
                fid_by_src[src].append(np.full(dI.size, fi, dtype=np.int32))
        dI_base = (sg_base[:, pair_js] - sb_base[:, pair_js]).ravel()
        entry["base"] = _src_stats(dI_base)
        if dI_base.size:
            base_raw.append(dI_base.astype(np.float32))
            base_fid.append(np.full(dI_base.size, fi, dtype=np.int32))
        per_fact.append(entry)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(
            {
                "method": exp.attribution.method,
                "seed": seed,
                "n_facts": len(per_fact),
                "n_base_sub": int(sg_base.shape[0]),
                "atomic_src_cats": list(_ATOMIC_SRC_CATS),
                "score_std": score_std,
                "per_fact": per_fact,
            },
            fh,
            indent=2,
        )

    def _cat(arrs: list[np.ndarray], dtype) -> np.ndarray:
        return np.concatenate(arrs) if arrs else np.empty(0, dtype=dtype)

    npz_out = out.with_suffix(".npz")
    savez_kwargs: dict[str, np.ndarray] = {
        "fact_keys": np.array(fact_key_list),
        "score_std": np.array(score_std, dtype=np.float32),
        "base_vals": _cat(base_raw, np.float32),
        "base_fact_ids": _cat(base_fid, np.int32),
    }
    for src in _ATOMIC_SRC_CATS:
        savez_kwargs[f"{src}_vals"] = _cat(raw_by_src[src], np.float32)
        savez_kwargs[f"{src}_fact_ids"] = _cat(fid_by_src[src], np.int32)
    np.savez(str(npz_out), **savez_kwargs)

    _mark_file(out, pp_cfg)
    logger.info(
        "[seed=%d] Per-fact per-source ΔI → %s (+ raw %s, %d facts, σ_scores=%.6g)",
        seed,
        out,
        npz_out.name,
        len(per_fact),
        score_std,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Method: bm25
# ═══════════════════════════════════════════════════════════════════════════

_BM25_PAIR_CHUNK = 1024  # query pairs scored per batch — bounds transient RAM
# to ~chunk × N × 4 B × 2 (e.g. ~29 GB at N≈7M). Each pair is still scored via
# bm25.get_scores() exactly once (same cost as the original single-pass
# design); only the (N, m) matrices that used to get materialized/persisted
# per consumer are gone. At Wikipedia scale those were ~191 GB each — writing
# both to /scratch overran it and killed the job with a SIGBUS mid-scoring.


def _attribute_bm25(exp: ExperimentCfg, seed: int) -> None:
    """BM25 retrieval baseline (Okapi BM25).

    Scores each training document against every query pair using BM25 built
    over the training corpus, in batches of _BM25_PAIR_CHUNK pairs. Each pair
    is scored exactly once and immediately folded into every consumer — ΔI
    stats, top-inspect, Prec@k (combined + learned/not-learned splits + exact-
    country), and per-fact-per-source ΔI — instead of persisting the full
    (N, m) score matrices to disk.

    ΔI(z, j) = BM25(s⁺_j, z) − BM25(s⁻_j, z)

    Output: delta_I_stats.npz, top_inspect*.parquet, prec_at_k.json (+ learned
    split and per-fact-per-source outputs where configured).
    """
    import bm25s as bm25s_lib

    train_path = _get_scoring_train_path(exp)
    train_ds = datasets.load_from_disk(str(train_path))
    queries_ds = datasets.load_from_disk(str(exp.queries_path()))

    n = len(train_ds)
    m = len(queries_ds)

    stats_out = exp.delta_I_stats_path(seed)
    inspect_out = exp.top_inspect_path(seed)
    inspect_good_out = exp.top_inspect_good_path(seed)
    inspect_bad_out = exp.top_inspect_bad_path(seed)
    prec_out = exp.prec_at_k_path(seed)
    pp_cfg = _postproc_cfg(exp, seed)

    cats_arr = np.array(train_ds["category"])
    dx_indices = np.where(cats_arr == "X")[0]
    base_indices = np.where(cats_arr == "base")[0]
    labels = _load_pair_labels(exp, queries_ds)
    fact_lk = _build_fact_lookups(train_ds, queries_ds)
    view_groups = labels.view_groups() if labels is not None else {}
    split_targets = {label: labels.path_for(exp, seed, label) for label in view_groups}
    ps_out = exp.delta_I_per_source_path(seed) if fact_lk is not None else None

    _write_lookups(exp, train_ds, queries_ds)

    needs_stats = not _up_to_date_file(stats_out, pp_cfg)
    needs_inspect = not _up_to_date_file(inspect_out, pp_cfg)
    needs_inspect_good = not _up_to_date_file(inspect_good_out, pp_cfg)
    needs_inspect_bad = not _up_to_date_file(inspect_bad_out, pp_cfg)
    needs_prec = not _up_to_date_file(prec_out, pp_cfg)
    needs_split = {
        label: not (_up_to_date_file(sp, pp_cfg) and _up_to_date_file(pp, pp_cfg))
        for label, (sp, pp) in split_targets.items()
    }
    needs_ps = ps_out is not None and not _up_to_date_file(ps_out, pp_cfg)

    if not (
        needs_stats
        or needs_inspect
        or needs_inspect_good
        or needs_inspect_bad
        or needs_prec
        or any(needs_split.values())
        or needs_ps
    ):
        logger.info(
            "[seed=%d][bm25] All attribution outputs up to date, skipping.", seed
        )
        return

    logger.info(
        "[seed=%d][bm25] D_X=%d  D_base=%d  pairs=%d",
        seed,
        len(dx_indices),
        len(base_indices),
        m,
    )

    logger.info("Building BM25 index over %d training documents …", n)
    corpus_tokens = bm25s_lib.tokenize(
        train_ds["text"], lower=True, stopwords=None, show_progress=False
    )
    bm25 = bm25s_lib.BM25()
    bm25.index(corpus_tokens, show_progress=False)
    del corpus_tokens
    gc.collect()

    good_texts = [str(ex["text_good"]) for ex in queries_ds]
    bad_texts = [str(ex["text_bad"]) for ex in queries_ds]
    logger.info("Tokenizing %d query pairs …", m)
    good_tokens = bm25s_lib.tokenize(
        good_texts, lower=True, stopwords=None, show_progress=False
    )
    bad_tokens = bm25s_lib.tokenize(
        bad_texts, lower=True, stopwords=None, show_progress=False
    )
    good_str = bm25s_lib.tokenization.convert_tokenized_to_string_list(good_tokens)
    bad_str = bm25s_lib.tokenization.convert_tokenized_to_string_list(bad_tokens)
    del good_tokens, bad_tokens

    # ── Static (score-independent) structures ──────────────────────────────
    n_avail = len(base_indices)
    n_sub = (
        exp.attribution.stats_base_size
        if exp.attribution.stats_base_size > 0
        else n_avail
    )
    n_sub = min(n_sub, n_avail)
    if n_sub < n_avail:
        rng = np.random.default_rng(_STATS_SUBSAMPLE_SEED)
        base_sub = np.sort(rng.choice(base_indices, n_sub, replace=False))
    else:
        base_sub = base_indices

    pool_indices, prec_mode = _prec_pool_indices(
        dx_indices, base_indices, exp.attribution.prec_base_size
    )
    is_dx_full = (cats_arr == "X").astype(np.int8)
    is_dx_pool = is_dx_full[pool_indices] if pool_indices is not None else None

    country_meta = None
    if exp.data.experiment_type == "facts":
        country_meta = _load_facts_country_metadata(
            Path(exp.data.data_dir), train_ds, m
        )
    if country_meta is not None:
        train_countries, query_countries = country_meta
        tc_arr = np.array(train_countries, dtype=object)
        sub_tc = tc_arr[pool_indices] if pool_indices is not None else tc_arr
    else:
        query_countries = sub_tc = None

    unit_norm = exp.attribution.unit_norm
    k_values = exp.attribution.k_precisions
    max_k = max(k_values)
    inspect_top_k = exp.attribution.inspect_top_k

    label_mask = {label: np.zeros(m, dtype=bool) for label in view_groups}
    for label, pair_js in view_groups.items():
        label_mask[label][pair_js] = True

    # ── RAM accumulators (replace the old (N, m) disk memmaps) ─────────────
    n_dx, n_base = len(dx_indices), len(base_sub)
    dx_good = np.empty((n_dx, m), dtype=np.float32)
    dx_bad = np.empty((n_dx, m), dtype=np.float32)
    base_good = np.empty((n_base, m), dtype=np.float32)
    base_bad = np.empty((n_base, m), dtype=np.float32)
    col_norms_good = np.empty(m, dtype=np.float32) if unit_norm else None
    col_norms_bad = np.empty(m, dtype=np.float32) if unit_norm else None

    if pool_indices is not None:
        pool_good = np.empty((len(pool_indices), m), dtype=np.float32)
        pool_bad = np.empty((len(pool_indices), m), dtype=np.float32)
    else:
        pool_good = pool_bad = None

    per_k = {str(k): [] for k in k_values}
    per_k_good = {str(k): [] for k in k_values}
    per_k_bad = {str(k): [] for k in k_values}
    per_k_split = {label: {str(k): [] for k in k_values} for label in view_groups}
    per_k_good_split = {label: {str(k): [] for k in k_values} for label in view_groups}
    per_k_bad_split = {label: {str(k): [] for k in k_values} for label in view_groups}
    per_k_exact = {str(k): [] for k in k_values} if country_meta is not None else None

    inspect_needed = needs_inspect or needs_inspect_good or needs_inspect_bad
    inspect_rows: dict[str, dict[str, list]] = {
        side: {"pair_idx": [], "train_idx": [], "rank": [], "scores": []}
        for side in ("contrast", "good", "bad")
    }
    idx_all = np.arange(n, dtype=np.int32)
    idx_pool = pool_indices.astype(np.int32) if pool_indices is not None else None

    def _prec_at_ks(scores: np.ndarray, is_dx: np.ndarray, per_k_dict: dict) -> None:
        actual_max_k = min(max_k, len(scores))
        top_idx = np.argpartition(scores, -actual_max_k)[-actual_max_k:]
        order = np.argsort(-scores[top_idx])
        cum_isdx = np.cumsum(is_dx[top_idx[order]])
        for k in k_values:
            actual_k = min(k, len(scores))
            per_k_dict[str(k)].append(float(cum_isdx[actual_k - 1]) / actual_k)

    def _accumulate_inspect(
        side: str, sc: np.ndarray, global_idx: np.ndarray, j: int
    ) -> None:
        acc = inspect_rows[side]
        kk = min(inspect_top_k, len(sc))
        if side == "bad":
            local = np.argpartition(sc, kk)[:kk]
            local = local[np.argsort(sc[local])]
        else:
            local = np.argpartition(sc, -kk)[-kk:]
            local = local[np.argsort(-sc[local])]
        acc["pair_idx"].append(np.full(kk, j, dtype=np.int32))
        acc["train_idx"].append(global_idx[local].astype(np.int32))
        acc["rank"].append(np.arange(kk, dtype=np.int16))
        acc["scores"].append(sc[local].astype(np.float32))

    def _process_column(
        j: int,
        sg_n: np.ndarray,
        sb_n: np.ndarray,
        is_dx: np.ndarray,
        global_idx: np.ndarray,
        sub_tc_arr: np.ndarray | None,
    ) -> None:
        """Fold one (already-normalized) column into Prec@k / top-inspect for
        the combined view and any learned/not-learned split it belongs to."""
        dI_j = sg_n - sb_n
        _prec_at_ks(dI_j, is_dx, per_k)
        _prec_at_ks(sg_n, is_dx, per_k_good)
        _prec_at_ks(-sb_n, is_dx, per_k_bad)
        for label, mask in label_mask.items():
            if mask[j]:
                _prec_at_ks(dI_j, is_dx, per_k_split[label])
                _prec_at_ks(sg_n, is_dx, per_k_good_split[label])
                _prec_at_ks(-sb_n, is_dx, per_k_bad_split[label])
        if per_k_exact is not None:
            is_exact = (sub_tc_arr == query_countries[j]).astype(np.int8)
            for k in k_values:
                actual_k = min(k, len(dI_j))
                top = np.argpartition(dI_j, -actual_k)[-actual_k:]
                per_k_exact[str(k)].append(float(is_exact[top].sum()) / actual_k)
        if inspect_needed:
            if needs_inspect:
                _accumulate_inspect("contrast", dI_j, global_idx, j)
            if needs_inspect_good:
                _accumulate_inspect("good", sg_n, global_idx, j)
            if needs_inspect_bad:
                _accumulate_inspect("bad", sb_n, global_idx, j)

    _zero_col = np.zeros(n, dtype=np.float32)
    logger.info(
        "Scoring %d query pairs against %d training docs (batches of %d) …",
        m,
        n,
        _BM25_PAIR_CHUNK,
    )
    for c0 in range(0, m, _BM25_PAIR_CHUNK):
        c1 = min(c0 + _BM25_PAIR_CHUNK, m)
        width = c1 - c0
        logger.info("  BM25 pairs %d-%d / %d", c0, c1, m)

        sg_chunk = np.empty((n, width), dtype=np.float32)
        sb_chunk = np.empty((n, width), dtype=np.float32)
        for pos, j in enumerate(range(c0, c1)):
            sg_chunk[:, pos] = (
                bm25.get_scores(good_str[j]) if good_str[j] else _zero_col
            )
            sb_chunk[:, pos] = bm25.get_scores(bad_str[j]) if bad_str[j] else _zero_col

        if unit_norm:
            ng = np.linalg.norm(sg_chunk, axis=0)
            nb = np.linalg.norm(sb_chunk, axis=0)
            ng = np.where(ng < 1e-12, 1.0, ng)
            nb = np.where(nb < 1e-12, 1.0, nb)
            col_norms_good[c0:c1] = ng
            col_norms_bad[c0:c1] = nb
            sg_chunk /= ng[np.newaxis, :]
            sb_chunk /= nb[np.newaxis, :]

        dx_good[:, c0:c1] = sg_chunk[dx_indices]
        dx_bad[:, c0:c1] = sb_chunk[dx_indices]
        base_good[:, c0:c1] = sg_chunk[base_sub]
        base_bad[:, c0:c1] = sb_chunk[base_sub]

        if pool_indices is not None:
            pool_good[:, c0:c1] = sg_chunk[pool_indices]
            pool_bad[:, c0:c1] = sb_chunk[pool_indices]
        else:
            for pos, j in enumerate(range(c0, c1)):
                _process_column(
                    j, sg_chunk[:, pos], sb_chunk[:, pos], is_dx_full, idx_all, sub_tc
                )

        del sg_chunk, sb_chunk
        gc.collect()

    del bm25
    gc.collect()

    if pool_indices is not None:
        for j in range(m):
            _process_column(
                j, pool_good[:, j], pool_bad[:, j], is_dx_pool, idx_pool, sub_tc
            )

    # ── Write outputs ────────────────────────────────────────────────────
    if needs_stats:
        dx_dI = (dx_good - dx_bad).flatten()
        base_dI = (base_good - base_bad).flatten()
        all_scores = np.concatenate(
            [
                dx_good.flatten(),
                dx_bad.flatten(),
                base_good.flatten(),
                base_bad.flatten(),
            ]
        )
        score_n = int(all_scores.size)
        score_mean = float(all_scores.mean())
        score_M2 = float(all_scores.var(ddof=0)) * score_n
        _write_delta_I_stats(
            stats_out, dx_dI, base_dI, score_n, score_mean, score_M2, m
        )
        _mark_file(stats_out, pp_cfg)
        logger.info(
            "[seed=%d][bm25] Stats → %s (%d D_X × %d pairs, %d base × %d pairs)",
            seed,
            stats_out,
            n_dx,
            m,
            n_base,
            m,
        )
        del dx_dI, base_dI, all_scores
        gc.collect()

    if inspect_needed:
        cols = queries_ds.column_names
        features = queries_ds["feature"] if "feature" in cols else [""] * m
        score_col_name = {
            "contrast": "delta_I",
            "good": "score_good",
            "bad": "score_bad",
        }
        for path, side, needed in [
            (inspect_out, "contrast", needs_inspect),
            (inspect_good_out, "good", needs_inspect_good),
            (inspect_bad_out, "bad", needs_inspect_bad),
        ]:
            if not needed:
                continue
            acc = inspect_rows[side]
            pair_idx_arr = (
                np.concatenate(acc["pair_idx"])
                if acc["pair_idx"]
                else np.empty(0, dtype=np.int32)
            )
            train_idx_arr = (
                np.concatenate(acc["train_idx"])
                if acc["train_idx"]
                else np.empty(0, dtype=np.int32)
            )
            ranks_arr = (
                np.concatenate(acc["rank"])
                if acc["rank"]
                else np.empty(0, dtype=np.int16)
            )
            scores_arr = (
                np.concatenate(acc["scores"])
                if acc["scores"]
                else np.empty(0, dtype=np.float32)
            )
            feats_out = [str(features[j]) for j in pair_idx_arr]
            texts = [str(train_ds[int(ti)]["text"]) for ti in train_idx_arr]
            if fact_lk is not None:
                cats_final = [
                    _fact_category(int(ti), int(pi), txt, fact_lk)
                    for ti, pi, txt in zip(train_idx_arr, pair_idx_arr, texts)
                ]
            else:
                cats_final = cats_arr[train_idx_arr].tolist()
            columns: dict[str, object] = {
                "pair_idx": pair_idx_arr,
                "feature": feats_out,
                "rank": ranks_arr,
                "train_idx": train_idx_arr,
                "category": cats_final,
                score_col_name[side]: scores_arr,
                "text": texts,
            }
            _inject_query_label_columns(
                columns, pair_idx_arr, queries_ds, labels, side=side
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns).to_parquet(str(path), index=False, compression="zstd")
            _mark_file(path, pp_cfg)
            logger.info(
                "[seed=%d][bm25] Top-%d/pair inspect (%s) → %s",
                seed,
                inspect_top_k,
                side,
                path,
            )

    if needs_prec:
        prec = {k: float(np.mean(v)) for k, v in per_k.items()}
        prec_good = {k: float(np.mean(v)) for k, v in per_k_good.items()}
        prec_bad = {k: float(np.mean(v)) for k, v in per_k_bad.items()}
        result: dict = {
            "seed": seed,
            "condition": exp.data.condition,
            "mode": prec_mode,
            "prec_at_k": prec,
            "prec_at_k_good": prec_good,
            "prec_at_k_bad": prec_bad,
        }
        if per_k_exact is not None:
            prec_exact = {k: float(np.mean(v)) for k, v in per_k_exact.items()}
            result["prec_at_k_exact"] = prec_exact
            logger.info(
                "[seed=%d][bm25] Exact Prec@k: %s",
                seed,
                "  ".join(f"Prec@{k}={v:.3f}" for k, v in prec_exact.items()),
            )
        prec_out.parent.mkdir(parents=True, exist_ok=True)
        with open(prec_out, "w") as fh:
            json.dump(result, fh, indent=2)
        _mark_file(prec_out, pp_cfg)
        logger.info(
            "[seed=%d][bm25] Prec@k (%s)\n  contrast:  %s\n  good (s⁺): %s\n  bad  (s⁻): %s",
            seed,
            prec_mode,
            "  ".join(f"P@{k}={v:.7f}" for k, v in prec.items()),
            "  ".join(f"P@{k}={v:.7f}" for k, v in prec_good.items()),
            "  ".join(f"P@{k}={v:.7f}" for k, v in prec_bad.items()),
        )

    for label, pair_js in view_groups.items():
        if len(pair_js) == 0:
            logger.info(
                "[seed=%d][bm25/%s] 0 pairs in this split — skipping.", seed, label
            )
            continue
        stats_split, prec_split = split_targets[label]
        if not _up_to_date_file(stats_split, pp_cfg):
            dI_dx = (dx_good[:, pair_js] - dx_bad[:, pair_js]).flatten()
            dI_base = (base_good[:, pair_js] - base_bad[:, pair_js]).flatten()
            all_s = np.concatenate(
                [
                    dx_good[:, pair_js].flatten(),
                    dx_bad[:, pair_js].flatten(),
                    base_good[:, pair_js].flatten(),
                    base_bad[:, pair_js].flatten(),
                ]
            )
            score_n = int(all_s.size)
            score_mean = float(all_s.mean())
            score_M2 = float(all_s.var(ddof=0)) * score_n
            _write_delta_I_stats(
                stats_split, dI_dx, dI_base, score_n, score_mean, score_M2, len(pair_js)
            )
            _mark_file(stats_split, pp_cfg)
            logger.info(
                "[seed=%d][bm25/%s] Stats → %s (%d D_X × %d pairs, %d base × %d pairs)",
                seed,
                label,
                stats_split,
                n_dx,
                len(pair_js),
                n_base,
                len(pair_js),
            )
        if not _up_to_date_file(prec_split, pp_cfg):
            prec = {k: float(np.mean(v)) for k, v in per_k_split[label].items()}
            prec_good = {
                k: float(np.mean(v)) for k, v in per_k_good_split[label].items()
            }
            prec_bad = {k: float(np.mean(v)) for k, v in per_k_bad_split[label].items()}
            result = {
                "seed": seed,
                "condition": exp.data.condition,
                "mode": prec_mode,
                "prec_at_k": prec,
                "prec_at_k_good": prec_good,
                "prec_at_k_bad": prec_bad,
            }
            prec_split.parent.mkdir(parents=True, exist_ok=True)
            with open(prec_split, "w") as fh:
                json.dump(result, fh, indent=2)
            _mark_file(prec_split, pp_cfg)
            logger.info(
                "[seed=%d][bm25/%s] Prec@k (%s)\n  contrast:  %s\n  good (s⁺): %s\n  bad  (s⁻): %s",
                seed,
                label,
                prec_mode,
                "  ".join(f"P@{k}={v:.7f}" for k, v in prec.items()),
                "  ".join(f"P@{k}={v:.7f}" for k, v in prec_good.items()),
                "  ".join(f"P@{k}={v:.7f}" for k, v in prec_bad.items()),
            )

    # ── Per-fact per-source delta_I (BEAR facts only) ────────────────────
    if fact_lk is not None:
        _write_per_fact_src_delta_I(
            exp,
            seed,
            dx_good,
            dx_bad,
            base_good,
            base_bad,
            dx_indices,
            fact_lk,
            queries_ds,
            pp_cfg,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════


def _uses_scored_memmap(exp: ExperimentCfg) -> bool:
    """True when the method routes through _process_scored_memmap (Family A)."""
    a = exp.attribution
    return a.method in ("vanilla", "trackstar") or (
        a.method == "ekfac" and (a.ekfac_sketch or not a.ekfac_aggregate_query)
    )


def run(exp: ExperimentCfg, seed: int) -> None:
    """Run Step 2 attribution for one seed using the configured method."""
    stats_out = exp.delta_I_stats_path(seed)
    inspect_out = exp.top_inspect_path(seed)
    prec_out = exp.prec_at_k_path(seed)

    pp_cfg = _postproc_cfg(exp, seed)
    up_to_date = (
        _up_to_date_file(stats_out, pp_cfg)
        and _up_to_date_file(inspect_out, pp_cfg)
        and _up_to_date_file(prec_out, pp_cfg)
    )
    if up_to_date and (_uses_scored_memmap(exp) or exp.attribution.method == "bm25"):
        up_to_date = _up_to_date_file(
            exp.top_inspect_good_path(seed), pp_cfg
        ) and _up_to_date_file(exp.top_inspect_bad_path(seed), pp_cfg)
    if up_to_date:
        logger.info("[seed=%d] All attribution outputs up to date, skipping.", seed)
        return

    if exp.attribution.use_tf32_matmuls:
        torch.set_float32_matmul_precision("high")

    method = exp.attribution.method
    logger.info("[seed=%d] Attribution method: %s", seed, method)

    dispatch = {
        "vanilla": _attribute_vanilla,
        "trackstar": _attribute_trackstar,
        "ekfac": _attribute_ekfac,
        "bm25": _attribute_bm25,
    }

    if method not in dispatch:
        raise ValueError(
            f"Unknown attribution method: {method!r}. Choose from {list(dispatch)}"
        )

    dispatch[method](exp, seed)
