from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

from simple_parsing.helpers import Serializable

LEARNED_THRESHOLD = 0.5
"""softmax_prob_good threshold above which a query pair counts as 'learned' for
the learned/not-learned split and top_inspect filenames below. Mirrors
probe_models.py's SOFTMAX_PROB_LEARNED_THRESHOLD."""


@dataclass
class DataCfg(Serializable):
    """Experiment-level data settings.

    Expected folder structure under data_dir:
        data_training/base/*.txt       → D_base
        data_training/X/*.txt          → D_X
        data_test/X/good/*.txt         → s+ minimal pairs
        data_test/X/bad/*.txt          → s- minimal pairs
        data_test/base/*.txt           → base test samples (PPL sanity check)
    """

    data_dir: Path = Path("")
    """Root directory containing data_training/ and data_test/."""

    max_samples_base: int = 0
    """Cap on D_base examples used for training. 0 = use all available."""

    max_samples_X: int = 0
    """Cap on D_X examples. 0 = use all (D_X is small, ~919 sentences)."""

    oversample_X: bool = False
    """If True, repeat D_X in full rounds to reach exactly max_samples_X.
    Requires max_samples_X > 0 (raises ValueError otherwise)."""

    shuffle_within_blocks: bool = True
    """Deterministically shuffle rows within D_base and D_X before mixing."""

    condition: Literal["full", "base_only", ""] = ""
    """Training condition:
      full       — train on D_base ∪ D_X  (f_θ)
      base_only  — train on D_base only   (f_θ⁻)
    """

    mixing_strategy: Literal["start", "end", "middle", "full_shuffle", ""] = ""
    """How to place D_X within D_base when ``condition='full'``.

      start         — D_X followed by D_base
      end           — D_base followed by D_X
      middle        — D_X inserted as one contiguous block halfway through D_base
      full_shuffle  — randomly interleave D_X into D_base with the experiment
                      seed, while preserving the original order of D_base

    Ignored when ``condition='base_only'``.
    """

    def __post_init__(self) -> None:
        if self.oversample_X and self.max_samples_X == 0:
            raise ValueError("oversample_X=True requires max_samples_X > 0")

    experiment_type: Literal["linguistic", "facts"] = "linguistic"
    """Experiment type, governs evaluation logic:
      linguistic — BLiMP-style minimal pairs (grammatical vs. ungrammatical).
                   Reports PPL, full-sentence, one-prefix, two-prefix accuracy.
      facts      — Cloze-style fact queries (correct vs. wrong capital).
                   Same scoring metrics plus top-1 generation accuracy:
                   given the shared prefix, does argmax give the correct token?
    """


@dataclass
class ModelCfg(Serializable):
    """Language model specification."""

    name: str = "gpt2"
    revision: str = "main"
    max_length: int = 256
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"


@dataclass
class AttributionCfg(Serializable):
    """Bergson influence function settings."""

    method: Literal["vanilla", "trackstar", "ekfac", "bm25"] = "vanilla"
    """Attribution method.

      vanilla    — projected gradient cosine similarity (random projection).
                   Optionally normalised by Adam's exp_avg_sq moments when
                   use_optimizer_state=True.  Gives per-pair ΔI.
      trackstar  — TrackStar (Chang et al., 2024): projected gradients with a
                   curvature-aware mixed preconditioner.  Gives per-pair ΔI.
      ekfac      — EK-FAC preconditioned influence.  Runs the full Hessian
                   pipeline twice (once with aggregated s⁺ query, once with
                   aggregated s⁻ query) and subtracts.  Gives *global* ΔI
                   (no per-pair resolution; pair_idx=0 in output).
      bm25       — BM25 retrieval baseline (Okapi BM25).  Model-independent:
                   scores are shared across seeds and computed once.
                   Gives per-pair ΔI(z, j) = BM25(s⁺_j, z) − BM25(s⁻_j, z).
    """

    # ── shared: unit normalization (all methods) ──────────────────────────
    unit_norm: bool = True
    """Normalize scores so that methods are on comparable scales.

    How this is applied per method:
      vanilla, trackstar, ekfac (sketched & non-sketched)
        → bergson PreprocessConfig.unit_normalize=True on query-index build
          and scoring, giving (preconditioner-weighted) cosine-similarity scores.
              independently before subtracting to form ΔI(z).
      bm25  → post-hoc: L2-normalize each query column of score_good/score_bad
              (length N) before computing ΔI(z, j).
    False means raw (dot-product / BM25) scores with no normalization."""

    projection_dim: int = 16
    """Random-projection dimension (vanilla & trackstar only)."""

    use_optimizer_state: bool = False
    """vanilla only: normalise gradients by Adam's exp_avg_sq moments."""

    # ── trackstar ─────────────────────────────────────────────────────────
    trackstar_downweight_components: int = 1000
    """TrackStar: number of components to downweight in the mixed preconditioner
    (§A.1.3 of Chang et al., 2024).  Typical value ~1000 out of ~65k total."""

    # ── ekfac ─────────────────────────────────────────────────────────────
    ekfac_query_chunk_size: int = 0
    """Split the EK-FAC query IVHP/scoring path into chunks of this many
    queries.  Keeps peak RAM at roughly chunk_size × largest_module_size
    during IVHP application and chunk_size × param_size during full-gradient
    scoring.  0 = process all queries at once.  Typical value: 16."""

    ekfac_query_low_rank: int = 0
    """Approximate each preconditioned query gradient matrix with a truncated
    SVD of this rank before scoring.  Compresses query memory by ~O*I/(r*(O+I))
    per module (e.g. ~19× for GPT-2 MLP layers at rank 32).  Enables fitting
    all queries on the GPU during scoring, avoiding multi-pass scoring.
    IVHP application may still need ekfac_query_chunk_size.
    0 = store full query gradients (default).  Typical value: 32."""

    token_batch_size: int = 0
    """Token batch size for vanilla and trackstar gradient collection.
    allocate_batches packs sequences so that max_len_in_batch * n_seqs <=
    token_batch_size — sequences are still truncated to model.max_length.
    Larger values mean more sequences per GPU pass and much better utilisation.
    0 = use model.max_length (one seq/pass for full-length sequences).
    Try 4096–16384 for speed on modern GPUs."""

    trackstar_score_token_batch_size: int = 0
    """Token batch size for trackstar's Step 5/5 scoring pass only.  Step 5
    does strictly more per-example GPU work than Step 1 (applies the mixed
    preconditioner and scores against every query gradient column), so it
    can need a smaller batch than token_batch_size to avoid OOM.
    0 = fall back to token_batch_size."""

    max_batch_size: int = 0
    """Cap on the number of documents packed into a single vanilla gradient-
    collection batch (query index build + train scoring), independent of
    token_batch_size.  allocate_batches packs by token budget alone, so a
    batch of many short documents (e.g. single sentences) can pack far more
    documents than a batch of few long ones, even under the same token
    budget — this bounds that per-example collection cost directly.
    0 = no cap."""

    use_tf32_matmuls: bool = False
    """Enable TF32 matrix multiplications (Ampere/Ada Lovelace GPUs).
    ~2–3× matmul throughput with negligible effect on influence rankings."""

    ekfac_token_batch_size: int = 0
    """Token batch size for EK-FAC gradient collection passes.  Controls how
    many tokens are packed into a single forward/backward — fewer tokens means
    lower peak GPU memory at the cost of more iterations.
    0 = use model.max_length (default).  Try 128 or 64 if OOM."""

    ekfac_score_token_batch_size: int = 0
    """Token batch size for the ekfac scoring pass only (Step 3/4 of
    _attribute_ekfac_sketched, Step 4 of _attribute_ekfac).  The scoring pass has
    lower peak GPU memory than Hessian fitting because it does not accumulate
    Kronecker factors, so it can safely use a larger batch than
    ekfac_token_batch_size.
    0 = fall back to ekfac_token_batch_size.  Try 8192–16384 for a 4–8× speed-
    up over a conservative ekfac_token_batch_size without risking OOM during
    Hessian fitting."""

    ekfac_query_token_batch_size: int = 0
    """Token batch size for the query-gradient index build in the query-batching
    pipeline (Step 1 of _attribute_ekfac) only.

    That step is the memory-critical one: it materialises *full-parameter*
    gradients (projection_dim=0) for each query sentence, so it needs a far
    smaller batch than the Hessian fit or the scoring pass over D_train.
    Previously the same conservative value was pushed through
    ekfac_token_batch_size and therefore applied to every stage, making the
    Hessian fit ~25x slower than the identical fit in the sketched pipeline
    (41min at 4096 vs 11h+ at 64, measured on goldfish_10b binding).
    0 = fall back to ekfac_token_batch_size."""

    ekfac_method: Literal["kfac", "ekfac", "tkfac", "shampoo"] = "ekfac"
    """Hessian approximation backend for the EK-FAC pipeline.

    ``ekfac`` — Bergson's ``kfac`` factors with eigenvalue correction enabled.
    ``kfac``  — plain K-FAC without eigenvalue correction; required for
                ``ekfac_modified_projections=True`` (the P-SIFT path).
    ``tkfac`` / ``shampoo`` — alternative Kronecker-factored approximations.
    """

    ekfac_modified_projections: bool = False
    """P-SIFT: precompute M = R · cov^{-1/2} per side and apply it during
    gradient collection instead of whitening first and projecting after.

    When True, both query and training gradients are sketched in a single
    matmul per side; no separate iVHP step is needed.

    Requires ``ekfac_method='kfac'`` and ``ekfac_sketch=True``.
    Incompatible with ``ekfac_method='ekfac'`` (eigenvalue correction breaks
    the per-side decomposition).
    """

    ekfac_lambda_damp: float = 0.1
    """EK-FAC damping factor for eigenvalue correction."""

    ekfac_aggregate_query: bool = False
    """False (default): build a 2m-entry query index (aggregation='none') so
    H⁻¹ is applied per sentence → per-pair ΔI, same granularity as vanilla.
    Disk cost: 2m × full-param gradient size (e.g. ~100 GB for m=100 on GPT-2).
    True: collapse s⁺ and s⁻ each to a single mean gradient → global ΔI only.
    Disk cost: 2 × full-param gradient size (~1 GB on GPT-2)."""

    ekfac_filter_modules: str | None = None
    """Glob pattern of modules to exclude from EK-FAC Hessian and gradient index.
    Use 'lm_head' for GPT-2 or 'embed_out' for Pythia to skip the vocab-sized
    output projection (50k×50k S_cov → ~40 GB eigendecomp peak)."""

    ekfac_sketch: bool = False
    """Sketched version: use random projection to compute sketched EK-FAC influence
    (Φ H^{-1/2} g). When True, both query and train gradients are whitened in
    [O, I] space then projected with a shared random matrix Φ of dimension
    ``projection_dim``. Skips the iVHP step (whitening is fused into gradient
    collection). Score is cosine-normalised I_norm(z,q)=I(z,q)/sqrt(I(z,z)·I(q,q))
    to match TrackStar's cosine convention.  Requires aggregate_query=False,
    ekfac_filter_modules ⊇ bias/embedding layers."""

    # ── attribution granularity ───────────────────────────────────────────
    sentence_level: bool = False
    """Build the training gradient index at sentence granularity: one gradient
    entry per raw sentence (one line in the source .txt files) instead of per
    packed 512-token sequence.  Produces ~10-30× more index entries but makes
    Prec@k and top_inspect.parquet directly interpretable as individual sentences.
    scoring_base_size still applies (caps D_base sentences, not sequences)."""

    # ── offline index-based scoring ───────────────────────────────────────
    train_chunk_size: int = 2048
    """Rows loaded from the prebuilt training gradient index per GPU step when
    using offline index-based scoring (score_from_index).  Tune to fit VRAM:
    memory per chunk ≈ train_chunk_size × total_grad_dim × 4 B.
    Has no effect when score_from_index is not used."""

    # ── corpus subsampling for scoring ────────────────────────────────────
    scoring_base_size: int = 0
    """D_base examples included in the scoring pass (and Hessian fitting for
    ekfac).  0 = full D_base corpus.
    When > 0 a deterministic D_base subset is built once (shared across seeds)
    and reused for scoring, Hessian fitting, and stats analysis — so
    all three stages always operate on the same examples."""

    # ── common post-processing ────────────────────────────────────────────
    stats_base_size: int = 1000
    """Max D_base examples included in delta_I_stats.npz for statistics
    (Wilcoxon / mean ΔI).
    When scoring_base_size > 0 this draws from the already-subsampled D_base
    (so stats_base_size ≤ scoring_base_size is enforced automatically).
    0 = include all available scored D_base examples."""

    prec_base_size: int = 0
    """D_base examples included when ranking for Prec@k.
    0  = full scored D_train (streams two columns at a time for
         vanilla/trackstar; full ΔI array for ekfac).
    >0 = D_X ∪ this many random D_base examples (approximate but fast)."""

    k_precisions: list[int] = field(default_factory=lambda: [10, 20])
    """k values for Prec@k under contrastive ΔI ranking."""

    inspect_top_k: int = 50
    """Number of top training examples stored per query pair in top_inspect*.parquet."""

    # ── learned / not-learned query-pair split ────────────────────────────
    full_pairs_parquet: str | None = None
    """Path to probe_models.py's --pairs-output parquet for the FULL model
    on this experiment's phenomenon (columns: phenomenon, file_stem, pair_idx,
    text_good, text_bad, logp_good, logp_bad, softmax_prob_good, learned_x).
    Provides per-pair learned/not-learned labels (softmax_prob_good > 0.5) for
    the learned/not_learned split. None = skip the split (combined-only
    outputs, as before)."""

    no_x_pairs_parquet: str | None = None
    """Path to probe_models.py's --pairs-output parquet for the matching
    no-X (ablated) model — same phenomenon, model never trained on D_X.
    Combined with full_pairs_parquet to additionally split into
    learned_specific (full model learned, no-X model did not — the ablation
    worked as intended) and learned_confounded (full model learned, no-X model
    also learned — already known without X, so attribution can't be credited
    to X). None = skip these two extra views."""

    def __post_init__(self) -> None:
        if self.ekfac_modified_projections:
            if not self.ekfac_sketch:
                raise ValueError(
                    "ekfac_modified_projections=True requires ekfac_sketch=True."
                )
            if self.ekfac_method == "ekfac":
                raise ValueError(
                    "ekfac_modified_projections=True is incompatible with "
                    "ekfac_method='ekfac' (eigenvalue correction breaks the "
                    "per-side decomposition). Use ekfac_method='kfac'."
                )

        if (
            self.method in ("vanilla", "trackstar")
            and self.scoring_base_size > 0
            and self.prec_base_size > self.scoring_base_size
        ):
            logger.warning(
                "prec_base_size (%d) > scoring_base_size (%d) for method '%s': "
                "the score memmap only covers %d scored D_base examples, so "
                "Prec@k may index unscored entries. "
                "Set prec_base_size <= scoring_base_size to avoid this.",
                self.prec_base_size,
                self.scoring_base_size,
                self.method,
                self.scoring_base_size,
            )


@dataclass
class ExperimentCfg(Serializable):
    """Full experiment configuration."""

    name: str = "default"
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    artifacts_dir: Path = Path("artifacts")

    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    attribution: AttributionCfg = field(default_factory=AttributionCfg)

    # ── Derived path helpers ──────────────────────────────────────────────

    def shared_dir(self) -> Path:
        """Artifacts shared across seeds and conditions."""
        return self.artifacts_dir / self.name / "shared"

    def data_variant_tag(self) -> str:
        """Stable tag encoding condition-specific data construction."""
        if not self.data.condition and not self.data.mixing_strategy:
            return ""
        shuffle_tag = "_blockshuffle" if self.data.shuffle_within_blocks else ""
        if self.data.condition == "full":
            return f"{self.data.condition}_{self.data.mixing_strategy}{shuffle_tag}"
        return f"{self.data.condition}{shuffle_tag}"

    def train_data_path(self) -> Path:
        """Condition-specific training dataset (HuggingFace on-disk format)."""
        tag = self.data_variant_tag()
        return self.shared_dir() / (f"train_{tag}" if tag else "train")

    def attribution_sentence_data_path(self) -> Path:
        """Sentence-level training dataset for attribution (one row per raw sentence).
        Built from data_training/{base,X}/*.txt; scoring_base_size caps D_base sentences.
        Shared across seeds and conditions (the raw text is condition-independent)."""
        n = self.attribution.scoring_base_size
        tag = self.data_variant_tag()
        base = f"attribution_sentences_{tag}" if tag else "attribution_sentences"
        suffix = f"_base{n}" if n > 0 else ""
        return self.shared_dir() / (base + suffix)

    def attribution_train_subset_path(self) -> Path:
        """D_train subsampled to D_X + scoring_base_size D_base examples.
        Shared across seeds (deterministic, fixed internal seed).
        Only meaningful when scoring_base_size > 0."""
        n = self.attribution.scoring_base_size
        tag = self.data_variant_tag()
        name = (
            f"attribution_train_{tag}_base{n}" if tag else f"attribution_train_base{n}"
        )
        return self.shared_dir() / name

    def queries_path(self) -> Path:
        """Minimal-pair queries — condition-independent, prepared once."""
        return self.shared_dir() / "queries"

    def base_test_path(self) -> Path:
        """Base corpus held-out sentences."""
        return self.shared_dir() / "base_test"

    # ── Query-text datasets (shared, condition-independent) ───────────────

    def query_texts_path(self) -> Path:
        """Interleaved s⁺/s⁻ dataset (2m sentences) — vanilla & trackstar."""
        return self.shared_dir() / "query_texts"

    def query_plus_texts_path(self) -> Path:
        """s⁺-only dataset (m sentences) — ekfac only."""
        return self.shared_dir() / "query_plus_texts"

    def query_minus_texts_path(self) -> Path:
        """s⁻-only dataset (m sentences) — ekfac only."""
        return self.shared_dir() / "query_minus_texts"

    # ── Per-(condition, seed) directories ────────────────────────────────

    def seed_dir(self, seed: int) -> Path:
        return self.artifacts_dir / self.name / self.data_variant_tag() / f"seed_{seed}"

    def model_dir(self, seed: int) -> Path:
        return self.seed_dir(seed) / "model"

    def grad_dir(self, seed: int) -> Path:
        """Training-time gradient artifacts (compat helper for train command)."""
        return self.seed_dir(seed) / "gradients"

    def scores_path(self, seed: int) -> Path:
        return self.seed_dir(seed) / "scores.parquet"

    # ── Attribution method directory (method-specific) ────────────────────

    def attribution_method_dir(self, seed: int) -> Path:
        """Root for all artifacts of the current attribution method."""
        a = self.attribution
        name = a.method
        if a.method == "ekfac":
            if a.ekfac_sketch:
                if a.ekfac_modified_projections:
                    name = "kfac_effic_precond"
                elif a.ekfac_method == "ekfac":
                    name = "ekfac_full_precond"
                else:  # ekfac_method == "kfac", modified_projections=False
                    name = "kfac_full_precond"
            elif a.ekfac_method == "kfac":  # non-sketched kfac query batching
                name = "kfac"
        return self.seed_dir(seed) / "attribution" / name

    # ── Family A: vanilla & trackstar (per-pair ΔI) ───────────────────────

    def query_grad_dir(self, seed: int) -> Path:
        """Per-sentence query gradient index."""
        return self.attribution_method_dir(seed) / "query_index"

    def scores_raw_dir(self, seed: int) -> Path:
        """(N, 2m) bergson score memmap."""
        return self.attribution_method_dir(seed) / "scores_raw"

    # ── TrackStar specific ────────────────────────────────────────────────

    def train_grad_index_dir(self, seed: int) -> Path:
        """Prebuilt training gradient index shared across attribution methods."""
        return self.seed_dir(seed) / "train_index"

    def ekfac_train_index_dir(self, seed: int) -> Path:
        """Whitened training gradient index for EKFac-sketched (H^{-1/2} baked in).
        Separate from train_grad_index_dir because the stored gradients are
        hessian-specific and cannot be shared with vanilla/trackstar."""
        return self.seed_dir(seed) / "ekfac_train_index"

    def ts_value_precond_dir(self, seed: int) -> Path:
        return self.attribution_method_dir(seed) / "value_precond"

    def ts_query_precond_dir(self, seed: int) -> Path:
        return self.attribution_method_dir(seed) / "query_precond"

    def ts_mixed_precond_dir(self, seed: int) -> Path:
        return self.attribution_method_dir(seed) / "mixed_precond"

    # ── EK-FAC specific ───────────────────────────────────────────────────

    def ekfac_query_agg_dir(self, seed: int, sign: str) -> Path:
        """Aggregated (mean) query gradient for s⁺ or s⁻."""
        return self.attribution_method_dir(seed) / f"query_{sign}_agg"

    def ekfac_hessian_dir(self, seed: int) -> Path:
        """Kronecker-factor Hessian approximation — shared across all ekfac methods and kfac/ekfac variants."""
        return self.seed_dir(seed) / "kfac_hessian"

    def ekfac_query_ivhp_dir(self, seed: int, sign: str) -> Path:
        """H⁻¹-transformed aggregated query gradient (aggregated mode)."""
        return self.attribution_method_dir(seed) / f"query_{sign}_ivhp"

    def ekfac_ivhp_dir(self, seed: int) -> Path:
        """H⁻¹-transformed per-sentence query index (per-pair mode)."""
        return self.attribution_method_dir(seed) / "query_ivhp"

    def ekfac_scores_dir(self, seed: int, sign: str) -> Path:
        """(N, 1) bergson score memmap for the plus or minus query."""
        return self.attribution_method_dir(seed) / f"scores_{sign}"

    # ── Final outputs (all methods) ───────────────────────────────────────

    def delta_I_stats_path(self, seed: int) -> Path:
        """Lean npz with ΔI arrays (D_X + D_base sample) and Welford score state.
        Covers all query pairs combined — see _learned/_not_learned variants below."""
        return self.attribution_method_dir(seed) / "delta_I_stats.npz"

    def delta_I_stats_learned_path(self, seed: int) -> Path:
        """Same as delta_I_stats_path, restricted to pairs the model has learned
        (correct_fullsentence=True in scores.parquet). Family A methods only."""
        return self.attribution_method_dir(seed) / "delta_I_stats_learned.npz"

    def delta_I_stats_not_learned_path(self, seed: int) -> Path:
        """Same as delta_I_stats_path, restricted to pairs the model has not learned."""
        return self.attribution_method_dir(seed) / "delta_I_stats_not_learned.npz"

    def delta_I_stats_not_learned_no_x_path(self, seed: int) -> Path:
        """Restricted to pairs the no-X (initial) model did not learn — a reference
        group where D_X had potential causal influence on the full model's outcome."""
        return self.attribution_method_dir(seed) / "delta_I_stats_not_learned_no_x.npz"

    def delta_I_stats_learned_specific_path(self, seed: int) -> Path:
        """Restricted to pairs the full model learned AND the no-X model did
        not (the ablation worked as intended — learning is specific to X)."""
        return self.attribution_method_dir(seed) / "delta_I_stats_learned_specific.npz"

    def delta_I_stats_learned_confounded_path(self, seed: int) -> Path:
        """Restricted to pairs the full model learned AND the no-X model also
        learned (already known without X — attribution can't be credited to X)."""
        return (
            self.attribution_method_dir(seed) / "delta_I_stats_learned_confounded.npz"
        )

    def top_inspect_path(self, seed: int) -> Path:
        """Small parquet: top-K D_X examples by mean ΔI with text, for inspection."""
        return self.attribution_method_dir(seed) / "top_inspect.parquet"

    def top_inspect_good_path(self, seed: int) -> Path:
        """Top-K D_X examples by mean I(s⁺, z) — most influential for the correct sentence."""
        return self.attribution_method_dir(seed) / "top_inspect_good.parquet"

    def top_inspect_bad_path(self, seed: int) -> Path:
        """Top-K D_X examples by most negative mean I(s⁻, z) — most harmful for the wrong sentence."""
        return self.attribution_method_dir(seed) / "top_inspect_bad.parquet"

    def prec_at_k_path(self, seed: int) -> Path:
        """Pre-computed Prec@k results (JSON). Covers all query pairs combined —
        see _learned/_not_learned variants below."""
        return self.attribution_method_dir(seed) / "prec_at_k.json"

    def prec_at_k_learned_path(self, seed: int) -> Path:
        """Same as prec_at_k_path, restricted to pairs the model has learned
        (correct_fullsentence=True in scores.parquet). Family A methods only."""
        return self.attribution_method_dir(seed) / "prec_at_k_learned.json"

    def prec_at_k_not_learned_path(self, seed: int) -> Path:
        """Same as prec_at_k_path, restricted to pairs the model has not learned."""
        return self.attribution_method_dir(seed) / "prec_at_k_not_learned.json"

    def prec_at_k_not_learned_no_x_path(self, seed: int) -> Path:
        """Same as prec_at_k_path, restricted to pairs the no-X (initial) model did not learn."""
        return self.attribution_method_dir(seed) / "prec_at_k_not_learned_no_x.json"

    def prec_at_k_learned_specific_path(self, seed: int) -> Path:
        """Restricted to pairs the full model learned AND the no-X model did
        not (the ablation worked as intended — learning is specific to X)."""
        return self.attribution_method_dir(seed) / "prec_at_k_learned_specific.json"

    def prec_at_k_learned_confounded_path(self, seed: int) -> Path:
        """Restricted to pairs the full model learned AND the no-X model also
        learned (already known without X — attribution can't be credited to X)."""
        return self.attribution_method_dir(seed) / "prec_at_k_learned_confounded.json"

    def delta_I_per_source_path(self, seed: int) -> Path:
        """Per-fact per-source delta_I stats for BEAR facts (JSON). Computed over
        all D_X items of each source category — unbiased, unlike the top_inspect
        parquet which only covers the top-k ranked items."""
        return self.attribution_method_dir(seed) / "delta_I_per_source.json"

    def results_dir(self) -> Path:
        return self.artifacts_dir / self.name / "results"
