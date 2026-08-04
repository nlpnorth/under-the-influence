"""Backward-compatible wrapper for semantic experiments.

This module re-exports all functions from the semantic package for backward
compatibility. New code should import directly from examples.semantic instead.
"""

# Re-export everything from the semantic package
from semantic import (
    build_style_indices,
    build_style_lookup,
    compute_between_preconditioner,
    compute_between_preconditioner_covariance,
    compute_between_preconditioner_means,
    compute_metrics,
    compute_metrics_groupwise,
    compute_mixed_preconditioner,
    compute_scores_fast,
    compute_scores_with_bergson,
    create_data,
    create_index,
    create_qwen_only_dataset,
    finetune,
    load_scores_matrix,
    main,
    reword,
    run_preconditioner_comparison,
)

__all__ = [
    # Data creation
    "reword",
    "create_data",
    "create_qwen_only_dataset",
    # Scoring
    "load_scores_matrix",
    "compute_scores_fast",
    "compute_scores_with_bergson",
    # Metrics
    "build_style_lookup",
    "compute_metrics_groupwise",
    "compute_metrics",
    # Preconditioners
    "build_style_indices",
    "compute_between_preconditioner_covariance",
    "compute_between_preconditioner_means",
    "compute_between_preconditioner",
    "compute_mixed_preconditioner",
    # Experiment
    "create_index",
    "finetune",
    "run_preconditioner_comparison",
    "main",
]

if __name__ == "__main__":
    main()
