"""Common utilities for EKFAC tests."""

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor


def add_tensor_dicts(a: dict[str, Tensor], b: dict[str, Tensor]) -> dict[str, Tensor]:
    """Add two dictionaries of tensors element-wise."""
    assert set(a.keys()) == set(b.keys()), "Keys must match"
    return {k: a[k] + b[k] for k in a}


def tensor_dict_to_device(
    d: dict[str, Tensor], device: str | torch.device
) -> dict[str, Tensor]:
    """Move all tensors in a dictionary to the specified device."""
    return {k: v.to(device) for k, v in d.items()}


def load_sharded_covariances(sharded_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load and concatenate sharded covariance files.

    Args:
        sharded_dir: Directory containing shard_0.safetensors, shard_1.safetensors, etc.

    Returns:
        Dictionary mapping layer names to concatenated covariance tensors.
    """
    sharded_dir = Path(sharded_dir)
    shard_files = sorted(sharded_dir.glob("shard_*.safetensors"))

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {sharded_dir}")

    shards = [load_file(str(f)) for f in shard_files]

    # Concatenate shards along first dimension
    result = {}
    for key in shards[0]:
        result[key] = torch.cat([shard[key] for shard in shards], dim=0)

    return result


def compute_eigenvector_cosine_similarity(
    gt: dict[str, torch.Tensor],
    run: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute column-wise cosine similarity between eigenvector matrices.

    Eigenvectors are only defined up to sign, so we check |cosine_similarity| ≈ 1.

    Args:
        gt: Ground truth eigenvector dictionary (each value is a matrix).
        run: Run eigenvector dictionary.

    Returns:
        Tuple of (all_abs_cosine_sims, signs) where:
        - all_abs_cosine_sims: flattened tensor of |cos_sim| for all columns
        - signs[k]: sign of cosine similarity per column (for λ correction alignment)
    """
    all_cos_sims = []
    signs = {}
    for k in sorted(gt.keys()):
        cos_sim = F.cosine_similarity(gt[k], run[k], dim=0)
        all_cos_sims.append(cos_sim.abs())
        signs[k] = torch.sign(cos_sim)
    return torch.cat(all_cos_sims), signs


def format_per_layer_errors(
    gt: dict[str, torch.Tensor],
    run: dict[str, torch.Tensor],
) -> str:
    """Format per-layer error details for debugging."""
    lines = []
    for k in sorted(gt.keys()):
        rel_err = (gt[k] - run[k]).norm() / gt[k].norm()
        lines.append(f"  {k}: rel_error={rel_err:.2e}")
    return "\n".join(lines)


def format_per_layer_cosine_similarity(
    gt: dict[str, torch.Tensor],
    run: dict[str, torch.Tensor],
) -> str:
    """Format per-layer |cosine_similarity| stats for eigenvector debugging."""
    lines = []
    for k in sorted(gt.keys()):
        abs_cos_sim = F.cosine_similarity(gt[k], run[k], dim=0).abs()
        lines.append(
            f"  {k}: min={abs_cos_sim.min():.6f}, "
            f"avg={abs_cos_sim.mean():.6f}, "
            f"med={abs_cos_sim.median():.6f}, "
            f"max={abs_cos_sim.max():.6f}"
        )
    return "\n".join(lines)


def load_covariances(
    run_path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    """Load activation and gradient covariances from an EKFAC run.

    Args:
        run_path: Path to the run directory containing influence_results/.

    Returns:
        Tuple of (activation_covariances, gradient_covariances, total_processed).
    """
    run_path = Path(run_path)
    results_path = run_path / "influence_results"

    A_cov = load_sharded_covariances(results_path / "activation_sharded")
    G_cov = load_sharded_covariances(results_path / "gradient_sharded")
    total_processed = torch.load(results_path / "total_processed.pt").item()

    return A_cov, G_cov, total_processed
