import os

import pytest
from safetensors.torch import load_file

from tests.ekfac_tests.test_utils import (
    compute_eigenvector_cosine_similarity,
    format_per_layer_cosine_similarity,
    load_sharded_covariances,
)


@pytest.mark.parametrize("eigenvector_type", ["activation", "gradient"])
def test_eigenvectors(
    ekfac_results_path: str,
    ground_truth_eigenvectors_path: str,
    eigenvector_type: str,
    world_size: int,
) -> None:
    """Test eigenvectors against ground truth."""
    print(f"\nTesting {eigenvector_type} eigenvectors...")

    eigenvectors_ground_truth_path = os.path.join(
        ground_truth_eigenvectors_path, f"eigenvectors_{eigenvector_type}s.safetensors"
    )
    eigenvectors_run_path = os.path.join(
        ekfac_results_path, f"eigen_{eigenvector_type}_sharded"
    )

    # load ground_truth
    ground_truth_eigenvectors = load_file(eigenvectors_ground_truth_path)

    # load run eigenvectors (sharded) and concatenate
    run_eigenvectors = load_sharded_covariances(eigenvectors_run_path)

    # Eigenvectors are only defined up to sign, so check |cosine_similarity| ≈ 1
    abs_cos_sims, _ = compute_eigenvector_cosine_similarity(
        ground_truth_eigenvectors, run_eigenvectors
    )

    avg_cos_sim = abs_cos_sims.mean().item()
    # Looser tolerance for distributed runs
    atol = 1e-3 if world_size > 1 else 1e-6
    assert avg_cos_sim > 1 - atol, (
        f"{eigenvector_type} eigenvectors: avg |cos_sim|={avg_cos_sim:.6f}, "
        f"expected > {1 - atol}\n"
        + format_per_layer_cosine_similarity(
            ground_truth_eigenvectors, run_eigenvectors
        )
    )
    print(f"{eigenvector_type} eigenvectors match (avg |cos_sim|={avg_cos_sim:.6f})")
