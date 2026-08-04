import os

import pytest
import torch
from safetensors.torch import load_file

from tests.ekfac_tests.test_utils import (
    format_per_layer_errors,
    load_sharded_covariances,
)


@pytest.mark.parametrize("covariance_type", ["activation", "gradient"])
def test_covariances(
    ekfac_results_path: str,
    ground_truth_covariances_path: str,
    covariance_type: str,
) -> None:
    """Test covariances against ground truth."""
    print(f"\nTesting {covariance_type} covariances...")

    covariances_ground_truth_path = os.path.join(
        ground_truth_covariances_path, f"{covariance_type}_covariance.safetensors"
    )
    covariances_run_path = os.path.join(
        ekfac_results_path, f"{covariance_type}_sharded"
    )

    ground_truth_covariances = load_file(covariances_ground_truth_path)
    run_covariances = load_sharded_covariances(covariances_run_path)

    # Concatenate and check relative error
    gt_all = torch.cat(
        [
            ground_truth_covariances[k].flatten()
            for k in sorted(ground_truth_covariances)
        ]
    )
    run_all = torch.cat([run_covariances[k].flatten() for k in sorted(run_covariances)])

    rel_error = (gt_all - run_all).norm() / gt_all.norm()

    atol = 1e-4
    assert rel_error < atol, (
        f"{covariance_type} covariances: rel_error={rel_error:.2e}, expected < {atol}\n"
        + format_per_layer_errors(ground_truth_covariances, run_covariances)
    )
    print(f"{covariance_type} covariances match (rel_error={rel_error:.2e})")
