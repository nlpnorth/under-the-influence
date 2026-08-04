import os

import torch
from safetensors.torch import load_file

from tests.ekfac_tests.test_utils import (
    compute_eigenvector_cosine_similarity,
    format_per_layer_errors,
    load_sharded_covariances,
)


def test_eigenvalue_corrections(
    ground_truth_eigenvalue_corrections_path: str,
    ground_truth_eigenvectors_path: str,
    ekfac_results_path: str,
    world_size: int,
) -> None:
    """Test eigenvalue corrections against ground truth."""
    print("\nTesting eigenvalue corrections...")

    lambda_ground_truth_path = os.path.join(
        ground_truth_eigenvalue_corrections_path, "eigenvalue_corrections.safetensors"
    )
    lambda_run_path = os.path.join(ekfac_results_path, "eigenvalue_correction_sharded")

    # load ground_truth
    lambda_ground_truth = load_file(lambda_ground_truth_path)

    # load run eigenvalue corrections (sharded)
    lambda_run = load_sharded_covariances(lambda_run_path)

    total_processed_run_path = os.path.join(ekfac_results_path, "total_processed.pt")
    lambda_device = lambda_run[list(lambda_run.keys())[0]].device
    total = torch.load(total_processed_run_path, map_location=lambda_device)

    # Normalize by total
    lambda_run = {k: v / total for k, v in lambda_run.items()}

    # Load eigenvectors to compute sign alignment.
    gt_act_eigenvectors = load_file(
        os.path.join(
            ground_truth_eigenvectors_path, "eigenvectors_activations.safetensors"
        )
    )
    gt_grad_eigenvectors = load_file(
        os.path.join(
            ground_truth_eigenvectors_path, "eigenvectors_gradients.safetensors"
        )
    )
    run_act_eigenvectors = load_sharded_covariances(
        os.path.join(ekfac_results_path, "eigen_activation_sharded")
    )
    run_grad_eigenvectors = load_sharded_covariances(
        os.path.join(ekfac_results_path, "eigen_gradient_sharded")
    )

    _, act_signs = compute_eigenvector_cosine_similarity(
        gt_act_eigenvectors, run_act_eigenvectors
    )
    _, grad_signs = compute_eigenvector_cosine_similarity(
        gt_grad_eigenvectors, run_grad_eigenvectors
    )

    # Align eigenvalue correction signs
    lambda_run_aligned = {}
    for k in lambda_run:
        sign_G = grad_signs[k][:, None]  # (d_out, 1)
        sign_A = act_signs[k][None, :]  # (1, d_in)
        lambda_run_aligned[k] = lambda_run[k] * sign_G * sign_A

    # Concatenate and check relative error
    gt_all = torch.cat(
        [lambda_ground_truth[k].flatten() for k in sorted(lambda_ground_truth)]
    )
    run_all = torch.cat(
        [lambda_run_aligned[k].flatten() for k in sorted(lambda_run_aligned)]
    )

    rel_error = (gt_all - run_all).norm() / gt_all.norm()

    # Looser tolerance for distributed runs
    atol = 0.2 if world_size > 1 else 1e-4
    assert rel_error < atol, (
        f"eigenvalue corrections: rel_error={rel_error:.2e}, expected < {atol}\n"
        + format_per_layer_errors(lambda_ground_truth, lambda_run_aligned)
    )
    print(f"Eigenvalue corrections match (rel_error={rel_error:.2e})")
