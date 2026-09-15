"""Explicit auxiliary losses; preserve E9's exact centered Pearson implementation."""

import torch
from evaluate_unified_temporal import centered_correlation
from torch.nn import functional as F

from experiments.finals_pre.e8_loss_comparison.protocol import AUX_WEIGHT


def auxiliary_loss(prediction, target, name):
    if name == "smooth_l1":
        return F.smooth_l1_loss(prediction, target, beta=1.0)
    if name == "l1":
        return F.l1_loss(prediction, target)
    if name == "l2_half":
        return 0.5 * F.mse_loss(prediction, target)
    raise ValueError(f"unknown loss: {name}")


def loss_components(prediction, target, name):
    correlation = centered_correlation(prediction, target)
    auxiliary = auxiliary_loss(prediction, target, name)
    total = -correlation + AUX_WEIGHT * auxiliary
    with torch.no_grad():
        error = prediction - target
        diagnostics = {
            "correlation": correlation.detach(),
            "auxiliary": auxiliary.detach(),
            "weighted_auxiliary": AUX_WEIGHT * auxiliary.detach(),
            "prediction_mean": prediction.mean(),
            "prediction_std": prediction.std(unbiased=False),
            "mean_absolute_error": error.abs().mean(),
            "large_error_fraction": (error.abs() > 1).float().mean(),
        }
    return total, diagnostics
