from __future__ import annotations

import torch

from bigalpha2026.alpha_models import (
    All156TemporalConfig,
    All156TemporalModel,
    All156TemporalNetwork,
    All618MLPConfig,
    All618MLPNetwork,
    ModelFactory,
)
from bigalpha2026.alpha_models.temporal import (
    CausalDepthwiseConv1d,
    TemporalSummary,
    masked_cross_sectional_zscore,
)


def small_config() -> All156TemporalConfig:
    return All156TemporalConfig(
        input_dim=6,
        model_dim=16,
        lookback=8,
        kernels=(3, 5),
        transformer_layers=1,
        attention_heads=4,
        feedforward_dim=32,
        dropout=0.0,
    )


def test_model_factory_creates_registered_temporal_model() -> None:
    model = ModelFactory.create(
        "all156_temporal",
        {
            "input_dim": 6,
            "model_dim": 16,
            "lookback": 8,
            "kernels": (3, 5),
            "transformer_layers": 1,
            "attention_heads": 4,
            "feedforward_dim": 32,
            "dropout": 0.0,
        },
    )
    assert isinstance(model, All156TemporalModel)


def test_cross_sectional_normalization_excludes_missing_and_padding() -> None:
    values = torch.tensor([[[[1.0]], [[3.0]], [[999.0]]]])
    observed = torch.tensor([[[[True]], [[True]], [[False]]]])
    stocks = torch.tensor([[True, True, False]])
    normalized = masked_cross_sectional_zscore(values, observed, stocks)
    torch.testing.assert_close(normalized[:, :2], torch.tensor([[[[-1.0]], [[1.0]]]]))
    assert normalized[0, 2, 0, 0] == 0


def test_causal_convolution_does_not_see_future() -> None:
    torch.manual_seed(3)
    layer = CausalDepthwiseConv1d(channels=2, kernel_size=3).eval()
    original = torch.randn(1, 2, 8)
    changed = original.clone()
    changed[:, :, 5:] += 100
    torch.testing.assert_close(layer(original)[:, :, :5], layer(changed)[:, :, :5])


def test_temporal_network_masks_padding_and_is_stock_permutation_equivariant() -> None:
    torch.manual_seed(7)
    network = All156TemporalNetwork(small_config()).eval()
    values = torch.randn(2, 5, 8, 6)
    observed = torch.rand_like(values) > 0.15
    values = values.masked_fill(~observed, float("nan"))
    stocks = torch.tensor([[True, True, True, True, False], [True, True, True, False, False]])

    scores = network(values, observed, stocks)
    assert scores.shape == (2, 5)
    assert torch.isfinite(scores).all()
    assert torch.count_nonzero(scores.masked_select(~stocks)) == 0

    permutation = torch.tensor([2, 0, 4, 1, 3])
    permuted = network(values[:, permutation], observed[:, permutation], stocks[:, permutation])
    torch.testing.assert_close(permuted, scores[:, permutation], atol=1e-5, rtol=1e-5)


def test_temporal_network_backward() -> None:
    network = All156TemporalNetwork(small_config()).train()
    values = torch.randn(1, 4, 8, 6)
    observed = torch.ones_like(values, dtype=torch.bool)
    stocks = torch.ones(1, 4, dtype=torch.bool)
    loss = network(values, observed, stocks).square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in network.parameters())


def test_all618_fusion_uses_masked_candidate_tower() -> None:
    config = All156TemporalConfig(
        input_dim=6,
        model_dim=16,
        lookback=8,
        kernels=(3, 5),
        transformer_layers=1,
        attention_heads=4,
        feedforward_dim=32,
        dropout=0.0,
        candidate_dim=7,
        candidate_hidden_dim=12,
    )
    network = All156TemporalNetwork(config).eval()
    values = torch.randn(1, 4, 8, 6)
    observed = torch.ones_like(values, dtype=torch.bool)
    stocks = torch.tensor([[True, True, True, False]])
    candidates = torch.randn(1, 4, 7)
    candidate_observed = torch.ones_like(candidates, dtype=torch.bool)
    candidates[0, 2, 3] = float("nan")
    candidate_observed[0, 2, 3] = False

    scores = network(
        values,
        observed,
        stocks,
        candidates,
        candidate_observed,
    )
    assert scores.shape == (1, 4)
    assert torch.isfinite(scores).all()
    assert scores[0, 3] == 0


def test_all618_mlp_uses_both_feature_towers() -> None:
    network = All618MLPNetwork(
        All618MLPConfig(
            bar_dim=6,
            candidate_dim=7,
            hidden_dims=(16, 8),
            dropout=0.0,
        )
    ).eval()
    bars = torch.randn(1, 4, 6)
    bar_observed = torch.ones_like(bars, dtype=torch.bool)
    candidates = torch.randn(1, 4, 7)
    candidate_observed = torch.ones_like(candidates, dtype=torch.bool)
    stocks = torch.tensor([[True, True, True, False]])
    scores = network(
        bars,
        bar_observed,
        stocks,
        candidates,
        candidate_observed,
    )
    assert scores.shape == (1, 4)
    assert torch.isfinite(scores).all()
    assert scores[0, 3] == 0


def test_temporal_summary_uses_last_real_position_with_left_padding() -> None:
    config = small_config()
    summary = TemporalSummary(config).eval()
    captured: dict[str, torch.Tensor] = {}
    summary.merge.register_forward_pre_hook(
        lambda _module, args: captured.setdefault("input", args[0])
    )
    values = torch.arange(8, dtype=torch.float32).view(1, 8, 1).expand(1, 8, 16)
    valid_time = torch.tensor([[False, False, False, True, True, True, True, True]])
    summary(values, valid_time)
    merged_input = captured["input"]
    torch.testing.assert_close(merged_input[:, :16], torch.full((1, 16), 7.0))
