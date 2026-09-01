from __future__ import annotations

"""E3 progressive-addition contracts."""

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def e3(load_experiment_module):
    return load_experiment_module(
        "e3_progressive_add/model_progressive.py", "pre_e3_model_progressive"
    )


def arm_config(e3, values):
    return e3.ProgressiveConfig(
        model_dim=8,
        max_minutes=8,
        kernels=(3,),
        tcn_blocks=1,
        tail_minutes=4,
        dropout=0.0,
        use_sequence_path=values["sequence"],
        use_statistics_path=values["statistics"],
        use_path_fusion=values["path_fusion"],
        use_cross_section=values["cross_section"],
        sequence_summary=values["sequence_summary"],
    )


@pytest.mark.parametrize(
    "arm",
    [
        "p0_statistics_head",
        "p1_add_deepsets",
        "p2_add_tcn_last",
        "p3_add_full_summaries",
    ],
)
def test_each_declared_arm_runs_and_preserves_stock_shape(e3, contract, arm):
    config = arm_config(e3, contract["experiments"]["E3"]["arms"][arm])
    model = e3.ProgressiveNetwork(config).eval()
    values = torch.randn(2, 5, 8, 17)
    observed = torch.ones_like(values, dtype=torch.bool)
    minutes = torch.ones(2, 5, 8, dtype=torch.bool)
    stocks = torch.tensor([[True] * 5, [True, True, True, False, False]])
    with torch.inference_mode():
        scores = model(values, observed, minutes, stocks)
    assert scores.shape == (2, 5)
    assert torch.equal(scores[1, 3:], torch.zeros(2))


def test_progressive_arms_add_the_intended_modules(e3, contract):
    arms = contract["experiments"]["E3"]["arms"]
    p0 = e3.ProgressiveNetwork(arm_config(e3, arms["p0_statistics_head"]))
    p1 = e3.ProgressiveNetwork(arm_config(e3, arms["p1_add_deepsets"]))
    p2 = e3.ProgressiveNetwork(arm_config(e3, arms["p2_add_tcn_last"]))
    p3 = e3.ProgressiveNetwork(arm_config(e3, arms["p3_add_full_summaries"]))

    assert not hasattr(p0, "feature_projection") and not hasattr(p0, "cross_section")
    assert not hasattr(p1, "feature_projection") and hasattr(p1, "cross_section")
    assert hasattr(p2, "feature_projection") and not hasattr(p2, "attention")
    assert hasattr(p3, "feature_projection") and hasattr(p3, "attention")


def test_launcher_flags_match_the_four_declared_arms(contract):
    from conftest import FINALS_PRE

    source = (FINALS_PRE / "e3_progressive_add/run.sh").read_text(encoding="utf-8")
    expected_fragments = {
        "p0_statistics_head": (
            "--disable-sequence-path --disable-path-fusion --disable-cross-section"
        ),
        "p1_add_deepsets": "--disable-sequence-path --disable-path-fusion",
        "p2_add_tcn_last": "--sequence-summary last",
        "p3_add_full_summaries": 'run_arm p3_add_full_summaries "$seed"',
    }
    assert set(expected_fragments) == set(contract["experiments"]["E3"]["arms"])
    for fragment in expected_fragments.values():
        assert fragment in source


def test_invalid_pathway_and_channel_configs_fail_closed(e3):
    with pytest.raises(ValueError, match="at least one pathway"):
        e3.ProgressiveConfig(use_sequence_path=False, use_statistics_path=False)
    with pytest.raises(ValueError, match="duplicates"):
        e3.ProgressiveConfig(keep_channels=(0, 0))
    with pytest.raises(ValueError, match="out-of-range"):
        e3.ProgressiveConfig(keep_channels=(17,))
