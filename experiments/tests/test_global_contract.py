from __future__ import annotations

"""Cross-experiment PRE contracts."""

import ast
from pathlib import Path

from conftest import FINALS_PRE


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a literal assignment in {path}")


def test_contract_is_explicitly_outcome_free(contract):
    assert contract["schema_version"] == 1
    assert contract["outcome_free"] is True
    serialized = str(contract).lower()
    assert "p_value" not in serialized
    assert "minimum_ic" not in serialized


def test_every_scorer_uses_the_same_o2c_label(contract):
    expected = contract["global"]["label_column"]
    paths = (
        "e1_o2c_walkforward/train.py",
        "e2_o2c_epoch_curve/score.py",
        "e4_private_fixed_oos/score_fixed.py",
        "e5_raw23_direct/e5_train_autodl.py",
        "e5_raw23_direct/score_autodl.py",
    )
    for relative in paths:
        assert literal_assignment(FINALS_PRE / relative, "LABEL") == expected


def test_seeded_experiments_share_the_declared_seeds(contract):
    expected = tuple(contract["global"]["seeds"])
    assert literal_assignment(FINALS_PRE / "e2_o2c_epoch_curve/score.py", "SEEDS") == expected
    assert literal_assignment(FINALS_PRE / "e5_raw23_direct/score_autodl.py", "SEEDS") == expected


def test_score_guards_pin_241_days_and_42_barra_regressors(contract):
    expected_days = str(contract["global"]["expected_oos_days"])
    expected_regressors = str(contract["global"]["barra_regressor_count"])
    for relative in (
        "e2_o2c_epoch_curve/score.py",
        "e4_private_fixed_oos/score_fixed.py",
        "e5_raw23_direct/score_autodl.py",
    ):
        source = (FINALS_PRE / relative).read_text(encoding="utf-8")
        assert expected_regressors in source
    for relative in ("e2_o2c_epoch_curve/score.py", "e5_raw23_direct/score_autodl.py"):
        assert expected_days in (FINALS_PRE / relative).read_text(encoding="utf-8")


def test_a_components_are_only_the_four_declared_scores(contract):
    assert contract["global"]["a_components"] == [
        "rank_ic",
        "rank_ic_ir",
        "long_short_sharpe",
        "stress_ic_ir",
    ]
    assert literal_assignment(
        FINALS_PRE / "e5_raw23_direct/score_autodl.py", "A_COLUMNS"
    ) == tuple(contract["global"]["a_components"])
