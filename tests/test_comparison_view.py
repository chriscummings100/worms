"""Competition scores describe cumulative food intake, independently of survival."""

import pytest

from worms.comparison_view import comparison_result


@pytest.mark.parametrize("score", [0.0, 0.5, 0.75, 1.0])
def test_energy_share_remains_scored_when_both_cohorts_later_die(score):
    result = comparison_result({"score": score, "current": 0, "reference": 0})

    assert result == f"Current energy share {score:.1%} (50% = equal)"


def test_survival_without_food_intake_is_explicitly_unscored():
    result = comparison_result(
        {"score": None, "current": 64, "reference": 64, "reason": "no_energy_collected"}
    )

    assert result == "No food energy collected: no score"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("no_current_population", "No live population to sample"),
        ("no_reference_population", "No reference population to sample"),
    ],
)
def test_missing_source_population_stays_distinct_from_zero_intake(reason, expected):
    assert comparison_result({"score": None, "reason": reason}) == expected
