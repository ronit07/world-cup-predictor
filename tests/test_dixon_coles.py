import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dixon_coles import elo_diff_to_goal_rates, match_probabilities, scoreline_matrix


def test_equal_teams_give_even_goal_rates():
    lam, mu = elo_diff_to_goal_rates(0.0)
    assert abs(lam - mu) < 1e-9


def test_stronger_team_has_higher_goal_rate():
    lam, mu = elo_diff_to_goal_rates(300.0)
    assert lam > mu


def test_scoreline_matrix_sums_to_one():
    matrix = scoreline_matrix(1.5, 1.1)
    total = sum(sum(row) for row in matrix)
    assert abs(total - 1.0) < 1e-6


def test_match_probabilities_sum_to_one():
    result = match_probabilities(1800, 1700, home_advantage=100.0)
    total = result["home_win"] + result["draw"] + result["away_win"]
    assert abs(total - 1.0) < 1e-6


def test_favourite_has_higher_win_probability():
    result = match_probabilities(1900, 1500, home_advantage=0.0)
    assert result["home_win"] > result["away_win"]
