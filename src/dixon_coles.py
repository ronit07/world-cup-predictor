"""Dixon-Coles adjusted bivariate Poisson scoreline model.

Converts an Elo rating gap into two expected-goal rates, then applies the
Dixon & Coles (1997) low-score correlation adjustment before deriving
win/draw/loss and full scoreline probabilities. rho < 0 slightly boosts
0-0 / 1-1 draws and slightly suppresses 1-0 / 0-1, matching the empirical
under-dispersion of low-scoring football matches relative to independent
Poisson.
"""

from __future__ import annotations

import math

AVG_GOALS_PER_TEAM = 1.35
ELO_TO_GOALS_SCALE = 400.0
DEFAULT_RHO = -0.05
MAX_GOALS = 8


def elo_diff_to_goal_rates(elo_diff: float, home_advantage: float = 0.0) -> tuple[float, float]:
    """Map an Elo gap to (lambda, mu) expected goals for the two sides."""
    adjusted_diff = elo_diff + home_advantage
    ratio = math.exp(adjusted_diff / ELO_TO_GOALS_SCALE)
    lam = AVG_GOALS_PER_TEAM * math.sqrt(ratio)
    mu = AVG_GOALS_PER_TEAM / math.sqrt(ratio)
    return lam, mu


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - (lam * mu * rho)
    if x == 0 and y == 1:
        return 1.0 + (lam * rho)
    if x == 1 and y == 0:
        return 1.0 + (mu * rho)
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate) * rate**k / math.factorial(k)


def scoreline_matrix(
    lam: float, mu: float, rho: float = DEFAULT_RHO, max_goals: int = MAX_GOALS
) -> list[list[float]]:
    """P(home scores i, away scores j) for i, j in [0, max_goals]."""
    matrix = []
    for i in range(max_goals + 1):
        row = []
        for j in range(max_goals + 1):
            p = _poisson_pmf(i, lam) * _poisson_pmf(j, mu) * _tau(i, j, lam, mu, rho)
            row.append(max(p, 0.0))
        matrix.append(row)

    total = sum(sum(row) for row in matrix)
    if total > 0:
        matrix = [[p / total for p in row] for row in matrix]
    return matrix


def outcome_probabilities(matrix: list[list[float]]) -> tuple[float, float, float]:
    """(home_win, draw, away_win) probabilities from a scoreline matrix."""
    home_win = draw = away_win = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
    return home_win, draw, away_win


def match_probabilities(
    home_elo: float,
    away_elo: float,
    home_advantage: float = 0.0,
    rho: float = DEFAULT_RHO,
) -> dict:
    lam, mu = elo_diff_to_goal_rates(home_elo - away_elo, home_advantage)
    matrix = scoreline_matrix(lam, mu, rho)
    home_win, draw, away_win = outcome_probabilities(matrix)
    return {
        "lambda_home": lam,
        "mu_away": mu,
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "scoreline_matrix": matrix,
    }
