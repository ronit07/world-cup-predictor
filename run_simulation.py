"""CLI entry point: simulate the 2026 World Cup many times and print each
team's odds of reaching every stage, sorted by title chances.

Usage:
    python run_simulation.py                      # 10,000 sims, top 20 teams
    python run_simulation.py --all                # all 48 teams
    python run_simulation.py --team USA Mexico    # just the teams you care about
    python run_simulation.py --sims 50000 --seed 7
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_loading import load_groups, load_seed_ratings  # noqa: E402
from elo import EloRatingSystem  # noqa: E402
from simulate import run_monte_carlo  # noqa: E402

COLUMNS = [
    ("Win it", "champion"),
    ("Final", "final"),
    ("Semis", "semifinal"),
    ("QF", "quarterfinal"),
    ("R16", "round_of_16"),
    ("R32", "round_of_32"),
]


def resolve_teams(requested: list[str], known: list[str]) -> list[str]:
    """Match team names case-insensitively; suggest close matches for typos."""
    by_lower = {t.lower(): t for t in known}
    resolved = []
    for name in requested:
        team = by_lower.get(name.lower())
        if team is None:
            hint = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
            suggestion = f" Did you mean: {', '.join(hint)}?" if hint else ""
            sys.exit(f"Unknown team '{name}'.{suggestion}")
        resolved.append(team)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo World Cup 2026 simulator (Elo + Dixon-Coles).",
        epilog="Each column is the chance of reaching at least that stage.",
    )
    parser.add_argument("--sims", type=int, default=10_000, help="number of tournaments to simulate (default 10000)")
    parser.add_argument("--seed", type=int, default=42, help="random seed, so results are reproducible (default 42)")
    show = parser.add_mutually_exclusive_group()
    show.add_argument("--top", type=int, default=20, help="how many teams to show (default 20)")
    show.add_argument("--all", action="store_true", help="show all 48 teams")
    show.add_argument("--team", nargs="+", metavar="NAME", help="show only these teams, e.g. --team Brazil USA")
    args = parser.parse_args()

    if args.sims < 1:
        parser.error("--sims must be at least 1")

    ratings = load_seed_ratings()
    groups = load_groups()
    group_of = {team: name.replace("Group ", "") for name, teams in groups.items() for team in teams}

    started = time.perf_counter()
    odds = run_monte_carlo(groups, EloRatingSystem(ratings=ratings), n_sims=args.sims, seed=args.seed)
    elapsed = time.perf_counter() - started

    ranked = sorted(odds, key=lambda t: odds[t]["champion"], reverse=True)
    if args.team:
        wanted = set(resolve_teams(args.team, ranked))
        rows = [t for t in ranked if t in wanted]
    elif args.all:
        rows = ranked
    else:
        rows = ranked[: max(args.top, 1)]

    print(f"Simulated {args.sims:,} tournaments in {elapsed:.1f}s (seed {args.seed}).")
    print("Each column is the chance of reaching at least that stage.\n")
    header = f"{'#':>3}  {'Team':<15}{'Grp':<5}{'Elo':>6}" + "".join(f"{label:>9}" for label, _ in COLUMNS)
    print(header)
    print("-" * len(header))
    for team in rows:
        line = f"{ranked.index(team) + 1:>3}  {team:<15}{group_of[team]:<5}{ratings[team]:>6.0f}"
        line += "".join(f"{odds[team][key] * 100:>8.1f}%" for _, key in COLUMNS)
        print(line)

    if not args.team and not args.all and len(rows) < len(ranked):
        print(f"\nShowing the top {len(rows)} of {len(ranked)}. Use --all for every team or --team NAME for specific ones.")


if __name__ == "__main__":
    main()
