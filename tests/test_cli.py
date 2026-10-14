import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_simulation import resolve_teams

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_resolve_teams_is_case_insensitive():
    assert resolve_teams(["usa", "BRAZIL"], ["Brazil", "USA"]) == ["USA", "Brazil"]


def test_resolve_teams_suggests_typo_fix():
    with pytest.raises(SystemExit, match="Did you mean: Brazil"):
        resolve_teams(["Brazl"], ["Brazil", "USA"])


def test_cli_runs_from_any_folder():
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "run_simulation.py"), "--sims", "50", "--team", "Argentina"],
        cwd=os.path.join(REPO_ROOT, "tests"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Simulated 50 tournaments" in result.stdout
    assert "Argentina" in result.stdout
