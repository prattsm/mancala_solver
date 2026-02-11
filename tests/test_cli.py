import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import cli
from mancala_solver import SearchResult


class TestCLI(unittest.TestCase):
    def test_read_move_opponent_enter_reprompts(self):
        with patch("builtins.input", side_effect=["", "2"]):
            raw = cli.read_move("Opponent move: ", allow_enter=False, default_move=None)
        self.assertEqual(raw, "2")

    def test_read_move_you_enter_requires_default(self):
        with patch("builtins.input", side_effect=["", "3"]):
            raw = cli.read_move("Your move: ", allow_enter=True, default_move=None)
        self.assertEqual(raw, "3")

    def test_read_move_you_enter_uses_default(self):
        with patch("builtins.input", side_effect=[""]):
            raw = cli.read_move("Your move: ", allow_enter=True, default_move=4)
        self.assertEqual(raw, "")

    def test_main_opponent_enter_does_not_use_stale_you_move(self):
        inputs = iter(["n", "", "q"])
        with (
            patch("builtins.input", side_effect=lambda _prompt="": next(inputs)),
            patch.object(cli, "default_cache_path", return_value=Path("/tmp/mancala_cli_cache_test.pkl.gz")),
            patch.object(cli, "load_tt", return_value={}),
            patch.object(cli, "save_tt", return_value=None),
            patch("atexit.register", side_effect=lambda _fn: None),
            patch("sys.stdout", new=StringIO()),
        ):
            rc = cli.main(["--seeds", "1", "--time-ms", "50"])
        self.assertEqual(rc, 0)

    def test_cli_defaults_to_timed_search(self):
        calls = []
        fake_result = SearchResult(
            best_move=1,
            score=2,
            top_moves=[(1, 2)],
            depth=1,
            complete=False,
            elapsed_ms=10,
            nodes=100,
        )
        inputs = iter(["y", "q"])

        def fake_solve(state, topn=3, tt=None, time_limit_ms=None, **kwargs):
            calls.append(time_limit_ms)
            return fake_result

        with (
            patch("builtins.input", side_effect=lambda _prompt="": next(inputs)),
            patch.object(cli, "default_cache_path", return_value=Path("/tmp/mancala_cli_cache_test.pkl.gz")),
            patch.object(cli, "load_tt", return_value={}),
            patch.object(cli, "save_tt", return_value=None),
            patch("atexit.register", side_effect=lambda _fn: None),
            patch.object(cli, "solve_best_move", side_effect=fake_solve),
            patch("sys.stdout", new=StringIO()),
        ):
            rc = cli.main(["--seeds", "1"])

        self.assertEqual(rc, 0)
        self.assertEqual(calls, [300])

    def test_cli_perfect_overrides_time_budget(self):
        calls = []
        fake_result = SearchResult(
            best_move=1,
            score=2,
            top_moves=[(1, 2)],
            depth=1,
            complete=True,
            elapsed_ms=10,
            nodes=100,
        )
        inputs = iter(["y", "q"])

        def fake_solve(state, topn=3, tt=None, time_limit_ms=None, **kwargs):
            calls.append(time_limit_ms)
            return fake_result

        with (
            patch("builtins.input", side_effect=lambda _prompt="": next(inputs)),
            patch.object(cli, "default_cache_path", return_value=Path("/tmp/mancala_cli_cache_test.pkl.gz")),
            patch.object(cli, "load_tt", return_value={}),
            patch.object(cli, "save_tt", return_value=None),
            patch("atexit.register", side_effect=lambda _fn: None),
            patch.object(cli, "solve_best_move", side_effect=fake_solve),
            patch("sys.stdout", new=StringIO()),
        ):
            rc = cli.main(["--seeds", "1", "--time-ms", "50", "--perfect"])

        self.assertEqual(rc, 0)
        self.assertEqual(calls, [None])


if __name__ == "__main__":
    unittest.main()
