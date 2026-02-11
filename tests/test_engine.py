import random
import tempfile
import unittest
from pathlib import Path

from mancala_engine import (
    DropLocation,
    OPP,
    YOU,
    State,
    apply_move,
    apply_move_fast,
    apply_move_fast_with_info,
    apply_move_with_info,
    initial_state,
    is_terminal,
    legal_moves,
)
from mancala_solver import (
    EXACT,
    INF,
    TTEntry,
    _tt_best_move,
    best_move,
    load_tt,
    ordered_children,
    save_tt,
    search,
    terminal_diff,
)


def make_state(to_move, pits_you, pits_opp, store_you=0, store_opp=0):
    return State(to_move, tuple(pits_you), tuple(pits_opp), store_you, store_opp)


def total_seeds(state):
    return sum(state.pits_you) + sum(state.pits_opp) + state.store_you + state.store_opp


class TestEngine(unittest.TestCase):
    def test_skip_opponent_store_on_you_move(self):
        state = make_state(YOU, [0, 0, 0, 0, 0, 13], [1, 0, 0, 0, 0, 0])
        before_total = total_seeds(state)
        new_state = apply_move(state, 6)
        self.assertEqual(new_state.store_opp, 0)
        self.assertEqual(total_seeds(new_state), before_total)
        info = apply_move_with_info(state, 6)
        forbidden = DropLocation("STORE", None, OPP)
        self.assertNotIn(forbidden, info.trace.drops)

    def test_skip_your_store_on_opp_move(self):
        state = make_state(OPP, [1, 0, 0, 0, 0, 0], [8, 1, 0, 0, 0, 0])
        before_total = total_seeds(state)
        new_state = apply_move(state, 1)
        self.assertEqual(new_state.store_you, 0)
        self.assertEqual(total_seeds(new_state), before_total)
        info = apply_move_with_info(state, 1)
        forbidden = DropLocation("STORE", None, YOU)
        self.assertNotIn(forbidden, info.trace.drops)

    def test_extra_turn_you(self):
        state = make_state(YOU, [0, 0, 0, 4, 0, 0], [1, 0, 0, 0, 0, 0])
        info = apply_move_with_info(state, 4)
        self.assertEqual(len(info.trace.drops), info.trace.picked_count)
        self.assertEqual(info.trace.last_drop.store, YOU)
        self.assertEqual(info.trace.last_drop.side, "STORE")
        self.assertEqual(info.state.store_you, 1)
        self.assertEqual(info.state.to_move, YOU)
        self.assertTrue(info.extra_turn)

    def test_extra_turn_opp(self):
        state = make_state(OPP, [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1])
        info = apply_move_with_info(state, 1)
        self.assertEqual(len(info.trace.drops), info.trace.picked_count)
        self.assertEqual(info.trace.last_drop.store, OPP)
        self.assertEqual(info.trace.last_drop.side, "STORE")
        self.assertEqual(info.state.store_opp, 1)
        self.assertEqual(info.state.to_move, OPP)
        self.assertTrue(info.extra_turn)

    def test_capture_you(self):
        state = make_state(YOU, [0, 0, 1, 0, 0, 1], [2, 0, 0, 0, 4, 0])
        info = apply_move_with_info(state, 3)
        self.assertEqual(info.state.store_you, 5)
        self.assertEqual(info.state.pits_you[1], 0)
        self.assertEqual(info.state.pits_opp[4], 0)
        self.assertEqual(info.state.pits_opp[0], 2)
        self.assertTrue(info.capture)
        self.assertIsNotNone(info.trace.capture)
        self.assertEqual(info.trace.capture.opposite_index, 5 - info.trace.capture.landing_index)

    def test_capture_opp_opposite_mapping(self):
        state = make_state(OPP, [1, 0, 0, 0, 0, 4], [0, 1, 0, 0, 0, 1])
        info = apply_move_with_info(state, 2)
        self.assertEqual(info.state.store_opp, 5)
        self.assertEqual(info.state.pits_opp[0], 0)
        self.assertEqual(info.state.pits_you[5], 0)
        self.assertEqual(info.state.pits_you[0], 1)
        self.assertTrue(info.capture)
        self.assertIsNotNone(info.trace.capture)
        self.assertEqual(info.trace.capture.opposite_index, 5 - info.trace.capture.landing_index)

    def test_capture_when_opposite_empty_moves_single_seed(self):
        state = make_state(YOU, [0, 0, 1, 0, 0, 1], [2, 0, 0, 0, 0, 0])
        info = apply_move_with_info(state, 3)
        self.assertTrue(info.capture)
        self.assertEqual(info.state.store_you, 1)
        self.assertEqual(info.state.pits_you[1], 0)
        self.assertEqual(info.state.pits_opp[4], 0)

    def test_no_capture_when_last_drop_is_store(self):
        state = make_state(YOU, [0, 0, 0, 4, 0, 0], [1, 0, 0, 0, 0, 0])
        info = apply_move_with_info(state, 4)
        self.assertTrue(info.extra_turn)
        self.assertFalse(info.capture)
        self.assertIsNone(info.trace.capture)

    def test_terminal_sweep_when_you_empty(self):
        state = make_state(YOU, [1, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2])
        new_state = apply_move(state, 1)
        self.assertEqual(new_state.pits_you, (0, 0, 0, 0, 0, 0))
        self.assertEqual(new_state.pits_opp, (0, 0, 0, 0, 0, 0))
        self.assertEqual(new_state.store_you, 1)
        self.assertEqual(new_state.store_opp, 12)

    def test_trace_sweep_values(self):
        state = make_state(YOU, [1, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2])
        info = apply_move_with_info(state, 1)
        self.assertTrue(info.trace.terminal_after)
        self.assertEqual(info.trace.sweep_opp, 12)
        self.assertEqual(info.trace.sweep_you, 0)

    def test_terminal_sweep_when_opp_empty(self):
        state = make_state(YOU, [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0])
        new_state = apply_move(state, 5)
        self.assertEqual(new_state.pits_you, (0, 0, 0, 0, 0, 0))
        self.assertEqual(new_state.pits_opp, (0, 0, 0, 0, 0, 0))
        self.assertEqual(new_state.store_you, 1)

    def test_illegal_move_empty_pit(self):
        state = make_state(YOU, [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            apply_move(state, 1)

    def test_extra_turn_fast_path(self):
        state = make_state(YOU, [0, 0, 0, 4, 0, 0], [1, 0, 0, 0, 0, 0])
        new_state, extra_turn, _ = apply_move_fast_with_info(state, 4)
        self.assertEqual(new_state.store_you, 1)
        self.assertEqual(new_state.to_move, YOU)
        self.assertTrue(extra_turn)

    def test_fast_vs_trace_state_equality(self):
        states = [
            make_state(YOU, [4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]),
            make_state(OPP, [0, 1, 0, 5, 2, 0], [3, 0, 4, 0, 1, 6], 2, 1),
            make_state(YOU, [0, 0, 7, 1, 0, 3], [2, 5, 0, 0, 4, 1], 10, 9),
        ]
        for state in states:
            for move in range(1, 7):
                pits = state.pits_you if state.to_move == YOU else state.pits_opp
                if pits[move - 1] == 0:
                    continue
                fast_state, _, _ = apply_move_fast_with_info(state, move)
                trace_state = apply_move_with_info(state, move).state
                self.assertEqual(fast_state, trace_state)

    def test_fast_vs_trace_random_reachable_states(self):
        rng = random.Random(11)
        state = initial_state(seeds=2, you_first=True)
        checked = 0

        while checked < 120:
            if is_terminal(state):
                state = initial_state(seeds=2, you_first=bool(rng.getrandbits(1)))
                continue

            moves = legal_moves(state)
            self.assertTrue(moves)
            move = rng.choice(moves)
            fast_state = apply_move_fast(state, move)
            trace_state = apply_move_with_info(state, move).state
            self.assertEqual(fast_state, trace_state)

            state = fast_state
            checked += 1

    def test_tt_best_move_valid_for_opp(self):
        rng = random.Random(0)
        tt = {}

        def random_reachable_opp_state():
            while True:
                state = initial_state(seeds=2, you_first=bool(rng.getrandbits(1)))
                plies = rng.randint(1, 18)
                for _ in range(plies):
                    if is_terminal(state):
                        break
                    move = rng.choice(legal_moves(state))
                    state = apply_move(state, move)
                if state.to_move == OPP and not is_terminal(state):
                    return state

        for _ in range(10):
            state = random_reachable_opp_state()
            search(state, -INF, INF, tt)
            best = _tt_best_move(state, tt)
            legal = legal_moves(state)
            if best is not None:
                self.assertIn(best, legal)
                apply_move_fast_with_info(state, best)
            children = ordered_children(state, tt)
            self.assertTrue(children)

    def test_best_move_value_matches_search(self):
        rng = random.Random(21)
        tt = {}

        def random_reachable_state():
            state = initial_state(seeds=2, you_first=bool(rng.getrandbits(1)))
            plies = rng.randint(0, 20)
            for _ in range(plies):
                if is_terminal(state):
                    break
                state = apply_move(state, rng.choice(legal_moves(state)))
            return state

        checked = 0
        while checked < 8:
            state = random_reachable_state()
            if is_terminal(state):
                continue
            move, value, _ = best_move(state, tt=tt)
            self.assertIn(move, legal_moves(state))
            self.assertEqual(value, search(state, -INF, INF, tt))
            checked += 1

    def test_best_move_value_matches_search_fresh_tt(self):
        rng = random.Random(22)

        for _ in range(4):
            state = initial_state(seeds=2, you_first=bool(rng.getrandbits(1)))
            plies = rng.randint(0, 16)
            for _ in range(plies):
                if is_terminal(state):
                    break
                state = apply_move(state, rng.choice(legal_moves(state)))
            if is_terminal(state):
                continue
            tt = {}
            move, value, _ = best_move(state, tt=tt)
            self.assertIn(move, legal_moves(state))
            self.assertEqual(value, search(state, -INF, INF, tt))

    def test_terminal_diff_virtual_sweep(self):
        state_you_empty = make_state(YOU, [0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6], 8, 2)
        self.assertEqual(terminal_diff(state_you_empty), 8 - (2 + 21))

        state_opp_empty = make_state(OPP, [1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0], 3, 9)
        self.assertEqual(terminal_diff(state_opp_empty), (3 + 21) - 9)

        non_terminal = make_state(YOU, [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], 4, 7)
        self.assertEqual(terminal_diff(non_terminal), -3)

    def test_tt_cache_round_trip(self):
        state_a = make_state(YOU, [1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], 7, 8)
        state_b = make_state(OPP, [0, 1, 0, 2, 0, 3], [3, 0, 2, 0, 1, 0], 4, 9)
        tt = {
            state_a: TTEntry(3, EXACT, 4),
            state_b: TTEntry(-2, EXACT, 1),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.pkl.gz"
            save_tt(tt, path)
            loaded = load_tt(path)
        self.assertEqual(loaded, tt)


if __name__ == "__main__":
    unittest.main()
