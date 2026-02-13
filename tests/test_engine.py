import gzip
import random
import tempfile
import unittest
from pathlib import Path

import mancala_solver as solver_mod
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
    SearchResult,
    TTEntry,
    _tt_best_move,
    best_move,
    load_tt,
    ordered_children,
    save_tt,
    search,
    search_depth,
    solve_best_move,
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

    def test_no_capture_when_opposite_empty_you(self):
        state = make_state(YOU, [0, 0, 1, 0, 0, 1], [2, 0, 0, 0, 0, 0])
        info = apply_move_with_info(state, 3)
        self.assertFalse(info.capture)
        self.assertIsNone(info.trace.capture)
        self.assertEqual(info.state.store_you, 0)
        self.assertEqual(info.state.pits_you[1], 1)
        self.assertEqual(info.state.pits_opp[4], 0)

    def test_no_capture_when_opposite_empty_opp(self):
        state = make_state(OPP, [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1])
        info = apply_move_with_info(state, 2)
        self.assertFalse(info.capture)
        self.assertIsNone(info.trace.capture)
        self.assertEqual(info.state.store_opp, 0)
        self.assertEqual(info.state.pits_opp[0], 1)
        self.assertEqual(info.state.pits_you[5], 0)

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

        for _ in range(2):
            state = initial_state(seeds=2, you_first=bool(rng.getrandbits(1)))
            plies = rng.randint(14, 20)
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
            state_a: TTEntry(3, EXACT, 4, 7),
            state_b: TTEntry(-2, EXACT, 1, 3, True),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.pkl.gz"
            save_tt(tt, path)
            loaded = load_tt(path)
        self.assertEqual(loaded, tt)

    def test_tt_prunes_when_over_limit(self):
        original_max = solver_mod.TT_MAX_ENTRIES
        original_prune_to = solver_mod.TT_PRUNE_TO
        try:
            solver_mod.TT_MAX_ENTRIES = 5
            solver_mod.TT_PRUNE_TO = 3
            tt = {}
            states = [
                make_state(YOU, [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], store_you=i, store_opp=0)
                for i in range(6)
            ]
            for idx, state in enumerate(states):
                solver_mod._tt_store(tt, state, TTEntry(idx, EXACT, 1, 1))

            self.assertLessEqual(len(tt), 3)
            self.assertNotIn(states[0], tt)
            self.assertNotIn(states[1], tt)
            self.assertNotIn(states[2], tt)
            self.assertIn(states[-1], tt)
        finally:
            solver_mod.TT_MAX_ENTRIES = original_max
            solver_mod.TT_PRUNE_TO = original_prune_to

    def test_tt_store_mutation_callback_only_when_tt_changes(self):
        state = make_state(YOU, [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], 0, 0)
        tt = {}
        mutations = {"count": 0}

        def on_mutation():
            mutations["count"] += 1

        solver_mod._tt_store(tt, state, TTEntry(3, EXACT, 1, 4), on_mutation=on_mutation)
        self.assertEqual(mutations["count"], 1)

        # Shallower entry should be ignored, so no mutation callback.
        solver_mod._tt_store(tt, state, TTEntry(5, EXACT, 2, 2), on_mutation=on_mutation)
        self.assertEqual(mutations["count"], 1)

        original_max = solver_mod.TT_MAX_ENTRIES
        original_prune_to = solver_mod.TT_PRUNE_TO
        try:
            solver_mod.TT_MAX_ENTRIES = 1
            solver_mod.TT_PRUNE_TO = 1
            state_2 = make_state(YOU, [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], 1, 0)
            solver_mod._tt_store(tt, state_2, TTEntry(2, EXACT, 2, 3), on_mutation=on_mutation)
            # One callback for store and one for prune.
            self.assertEqual(mutations["count"], 3)
        finally:
            solver_mod.TT_MAX_ENTRIES = original_max
            solver_mod.TT_PRUNE_TO = original_prune_to

    def test_tt_cutoff_requires_sufficient_depth(self):
        state = make_state(YOU, [2, 2, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0], 0, 0)
        norm_state, _ = solver_mod.normalize_state(state)
        tt = {
            norm_state: TTEntry(123456, EXACT, 1, 1),
        }
        value = search_depth(state, depth=2, tt=tt)
        self.assertNotEqual(value, 123456)

    def test_solve_best_move_scores_all_root_moves(self):
        state = initial_state(seeds=4, you_first=True)
        result = solve_best_move(state, topn=6, tt={}, time_limit_ms=50)
        self.assertEqual(len(result.top_moves), len(legal_moves(state)))

    def test_solve_best_move_uses_start_depth_and_guess_on_timeout(self):
        state = initial_state(seeds=2, you_first=True)
        norm_state, _ = solver_mod.normalize_state(state)
        tt = {
            norm_state: TTEntry(7, EXACT, 2, 4),
        }
        result = solve_best_move(
            state,
            topn=3,
            tt=tt,
            time_limit_ms=0,
            start_depth=5,
            guess_score=7,
        )
        self.assertEqual(result.depth, 4)
        self.assertFalse(result.complete)
        self.assertEqual(result.score, 7)
        self.assertEqual(result.best_move, 2)

    def test_solve_best_move_timeout_preserves_previous_result(self):
        state = initial_state(seeds=2, you_first=True)
        previous = SearchResult(
            best_move=4,
            score=5,
            top_moves=[(4, 5), (3, 4)],
            depth=16,
            complete=False,
            elapsed_ms=900,
            nodes=321_000,
        )
        result = solve_best_move(
            state,
            topn=3,
            tt={},
            time_limit_ms=0,
            start_depth=17,
            guess_score=5,
            previous_result=previous,
        )
        self.assertEqual(result.depth, previous.depth)
        self.assertEqual(result.score, previous.score)
        self.assertEqual(result.nodes, previous.nodes)
        self.assertEqual(result.best_move, previous.best_move)
        self.assertFalse(result.complete)

    def test_timeout_returns_last_completed_depth_result(self):
        state = initial_state(seeds=2, you_first=True)
        original = solver_mod._run_depth_iteration_with_aspiration
        calls = {"count": 0}

        def fake_iteration(state, topn, depth, guess_score, context):
            calls["count"] += 1
            if calls["count"] == 1:
                context.nodes += 17
                context.hit_depth_limit = True
                return solver_mod._DepthSearchResult(
                    best_move=4,
                    score=3,
                    top_moves=[(4, 3), (1, 2)],
                    fail_low=False,
                    fail_high=False,
                )
            raise solver_mod.SearchTimeout()

        try:
            solver_mod._run_depth_iteration_with_aspiration = fake_iteration
            result = solve_best_move(state, topn=3, tt={}, time_limit_ms=100)
        finally:
            solver_mod._run_depth_iteration_with_aspiration = original

        self.assertEqual(result.depth, 1)
        self.assertEqual(result.best_move, 4)
        self.assertEqual(result.score, 3)
        self.assertFalse(result.complete)
        self.assertEqual(result.top_moves[:1], [(4, 3)])

    def test_unproven_tt_exact_does_not_mark_complete(self):
        state = initial_state(seeds=4, you_first=True)
        tt = {}
        for move in legal_moves(state):
            child_state, _, _ = apply_move_fast_with_info(state, move)
            solver_mod.search_depth(child_state, depth=1, tt=tt)
        result = solve_best_move(state, topn=3, tt=tt, time_limit_ms=200, start_depth=2)
        self.assertFalse(result.complete)

    def test_depth_limited_exact_entry_is_not_marked_proven(self):
        state = make_state(YOU, [2, 2, 0, 0, 0, 0], [1, 2, 0, 0, 2, 0], 11, 16)
        tt = {}
        solver_mod.search_depth(state, depth=2, tt=tt)
        norm_state, _ = solver_mod.normalize_state(state)
        entry = tt.get(norm_state)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.flag, EXACT)
        self.assertFalse(entry.proven)

        context = solver_mod._SearchContext(tt=tt, deadline=None)
        _, proven = solver_mod._search_depth(state, -solver_mod.INF, solver_mod.INF, 2, context)
        self.assertFalse(proven)
        self.assertTrue(context.used_unproven_tt)

    def test_full_solve_interrupt_returns_partial_result(self):
        state = initial_state(seeds=4, you_first=True)
        result = solve_best_move(
            state,
            topn=3,
            tt={},
            time_limit_ms=None,
            interrupt_check=lambda: True,
        )
        self.assertFalse(result.complete)
        self.assertIsNotNone(result.best_move)

    def test_load_tt_handles_unexpected_deserialization_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.pkl.gz"
            with gzip.open(path, "wb") as handle:
                handle.write(b"not-a-pickle")

            original_pickle_load = solver_mod.pickle.load

            def boom(_handle):
                raise RuntimeError("boom")

            try:
                solver_mod.pickle.load = boom
                loaded = load_tt(path)
            finally:
                solver_mod.pickle.load = original_pickle_load

        self.assertEqual(loaded, {})

    def test_root_aspiration_researches_bound_scores(self):
        state = initial_state(seeds=1, you_first=True)
        child_a = make_state(YOU, [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        child_b = make_state(YOU, [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
        fake_children = [
            (1, child_a, False, False, 0),
            (2, child_b, False, False, 0),
        ]
        context = solver_mod._SearchContext(tt={}, deadline=None)
        original_children = solver_mod.ordered_children
        original_search = solver_mod._search_depth
        calls = []

        def fake_ordered_children(_state, _tt, context=None):
            return fake_children

        def fake_search(child_state, alpha, beta, depth, _context):
            calls.append((child_state, alpha, beta))
            if alpha == -solver_mod.INF and beta == solver_mod.INF:
                return (50 if child_state is child_a else 10, False)
            return (beta if child_state is child_a else alpha, False)

        try:
            solver_mod.ordered_children = fake_ordered_children
            solver_mod._search_depth = fake_search
            result = solver_mod._best_move_depth(state, topn=2, depth=3, context=context, alpha=-5, beta=5)
        finally:
            solver_mod.ordered_children = original_children
            solver_mod._search_depth = original_search

        self.assertEqual(result.best_move, 1)
        self.assertEqual(result.score, 50)
        self.assertIn((child_a, -solver_mod.INF, solver_mod.INF), calls)


if __name__ == "__main__":
    unittest.main()
