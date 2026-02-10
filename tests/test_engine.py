import unittest

from mancala_engine import OPP, YOU, State, apply_move, apply_move_fast_with_info, apply_move_with_info


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

    def test_skip_your_store_on_opp_move(self):
        state = make_state(OPP, [1, 0, 0, 0, 0, 0], [8, 1, 0, 0, 0, 0])
        before_total = total_seeds(state)
        new_state = apply_move(state, 1)
        self.assertEqual(new_state.store_you, 0)
        self.assertEqual(total_seeds(new_state), before_total)

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


if __name__ == "__main__":
    unittest.main()
