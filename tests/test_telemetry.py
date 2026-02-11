import unittest
from unittest.mock import patch

import mancala_solver as solver_mod
from mancala_engine import initial_state
from mancala_telemetry import TelemetryEnvelope


class _CollectSink:
    def __init__(self) -> None:
        self.events: list[TelemetryEnvelope] = []

    def emit(self, envelope: TelemetryEnvelope) -> None:
        self.events.append(envelope)

    def close(self) -> None:
        return


class TestTelemetry(unittest.TestCase):
    def test_solve_emits_core_telemetry_events(self):
        sink = _CollectSink()
        state = initial_state(seeds=2, you_first=True)
        with (
            patch.object(solver_mod, "TELEMETRY_NODE_MASK", 0),
            patch.object(solver_mod, "TELEMETRY_EMIT_INTERVAL_MS", 0),
        ):
            result = solver_mod.solve_best_move(
                state,
                topn=3,
                tt={},
                time_limit_ms=60,
                telemetry_sink=sink,
            )

        self.assertIsNotNone(result.best_move)
        names = [event.event for event in sink.events]
        self.assertIn("search_start", names)
        self.assertIn("iteration_start", names)
        self.assertIn("iteration_done", names)
        self.assertIn("pv_update", names)
        self.assertIn("node_batch", names)
        self.assertIn("search_end", names)

    def test_search_end_reason_timeout_on_zero_budget(self):
        sink = _CollectSink()
        state = initial_state(seeds=4, you_first=True)
        result = solver_mod.solve_best_move(
            state,
            topn=3,
            tt={},
            time_limit_ms=0,
            telemetry_sink=sink,
        )
        self.assertFalse(result.complete)
        search_end_events = [event for event in sink.events if event.event == "search_end"]
        self.assertTrue(search_end_events)
        self.assertEqual(search_end_events[-1].data.get("reason"), "timeout")


if __name__ == "__main__":
    unittest.main()
