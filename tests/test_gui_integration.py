import tempfile
import unittest
from unittest.mock import patch

try:
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtTest import QSignalSpy
    from PySide6.QtWidgets import QApplication

    import mancala_gui as gui_mod
    from mancala_engine import YOU, apply_move_with_info
    from mancala_solver import SearchResult

    HAS_QT = True
except Exception:
    HAS_QT = False


if HAS_QT:
    class _DummyWorker:
        def __init__(self) -> None:
            self.latest_request_id = 0
            self.cache = {}
            self.solve_calls = []
            self.try_snapshot_calls = 0
            self.tt_mutation_counter = 0
            self.tt_saved_counter = 0
            self.shutdown_calls = []
            self.shutdown_result = True
            self.wait_for_idle_calls = []
            self.wait_for_idle_result = True
            self.closed = False

        def set_latest_request_id(self, request_id: int) -> None:
            self.latest_request_id = request_id

        def solve(self, *args) -> None:
            self.solve_calls.append(args)

        def set_cache(self, tt) -> None:
            self.cache = dict(tt)
            self.tt_mutation_counter = 0
            self.tt_saved_counter = 0

        def cache_size(self) -> int:
            return len(self.cache)

        def snapshot_cache(self, max_entries=None) -> dict:
            if max_entries is None:
                return dict(self.cache)
            snapshot = {}
            for idx, (key, value) in enumerate(self.cache.items()):
                if idx >= max_entries:
                    break
                snapshot[key] = value
            return snapshot

        def try_snapshot_cache(self, max_entries=None):
            self.try_snapshot_calls += 1
            return self.snapshot_cache(max_entries=max_entries)

        def snapshot_cache_with_counter(self):
            return dict(self.cache), self.tt_mutation_counter

        def try_snapshot_cache_with_counter(self):
            self.try_snapshot_calls += 1
            return self.snapshot_cache_with_counter()

        def try_snapshot_cache_with_counter_budget(self, _budget_ms: int):
            self.try_snapshot_calls += 1
            return self.snapshot_cache_with_counter()

        def is_cache_dirty(self) -> bool:
            return self.tt_mutation_counter != self.tt_saved_counter

        def mark_cache_saved(self, mutation_counter: int) -> None:
            if mutation_counter > self.tt_saved_counter:
                self.tt_saved_counter = mutation_counter

        def shutdown(self, timeout_ms: int) -> bool:
            self.shutdown_calls.append(timeout_ms)
            return self.shutdown_result

        def wait_for_idle(self, timeout_ms: int) -> bool:
            self.wait_for_idle_calls.append(timeout_ms)
            return self.wait_for_idle_result

        def close(self) -> None:
            self.closed = True


    def _fake_setup_solver(window: "gui_mod.MancalaWindow") -> None:
        window.solver_worker = _DummyWorker()
        window.solver_worker.set_latest_request_id(window.solve_request_id)
        window.solve_requested.connect(window.solver_worker.solve)


    def _fake_start_cache_save(window: "gui_mod.MancalaWindow", _snapshot: dict, mutation_counter: int) -> bool:
        window.solver_worker.mark_cache_saved(mutation_counter)
        return True


    def _fake_shutdown_cache_worker(window: "gui_mod.MancalaWindow", _timeout_ms: int) -> None:
        window.cache_save_process = None
        window.cache_save_pending_counter = None


@unittest.skipUnless(HAS_QT, "PySide6 is required for GUI integration tests")
class TestGUIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.slice_patch = patch.object(gui_mod, "SOLVE_SLICE_MS", 40)
        self.requeue_patch = patch.object(gui_mod, "SOLVE_REQUEUE_DELAY_MS", 1)
        self.setup_patch = patch.object(gui_mod.MancalaWindow, "_setup_solver", _fake_setup_solver)
        self.start_cache_patch = patch.object(gui_mod.MancalaWindow, "_start_cache_save", _fake_start_cache_save)
        self.shutdown_cache_patch = patch.object(
            gui_mod.MancalaWindow,
            "_shutdown_cache_save_worker",
            _fake_shutdown_cache_worker,
        )
        self.slice_patch.start()
        self.requeue_patch.start()
        self.setup_patch.start()
        self.start_cache_patch.start()
        self.shutdown_cache_patch.start()

        self.window = gui_mod.MancalaWindow()
        self.window.slice_progress_timer.stop()
        self.window.cache_autosave_timer.stop()
        self.window.slice_start_time = None
        self.window.solving = False
        self.window.requeue_pending = False
        self.window.search_progress = None
        self.window.current_best_move = None
        self.window.current_top_moves = []

    def tearDown(self) -> None:
        if self.window is not None:
            self.window.close()
            self.app.processEvents()
        self.shutdown_cache_patch.stop()
        self.start_cache_patch.stop()
        self.setup_patch.stop()
        self.requeue_patch.stop()
        self.slice_patch.stop()

    def _prepare_pending_extra_turn(self):
        prev_state = self.window.state
        info = apply_move_with_info(prev_state, 4)
        self.assertEqual(info.state.to_move, YOU)
        pre_counts = gui_mod.CountsSnapshot(
            pits_you=tuple(prev_state.pits_you),
            pits_opp=tuple(prev_state.pits_opp),
            store_you=prev_state.store_you,
            store_opp=prev_state.store_opp,
        )
        self.window.pending_state = info.state
        self.window.pending_trace = info.trace
        self.window.pending_pre_counts = pre_counts
        return info.state

    def test_deferred_incomplete_result_requeues_after_commit(self) -> None:
        committed_state = self._prepare_pending_extra_turn()
        deferred = SearchResult(
            best_move=4,
            score=3,
            top_moves=[(4, 3)],
            depth=5,
            complete=False,
            elapsed_ms=10,
            nodes=100,
        )
        self.window.deferred_result = deferred
        self.window.deferred_result_state = committed_state
        self.window.solve_target_state = committed_state

        spy = QSignalSpy(self.window.solve_requested)
        self.window.commit_pending_move()

        self.assertTrue(spy.wait(1000))
        self.assertIsNotNone(self.window.search_progress)
        self.assertEqual(self.window.search_progress.depth, 5)
        self.assertFalse(self.window.search_progress.complete)

    def test_incomplete_result_requeues_from_on_solve_result(self) -> None:
        result = SearchResult(
            best_move=4,
            score=1,
            top_moves=[(4, 1), (3, 0)],
            depth=2,
            complete=False,
            elapsed_ms=8,
            nodes=64,
        )
        self.window.solve_target_state = self.window.state
        self.window.solving = True
        request_id = self.window.solve_request_id
        spy = QSignalSpy(self.window.solve_requested)

        self.window.on_solve_result(request_id, result)

        self.assertTrue(spy.wait(1000))
        self.assertIsNotNone(self.window.search_progress)
        self.assertEqual(self.window.search_progress.depth, 2)
        self.assertFalse(self.window.search_progress.complete)

    def test_schedule_preserves_progress_even_if_best_move_is_none(self) -> None:
        self.window.search_progress = SearchResult(
            best_move=None,
            score=2,
            top_moves=[],
            depth=7,
            complete=False,
            elapsed_ms=30,
            nodes=500,
        )
        self.window.current_best_move = None
        self.window.solve_target_state = self.window.state
        self.window.solving = False

        spy = QSignalSpy(self.window.solve_requested)
        self.window.schedule_solve_if_needed()

        self.assertTrue(spy.wait(1000))
        args = spy[0]
        self.assertEqual(args[3], 8)
        self.assertEqual(args[4], 2)
        self.assertIsNotNone(args[5])

    def test_regressive_same_state_result_does_not_replace_progress(self) -> None:
        previous = SearchResult(
            best_move=4,
            score=5,
            top_moves=[(4, 5), (3, 4)],
            depth=16,
            complete=False,
            elapsed_ms=800,
            nodes=240000,
        )
        self.window.search_progress = previous
        self.window.current_best_move = previous.best_move
        self.window.current_best_eval = previous.score
        self.window.current_top_moves = list(previous.top_moves)
        self.window.solve_target_state = self.window.state
        self.window.solving = True
        request_id = self.window.solve_request_id

        regressive = SearchResult(
            best_move=None,
            score=5,
            top_moves=[],
            depth=13,
            complete=False,
            elapsed_ms=805,
            nodes=0,
        )
        self.window.on_solve_result(request_id, regressive)

        self.assertIsNotNone(self.window.search_progress)
        self.assertEqual(self.window.search_progress.depth, 16)
        self.assertEqual(self.window.current_best_move, 4)

    def test_close_event_uses_bounded_solver_shutdown(self) -> None:
        self.window.solver_worker.tt_mutation_counter = 1
        event = QCloseEvent()

        self.window.closeEvent(event)

        self.assertEqual(self.window.solver_worker.wait_for_idle_calls, [gui_mod.SOLVER_CLOSE_TIMEOUT_MS])
        self.assertEqual(self.window.solver_worker.shutdown_calls, [gui_mod.SOLVER_CLOSE_TIMEOUT_MS, 0])
        self.assertEqual(self.window.solver_worker.tt_saved_counter, 1)
        self.assertTrue(self.window.solver_worker.closed)
        self.window = None

    def test_close_event_skips_save_when_not_dirty(self) -> None:
        self.window.solver_worker.tt_mutation_counter = 0
        self.window.solver_worker.tt_saved_counter = 0
        event = QCloseEvent()
        self.window.closeEvent(event)

        self.assertEqual(self.window.solver_worker.wait_for_idle_calls, [gui_mod.SOLVER_CLOSE_TIMEOUT_MS])
        self.assertEqual(self.window.solver_worker.shutdown_calls, [gui_mod.SOLVER_CLOSE_TIMEOUT_MS, 0])
        self.assertEqual(self.window.solver_worker.tt_saved_counter, 0)
        self.assertTrue(self.window.solver_worker.closed)
        self.window = None

    def test_close_event_skips_final_save_when_solver_shutdown_fails(self) -> None:
        self.window.solver_worker.wait_for_idle_result = False
        self.window.solver_worker.shutdown_result = False
        self.window.solver_worker.tt_mutation_counter = 3
        event = QCloseEvent()
        self.window.closeEvent(event)

        self.assertEqual(self.window.solver_worker.wait_for_idle_calls, [gui_mod.SOLVER_CLOSE_TIMEOUT_MS])
        self.assertEqual(self.window.solver_worker.shutdown_calls, [gui_mod.SOLVER_CLOSE_TIMEOUT_MS, 0])
        self.assertEqual(self.window.solver_worker.tt_saved_counter, 0)
        self.assertTrue(self.window.solver_worker.closed)
        self.window = None

    def test_state_change_requeues_solver_quickly(self) -> None:
        committed_state = self._prepare_pending_extra_turn()
        self.window.solve_target_state = committed_state
        self.window.solving = False
        spy = QSignalSpy(self.window.solve_requested)

        self.window.commit_pending_move()

        self.assertTrue(spy.wait(250))
        args = spy[0]
        self.assertEqual(args[0], committed_state)
        self.window = None


@unittest.skipUnless(HAS_QT, "PySide6 is required for GUI integration tests")
class TestGUISettingsPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.prev_format = QSettings.defaultFormat()
        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, self.temp_dir.name)

        self.org_patch = patch.object(gui_mod, "SETTINGS_ORG", "mancala_test_org")
        self.app_patch = patch.object(gui_mod, "SETTINGS_APP", "mancala_test_app")
        self.setup_patch = patch.object(gui_mod.MancalaWindow, "_setup_solver", _fake_setup_solver)
        self.start_cache_patch = patch.object(gui_mod.MancalaWindow, "_start_cache_save", _fake_start_cache_save)
        self.shutdown_cache_patch = patch.object(
            gui_mod.MancalaWindow,
            "_shutdown_cache_save_worker",
            _fake_shutdown_cache_worker,
        )
        self.org_patch.start()
        self.app_patch.start()
        self.setup_patch.start()
        self.start_cache_patch.start()
        self.shutdown_cache_patch.start()

        self.window = gui_mod.MancalaWindow()
        self.window.slice_progress_timer.stop()
        self.window.cache_autosave_timer.stop()
        self.window.solving = False

    def tearDown(self) -> None:
        if self.window is not None:
            self.window.close()
            self.app.processEvents()
        self.shutdown_cache_patch.stop()
        self.start_cache_patch.stop()
        self.setup_patch.stop()
        self.app_patch.stop()
        self.org_patch.stop()
        QSettings.setDefaultFormat(self.prev_format)
        self.temp_dir.cleanup()

    def test_view_settings_persist_across_windows(self) -> None:
        self.window.resize(1060, 760)
        self.window.anim_toggle.setChecked(True)
        self.window.show_numbers_check.setChecked(True)
        self.window.speed_combo.setCurrentText("Slow")
        self.window._save_persistent_settings()
        self.window.close()
        self.window = None

        restored = gui_mod.MancalaWindow()
        restored.slice_progress_timer.stop()

        self.assertTrue(restored.anim_toggle.isChecked())
        self.assertTrue(restored.show_numbers_check.isChecked())
        self.assertEqual(restored.speed_combo.currentText(), "Slow")

        settings = QSettings(gui_mod.SETTINGS_ORG, gui_mod.SETTINGS_APP)
        self.assertIsNotNone(settings.value("window/geometry"))
        self.assertGreaterEqual(restored.width(), 900)
        self.assertGreaterEqual(restored.height(), 720)

        restored.close()
        self.window = None


if __name__ == "__main__":
    unittest.main()
