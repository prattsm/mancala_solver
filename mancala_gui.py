"""PySide6 GUI for GamePigeon Mancala (Capture mode) solver."""

from __future__ import annotations

import sys
import multiprocessing as mp
import time
import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPauseAnimation,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QSequentialAnimationGroup,
    Qt,
    QObject,
    Signal,
    Slot,
    QSettings,
    QTimer,
)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QProgressBar,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mancala_engine import (
    OPP,
    YOU,
    CaptureInfo,
    DropLocation,
    MoveTrace,
    State,
    apply_move_with_info,
    final_diff,
    initial_state,
    is_terminal,
    legal_moves,
)
from mancala_solver import SearchResult, default_cache_path, load_tt, save_tt, solve_best_move
from mancala_telemetry import ThreadedTCPSink, parse_host_port

SOLVE_SLICE_MS = 300
SOLVE_SLICE_MS_FAST = 60
SOLVE_FAST_SLICES_AFTER_STATE_CHANGE = 4
SOLVE_REQUEUE_DELAY_MS = 15
CACHE_AUTOSAVE_CHECK_MS = 1000
CACHE_AUTOSAVE_INTERVAL_MS = 60_000
CACHE_AUTOSAVE_IDLE_MS = 5_000
CACHE_AUTOSAVE_SNAPSHOT_BUDGET_MS = 120
CACHE_CLOSE_SAVE_TIMEOUT_MS = 1_200
CACHE_CLOSE_SNAPSHOT_BUDGET_MS = 120
CACHE_SAVE_POLL_MS = 20
SOLVER_CLOSE_TIMEOUT_MS = 3_000
SETTINGS_ORG = "prattsm"
SETTINGS_APP = "mancala_solver"


@dataclass
class HistoryEntry:
    state: State
    last_move_desc: str
    last_capture: bool
    last_extra: bool
    last_trace: Optional[MoveTrace]
    last_pre_counts: Optional["CountsSnapshot"]


@dataclass(frozen=True)
class CountsSnapshot:
    pits_you: Tuple[int, ...]
    pits_opp: Tuple[int, ...]
    store_you: int
    store_opp: int


@dataclass(frozen=True)
class AnimationSettings:
    per_drop_ms: int
    max_total_ms: int
    show_step: bool
    highlight_capture: bool


def _cache_save_worker(requests: "mp.Queue[Optional[Tuple[str, dict, int]]]", results: "mp.Queue[Tuple[bool, int]]") -> None:
    while True:
        task = requests.get()
        if task is None:
            return
        path_raw, snapshot, mutation_counter = task
        ok = True
        try:
            save_tt(snapshot, Path(path_raw))
        except Exception:
            ok = False
        try:
            results.put_nowait((ok, mutation_counter))
        except queue.Full:
            try:
                _ = results.get_nowait()
            except queue.Empty:
                pass
            try:
                results.put_nowait((ok, mutation_counter))
            except queue.Full:
                pass


def _solver_process_worker(
    commands: "mp.Queue[dict]",
    events: "mp.Queue[tuple]",
    latest_request_id: "mp.Value",
    telemetry_endpoint: Optional[Tuple[str, int]],
) -> None:
    tt: dict = {}
    tt_mutation_counter = 0
    telemetry_sink: Optional[ThreadedTCPSink] = None
    if telemetry_endpoint is not None:
        telemetry_sink = ThreadedTCPSink(telemetry_endpoint[0], telemetry_endpoint[1])

    def _put_event(event: tuple) -> None:
        try:
            events.put_nowait(event)
        except queue.Full:
            return

    def _is_interrupted(request_id: int) -> bool:
        return request_id != latest_request_id.value

    def _record_tt_mutation() -> None:
        nonlocal tt_mutation_counter
        tt_mutation_counter += 1

    try:
        while True:
            cmd = commands.get()
            cmd_type = cmd.get("type")
            if cmd_type == "shutdown":
                break
            if cmd_type == "set_cache":
                raw_tt = cmd.get("tt")
                if isinstance(raw_tt, dict):
                    tt = dict(raw_tt)
                    tt_mutation_counter = 0
                _put_event(("cache_reset", tt_mutation_counter))
                continue
            if cmd_type == "snapshot":
                token = int(cmd.get("token", 0))
                max_entries = cmd.get("max_entries")
                budget_ms = int(cmd.get("budget_ms", 0))
                deadline = None
                if budget_ms > 0:
                    deadline = time.perf_counter() + (budget_ms / 1000.0)
                snapshot: Optional[dict]
                if max_entries is None or max_entries <= 0 or len(tt) <= max_entries:
                    if deadline is None:
                        snapshot = dict(tt)
                    else:
                        snapshot = {}
                        for idx, (state, entry) in enumerate(tt.items()):
                            snapshot[state] = entry
                            if (idx & 0xFF) == 0 and time.perf_counter() >= deadline:
                                snapshot = None
                                break
                else:
                    snapshot = {}
                    for idx, (state, entry) in enumerate(tt.items()):
                        if idx >= max_entries:
                            break
                        snapshot[state] = entry
                        if deadline is not None and (idx & 0xFF) == 0 and time.perf_counter() >= deadline:
                            snapshot = None
                            break
                _put_event(("snapshot", token, snapshot, tt_mutation_counter))
                continue
            if cmd_type == "solve":
                request_id = int(cmd.get("request_id", 0))
                state = cmd.get("state")
                if not isinstance(state, State):
                    _put_event(("error", request_id, "Invalid solve state payload"))
                    continue
                topn = int(cmd.get("topn", 3))
                start_depth = int(cmd.get("start_depth", 1))
                guess_score = cmd.get("guess_score")
                previous_result = cmd.get("previous_result")
                slice_ms = max(1, int(cmd.get("slice_ms", SOLVE_SLICE_MS)))

                def _on_progress(result: SearchResult) -> None:
                    if _is_interrupted(request_id):
                        raise InterruptedError()
                    _put_event(("progress", request_id, result, tt_mutation_counter))

                def _on_live(nodes: int, elapsed_ms: int) -> None:
                    if _is_interrupted(request_id):
                        raise InterruptedError()
                    _put_event(("live", request_id, int(nodes), int(elapsed_ms), tt_mutation_counter))

                try:
                    result = solve_best_move(
                        state,
                        topn=topn,
                        tt=tt,
                        time_limit_ms=slice_ms,
                        progress_callback=_on_progress,
                        start_depth=start_depth,
                        guess_score=guess_score,
                        previous_result=previous_result,
                        interrupt_check=lambda rid=request_id: _is_interrupted(rid),
                        tt_mutation_callback=_record_tt_mutation,
                        live_callback=_on_live,
                        telemetry_sink=telemetry_sink,
                    )
                except InterruptedError:
                    continue
                except Exception as exc:
                    if _is_interrupted(request_id):
                        continue
                    _put_event(("error", request_id, f"{type(exc).__name__}: {exc}"))
                    continue
                if _is_interrupted(request_id):
                    continue
                _put_event(("result", request_id, result, tt_mutation_counter))
    finally:
        if telemetry_sink is not None:
            telemetry_sink.close()


class SolverWorker(QObject):
    result_ready = Signal(int, object)
    progress = Signal(int, object)
    live_metrics = Signal(int, int, int)
    solve_failed = Signal(int, str)

    def __init__(self) -> None:
        super().__init__()
        self.latest_request_id = 0
        self.tt_mutation_counter = 0
        self.tt_saved_counter = 0
        self._cache_size_estimate = 0
        self._rpc_token = 0
        self._pending_snapshots: dict[int, Tuple[Optional[dict], int]] = {}
        self._command_queue: "mp.Queue[dict]" = mp.Queue()
        self._event_queue: "mp.Queue[tuple]" = mp.Queue(maxsize=512)
        self._latest_request_id_value = mp.Value("i", 0, lock=False)
        self._process: Optional[mp.Process] = None
        self._active_request_id: Optional[int] = None
        self._closed = False
        self._allow_process_restart = True
        self._event_timer = QTimer(self)
        self._event_timer.setInterval(10)
        self._event_timer.timeout.connect(self._drain_events)
        self._event_timer.start()
        self._telemetry_endpoint: Optional[Tuple[str, int]] = None
        endpoint_raw = os.environ.get("MANCALA_TELEMETRY", "").strip()
        endpoint = parse_host_port(endpoint_raw) if endpoint_raw else None
        if endpoint is not None:
            self._telemetry_endpoint = endpoint

    def _ensure_process(self) -> bool:
        if self._closed:
            return False
        if self._process is not None and self._process.is_alive():
            return True
        if not self._allow_process_restart:
            return False
        if self._process is not None:
            self._process.join(timeout=0.05)
            self._process = None
        try:
            process = mp.Process(
                target=_solver_process_worker,
                args=(
                    self._command_queue,
                    self._event_queue,
                    self._latest_request_id_value,
                    self._telemetry_endpoint,
                ),
                name="mancala-solver",
                daemon=True,
            )
            process.start()
        except Exception:
            self._process = None
            return False
        self._process = process
        return True

    def _next_rpc_token(self) -> int:
        self._rpc_token += 1
        return self._rpc_token

    def _send_command(self, cmd: dict) -> bool:
        if not self._ensure_process():
            return False
        try:
            self._command_queue.put_nowait(cmd)
            return True
        except queue.Full:
            try:
                self._command_queue.put(cmd, timeout=0.05)
                return True
            except Exception:
                return False

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break

            kind = event[0] if event else None
            if kind == "progress":
                _, request_id, result, mutation_counter = event
                self.tt_mutation_counter = max(self.tt_mutation_counter, int(mutation_counter))
                self.progress.emit(int(request_id), result)
                continue
            if kind == "live":
                _, request_id, nodes, elapsed_ms, mutation_counter = event
                self.tt_mutation_counter = max(self.tt_mutation_counter, int(mutation_counter))
                self.live_metrics.emit(int(request_id), int(nodes), int(elapsed_ms))
                continue
            if kind == "result":
                _, request_id, result, mutation_counter = event
                self.tt_mutation_counter = max(self.tt_mutation_counter, int(mutation_counter))
                if self._active_request_id == int(request_id):
                    self._active_request_id = None
                self.result_ready.emit(int(request_id), result)
                continue
            if kind == "error":
                _, request_id, text = event
                if self._active_request_id == int(request_id):
                    self._active_request_id = None
                self.solve_failed.emit(int(request_id), str(text))
                continue
            if kind == "snapshot":
                _, token, snapshot, mutation_counter = event
                self.tt_mutation_counter = max(self.tt_mutation_counter, int(mutation_counter))
                self._pending_snapshots[int(token)] = (snapshot, int(mutation_counter))
                if snapshot is not None:
                    self._cache_size_estimate = len(snapshot)
                continue
            if kind == "cache_reset":
                _, mutation_counter = event
                self.tt_mutation_counter = int(mutation_counter)
                self.tt_saved_counter = int(mutation_counter)
                self._cache_size_estimate = 0
                continue

    def set_latest_request_id(self, request_id: int) -> None:
        self.latest_request_id = request_id
        self._latest_request_id_value.value = int(request_id)
        if self._active_request_id is not None and self._active_request_id != int(request_id):
            self._active_request_id = None

    def wait_for_idle(self, timeout_ms: int) -> bool:
        deadline = time.perf_counter() + (max(0, timeout_ms) / 1000.0)
        while time.perf_counter() < deadline:
            self._drain_events()
            if self._active_request_id is None:
                return True
            time.sleep(0.01)
        self._drain_events()
        return self._active_request_id is None

    def _request_snapshot(self, budget_ms: int) -> Optional[Tuple[dict, int]]:
        if not self._ensure_process():
            return None
        token = self._next_rpc_token()
        if not self._send_command({"type": "snapshot", "token": token, "budget_ms": max(0, int(budget_ms))}):
            return None
        deadline = time.perf_counter() + (max(0, budget_ms) / 1000.0)
        while time.perf_counter() <= deadline:
            self._drain_events()
            payload = self._pending_snapshots.pop(token, None)
            if payload is not None:
                snapshot, mutation_counter = payload
                if snapshot is None:
                    return None
                return snapshot, mutation_counter
            time.sleep(0.002)
        self._drain_events()
        payload = self._pending_snapshots.pop(token, None)
        if payload is None:
            return None
        snapshot, mutation_counter = payload
        if snapshot is None:
            return None
        return snapshot, mutation_counter

    @Slot(object, int, int, int, object, object, int)
    def solve(
        self,
        state: State,
        topn: int,
        request_id: int,
        start_depth: int,
        guess_score: Optional[int],
        previous_result: Optional[SearchResult],
        slice_ms: int,
    ) -> None:
        if request_id != self.latest_request_id:
            return
        if not self._send_command(
            {
                "type": "solve",
                "state": state,
                "topn": int(topn),
                "request_id": int(request_id),
                "start_depth": int(start_depth),
                "guess_score": guess_score,
                "previous_result": previous_result,
                "slice_ms": max(1, int(slice_ms)),
            }
        ):
            self.solve_failed.emit(request_id, "Failed to start solver process")
            return
        self._active_request_id = int(request_id)

    def set_cache(self, tt) -> None:
        if not isinstance(tt, dict):
            tt = {}
        if not self._send_command({"type": "set_cache", "tt": dict(tt)}):
            self.solve_failed.emit(self.latest_request_id, "Failed to initialize solver cache")
            return
        self._cache_size_estimate = len(tt)
        self.tt_mutation_counter = 0
        self.tt_saved_counter = 0

    def cache_size(self) -> int:
        snapshot_with_counter = self.try_snapshot_cache_with_counter_budget(40)
        if snapshot_with_counter is not None:
            snapshot, _ = snapshot_with_counter
            self._cache_size_estimate = len(snapshot)
        return self._cache_size_estimate

    def snapshot_cache(self, max_entries: Optional[int] = None) -> dict:
        snapshot_with_counter = self.snapshot_cache_with_counter()
        if snapshot_with_counter is None:
            return {}
        snapshot, _ = snapshot_with_counter
        if max_entries is None or max_entries <= 0 or len(snapshot) <= max_entries:
            return snapshot
        limited = {}
        for idx, (state, entry) in enumerate(snapshot.items()):
            if idx >= max_entries:
                break
            limited[state] = entry
        return limited

    def snapshot_cache_with_counter(self) -> Tuple[dict, int]:
        snapshot_with_counter = self._request_snapshot(5_000)
        if snapshot_with_counter is None:
            return {}, self.tt_mutation_counter
        return snapshot_with_counter

    def try_snapshot_cache_with_counter(self) -> Optional[Tuple[dict, int]]:
        if self._active_request_id is not None:
            return None
        return self._request_snapshot(20)

    def try_snapshot_cache_with_counter_budget(self, budget_ms: int) -> Optional[Tuple[dict, int]]:
        if budget_ms <= 0:
            return None
        if self._active_request_id is not None:
            return None
        return self._request_snapshot(budget_ms)

    def try_snapshot_cache(self, max_entries: Optional[int] = None) -> Optional[dict]:
        snapshot_with_counter = self.try_snapshot_cache_with_counter()
        if snapshot_with_counter is None:
            return None
        snapshot, _ = snapshot_with_counter
        if max_entries is None or max_entries <= 0 or len(snapshot) <= max_entries:
            return snapshot
        limited = {}
        for idx, (state, entry) in enumerate(snapshot.items()):
            if idx >= max_entries:
                break
            limited[state] = entry
        return limited

    def is_cache_dirty(self) -> bool:
        return self.tt_mutation_counter != self.tt_saved_counter

    def mark_cache_saved(self, mutation_counter: int) -> None:
        if mutation_counter > self.tt_saved_counter:
            self.tt_saved_counter = mutation_counter

    def shutdown(self, timeout_ms: int) -> bool:
        self._allow_process_restart = False
        self._active_request_id = None
        process = self._process
        if process is None:
            return True
        self._event_timer.stop()
        self._send_command({"type": "shutdown"})
        process.join(timeout=max(0, timeout_ms) / 1000.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.25)
        stopped = not process.is_alive()
        self._process = None
        return stopped

    def close(self) -> None:
        self.shutdown(0)
        self._closed = True


class PitButton(QPushButton):
    def __init__(self, side: str, index: int, pit_num: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.side = side
        self.index = index
        self.pit_num = pit_num
        self.setProperty("recommended", False)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(64, 64)
        self.setSizePolicy(QPushButton().sizePolicy())

    def set_recommended(self, recommended: bool) -> None:
        if self.property("recommended") == recommended:
            return
        self.setProperty("recommended", recommended)
        self.style().unpolish(self)
        self.style().polish(self)


class StoreWidget(QFrame):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Store")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("StoreLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.count_label = QLabel("00")
        self.count_label.setObjectName("StoreCount")
        self.count_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.count_label)

    def set_count(self, value: int) -> None:
        self.count_label.setText(f"{value:02d}")


class MancalaWindow(QMainWindow):
    solve_requested = Signal(object, int, int, int, object, object, int)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GamePigeon Mancala — Capture Mode")
        self.setMinimumSize(900, 720)

        self.cache_path: Path = default_cache_path()
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        self.state: State = initial_state(seeds=4, you_first=True)
        self.history: List[HistoryEntry] = []
        self.last_move_desc = "-"
        self.last_capture = False
        self.last_extra = False
        self.last_trace: Optional[MoveTrace] = None
        self.last_pre_counts: Optional[CountsSnapshot] = None

        self.current_best_move: Optional[int] = None
        self.current_best_eval: Optional[int] = None
        self.current_top_moves: List[Tuple[int, int]] = []
        self.topn = 3

        self.solve_request_id = 0
        self.solve_target_state: Optional[State] = None
        self.deferred_result: Optional[SearchResult] = None
        self.deferred_result_state: Optional[State] = None
        self.search_progress: Optional[SearchResult] = None
        self.solving = False
        self.animating = False
        self.closing = False
        self.requeue_pending = False
        self.slice_start_time: Optional[float] = None
        self.slice_hide_token = 0
        self.active_start_depth = 1
        self.active_slice_ms = SOLVE_SLICE_MS
        self.fast_slices_remaining = SOLVE_FAST_SLICES_AFTER_STATE_CHANGE
        self.search_heartbeat_phase = 0
        self.last_best_update_time: Optional[float] = None
        self.panel_snapshot_key: Optional[Tuple[str, str]] = None
        self.solver_error_text: Optional[str] = None
        self.live_nodes = 0
        self.live_elapsed_ms = 0
        self.last_solver_activity = time.perf_counter()
        self.last_cache_save_time = 0.0
        self.autosave_enabled = True
        self.cache_save_requests: "mp.Queue[Optional[Tuple[str, dict, int]]]" = mp.Queue(maxsize=1)
        self.cache_save_results: "mp.Queue[Tuple[bool, int]]" = mp.Queue(maxsize=8)
        self.cache_save_process: Optional[mp.Process] = None
        self.cache_save_pending_counter: Optional[int] = None
        self.shutdown_dialog: Optional[QDialog] = None
        self.shutdown_label: Optional[QLabel] = None
        self.shutdown_progress: Optional[QProgressBar] = None

        self.anim_counts_you: Optional[List[int]] = None
        self.anim_counts_opp: Optional[List[int]] = None
        self.anim_store_you: Optional[int] = None
        self.anim_store_opp: Optional[int] = None

        self.pending_state: Optional[State] = None
        self.pending_trace: Optional[MoveTrace] = None
        self.pending_pre_counts: Optional[CountsSnapshot] = None

        self.you_buttons_by_index: List[Optional[PitButton]] = [None] * 6
        self.opp_buttons_by_index: List[Optional[PitButton]] = [None] * 6

        self.overlay: Optional[QWidget] = None
        self.anim_group: Optional[QSequentialAnimationGroup] = None

        self._build_ui()
        self._setup_solver()
        self.slice_progress_timer = QTimer(self)
        self.slice_progress_timer.setInterval(200)
        self.slice_progress_timer.timeout.connect(self._update_slice_progress_bar)
        self.cache_autosave_timer = QTimer(self)
        self.cache_autosave_timer.setInterval(CACHE_AUTOSAVE_CHECK_MS)
        self.cache_autosave_timer.timeout.connect(self._on_cache_autosave_timer)
        self.cache_autosave_timer.start()
        self._apply_style()
        self._load_persistent_settings()

        self.reset_game()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(16)

        board_layout = QVBoxLayout()
        board_layout.setSpacing(16)

        self.opp_store = StoreWidget("OPP STORE")
        board_layout.addWidget(self.opp_store)

        grid_frame = QFrame()
        grid_layout = QGridLayout(grid_frame)
        grid_layout.setContentsMargins(12, 12, 12, 12)
        grid_layout.setSpacing(12)

        for row in range(6):
            you_index = 5 - row
            you_pit_num = you_index + 1
            you_btn = PitButton(YOU, you_index, you_pit_num)
            you_btn.clicked.connect(lambda _, b=you_btn: self.handle_pit_click(b))
            grid_layout.addWidget(you_btn, row, 0)
            self.you_buttons_by_index[you_index] = you_btn

            opp_index = row
            opp_pit_num = opp_index + 1
            opp_btn = PitButton(OPP, opp_index, opp_pit_num)
            opp_btn.clicked.connect(lambda _, b=opp_btn: self.handle_pit_click(b))
            grid_layout.addWidget(opp_btn, row, 1)
            self.opp_buttons_by_index[opp_index] = opp_btn

        board_layout.addWidget(grid_frame)

        self.you_store = StoreWidget("YOUR STORE")
        board_layout.addWidget(self.you_store)

        side_widget = QFrame()
        side_widget.setObjectName("SidePanel")
        side_panel = QVBoxLayout(side_widget)
        side_panel.setContentsMargins(12, 12, 12, 12)
        side_panel.setSpacing(12)

        go_first_label = QLabel("Turn Order")
        go_first_label.setObjectName("SideHeader")
        side_panel.addWidget(go_first_label)

        self.radio_first = QRadioButton("I go first")
        self.radio_second = QRadioButton("I go second")
        self.radio_first.setChecked(True)
        order_group = QButtonGroup(self)
        order_group.addButton(self.radio_first)
        order_group.addButton(self.radio_second)
        side_panel.addWidget(self.radio_first)
        side_panel.addWidget(self.radio_second)

        seeds_label = QLabel("Seeds per pit")
        seeds_label.setObjectName("SideHeader")
        side_panel.addWidget(seeds_label)

        self.seeds_spin = QSpinBox()
        self.seeds_spin.setRange(0, 12)
        self.seeds_spin.setValue(4)
        side_panel.addWidget(self.seeds_spin)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_game)
        side_panel.addWidget(self.reset_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_move)
        side_panel.addWidget(self.undo_button)

        self.play_best_button = QPushButton("Play Best")
        self.play_best_button.clicked.connect(self.play_best)
        side_panel.addWidget(self.play_best_button)

        self.auto_play_check = QCheckBox("Auto-play my best move")
        self.auto_play_check.setChecked(False)
        side_panel.addWidget(self.auto_play_check)

        self.show_numbers_check = QCheckBox("Show pit numbers")
        self.show_numbers_check.setChecked(False)
        self.show_numbers_check.toggled.connect(self.update_pit_labels)
        side_panel.addWidget(self.show_numbers_check)

        self.anim_toggle = QToolButton()
        self.anim_toggle.setText("Animation")
        self.anim_toggle.setCheckable(True)
        self.anim_toggle.setChecked(False)
        self.anim_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.anim_toggle.setArrowType(Qt.RightArrow)
        self.anim_toggle.toggled.connect(self._toggle_animation_panel)
        side_panel.addWidget(self.anim_toggle)

        self.anim_panel = QFrame()
        anim_layout = QVBoxLayout(self.anim_panel)
        anim_layout.setContentsMargins(0, 0, 0, 0)
        anim_layout.setSpacing(8)

        self.animate_check = QCheckBox("Animate moves")
        self.animate_check.setChecked(True)
        anim_layout.addWidget(self.animate_check)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["Fast", "Normal", "Slow"])
        self.speed_combo.setCurrentText("Normal")
        anim_layout.addWidget(self.speed_combo)

        self.step_check = QCheckBox("Show step-by-step sowing")
        self.step_check.setChecked(True)
        anim_layout.addWidget(self.step_check)

        self.capture_check = QCheckBox("Highlight capture")
        self.capture_check.setChecked(True)
        anim_layout.addWidget(self.capture_check)

        self.replay_button = QPushButton("Replay last move")
        self.replay_button.clicked.connect(self.replay_last_move)
        anim_layout.addWidget(self.replay_button)

        self.anim_panel.setVisible(False)
        side_panel.addWidget(self.anim_panel)

        solver_header = QLabel("Solver")
        solver_header.setObjectName("SideHeader")
        side_panel.addWidget(solver_header)

        self.solve_state_label = QLabel("State: Idle")
        self.solve_state_label.setObjectName("SolveState")
        self.solve_state_label.setWordWrap(True)
        side_panel.addWidget(self.solve_state_label)

        self.solve_best_label = QLabel("Best: -")
        self.solve_best_label.setObjectName("SolveBest")
        self.solve_best_label.setWordWrap(True)
        side_panel.addWidget(self.solve_best_label)

        self.solve_depth_label = QLabel("Depth: 0 complete")
        self.solve_depth_label.setObjectName("SolveDepth")
        self.solve_depth_label.setWordWrap(True)
        side_panel.addWidget(self.solve_depth_label)

        self.solve_age_label = QLabel("Last best update: -")
        self.solve_age_label.setObjectName("SolveAge")
        self.solve_age_label.setWordWrap(True)
        side_panel.addWidget(self.solve_age_label)

        self.solve_heartbeat_label = QLabel("Solver heartbeat: -")
        self.solve_heartbeat_label.setObjectName("SolveHeartbeat")
        self.solve_heartbeat_label.setWordWrap(True)
        side_panel.addWidget(self.solve_heartbeat_label)

        self.solve_metrics_label = QLabel("Nodes 0 | NPS 0/s | Elapsed 0.0s")
        self.solve_metrics_label.setObjectName("SolveMetrics")
        self.solve_metrics_label.setWordWrap(True)
        side_panel.addWidget(self.solve_metrics_label)

        self.slice_progress_label = QLabel("Slice: -")
        self.slice_progress_label.setObjectName("SliceProgress")
        self.slice_progress_label.hide()
        side_panel.addWidget(self.slice_progress_label)

        self.slice_progress = QProgressBar()
        self.slice_progress.setRange(0, SOLVE_SLICE_MS)
        self.slice_progress.setValue(0)
        self.slice_progress.setTextVisible(False)
        self.slice_progress.setFixedHeight(12)
        self.slice_progress.hide()
        side_panel.addWidget(self.slice_progress)

        top_moves_header = QLabel("Top Moves")
        top_moves_header.setObjectName("SideHeader")
        side_panel.addWidget(top_moves_header)

        self.top_moves_label = QLabel("-")
        self.top_moves_label.setObjectName("TopMoves")
        self.top_moves_label.setWordWrap(True)
        side_panel.addWidget(self.top_moves_label)

        side_panel.addStretch(1)

        main_layout.addLayout(board_layout, 2)
        main_layout.addWidget(side_widget, 1)

        self.overlay = QWidget(root)
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.overlay.setAttribute(Qt.WA_NoSystemBackground, True)
        self.overlay.setStyleSheet("background: transparent;")
        self.overlay.setGeometry(root.rect())
        self.overlay.raise_()

    def _setup_solver(self) -> None:
        self.solver_worker = SolverWorker()
        self.solver_worker.set_latest_request_id(self.solve_request_id)
        self.solver_worker.set_cache(load_tt(self.cache_path))
        self.solve_requested.connect(self.solver_worker.solve)
        self.solver_worker.progress.connect(self.on_solve_progress)
        self.solver_worker.live_metrics.connect(self.on_solve_live)
        self.solver_worker.result_ready.connect(self.on_solve_result)
        self.solver_worker.solve_failed.connect(self.on_solve_failed)

    def _set_latest_request_id(self) -> None:
        if hasattr(self, "solver_worker"):
            self.solver_worker.set_latest_request_id(self.solve_request_id)

    def _start_slice_progress(self) -> None:
        self.slice_hide_token += 1
        self.slice_start_time = time.perf_counter()
        self.search_heartbeat_phase = 0
        slice_ms = max(1, self.active_slice_ms)
        self.slice_progress.setRange(0, slice_ms)
        self.slice_progress.setValue(0)
        self.slice_progress_label.setText(f"Slice: 0/{slice_ms} ms")
        self.slice_progress_label.show()
        self.slice_progress.show()
        if not self.slice_progress_timer.isActive():
            self.slice_progress_timer.start()
        self.update_status()

    def _update_slice_progress_bar(self) -> None:
        if self.slice_start_time is None:
            return
        slice_ms = max(1, self.active_slice_ms)
        elapsed_ms = int((time.perf_counter() - self.slice_start_time) * 1000)
        shown_ms = min(slice_ms, max(0, elapsed_ms))
        self.slice_progress.setValue(shown_ms)
        self.slice_progress_label.setText(f"Slice: {shown_ms}/{slice_ms} ms")
        self.search_heartbeat_phase = (self.search_heartbeat_phase + 1) % 4
        self.update_status()

    def _finish_slice_progress(self, hide_after_ms: Optional[int] = None) -> None:
        self.slice_start_time = None
        self.slice_progress_timer.stop()
        slice_ms = max(1, self.active_slice_ms)
        self.slice_progress.setValue(slice_ms)
        self.slice_progress_label.setText(f"Slice: {slice_ms}/{slice_ms} ms")
        if hide_after_ms is None:
            return
        token = self.slice_hide_token
        QTimer.singleShot(hide_after_ms, lambda: self._hide_slice_progress_if_idle(token))

    def _hide_slice_progress_if_idle(self, token: int) -> None:
        if token != self.slice_hide_token:
            return
        if self.solving:
            return
        if self.animating or self.state.to_move != YOU:
            self.slice_progress.hide()
            self.slice_progress_label.hide()
            return
        if self.search_progress is None or self.search_progress.complete:
            self.slice_progress.hide()
            self.slice_progress_label.hide()

    @staticmethod
    def _format_nodes(value: int) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return str(value)

    def _queue_requeue_solve(self) -> None:
        if self.closing or self.requeue_pending:
            return
        self.requeue_pending = True
        QTimer.singleShot(SOLVE_REQUEUE_DELAY_MS, self._run_requeue_solve)

    def _run_requeue_solve(self) -> None:
        self.requeue_pending = False
        if self.closing:
            return
        self.schedule_solve_if_needed()

    def _pump_ui_events(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _show_shutdown_dialog(self) -> None:
        if self.shutdown_dialog is not None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Closing")
        dialog.setModal(True)
        dialog.setWindowFlag(Qt.WindowCloseButtonHint, False)
        dialog.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        dialog.setMinimumWidth(360)
        dialog.setStyleSheet(
            """
            QDialog {
                background: #132126;
            }
            QLabel {
                color: #f3f4f6;
                font-size: 12px;
            }
            QProgressBar {
                border: 1px solid #2f4a54;
                border-radius: 6px;
                background: #0e171b;
            }
            QProgressBar::chunk {
                background: #49b17d;
                border-radius: 6px;
            }
            """
        )
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        label = QLabel("Closing app...")
        label.setWordWrap(True)
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setFormat("%p%")
        progress.setTextVisible(True)
        progress.setFixedHeight(12)
        layout.addWidget(label)
        layout.addWidget(progress)
        dialog.show()
        dialog.raise_()
        self.shutdown_dialog = dialog
        self.shutdown_label = label
        self.shutdown_progress = progress
        self._pump_ui_events()

    def _update_shutdown_status(self, text: str, progress: Optional[int] = None) -> None:
        if self.shutdown_label is not None:
            self.shutdown_label.setText(text)
        if progress is not None and self.shutdown_progress is not None:
            bounded = max(0, min(100, progress))
            self.shutdown_progress.setValue(bounded)
        self._pump_ui_events()

    def _hide_shutdown_dialog(self) -> None:
        dialog = self.shutdown_dialog
        if dialog is None:
            return
        dialog.hide()
        dialog.deleteLater()
        self.shutdown_dialog = None
        self.shutdown_label = None
        self.shutdown_progress = None

    def _ensure_cache_save_worker(self) -> bool:
        process = self.cache_save_process
        if process is not None and process.is_alive():
            return True
        if process is not None:
            process.join(timeout=0.05)
            self.cache_save_process = None
        try:
            process = mp.Process(
                target=_cache_save_worker,
                args=(self.cache_save_requests, self.cache_save_results),
                name="mancala-cache-save",
                daemon=True,
            )
            process.start()
        except Exception:
            self.cache_save_process = None
            return False
        self.cache_save_process = process
        return True

    def _cache_save_in_progress(self) -> bool:
        process = self.cache_save_process
        if self.cache_save_pending_counter is None:
            return False
        if process is None or not process.is_alive():
            self.cache_save_pending_counter = None
            return False
        return True

    def _enqueue_latest_cache_snapshot(self, snapshot: dict, mutation_counter: int) -> bool:
        if not self._ensure_cache_save_worker():
            return False
        payload = (str(self.cache_path), snapshot, mutation_counter)
        while True:
            try:
                self.cache_save_requests.put_nowait(payload)
                self.cache_save_pending_counter = mutation_counter
                return True
            except queue.Full:
                try:
                    _ = self.cache_save_requests.get_nowait()
                except queue.Empty:
                    return False

    def _start_cache_save(self, snapshot: dict, mutation_counter: int) -> bool:
        return self._enqueue_latest_cache_snapshot(snapshot, mutation_counter)

    def _shutdown_cache_save_worker(self, timeout_ms: int) -> None:
        process = self.cache_save_process
        if process is None:
            return
        try:
            while True:
                try:
                    _ = self.cache_save_requests.get_nowait()
                except queue.Empty:
                    break
            try:
                self.cache_save_requests.put_nowait(None)
            except queue.Full:
                try:
                    _ = self.cache_save_requests.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.cache_save_requests.put_nowait(None)
                except queue.Full:
                    pass
        except Exception:
            pass

        process.join(timeout=max(0, timeout_ms) / 1000.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.2)
        self.cache_save_process = None
        self.cache_save_pending_counter = None

        while True:
            try:
                _ = self.cache_save_results.get_nowait()
            except queue.Empty:
                break

    def _poll_cache_save_completion(self) -> bool:
        updated = False
        while True:
            try:
                ok, mutation_counter = self.cache_save_results.get_nowait()
            except queue.Empty:
                break
            updated = True
            if ok:
                self.solver_worker.mark_cache_saved(mutation_counter)
                self.last_cache_save_time = time.perf_counter()
            if self.cache_save_pending_counter is not None and mutation_counter >= self.cache_save_pending_counter:
                self.cache_save_pending_counter = None
        return updated

    def _wait_for_cache_save_completion(
        self,
        timeout_ms: int,
        phase_text: Optional[str] = None,
        progress_start: int = 60,
        progress_end: int = 95,
    ) -> bool:
        if timeout_ms <= 0:
            self._poll_cache_save_completion()
            return not self._cache_save_in_progress()
        start = time.perf_counter()
        deadline = time.perf_counter() + (timeout_ms / 1000.0)
        while time.perf_counter() < deadline:
            self._poll_cache_save_completion()
            if not self._cache_save_in_progress():
                return True
            if phase_text is not None:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                ratio = min(1.0, elapsed_ms / max(1, timeout_ms))
                progress = progress_start + int((progress_end - progress_start) * ratio)
                self._update_shutdown_status(
                    f"{phase_text} ({elapsed_ms / 1000.0:.1f}s elapsed, {timeout_ms / 1000.0:.1f}s budget)",
                    progress=progress,
                )
            else:
                self._pump_ui_events()
            time.sleep(CACHE_SAVE_POLL_MS / 1000.0)
        self._poll_cache_save_completion()
        return not self._cache_save_in_progress()

    def _on_cache_autosave_timer(self) -> None:
        if not hasattr(self, "solver_worker"):
            return
        self._poll_cache_save_completion()
        if self.closing or not self.autosave_enabled:
            return
        if self.solving or self.animating:
            return
        if self.state.to_move == YOU and not is_terminal(self.state):
            return
        now = time.perf_counter()
        if (now - self.last_solver_activity) * 1000 < CACHE_AUTOSAVE_IDLE_MS:
            return
        if self.last_cache_save_time > 0 and (now - self.last_cache_save_time) * 1000 < CACHE_AUTOSAVE_INTERVAL_MS:
            return
        if not self.solver_worker.is_cache_dirty():
            return
        snapshot_with_counter = self.solver_worker.try_snapshot_cache_with_counter_budget(
            CACHE_AUTOSAVE_SNAPSHOT_BUDGET_MS
        )
        if snapshot_with_counter is None:
            return
        snapshot, mutation_counter = snapshot_with_counter
        self._start_cache_save(snapshot, mutation_counter)

    def _apply_style(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyle("Fusion")
            app.setFont(QFont("Avenir", 11))

        self.setStyleSheet(
            """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2f5d43, stop:1 #3f7356);
            }
            QLabel { color: #f7f3ea; }
            QLabel#SideHeader { font-weight: 600; margin-top: 8px; }
            QLabel#TopMoves { color: #f0e6d6; }
            QLabel#SolveState { color: #f2e8d7; font-weight: 600; }
            QLabel#SolveBest { color: #f6efdf; }
            QLabel#SolveDepth { color: #e9ddc8; }
            QLabel#SolveAge { color: #dbc8a8; font-size: 10px; }
            QLabel#SolveHeartbeat { color: #d9c8aa; font-size: 10px; }
            QLabel#SolveMetrics { color: #d8ccb9; font-size: 10px; }
            QLabel#SliceProgress { color: #e9ddc8; font-size: 10px; font-weight: 600; }
            QFrame#SidePanel {
                background: rgba(15, 28, 20, 0.35);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 14px;
            }
            QFrame#Store {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #b8834f, stop:1 #a46f3e);
                border-radius: 28px;
                border: 2px solid #7a4b23;
            }
            QLabel#StoreLabel { font-size: 12px; letter-spacing: 1px; }
            QLabel#StoreCount { font-size: 28px; font-weight: 700; color: #fff7e6; }
            QPushButton {
                background: #f2e0c2;
                border: 2px solid #b08a5a;
                border-radius: 28px;
                min-height: 56px;
                min-width: 56px;
                color: #3b2a1a;
                font-weight: 600;
            }
            QPushButton:disabled {
                background: #e1d2b8;
                color: #9c8c78;
                border-color: #c7b193;
            }
            QPushButton[recommended="true"] {
                border: 3px solid #2f7a45;
                background: #f7e9c8;
            }
            QPushButton[flash="true"] {
                border: 3px solid #f4c542;
                background: #fff0c8;
            }
            QFrame#Store[flash="true"] {
                border: 3px solid #f4c542;
            }
            QCheckBox { color: #f7f3ea; }
            QRadioButton { color: #f7f3ea; }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid #d8c7b0;
                background: #f2e6d4;
            }
            QRadioButton::indicator:checked {
                background: #c6b097;
                border: 2px solid #8b6a3f;
            }
            QSpinBox {
                background: #f7f3ea;
                color: #2d2013;
                border-radius: 6px;
                padding: 4px 6px;
            }
            QComboBox {
                background: #f7f3ea;
                color: #2d2013;
                border-radius: 6px;
                padding: 4px 6px;
            }
            QProgressBar {
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 4px;
                background: rgba(15, 28, 20, 0.35);
            }
            QProgressBar::chunk {
                background: #d9ba7a;
                border-radius: 3px;
            }
            QLabel#SeedToken {
                background: #f7e9c8;
                border: 1px solid #b08a5a;
                border-radius: 6px;
            }
            QLabel#SeedBundle {
                background: #f1d08c;
                border: 1px solid #9c7b4a;
                border-radius: 10px;
                color: #3b2a1a;
                font-weight: 700;
                font-size: 10px;
            }
            QLabel#SkipLabel {
                background: rgba(10, 20, 15, 0.6);
                color: #f7f3ea;
                border-radius: 8px;
                padding: 4px 8px;
                font-weight: 600;
            }
            QLabel#ExtraLabel {
                background: rgba(10, 20, 15, 0.7);
                color: #f7f3ea;
                border-radius: 8px;
                padding: 4px 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#GameOverLabel {
                background: rgba(15, 10, 5, 0.75);
                color: #f7f3ea;
                border-radius: 12px;
                padding: 8px 16px;
                font-weight: 700;
                font-size: 16px;
            }
            """
        )

    def reset_game(self) -> None:
        if self.animating:
            return
        you_first = self.radio_first.isChecked()
        seeds = self.seeds_spin.value()
        self.state = initial_state(seeds=seeds, you_first=you_first)
        self.history.clear()
        self.last_move_desc = "-"
        self.last_capture = False
        self.last_extra = False
        self.last_trace = None
        self.last_pre_counts = None
        self.pending_state = None
        self.pending_trace = None
        self.pending_pre_counts = None
        self.current_best_move = None
        self.current_best_eval = None
        self.current_top_moves = []
        self.solve_target_state = None
        self.deferred_result = None
        self.deferred_result_state = None
        self.search_progress = None
        self.last_best_update_time = None
        self.panel_snapshot_key = None
        self.solver_error_text = None
        self.live_nodes = 0
        self.live_elapsed_ms = 0
        self.solve_request_id += 1
        self._set_latest_request_id()
        self.solving = False
        self.requeue_pending = False
        self.slice_progress.hide()
        self.slice_progress_label.hide()
        self.slice_progress_timer.stop()
        self.slice_start_time = None
        self.active_slice_ms = SOLVE_SLICE_MS
        self.fast_slices_remaining = SOLVE_FAST_SLICES_AFTER_STATE_CHANGE
        self._clear_anim_counts()
        self.refresh_ui()
        self.schedule_solve_if_needed()

    def undo_move(self) -> None:
        if self.animating or not self.history:
            return
        entry = self.history.pop()
        self.state = entry.state
        self.last_move_desc = entry.last_move_desc
        self.last_capture = entry.last_capture
        self.last_extra = entry.last_extra
        self.last_trace = entry.last_trace
        self.last_pre_counts = entry.last_pre_counts
        self.pending_state = None
        self.pending_trace = None
        self.pending_pre_counts = None
        self.current_best_move = None
        self.current_best_eval = None
        self.current_top_moves = []
        self.solve_target_state = None
        self.deferred_result = None
        self.deferred_result_state = None
        self.search_progress = None
        self.last_best_update_time = None
        self.panel_snapshot_key = None
        self.solver_error_text = None
        self.live_nodes = 0
        self.live_elapsed_ms = 0
        self.solve_request_id += 1
        self._set_latest_request_id()
        self.solving = False
        self.requeue_pending = False
        self.slice_progress.hide()
        self.slice_progress_label.hide()
        self.slice_progress_timer.stop()
        self.slice_start_time = None
        self.active_slice_ms = SOLVE_SLICE_MS
        self.fast_slices_remaining = SOLVE_FAST_SLICES_AFTER_STATE_CHANGE
        self._clear_anim_counts()
        self.refresh_ui()
        self.schedule_solve_if_needed()

    def play_best(self) -> None:
        if self.animating:
            return
        if self.current_best_move is None:
            return
        if self.state.to_move != YOU:
            return
        self.apply_move(self.current_best_move)

    def handle_pit_click(self, button: PitButton) -> None:
        if self.animating or is_terminal(self.state):
            return
        if button.side != self.state.to_move:
            return
        if button.pit_num not in legal_moves(self.state):
            return
        self.apply_move(button.pit_num)

    def apply_move(self, pit_num: int) -> None:
        if self.animating:
            return
        prev_state = self.state
        move_side = prev_state.to_move
        info = apply_move_with_info(prev_state, pit_num)
        pre_counts = CountsSnapshot(
            pits_you=tuple(prev_state.pits_you),
            pits_opp=tuple(prev_state.pits_opp),
            store_you=prev_state.store_you,
            store_opp=prev_state.store_opp,
        )

        self.history.append(
            HistoryEntry(
                prev_state,
                self.last_move_desc,
                self.last_capture,
                self.last_extra,
                self.last_trace,
                self.last_pre_counts,
            )
        )

        self.pending_state = info.state
        self.pending_trace = info.trace
        self.pending_pre_counts = pre_counts

        self.last_move_desc = f"{move_side} pit {pit_num}"
        self.last_capture = info.capture
        self.last_extra = info.extra_turn

        self.current_best_move = None
        self.current_best_eval = None
        self.current_top_moves = []
        self.solve_target_state = None
        self.deferred_result = None
        self.deferred_result_state = None
        self.search_progress = None
        self.last_best_update_time = None
        self.panel_snapshot_key = None
        self.solver_error_text = None
        self.live_nodes = 0
        self.live_elapsed_ms = 0
        self.solving = False
        self.solve_request_id += 1
        self._set_latest_request_id()
        self.requeue_pending = False
        self.slice_progress.hide()
        self.slice_progress_label.hide()
        self.slice_progress_timer.stop()
        self.slice_start_time = None
        self.active_slice_ms = SOLVE_SLICE_MS
        self.fast_slices_remaining = SOLVE_FAST_SLICES_AFTER_STATE_CHANGE

        if self.animate_check.isChecked():
            self.start_animation()
        else:
            self.commit_pending_move()

    def commit_pending_move(self) -> None:
        if self.pending_state is None or self.pending_trace is None or self.pending_pre_counts is None:
            return
        self.state = self.pending_state
        self.last_trace = self.pending_trace
        self.last_pre_counts = self.pending_pre_counts
        self.pending_state = None
        self.pending_trace = None
        self.pending_pre_counts = None
        self._clear_anim_counts()
        self.animating = False
        if self.deferred_result_state == self.state and self.deferred_result is not None:
            deferred_result = self.deferred_result
            if not self._is_regressive_same_state_result(deferred_result):
                self._apply_search_result(deferred_result)
            self.deferred_result = None
            self.deferred_result_state = None
            self.refresh_ui()
            self._maybe_autoplay(self.solve_request_id)
            result_for_requeue = self.search_progress if self.search_progress is not None else deferred_result
            if self._should_requeue_result(result_for_requeue):
                self._queue_requeue_solve()
            return
        self.refresh_ui()
        self.schedule_solve_if_needed()

    def replay_last_move(self) -> None:
        if self.animating or self.last_trace is None or self.last_pre_counts is None:
            return
        self.pending_state = None
        self.pending_trace = self.last_trace
        self.pending_pre_counts = self.last_pre_counts
        self.start_animation(commit=False)

    def _clear_anim_counts(self) -> None:
        self.anim_counts_you = None
        self.anim_counts_opp = None
        self.anim_store_you = None
        self.anim_store_opp = None

    def start_animation(self, commit: bool = True) -> None:
        if self.pending_trace is None or self.pending_pre_counts is None:
            return
        trace = self.pending_trace
        pre_counts = self.pending_pre_counts
        settings = self._snapshot_animation_settings()

        self.animating = True
        self._set_anim_counts_from_snapshot(pre_counts)
        self.refresh_ui()
        self.update_controls()
        if self.overlay is not None:
            self.overlay.raise_()
        if commit and self.pending_state is not None and self.pending_state.to_move == YOU:
            self._start_solve_for(self.pending_state)

        if self.anim_group is not None:
            try:
                self.anim_group.finished.disconnect()
            except (RuntimeError, TypeError):
                pass
            self.anim_group.stop()
            self.anim_group.deleteLater()
            self.anim_group = None
        self.anim_group = QSequentialAnimationGroup(self)

        picked_widget = self._widget_for_pit(trace.mover, trace.picked_index)
        if picked_widget is not None:
            self._add_flash_step(self.anim_group, picked_widget, 240)
            self._add_pick_label_step(self.anim_group, picked_widget, trace.picked_count)

        self._add_sowing_steps(self.anim_group, trace, settings)

        landing_widget = self._widget_for_drop(trace.last_drop)
        if landing_widget is not None:
            self._add_flash_step(self.anim_group, landing_widget, 400)

        if trace.extra_turn:
            self._add_extra_turn_step(self.anim_group, trace.mover)

        if trace.capture and settings.highlight_capture:
            self._add_capture_steps(self.anim_group, trace.capture, settings)
        elif trace.capture and settings.show_step:
            self._apply_capture_counts(trace.capture)

        if trace.terminal_after:
            self._add_sweep_steps(self.anim_group, trace, settings)

        if self.anim_group.animationCount() == 0:
            self._finish_animation(commit)
            return

        self.anim_group.finished.connect(lambda: self._on_anim_group_finished(commit))
        self.anim_group.start()

    def _on_anim_group_finished(self, commit: bool) -> None:
        if self.anim_group is not None:
            self.anim_group.deleteLater()
            self.anim_group = None
        self._finish_animation(commit)

    def _finish_animation(self, commit: bool) -> None:
        self.animating = False
        if commit:
            self.commit_pending_move()
        else:
            self.pending_state = None
            self.pending_trace = None
            self.pending_pre_counts = None
            self._clear_anim_counts()
            self.refresh_ui()
            self.update_controls()

    def _snapshot_animation_settings(self) -> AnimationSettings:
        speed = self.speed_combo.currentText()
        if speed == "Fast":
            per_drop_ms = 60
        elif speed == "Slow":
            per_drop_ms = 200
        else:
            per_drop_ms = 120

        max_total_ms = 7000 if speed == "Slow" else 5000
        return AnimationSettings(
            per_drop_ms=per_drop_ms,
            max_total_ms=max_total_ms,
            show_step=self.step_check.isChecked(),
            highlight_capture=self.capture_check.isChecked(),
        )

    def _set_anim_counts_from_snapshot(self, snapshot: CountsSnapshot) -> None:
        self.anim_counts_you = list(snapshot.pits_you)
        self.anim_counts_opp = list(snapshot.pits_opp)
        self.anim_store_you = snapshot.store_you
        self.anim_store_opp = snapshot.store_opp

    def _center_for_widget(self, widget: QWidget) -> QPoint:
        if self.overlay is None:
            return QPoint(0, 0)
        center = widget.rect().center()
        global_center = widget.mapToGlobal(center)
        return self.overlay.mapFromGlobal(global_center)

    def _widget_for_pit(self, side: str, index: int) -> Optional[QWidget]:
        if index < 0 or index >= 6:
            return None
        if side == YOU:
            return self.you_buttons_by_index[index]
        if side == OPP:
            return self.opp_buttons_by_index[index]
        return None

    def _widget_for_drop(self, drop: DropLocation) -> Optional[QWidget]:
        if drop.side == "STORE":
            if drop.store == YOU:
                return self.you_store
            if drop.store == OPP:
                return self.opp_store
            return None
        if drop.index is None:
            return None
        return self._widget_for_pit(drop.side, drop.index)

    def _create_token(self, center: QPoint, size: int, text: str = "", object_name: str = "SeedToken") -> QLabel:
        if self.overlay is None:
            raise RuntimeError("overlay not initialized")
        token = QLabel(text, self.overlay)
        token.setObjectName(object_name)
        token.setAlignment(Qt.AlignCenter)
        token.setFixedSize(size, size)
        token.move(center.x() - size // 2, center.y() - size // 2)
        token.show()
        return token

    def _add_flash_step(self, group: QSequentialAnimationGroup, widget: QWidget, duration_ms: int) -> None:
        widget.setProperty("flash", True)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        pause = QPauseAnimation(duration_ms)

        def clear_flash() -> None:
            widget.setProperty("flash", False)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

        pause.finished.connect(clear_flash)
        group.addAnimation(pause)

    def _add_pick_label_step(self, group: QSequentialAnimationGroup, widget: QWidget, picked_count: int) -> None:
        if picked_count <= 1 or self.overlay is None:
            return
        label = QLabel(f"x{picked_count}", self.overlay)
        label.setObjectName("SeedBundle")
        label.setAlignment(Qt.AlignCenter)
        label.setFixedSize(28, 18)
        center = self._center_for_widget(widget)
        label.move(center.x() - 14, center.y() - 30)
        label.show()
        pause = QPauseAnimation(320)
        pause.finished.connect(label.deleteLater)
        group.addAnimation(pause)

    def _add_sowing_steps(
        self, group: QSequentialAnimationGroup, trace: MoveTrace, settings: AnimationSettings
    ) -> None:
        drops = list(trace.drops)
        if not drops:
            return

        total_ms = len(drops) * settings.per_drop_ms
        compress = total_ms > settings.max_total_ms and len(drops) > 12
        head = drops[:6] if compress else drops
        tail = drops[-6:] if compress else []
        middle = drops[6:-6] if compress else []

        picked_widget = self._widget_for_pit(trace.mover, trace.picked_index)
        if picked_widget is None:
            return
        current_pos = self._center_for_widget(picked_widget)

        def add_drop_animation(drop: DropLocation, start_pos: QPoint) -> QPoint:
            dest_widget = self._widget_for_drop(drop)
            if dest_widget is None:
                if settings.show_step:
                    self._apply_drop_to_anim_counts(drop)
                return start_pos
            end_pos = self._center_for_widget(dest_widget)
            token = self._create_token(start_pos, 12)
            anim = QPropertyAnimation(token, b"pos")
            anim.setDuration(settings.per_drop_ms)
            anim.setStartValue(start_pos - QPoint(6, 6))
            anim.setEndValue(end_pos - QPoint(6, 6))
            anim.setEasingCurve(QEasingCurve.InOutQuad)

            def on_finished() -> None:
                token.deleteLater()
                if settings.show_step:
                    self._apply_drop_to_anim_counts(drop)
                self._flash_widget_quick(dest_widget, 80)

            anim.finished.connect(on_finished)
            group.addAnimation(anim)
            return end_pos

        for drop in head:
            current_pos = add_drop_animation(drop, current_pos)

        if compress and middle:
            if settings.show_step:
                self._apply_bulk_counts(middle)
            self._add_skip_label_step(group)
            middle_widget = self._widget_for_drop(middle[-1])
            if middle_widget is not None:
                current_pos = self._center_for_widget(middle_widget)

        for drop in tail:
            current_pos = add_drop_animation(drop, current_pos)

    def _add_skip_label_step(self, group: QSequentialAnimationGroup) -> None:
        if self.overlay is None:
            return
        label = QLabel("...", self.overlay)
        label.setObjectName("SkipLabel")
        label.setAlignment(Qt.AlignCenter)
        label.adjustSize()
        center = self.overlay.rect().center()
        label.move(center.x() - label.width() // 2, center.y() - label.height() // 2)
        label.show()
        pause = QPauseAnimation(280)
        pause.finished.connect(label.deleteLater)
        group.addAnimation(pause)

    def _add_extra_turn_step(self, group: QSequentialAnimationGroup, mover: str) -> None:
        if self.overlay is None:
            return
        store_widget = self.you_store if mover == YOU else self.opp_store
        center = self._center_for_widget(store_widget)
        label = QLabel("EXTRA TURN", self.overlay)
        label.setObjectName("ExtraLabel")
        label.adjustSize()
        label.move(center.x() - label.width() // 2, center.y() - 40)
        label.show()
        pause = QPauseAnimation(600)
        pause.finished.connect(label.deleteLater)
        group.addAnimation(pause)

    def _add_capture_steps(
        self, group: QSequentialAnimationGroup, capture: CaptureInfo, settings: AnimationSettings
    ) -> None:
        landing_widget = self._widget_for_pit(capture.landing_side, capture.landing_index)
        opposite_widget = self._widget_for_pit(capture.opposite_side, capture.opposite_index)
        store_widget = self.you_store if capture.to_store == YOU else self.opp_store
        if landing_widget is None or opposite_widget is None:
            return

        self._add_dual_flash_step(group, landing_widget, opposite_widget, 150)

        start_center = self._center_for_widget(landing_widget)
        start_center_opp = self._center_for_widget(opposite_widget)
        end_center = self._center_for_widget(store_widget)

        opposite_count = max(0, capture.captured_count - 1)
        opposite_text = f"x{opposite_count}" if opposite_count > 0 else ""
        token_a = self._create_token(start_center, 18, "x1", "SeedBundle")
        token_b = self._create_token(start_center_opp, 18, opposite_text, "SeedBundle")

        anim_a = QPropertyAnimation(token_a, b"pos")
        anim_a.setDuration(400)
        anim_a.setStartValue(start_center - QPoint(9, 9))
        anim_a.setEndValue(end_center - QPoint(9, 9))
        anim_a.setEasingCurve(QEasingCurve.InOutQuad)

        anim_b = QPropertyAnimation(token_b, b"pos")
        anim_b.setDuration(400)
        anim_b.setStartValue(start_center_opp - QPoint(9, 9))
        anim_b.setEndValue(end_center - QPoint(9, 9))
        anim_b.setEasingCurve(QEasingCurve.InOutQuad)

        parallel = QParallelAnimationGroup()
        parallel.addAnimation(anim_a)
        parallel.addAnimation(anim_b)
        group.addAnimation(parallel)

        def on_capture_finished() -> None:
            token_a.deleteLater()
            token_b.deleteLater()
            if settings.show_step:
                self._apply_capture_counts(capture)
            self._flash_widget_quick(store_widget, 120)

        parallel.finished.connect(on_capture_finished)

    def _add_sweep_steps(
        self, group: QSequentialAnimationGroup, trace: MoveTrace, settings: AnimationSettings
    ) -> None:
        if trace.sweep_you <= 0 and trace.sweep_opp <= 0:
            return
        if self.overlay is None:
            return

        if trace.sweep_you > 0:
            start_widget = self.you_buttons_by_index[2]
            if start_widget is not None:
                start = self._center_for_widget(start_widget)
                end = self._center_for_widget(self.you_store)
                token = self._create_token(start, 20, f"x{trace.sweep_you}", "SeedBundle")
                anim = QPropertyAnimation(token, b"pos")
                anim.setDuration(400)
                anim.setStartValue(start - QPoint(10, 10))
                anim.setEndValue(end - QPoint(10, 10))
                anim.setEasingCurve(QEasingCurve.InOutQuad)
                group.addAnimation(anim)

                def sweep_you_done() -> None:
                    token.deleteLater()
                    if settings.show_step:
                        self._apply_sweep_counts(trace.sweep_you, 0)
                    self._flash_widget_quick(self.you_store, 120)

                anim.finished.connect(sweep_you_done)

        if trace.sweep_opp > 0:
            start_widget = self.opp_buttons_by_index[3]
            if start_widget is not None:
                start = self._center_for_widget(start_widget)
                end = self._center_for_widget(self.opp_store)
                token = self._create_token(start, 20, f"x{trace.sweep_opp}", "SeedBundle")
                anim = QPropertyAnimation(token, b"pos")
                anim.setDuration(400)
                anim.setStartValue(start - QPoint(10, 10))
                anim.setEndValue(end - QPoint(10, 10))
                anim.setEasingCurve(QEasingCurve.InOutQuad)
                group.addAnimation(anim)

                def sweep_opp_done() -> None:
                    token.deleteLater()
                    if settings.show_step:
                        self._apply_sweep_counts(0, trace.sweep_opp)
                    self._flash_widget_quick(self.opp_store, 120)

                anim.finished.connect(sweep_opp_done)

        label = QLabel("GAME OVER", self.overlay)
        label.setObjectName("GameOverLabel")
        label.adjustSize()
        center = self.overlay.rect().center()
        label.move(center.x() - label.width() // 2, center.y() - label.height() // 2)
        label.show()
        pause = QPauseAnimation(900)
        pause.finished.connect(label.deleteLater)
        group.addAnimation(pause)

    def _flash_widget_quick(self, widget: QWidget, duration_ms: int) -> None:
        widget.setProperty("flash", True)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        QTimer.singleShot(
            duration_ms,
            lambda: self._clear_flash(widget),
        )

    def _clear_flash(self, widget: QWidget) -> None:
        widget.setProperty("flash", False)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _apply_drop_to_anim_counts(self, drop: DropLocation) -> None:
        if self.anim_counts_you is None or self.anim_counts_opp is None:
            return
        if drop.side == YOU and drop.index is not None:
            self.anim_counts_you[drop.index] += 1
            self._set_pit_text(self.you_buttons_by_index[drop.index], self.anim_counts_you[drop.index])
        elif drop.side == OPP and drop.index is not None:
            self.anim_counts_opp[drop.index] += 1
            self._set_pit_text(self.opp_buttons_by_index[drop.index], self.anim_counts_opp[drop.index])
        elif drop.side == "STORE":
            if drop.store == YOU:
                self.anim_store_you = (self.anim_store_you or 0) + 1
                self.you_store.set_count(self.anim_store_you)
            else:
                self.anim_store_opp = (self.anim_store_opp or 0) + 1
                self.opp_store.set_count(self.anim_store_opp)

    def _apply_bulk_counts(self, drops: List[DropLocation]) -> None:
        for drop in drops:
            self._apply_drop_to_anim_counts(drop)

    def _apply_capture_counts(self, capture: CaptureInfo) -> None:
        if self.anim_counts_you is None or self.anim_counts_opp is None:
            return
        if capture.landing_side == YOU:
            self.anim_counts_you[capture.landing_index] = 0
            self.anim_counts_opp[capture.opposite_index] = 0
            self._set_pit_text(self.you_buttons_by_index[capture.landing_index], 0)
            self._set_pit_text(self.opp_buttons_by_index[capture.opposite_index], 0)
            self.anim_store_you = (self.anim_store_you or 0) + capture.captured_count
            self.you_store.set_count(self.anim_store_you)
        else:
            self.anim_counts_opp[capture.landing_index] = 0
            self.anim_counts_you[capture.opposite_index] = 0
            self._set_pit_text(self.opp_buttons_by_index[capture.landing_index], 0)
            self._set_pit_text(self.you_buttons_by_index[capture.opposite_index], 0)
            self.anim_store_opp = (self.anim_store_opp or 0) + capture.captured_count
            self.opp_store.set_count(self.anim_store_opp)

    def _apply_sweep_counts(self, sweep_you: int, sweep_opp: int) -> None:
        if self.anim_counts_you is None or self.anim_counts_opp is None:
            return
        if sweep_you > 0:
            self.anim_counts_you = [0] * 6
            for idx, button in enumerate(self.you_buttons_by_index):
                self._set_pit_text(button, 0)
            self.anim_store_you = (self.anim_store_you or 0) + sweep_you
            self.you_store.set_count(self.anim_store_you)
        if sweep_opp > 0:
            self.anim_counts_opp = [0] * 6
            for idx, button in enumerate(self.opp_buttons_by_index):
                self._set_pit_text(button, 0)
            self.anim_store_opp = (self.anim_store_opp or 0) + sweep_opp
            self.opp_store.set_count(self.anim_store_opp)

    def _add_dual_flash_step(
        self, group: QSequentialAnimationGroup, widget_a: QWidget, widget_b: QWidget, duration_ms: int
    ) -> None:
        widget_a.setProperty("flash", True)
        widget_b.setProperty("flash", True)
        widget_a.style().unpolish(widget_a)
        widget_a.style().polish(widget_a)
        widget_b.style().unpolish(widget_b)
        widget_b.style().polish(widget_b)
        pause = QPauseAnimation(duration_ms)

        def clear_flash() -> None:
            widget_a.setProperty("flash", False)
            widget_b.setProperty("flash", False)
            widget_a.style().unpolish(widget_a)
            widget_a.style().polish(widget_a)
            widget_b.style().unpolish(widget_b)
            widget_b.style().polish(widget_b)

        pause.finished.connect(clear_flash)
        group.addAnimation(pause)

    def schedule_solve_if_needed(self) -> None:
        if self.closing:
            self.solving = False
            self.requeue_pending = False
            return
        if self.animating or is_terminal(self.state) or self.state.to_move != YOU:
            self.solving = False
            self.requeue_pending = False
            self._finish_slice_progress(hide_after_ms=120)
            self.update_recommendations()
            self.update_status()
            return

        has_result_for_state = self.solve_target_state == self.state and self.search_progress is not None
        if (
            has_result_for_state
            and not self.solving
            and self.search_progress is not None
            and self.search_progress.complete
        ):
            self.requeue_pending = False
            self._finish_slice_progress(hide_after_ms=120)
            self.update_recommendations()
            self.update_status()
            return

        preserve_progress = (
            has_result_for_state
            and not self.solving
            and self.search_progress is not None
            and not self.search_progress.complete
        )
        if preserve_progress and self.search_progress is not None:
            next_depth = max(1, self.search_progress.depth + 1)
            self._start_solve_for(
                self.state,
                preserve_progress=True,
                start_depth=next_depth,
                guess_score=self.search_progress.score,
                previous_result=self.search_progress,
            )
        else:
            self._start_solve_for(
                self.state,
                preserve_progress=False,
                start_depth=1,
                guess_score=None,
                previous_result=None,
            )

        self.update_recommendations()
        self.update_status()

    def _should_requeue_result(self, result: SearchResult) -> bool:
        return (
            not self.closing
            and not result.complete
            and not self.animating
            and self.state.to_move == YOU
            and not is_terminal(self.state)
            and self.solve_target_state == self.state
        )

    def _is_regressive_same_state_result(self, result: SearchResult) -> bool:
        if self.solve_target_state != self.state:
            return False
        previous = self.search_progress
        if previous is None:
            return False
        if previous.complete and not result.complete:
            return True
        if result.depth < previous.depth:
            return True
        if result.depth == previous.depth and result.nodes == 0 and previous.nodes > 0:
            return True
        return False

    def _start_solve_for(
        self,
        state: State,
        preserve_progress: bool = False,
        start_depth: int = 1,
        guess_score: Optional[int] = None,
        previous_result: Optional[SearchResult] = None,
    ) -> None:
        if self.closing:
            return
        self.last_solver_activity = time.perf_counter()
        self.requeue_pending = False
        self.solve_request_id += 1
        self._set_latest_request_id()
        self.solve_target_state = state
        self.deferred_result = None
        self.deferred_result_state = None
        if not preserve_progress:
            self.search_progress = None
            self.last_best_update_time = None
            self.panel_snapshot_key = None
            self.solver_error_text = None
            self.live_nodes = 0
            self.live_elapsed_ms = 0
        if self.fast_slices_remaining > 0:
            self.active_slice_ms = SOLVE_SLICE_MS_FAST
            self.fast_slices_remaining -= 1
        else:
            self.active_slice_ms = SOLVE_SLICE_MS
        self.solving = True
        self.active_start_depth = max(1, start_depth)
        self._start_slice_progress()
        self.solve_requested.emit(
            state,
            self.topn,
            self.solve_request_id,
            start_depth,
            guess_score,
            previous_result,
            self.active_slice_ms,
        )

    def _apply_search_result(self, result: SearchResult) -> None:
        prev = self.search_progress
        self.search_progress = result
        self.current_best_move = result.best_move
        self.current_best_eval = result.score
        self.current_top_moves = list(result.top_moves)
        self.live_nodes = result.nodes
        self.live_elapsed_ms = result.elapsed_ms
        self.solver_error_text = None
        if result.best_move is not None:
            if (
                prev is None
                or result.depth != prev.depth
                or result.best_move != prev.best_move
                or result.score != prev.score
            ):
                self.last_best_update_time = time.perf_counter()
        self.panel_snapshot_key = None

    @Slot(int, object)
    def on_solve_progress(self, request_id: int, result: object) -> None:
        if self.closing:
            return
        if request_id != self.solve_request_id:
            return
        if not isinstance(result, SearchResult):
            return
        if self.animating:
            return
        if self.solve_target_state != self.state:
            return
        self.last_solver_activity = time.perf_counter()
        if self._is_regressive_same_state_result(result):
            self.update_status()
            return
        self._apply_search_result(result)
        self.update_recommendations()
        self.update_status()

    @Slot(int, int, int)
    def on_solve_live(self, request_id: int, nodes: int, elapsed_ms: int) -> None:
        if self.closing:
            return
        if request_id != self.solve_request_id:
            return
        if self.animating:
            return
        if self.solve_target_state != self.state:
            return
        self.last_solver_activity = time.perf_counter()
        self.live_nodes = max(self.live_nodes, nodes)
        self.live_elapsed_ms = max(self.live_elapsed_ms, elapsed_ms)
        self.update_status()

    @Slot(int, object)
    def on_solve_result(self, request_id: int, result: object) -> None:
        if self.closing:
            return
        if request_id != self.solve_request_id:
            return
        if not isinstance(result, SearchResult):
            return
        self.last_solver_activity = time.perf_counter()
        self.solving = False
        hide_after_ms: Optional[int] = None
        if result.complete or self.animating or self.state.to_move != YOU:
            hide_after_ms = 180
        self._finish_slice_progress(hide_after_ms=hide_after_ms)
        if self.animating and self.pending_state is not None and self.solve_target_state == self.pending_state:
            self.deferred_result = result
            self.deferred_result_state = self.pending_state
            return

        if self.solve_target_state != self.state:
            return
        if self._is_regressive_same_state_result(result):
            self.update_status()
            if self._should_requeue_result(result):
                self._queue_requeue_solve()
            return

        self._apply_search_result(result)
        self.update_recommendations()
        self.update_status()
        self._maybe_autoplay(request_id)
        if self._should_requeue_result(result):
            self._queue_requeue_solve()

    @Slot(int, str)
    def on_solve_failed(self, request_id: int, error_text: str) -> None:
        if self.closing:
            return
        if request_id != self.solve_request_id:
            return
        self.last_solver_activity = time.perf_counter()
        self.solving = False
        self.requeue_pending = False
        self._finish_slice_progress(hide_after_ms=180)
        self.search_progress = None
        self.current_best_move = None
        self.current_best_eval = None
        self.current_top_moves = []
        self.last_best_update_time = None
        self.panel_snapshot_key = None
        self.live_nodes = 0
        self.live_elapsed_ms = 0
        self.solver_error_text = error_text
        self.update_recommendations()
        self.update_status()

    def update_pit_labels(self) -> None:
        pits_you, pits_opp, _, _ = self._display_counts()
        for idx, button in enumerate(self.you_buttons_by_index):
            self._set_pit_text(button, pits_you[idx])
        for idx, button in enumerate(self.opp_buttons_by_index):
            self._set_pit_text(button, pits_opp[idx])

    def _set_pit_text(self, button: Optional[PitButton], count: int) -> None:
        if button is None:
            return
        if self.show_numbers_check.isChecked():
            button.setText(f"{button.pit_num}\n{count:02d}")
        else:
            button.setText(f"{count:02d}")

    def _display_counts(self) -> Tuple[List[int], List[int], int, int]:
        if self.anim_counts_you is not None and self.anim_counts_opp is not None:
            store_you = self.anim_store_you if self.anim_store_you is not None else 0
            store_opp = self.anim_store_opp if self.anim_store_opp is not None else 0
            return list(self.anim_counts_you), list(self.anim_counts_opp), store_you, store_opp
        return (
            list(self.state.pits_you),
            list(self.state.pits_opp),
            self.state.store_you,
            self.state.store_opp,
        )

    def refresh_ui(self) -> None:
        self.update_board()
        self.update_recommendations()
        self.update_status()
        self.update_controls()

    def update_controls(self) -> None:
        allow_actions = not self.animating
        self.reset_button.setEnabled(allow_actions)
        self.undo_button.setEnabled(allow_actions and bool(self.history))
        self.replay_button.setEnabled(
            allow_actions and self.last_trace is not None and self.last_pre_counts is not None
        )

    def _toggle_animation_panel(self, checked: bool) -> None:
        if hasattr(self, "anim_panel"):
            self.anim_panel.setVisible(checked)
        self.anim_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    @staticmethod
    def _to_bool(value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value != 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return default

    def _load_persistent_settings(self) -> None:
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        expanded = self._to_bool(
            self.settings.value("view/animation_expanded"),
            self.anim_toggle.isChecked(),
        )
        self.anim_toggle.setChecked(expanded)

        show_numbers = self._to_bool(
            self.settings.value("view/show_pit_numbers"),
            self.show_numbers_check.isChecked(),
        )
        self.show_numbers_check.setChecked(show_numbers)

        speed = self.settings.value("animation/speed")
        if isinstance(speed, str) and self.speed_combo.findText(speed) >= 0:
            self.speed_combo.setCurrentText(speed)

    def _save_persistent_settings(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("view/animation_expanded", self.anim_toggle.isChecked())
        self.settings.setValue("view/show_pit_numbers", self.show_numbers_check.isChecked())
        self.settings.setValue("animation/speed", self.speed_combo.currentText())
        self.settings.sync()

    def _maybe_autoplay(self, expected_request: int) -> None:
        if (
            self.auto_play_check.isChecked()
            and self.state.to_move == YOU
            and self.current_best_move is not None
            and not is_terminal(self.state)
            and not self.animating
        ):
            move_to_play = self.current_best_move

            def _auto_play() -> None:
                if self.solve_request_id != expected_request:
                    return
                if self.state.to_move != YOU or is_terminal(self.state) or self.animating:
                    return
                if self.current_best_move != move_to_play:
                    return
                self.apply_move(move_to_play)

            QTimer.singleShot(0, _auto_play)

    def update_board(self) -> None:
        pits_you, pits_opp, store_you, store_opp = self._display_counts()
        self.opp_store.set_count(store_opp)
        self.you_store.set_count(store_you)

        for idx, button in enumerate(self.you_buttons_by_index):
            count = pits_you[idx]
            self._set_pit_text(button, count)
            if button is None:
                continue
            enabled = (
                not self.animating
                and self.state.to_move == YOU
                and count > 0
                and not is_terminal(self.state)
            )
            button.setEnabled(enabled)

        for idx, button in enumerate(self.opp_buttons_by_index):
            count = pits_opp[idx]
            self._set_pit_text(button, count)
            if button is None:
                continue
            enabled = (
                not self.animating
                and self.state.to_move == OPP
                and count > 0
                and not is_terminal(self.state)
            )
            button.setEnabled(enabled)

    def update_recommendations(self) -> None:
        for button in self.you_buttons_by_index:
            if button is not None:
                button.set_recommended(False)

        if not self.animating and self.state.to_move == YOU and self.current_best_move is not None:
            idx = self.current_best_move - 1
            if 0 <= idx < 6:
                button = self.you_buttons_by_index[idx]
                if button is not None:
                    button.set_recommended(True)

        if is_terminal(self.state):
            self.top_moves_label.setText("-")
            self.play_best_button.setEnabled(False)
            return

        if self.animating:
            self.top_moves_label.setText("-")
            self.play_best_button.setEnabled(False)
            return

        if self.state.to_move != YOU:
            self.top_moves_label.setText("-")
            self.play_best_button.setEnabled(False)
            return

        if not self.current_top_moves:
            self.top_moves_label.setText("-")
            self.play_best_button.setEnabled(False)
            return

        parts = [f"pit {move} ({score:+d})" for move, score in self.current_top_moves]
        self.top_moves_label.setText(" | ".join(parts))
        self.play_best_button.setEnabled(self.current_best_move is not None)

    def update_status(self) -> None:
        completed_depth = 0
        probe_depth: Optional[int] = None
        nodes = 0
        elapsed_ms = 0
        best_text = "-"
        solved = False

        if self.search_progress is not None:
            completed_depth = self.search_progress.depth
            nodes = self.search_progress.nodes
            elapsed_ms = self.search_progress.elapsed_ms
            solved = self.search_progress.complete
            if self.search_progress.best_move is not None:
                best_text = f"pit {self.search_progress.best_move} ({self.search_progress.score:+d})"

        if self.solving:
            nodes = max(nodes, self.live_nodes)
            elapsed_ms = max(elapsed_ms, self.live_elapsed_ms)

        turn_text = "Game over" if is_terminal(self.state) else ("Your turn" if self.state.to_move == YOU else "Opponent turn")

        if is_terminal(self.state):
            state_text = "State: Game over"
            solved = True
            if best_text == "-":
                best_text = f"diff {final_diff(self.state):+d}"
        else:
            if self.solver_error_text is not None and not self.solving:
                state_text = f"State: Solver error (see logs) | Turn: {turn_text}"
            elif solved:
                state_text = f"State: Solved (perfect) | Turn: {turn_text}"
            elif self.solving and self.state.to_move == YOU:
                state_text = f"State: Searching (best so far) | Turn: {turn_text}"
            elif self.search_progress is not None and self.state.to_move == YOU:
                state_text = f"State: Best so far | Turn: {turn_text}"
            else:
                state_text = f"State: Idle | Turn: {turn_text}"

        if self.solving and self.state.to_move == YOU and not solved:
            if self.search_progress is None:
                probe_depth = self.active_start_depth
            else:
                probe_depth = max(self.active_start_depth, completed_depth + 1)

        depth_text = f"Depth: {completed_depth} complete"
        if probe_depth is not None:
            depth_text += f", probing {probe_depth}"

        best_line = f"Best: {best_text}"
        panel_key = (best_line, depth_text)
        if panel_key != self.panel_snapshot_key:
            self.panel_snapshot_key = panel_key
            self.solve_best_label.setText(best_line)
            self.solve_depth_label.setText(depth_text)

        nps = 0
        if elapsed_ms > 0:
            nps = int(nodes * 1000 / elapsed_ms)
        elapsed_s = elapsed_ms / 1000.0
        self.solve_state_label.setText(state_text)

        if self.last_best_update_time is None:
            self.solve_age_label.setText("Last best update: -")
        else:
            age_s = max(0.0, time.perf_counter() - self.last_best_update_time)
            self.solve_age_label.setText(f"Last best update: {age_s:.1f}s ago")

        callback_age_s = max(0.0, time.perf_counter() - self.last_solver_activity)
        if self.solving:
            spinner = "|/-\\"[self.search_heartbeat_phase % 4]
            if callback_age_s < 1.5:
                heartbeat_text = f"Solver heartbeat: active {spinner} (last callback {callback_age_s:.1f}s ago)"
            elif callback_age_s < 5.0:
                heartbeat_text = f"Solver heartbeat: working {spinner} (last callback {callback_age_s:.1f}s ago)"
            else:
                heartbeat_text = (
                    "Solver heartbeat: heavy branch "
                    f"{spinner} (last callback {callback_age_s:.1f}s ago)"
                )
        else:
            heartbeat_text = "Solver heartbeat: idle"
        self.solve_heartbeat_label.setText(heartbeat_text)

        self.solve_metrics_label.setText(
            f"Nodes {self._format_nodes(nodes)} | NPS {self._format_nodes(nps)}/s | Elapsed {elapsed_s:.1f}s"
        )

    def _format_drop_location(self, drop: DropLocation) -> str:
        if drop.side == "STORE":
            return "YOUR STORE" if drop.store == YOU else "OPP STORE"
        if drop.index is None:
            return f"{drop.side} pit ?"
        pit_num = drop.index + 1
        return f"{drop.side} pit {pit_num}"

    def _finalize_cache_save_before_close(self, timeout_ms: int) -> None:
        deadline = time.perf_counter() + (max(0, timeout_ms) / 1000.0)

        def _remaining_ms() -> int:
            return max(0, int((deadline - time.perf_counter()) * 1000))

        self._update_shutdown_status("Finalizing cache save...", progress=60)
        self._poll_cache_save_completion()
        if self.solver_worker.is_cache_dirty() and not self._cache_save_in_progress():
            snapshot_budget_ms = min(CACHE_CLOSE_SNAPSHOT_BUDGET_MS, _remaining_ms())
            snapshot_with_counter = self.solver_worker.try_snapshot_cache_with_counter_budget(snapshot_budget_ms)
            if snapshot_with_counter is not None:
                snapshot, mutation_counter = snapshot_with_counter
                self._start_cache_save(snapshot, mutation_counter)
                self._update_shutdown_status("Saving cache snapshot...", progress=66)
            else:
                self._update_shutdown_status(
                    "Skipping final snapshot (time budget or solver busy).",
                    progress=84,
                )
        else:
            if self.solver_worker.is_cache_dirty():
                self._update_shutdown_status("Cache save already in progress...", progress=66)
            else:
                self._update_shutdown_status("Cache already up to date", progress=95)

        self._wait_for_cache_save_completion(
            _remaining_ms(),
            phase_text="Saving cache...",
            progress_start=62,
            progress_end=95,
        )
        self._poll_cache_save_completion()
        self._shutdown_cache_save_worker(_remaining_ms())

    def closeEvent(self, event) -> None:
        if self.closing:
            event.accept()
            return
        self.closing = True
        self._show_shutdown_dialog()
        self._update_shutdown_status("Preparing to close...", progress=5)
        self.autosave_enabled = False
        self.cache_autosave_timer.stop()
        self.solving = False
        self.requeue_pending = False
        self._save_persistent_settings()
        self.solve_request_id += 1
        self._set_latest_request_id()
        self.slice_progress_timer.stop()
        self.slice_start_time = None
        self.slice_progress.hide()
        self.slice_progress_label.hide()
        if self.anim_group is not None:
            try:
                self.anim_group.finished.disconnect()
            except (RuntimeError, TypeError):
                pass
            self.anim_group.stop()
            self.anim_group.deleteLater()
            self.anim_group = None
        try:
            self.solve_requested.disconnect(self.solver_worker.solve)
        except (RuntimeError, TypeError):
            pass
        try:
            self._update_shutdown_status("Stopping solver...", progress=10)
            idle = self.solver_worker.wait_for_idle(SOLVER_CLOSE_TIMEOUT_MS)
            if idle:
                self._finalize_cache_save_before_close(CACHE_CLOSE_SAVE_TIMEOUT_MS)
            else:
                self._update_shutdown_status("Solver still busy; skipping final cache save", progress=88)
            stopped = self.solver_worker.shutdown(SOLVER_CLOSE_TIMEOUT_MS)
            if not stopped:
                self._update_shutdown_status("Solver did not stop cleanly", progress=94)
            self._update_shutdown_status("Finishing...", progress=98)
            self.solver_worker.close()
            self._update_shutdown_status("Closed", progress=100)
        finally:
            self._hide_shutdown_dialog()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.overlay is not None and self.centralWidget() is not None:
            self.overlay.setGeometry(self.centralWidget().rect())


def main() -> int:
    app = QApplication(sys.argv)
    window = MancalaWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
