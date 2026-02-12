"""PySide6 GUI for GamePigeon Mancala (Capture mode) solver."""

from __future__ import annotations

import sys
import threading
import time
import traceback
import os
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
    QThread,
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

SOLVE_SLICE_MS = 500
SOLVE_REQUEUE_DELAY_MS = 50
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


class SolverWorker(QObject):
    result_ready = Signal(int, object)
    progress = Signal(int, object)
    live_metrics = Signal(int, int, int)
    solve_failed = Signal(int, str)

    def __init__(self) -> None:
        super().__init__()
        self.tt = {}
        self.tt_lock = threading.Lock()
        self.latest_request_id = 0
        self.telemetry_sink: Optional[ThreadedTCPSink] = None
        endpoint_raw = os.environ.get("MANCALA_TELEMETRY", "").strip()
        endpoint = parse_host_port(endpoint_raw) if endpoint_raw else None
        if endpoint is not None:
            self.telemetry_sink = ThreadedTCPSink(endpoint[0], endpoint[1])

    def set_latest_request_id(self, request_id: int) -> None:
        self.latest_request_id = request_id

    @Slot(object, int, int, int, object, object)
    def solve(
        self,
        state: State,
        topn: int,
        request_id: int,
        start_depth: int,
        guess_score: Optional[int],
        previous_result: Optional[SearchResult],
    ) -> None:
        def _is_interrupted() -> bool:
            if QThread.currentThread().isInterruptionRequested():
                return True
            return request_id != self.latest_request_id

        if _is_interrupted():
            return

        def _on_progress(result: SearchResult) -> None:
            if _is_interrupted():
                raise InterruptedError()
            self.progress.emit(request_id, result)

        def _on_live(nodes: int, elapsed_ms: int) -> None:
            if _is_interrupted():
                raise InterruptedError()
            self.live_metrics.emit(request_id, nodes, elapsed_ms)

        try:
            with self.tt_lock:
                result = solve_best_move(
                    state,
                    topn=topn,
                    tt=self.tt,
                    time_limit_ms=SOLVE_SLICE_MS,
                    progress_callback=_on_progress,
                    start_depth=start_depth,
                    guess_score=guess_score,
                    previous_result=previous_result,
                    interrupt_check=_is_interrupted,
                    live_callback=_on_live,
                    telemetry_sink=self.telemetry_sink,
                )
        except InterruptedError:
            return
        except Exception as exc:
            traceback.print_exc()
            if _is_interrupted():
                return
            self.solve_failed.emit(request_id, f"{type(exc).__name__}: {exc}")
            return

        if _is_interrupted():
            return
        self.result_ready.emit(request_id, result)

    def set_cache(self, tt) -> None:
        with self.tt_lock:
            self.tt = tt

    def snapshot_cache(self) -> dict:
        with self.tt_lock:
            return dict(self.tt)

    def try_snapshot_cache(self) -> Optional[dict]:
        if not self.tt_lock.acquire(blocking=False):
            return None
        try:
            return dict(self.tt)
        finally:
            self.tt_lock.release()

    def close(self) -> None:
        if self.telemetry_sink is not None:
            self.telemetry_sink.close()


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
    solve_requested = Signal(object, int, int, int, object, object)

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
        self.search_heartbeat_phase = 0
        self.last_best_update_time: Optional[float] = None
        self.panel_snapshot_key: Optional[Tuple[str, str]] = None
        self.solver_error_text: Optional[str] = None
        self.live_nodes = 0
        self.live_elapsed_ms = 0

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
        self.solver_thread = QThread(self)
        self.solver_worker = SolverWorker()
        self.solver_worker.set_latest_request_id(self.solve_request_id)
        self.solver_worker.set_cache(load_tt(self.cache_path))
        self.solver_worker.moveToThread(self.solver_thread)
        self.solve_requested.connect(self.solver_worker.solve)
        self.solver_worker.progress.connect(self.on_solve_progress)
        self.solver_worker.live_metrics.connect(self.on_solve_live)
        self.solver_worker.result_ready.connect(self.on_solve_result)
        self.solver_worker.solve_failed.connect(self.on_solve_failed)
        self.solver_thread.start()

    def _set_latest_request_id(self) -> None:
        if hasattr(self, "solver_worker"):
            self.solver_worker.set_latest_request_id(self.solve_request_id)

    def _start_slice_progress(self) -> None:
        self.slice_hide_token += 1
        self.slice_start_time = time.perf_counter()
        self.search_heartbeat_phase = 0
        self.slice_progress.setRange(0, SOLVE_SLICE_MS)
        self.slice_progress.setValue(0)
        self.slice_progress_label.setText(f"Slice: 0/{SOLVE_SLICE_MS} ms")
        self.slice_progress_label.show()
        self.slice_progress.show()
        if not self.slice_progress_timer.isActive():
            self.slice_progress_timer.start()
        self.update_status()

    def _update_slice_progress_bar(self) -> None:
        if self.slice_start_time is None:
            return
        elapsed_ms = int((time.perf_counter() - self.slice_start_time) * 1000)
        shown_ms = min(SOLVE_SLICE_MS, max(0, elapsed_ms))
        self.slice_progress.setValue(shown_ms)
        self.slice_progress_label.setText(f"Slice: {shown_ms}/{SOLVE_SLICE_MS} ms")
        self.search_heartbeat_phase = (self.search_heartbeat_phase + 1) % 4
        self.update_status()

    def _finish_slice_progress(self, hide_after_ms: Optional[int] = None) -> None:
        self.slice_start_time = None
        self.slice_progress_timer.stop()
        self.slice_progress.setValue(SOLVE_SLICE_MS)
        self.slice_progress_label.setText(f"Slice: {SOLVE_SLICE_MS}/{SOLVE_SLICE_MS} ms")
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

    def closeEvent(self, event) -> None:
        self.closing = True
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
        self.solver_thread.requestInterruption()
        self.solver_thread.quit()
        stopped = self.solver_thread.wait(3000)
        if not stopped:
            self.solver_thread.terminate()
            self.solver_thread.wait(500)
        self.solver_worker.close()
        cache_snapshot = self.solver_worker.try_snapshot_cache()
        if cache_snapshot is not None:
            save_tt(cache_snapshot, self.cache_path)
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
