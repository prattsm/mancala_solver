"""PySide6 GUI for GamePigeon Mancala (Capture mode) solver."""

from __future__ import annotations

import sys
import threading
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
from mancala_solver import best_move, default_cache_path, load_tt, save_tt


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

    def __init__(self) -> None:
        super().__init__()
        self.tt = {}
        self.tt_lock = threading.Lock()

    @Slot(object, int, int)
    def solve(self, state: State, topn: int, request_id: int) -> None:
        with self.tt_lock:
            move, eval_score, top_moves = best_move(state, topn=topn, tt=self.tt)
        self.result_ready.emit(request_id, (move, eval_score, top_moves))

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
    solve_requested = Signal(object, int, int)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GamePigeon Mancala — Capture Mode")
        self.setMinimumSize(900, 720)

        self.cache_path: Path = default_cache_path()

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
        self.deferred_result: Optional[Tuple[Optional[int], int, List[Tuple[int, int]]]] = None
        self.deferred_result_state: Optional[State] = None
        self.solving = False
        self.animating = False

        self.anim_counts_you: Optional[List[int]] = None
        self.anim_counts_opp: Optional[List[int]] = None
        self.anim_store_you: Optional[int] = None
        self.anim_store_opp: Optional[int] = None

        self.pending_state: Optional[State] = None
        self.pending_trace: Optional[MoveTrace] = None
        self.pending_pre_counts: Optional[CountsSnapshot] = None

        self.you_buttons_by_index: List[PitButton] = [None] * 6
        self.opp_buttons_by_index: List[PitButton] = [None] * 6

        self.overlay: Optional[QWidget] = None
        self.anim_group: Optional[QSequentialAnimationGroup] = None

        self._build_ui()
        self._setup_solver()
        self._apply_style()

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
        self.anim_toggle.setChecked(True)
        self.anim_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.anim_toggle.setArrowType(Qt.DownArrow)
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

        side_panel.addWidget(self.anim_panel)

        self.top_moves_label = QLabel("Top moves: -")
        self.top_moves_label.setObjectName("TopMoves")
        self.top_moves_label.setWordWrap(True)
        side_panel.addWidget(self.top_moves_label)

        self.status_label = QLabel("Status: -")
        self.status_label.setObjectName("Status")
        self.status_label.setWordWrap(True)
        side_panel.addWidget(self.status_label)

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
        self.solver_worker.set_cache(load_tt(self.cache_path))
        self.solver_worker.moveToThread(self.solver_thread)
        self.solve_requested.connect(self.solver_worker.solve)
        self.solver_worker.result_ready.connect(self.on_solve_result)
        self.solver_thread.start()

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
            QLabel#Status { padding-top: 6px; }
            QLabel#TopMoves { color: #f0e6d6; }
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
        self.solve_request_id += 1
        self.solving = False
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
        self.solve_request_id += 1
        self.solving = False
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
        self.solving = False
        self.solve_request_id += 1

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
            move, eval_score, top_moves = self.deferred_result
            self.current_best_move = move
            self.current_best_eval = eval_score
            self.current_top_moves = top_moves
            self.deferred_result = None
            self.deferred_result_state = None
            self.refresh_ui()
            self._maybe_autoplay(self.solve_request_id)
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
            self.anim_group.stop()
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

        self.anim_group.finished.connect(lambda: self._finish_animation(commit))
        self.anim_group.start()

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
        if side == YOU:
            return self.you_buttons_by_index[index]
        if side == OPP:
            return self.opp_buttons_by_index[index]
        return None

    def _widget_for_drop(self, drop: DropLocation) -> Optional[QWidget]:
        if drop.side == "STORE":
            return self.you_store if drop.store == YOU else self.opp_store
        return self._widget_for_pit(drop.side, drop.index or 0)

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
            end_pos = self._center_for_widget(self._widget_for_drop(drop))
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
                dest_widget = self._widget_for_drop(drop)
                if dest_widget is not None:
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
            current_pos = self._center_for_widget(self._widget_for_drop(middle[-1]))

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

        bundle_text = f"x{capture.captured_count}" if capture.captured_count > 1 else ""
        token_a = self._create_token(start_center, 18, bundle_text, "SeedBundle")
        token_b = self._create_token(start_center_opp, 18, bundle_text, "SeedBundle")

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
            start = self._center_for_widget(self.you_buttons_by_index[2])
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
            start = self._center_for_widget(self.opp_buttons_by_index[3])
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
        if self.animating or is_terminal(self.state) or self.state.to_move != YOU:
            self.solving = False
            self.update_recommendations()
            self.update_status()
            return

        if self.solve_target_state == self.state and self.current_best_move is not None and not self.solving:
            self.update_recommendations()
            self.update_status()
            return

        self._start_solve_for(self.state)

        self.update_recommendations()
        self.update_status()

    def _start_solve_for(self, state: State) -> None:
        self.solve_request_id += 1
        self.solve_target_state = state
        self.deferred_result = None
        self.deferred_result_state = None
        self.solving = True
        self.solve_requested.emit(state, self.topn, self.solve_request_id)

    @Slot(int, object)
    def on_solve_result(self, request_id: int, result: object) -> None:
        if request_id != self.solve_request_id:
            return
        move, eval_score, top_moves = result
        self.solving = False
        if self.animating and self.pending_state is not None and self.solve_target_state == self.pending_state:
            self.deferred_result = (move, eval_score, top_moves)
            self.deferred_result_state = self.pending_state
            return

        if self.solve_target_state != self.state:
            return

        self.current_best_move = move
        self.current_best_eval = eval_score
        self.current_top_moves = top_moves
        self.update_recommendations()
        self.update_status()
        self._maybe_autoplay(request_id)

    def update_pit_labels(self) -> None:
        pits_you, pits_opp, _, _ = self._display_counts()
        for idx, button in enumerate(self.you_buttons_by_index):
            self._set_pit_text(button, pits_you[idx])
        for idx, button in enumerate(self.opp_buttons_by_index):
            self._set_pit_text(button, pits_opp[idx])

    def _set_pit_text(self, button: PitButton, count: int) -> None:
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
            enabled = (
                not self.animating
                and self.state.to_move == OPP
                and count > 0
                and not is_terminal(self.state)
            )
            button.setEnabled(enabled)

    def update_recommendations(self) -> None:
        for button in self.you_buttons_by_index:
            button.set_recommended(False)

        if not self.animating and self.state.to_move == YOU and self.current_best_move is not None:
            idx = self.current_best_move - 1
            if 0 <= idx < 6:
                self.you_buttons_by_index[idx].set_recommended(True)

        if is_terminal(self.state):
            self.top_moves_label.setText("Top moves: -")
            self.play_best_button.setEnabled(False)
            return

        if self.animating:
            self.top_moves_label.setText("Top moves: animating...")
            self.play_best_button.setEnabled(False)
            return

        if self.state.to_move != YOU:
            self.top_moves_label.setText("Top moves: waiting for opponent")
            self.play_best_button.setEnabled(False)
            return

        if self.solving:
            self.top_moves_label.setText("Top moves: solving...")
            self.play_best_button.setEnabled(False)
            return

        if not self.current_top_moves:
            self.top_moves_label.setText("Top moves: -")
            self.play_best_button.setEnabled(False)
            return

        parts = []
        for idx, (move, score) in enumerate(self.current_top_moves):
            if idx == 0:
                parts.append(f"Best: pit {move} ({score:+d})")
            elif idx == 1:
                parts.append(f"Next: pit {move} ({score:+d})")
            else:
                parts.append(f"pit {move} ({score:+d})")
        self.top_moves_label.setText(" | ".join(parts))
        self.play_best_button.setEnabled(self.current_best_move is not None and not self.animating)

    def update_status(self) -> None:
        if is_terminal(self.state):
            self.status_label.setText(
                f"Game over. You {self.state.store_you} - Opponent {self.state.store_opp} "
                f"(diff {final_diff(self.state):+d})"
            )
            return

        turn_text = "Your turn" if self.state.to_move == YOU else "Opponent turn"
        status_parts = [f"Turn: {turn_text}"]

        trace = self.pending_trace if self.animating and self.pending_trace is not None else self.last_trace
        if trace is not None:
            landed = self._format_drop_location(trace.last_drop)
            status_parts.append(f"Moved from: {trace.mover} pit {trace.picked_pit} -> Landed: {landed}")
            if trace.capture is not None:
                status_parts.append(
                    f"Capture: YES (captured {trace.capture.captured_count} from opposite pit)"
                )
            else:
                status_parts.append("Capture: NO")
            status_parts.append(f"Extra turn: {'YES' if trace.extra_turn else 'NO'}")
        else:
            status_parts.append(f"Last: {self.last_move_desc}")

        if self.animating:
            status_parts.append("Animating...")
        if self.solving and self.state.to_move == YOU:
            status_parts.append("Solving...")
        self.status_label.setText(" | ".join(status_parts))

    def _format_drop_location(self, drop: DropLocation) -> str:
        if drop.side == "STORE":
            return "YOUR STORE" if drop.store == YOU else "OPP STORE"
        pit_num = (drop.index or 0) + 1
        return f"{drop.side} pit {pit_num}"

    def closeEvent(self, event) -> None:
        cache_snapshot = self.solver_worker.try_snapshot_cache()
        if cache_snapshot is not None:
            save_tt(cache_snapshot, self.cache_path)
        self.solver_thread.requestInterruption()
        self.solver_thread.quit()
        self.solver_thread.wait(0)
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
