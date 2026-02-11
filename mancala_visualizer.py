"""Live telemetry visualizer for Mancala solver search."""

from __future__ import annotations

import argparse
from collections import deque
import json
import queue
import socket
import threading
import time
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from mancala_engine import apply_move_fast_with_info, initial_state, is_terminal, legal_moves, pretty_print
from mancala_solver import key_to_state, solve_best_move
from mancala_telemetry import QueueTelemetrySink, TelemetryEnvelope, parse_host_port

try:
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    tk = None
    ttk = None


SPINNER = ("|", "/", "-", "\\")


class TelemetryTCPServer:
    def __init__(self, host: str, port: int, event_queue: "queue.Queue[object]") -> None:
        self.host = host
        self.port = port
        self.event_queue = event_queue
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="mancala-telemetry-server", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=0.6)

    def _run(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(0.5)
        self.event_queue.put({"event": "_listen", "data": {"host": self.host, "port": self.port}})
        try:
            while not self.stop_event.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                self.event_queue.put(
                    {
                        "event": "_connection",
                        "data": {"connected": True, "peer": f"{addr[0]}:{addr[1]}"},
                    }
                )
                conn.settimeout(0.5)
                buffer = b""
                try:
                    while not self.stop_event.is_set():
                        try:
                            chunk = conn.recv(4096)
                        except socket.timeout:
                            continue
                        if not chunk:
                            break
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            if not line:
                                continue
                            try:
                                payload = json.loads(line.decode("utf-8"))
                            except json.JSONDecodeError:
                                continue
                            if isinstance(payload, dict):
                                self.event_queue.put(payload)
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
                self.event_queue.put({"event": "_connection", "data": {"connected": False}})
        finally:
            try:
                server.close()
            except OSError:
                pass


class DemoRunner:
    def __init__(
        self,
        event_queue: "queue.Queue[object]",
        seeds: int,
        topn: int,
        slice_ms_getter,
    ) -> None:
        self.event_queue = event_queue
        self.seeds = max(0, seeds)
        self.topn = max(1, topn)
        self.slice_ms_getter = slice_ms_getter
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="mancala-visualizer-demo", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=0.6)

    def _run(self) -> None:
        sink = QueueTelemetrySink(self.event_queue)
        state = initial_state(seeds=self.seeds, you_first=True)
        tt = {}
        start_depth = 1
        guess_score: Optional[int] = None
        self.event_queue.put({"event": "_connection", "data": {"connected": True, "peer": "demo"}})
        try:
            while not self.stop_event.is_set():
                slice_ms = max(20, int(self.slice_ms_getter()))
                result = solve_best_move(
                    state,
                    topn=self.topn,
                    tt=tt,
                    time_limit_ms=slice_ms,
                    start_depth=start_depth,
                    guess_score=guess_score,
                    telemetry_sink=sink,
                )
                if self.stop_event.is_set():
                    break
                if result.complete:
                    if result.best_move is not None and result.best_move in legal_moves(state):
                        state, _, _ = apply_move_fast_with_info(state, result.best_move)
                    if is_terminal(state):
                        state = initial_state(seeds=self.seeds, you_first=True)
                    start_depth = 1
                    guess_score = None
                else:
                    start_depth = max(1, result.depth + 1)
                    guess_score = result.score
        finally:
            self.event_queue.put({"event": "_connection", "data": {"connected": False}})


class VisualizerApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.root.title(args.title)
        self.root.geometry("1220x860")
        self.root.minsize(980, 720)

        self.event_queue: "queue.Queue[object]" = queue.Queue()
        self.server: Optional[TelemetryTCPServer] = None
        self.demo_runner: Optional[DemoRunner] = None
        self.connected = False
        self.peer_label_text = "-"
        self.listen_label_text = "-"

        self.state_key = "-"
        self.board_text = "-"
        self.best_move: Optional[int] = None
        self.best_score = 0
        self.best_complete = False
        self.best_changed_at: Optional[float] = None
        self.search_reason = "-"
        self.depth_complete = 0
        self.depth_probing = 0
        self.last_elapsed_ms = 0
        self.nodes_total = 0
        self.nps = 0
        self.tt_hit_rate = 0.0
        self.tt_hits = 0
        self.tt_probes = 0
        self.tt_exact_reuse = 0
        self.tt_bound_reuse = 0
        self.cutoffs = 0
        self.eval_calls = 0
        self.branching = 0.0
        self.asp_window = 0
        self.asp_retries = 0
        self.max_depth = 0
        self.pv_moves: List[int] = []
        self.pv_scored: List[Tuple[int, Optional[int], str]] = []
        self.root_scores: Dict[int, int] = {}
        self.depth_hist: Dict[int, int] = {}
        self.cutoff_hist: Dict[int, int] = {}
        self.heartbeat_idx = 0
        self.last_batch_wall = 0.0

        self.nodes_series: Deque[Tuple[float, float]] = deque(maxlen=320)
        self.nps_series: Deque[Tuple[float, float]] = deque(maxlen=320)
        self.depth_series: Deque[Tuple[float, float]] = deque(maxlen=320)
        self.hit_rate_series: Deque[Tuple[float, float]] = deque(maxlen=320)

        self.slice_ms_var = tk.IntVar(value=max(40, args.slice_ms))
        self.slow_mode_var = tk.BooleanVar(value=False)

        self._build_ui()
        self._schedule_poll()

        if args.demo:
            self.start_demo()
        else:
            self.start_listener()

    def _build_ui(self) -> None:
        root = ttk.Frame(self.root, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(root)
        controls.pack(fill=tk.X, pady=(0, 8))
        self.connection_label = ttk.Label(controls, text="Disconnected")
        self.connection_label.pack(side=tk.LEFT, padx=(0, 8))
        self.peer_label = ttk.Label(controls, text="peer: -")
        self.peer_label.pack(side=tk.LEFT, padx=(0, 8))
        self.listen_label = ttk.Label(controls, text="listen: -")
        self.listen_label.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Button(controls, text="Start Listen", command=self.start_listener).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Stop Listen", command=self.stop_listener).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Start Demo", command=self.start_demo).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Stop Demo", command=self.stop_demo).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(controls, text="Slow-motion", variable=self.slow_mode_var).pack(side=tk.LEFT, padx=(16, 4))
        ttk.Scale(controls, from_=40, to=1200, variable=self.slice_ms_var, orient=tk.HORIZONTAL).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4)
        )
        self.slice_label = ttk.Label(controls, text="slice ms")
        self.slice_label.pack(side=tk.LEFT)

        status = ttk.Frame(root)
        status.pack(fill=tk.X, pady=(0, 8))
        self.status_line = ttk.Label(status, text="State: idle")
        self.status_line.pack(anchor=tk.W)
        self.metrics_line = ttk.Label(status, text="Nodes 0 | NPS 0 | TT hit 0.0%")
        self.metrics_line.pack(anchor=tk.W)
        self.best_line = ttk.Label(status, text="Best: -")
        self.best_line.pack(anchor=tk.W)
        self.pv_line = ttk.Label(status, text="PV: -")
        self.pv_line.pack(anchor=tk.W)

        panes = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(panes)
        right = ttk.Frame(panes)
        panes.add(left, weight=3)
        panes.add(right, weight=2)

        charts = ttk.Frame(left)
        charts.pack(fill=tk.BOTH, expand=True)

        self.nodes_canvas = tk.Canvas(charts, width=560, height=150, bg="#10151c", highlightthickness=0)
        self.nodes_canvas.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.nps_canvas = tk.Canvas(charts, width=560, height=150, bg="#10151c", highlightthickness=0)
        self.nps_canvas.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.depth_canvas = tk.Canvas(charts, width=560, height=150, bg="#10151c", highlightthickness=0)
        self.depth_canvas.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        self.hit_rate_canvas = tk.Canvas(charts, width=560, height=150, bg="#10151c", highlightthickness=0)
        self.hit_rate_canvas.grid(row=3, column=0, sticky="nsew", padx=4, pady=4)
        charts.rowconfigure((0, 1, 2, 3), weight=1)
        charts.columnconfigure(0, weight=1)

        right_top = ttk.Frame(right)
        right_top.pack(fill=tk.BOTH, expand=True)

        self.board_label = tk.Label(
            right_top,
            text="-",
            justify=tk.LEFT,
            anchor="nw",
            font=("Courier New", 10),
            bg="#0f131a",
            fg="#d7dde7",
            padx=8,
            pady=8,
        )
        self.board_label.pack(fill=tk.X, pady=(0, 8))

        self.heartbeat_label = ttk.Label(right_top, text="heartbeat: idle")
        self.heartbeat_label.pack(anchor=tk.W, pady=(0, 8))

        self.root_scores_label = tk.Label(
            right_top,
            text="Root scores: -",
            justify=tk.LEFT,
            anchor="nw",
            font=("Courier New", 10),
            bg="#111820",
            fg="#d7dde7",
            padx=8,
            pady=8,
        )
        self.root_scores_label.pack(fill=tk.X, pady=(0, 8))

        self.pv_tree_label = tk.Label(
            right_top,
            text="PV tree: -",
            justify=tk.LEFT,
            anchor="nw",
            font=("Courier New", 10),
            bg="#111820",
            fg="#d7dde7",
            padx=8,
            pady=8,
        )
        self.pv_tree_label.pack(fill=tk.X, pady=(0, 8))

        self.depth_hist_canvas = tk.Canvas(right_top, width=360, height=140, bg="#10151c", highlightthickness=0)
        self.depth_hist_canvas.pack(fill=tk.X, pady=4)
        self.cutoff_hist_canvas = tk.Canvas(right_top, width=360, height=140, bg="#10151c", highlightthickness=0)
        self.cutoff_hist_canvas.pack(fill=tk.X, pady=4)

    def _schedule_poll(self) -> None:
        self._drain_events()
        self._render()
        self.root.after(50, self._schedule_poll)

    def _drain_events(self) -> None:
        while True:
            try:
                raw = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(raw, TelemetryEnvelope):
                self._handle_event(raw.event, raw.data)
            elif isinstance(raw, dict):
                event_name = raw.get("event")
                data = raw.get("data", {})
                if isinstance(event_name, str) and isinstance(data, dict):
                    self._handle_event(event_name, data)

    def _handle_event(self, event: str, data: Dict[str, object]) -> None:
        if event == "_listen":
            self.listen_label_text = f"{data.get('host')}:{data.get('port')}"
            return
        if event == "_connection":
            self.connected = bool(data.get("connected"))
            peer = data.get("peer")
            self.peer_label_text = str(peer) if peer is not None else "-"
            return

        if event == "search_start":
            state_key = str(data.get("state_key", "-"))
            start_depth = int(data.get("start_depth", 1) or 1)
            if state_key != self.state_key or start_depth <= 1:
                self.nodes_series.clear()
                self.nps_series.clear()
                self.depth_series.clear()
                self.hit_rate_series.clear()
                self.root_scores.clear()
                self.pv_moves = []
                self.pv_scored = []
                self.search_reason = "-"
            self.state_key = state_key
            parsed = key_to_state(state_key)
            if parsed is not None:
                self.board_text = pretty_print(parsed)
            else:
                self.board_text = state_key
            return

        if event == "iteration_start":
            self.depth_probing = int(data.get("depth", 0) or 0)
            return

        if event == "iteration_done":
            self.depth_complete = int(data.get("depth", 0) or 0)
            self.best_complete = bool(data.get("complete"))
            move_raw = data.get("best_move")
            self.best_move = int(move_raw) if isinstance(move_raw, int) else None
            self.best_score = int(data.get("score", 0) or 0)
            self.max_depth = max(self.max_depth, int(data.get("max_depth", 0) or 0))
            self.asp_window = int(data.get("aspiration_window", 0) or 0)
            self.asp_retries = int(data.get("aspiration_retries", 0) or 0)
            elapsed_ms = int(data.get("elapsed_ms", self.last_elapsed_ms) or self.last_elapsed_ms)
            self.last_elapsed_ms = max(self.last_elapsed_ms, elapsed_ms)
            self.depth_series.append((elapsed_ms / 1000.0, float(self.depth_complete)))
            root_scores_payload = data.get("root_scores")
            self.root_scores.clear()
            if isinstance(root_scores_payload, list):
                for entry in root_scores_payload:
                    if isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], int):
                        self.root_scores[entry[0]] = int(entry[1])
                    elif isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], int):
                        self.root_scores[entry[0]] = int(entry[1])
            if self.best_move is not None:
                self.best_changed_at = time.perf_counter()
            return

        if event == "pv_update":
            pv_raw = data.get("pv_moves")
            pv: List[int] = []
            if isinstance(pv_raw, list):
                for value in pv_raw:
                    if isinstance(value, int):
                        pv.append(value)
            self.pv_moves = pv
            scored_raw = data.get("pv_scored")
            parsed_scored: List[Tuple[int, Optional[int], str]] = []
            if isinstance(scored_raw, list):
                for entry in scored_raw:
                    if isinstance(entry, (list, tuple)) and len(entry) == 3 and isinstance(entry[0], int):
                        move = entry[0]
                        score_raw = entry[1]
                        bound_raw = entry[2]
                        score = int(score_raw) if isinstance(score_raw, int) else None
                        bound = str(bound_raw) if isinstance(bound_raw, str) else "exact"
                        parsed_scored.append((move, score, bound))
            self.pv_scored = parsed_scored
            return

        if event == "node_batch":
            self.nodes_total = int(data.get("nodes_total", self.nodes_total) or self.nodes_total)
            self.nps = int(data.get("nps_estimate", self.nps) or self.nps)
            self.tt_hits = int(data.get("tt_hits", self.tt_hits) or self.tt_hits)
            self.tt_probes = int(data.get("tt_probes", self.tt_probes) or self.tt_probes)
            self.tt_exact_reuse = int(data.get("tt_exact_reuse", self.tt_exact_reuse) or self.tt_exact_reuse)
            self.tt_bound_reuse = int(data.get("tt_bound_reuse", self.tt_bound_reuse) or self.tt_bound_reuse)
            self.cutoffs = int(data.get("cutoffs", self.cutoffs) or self.cutoffs)
            self.eval_calls = int(data.get("eval_calls", self.eval_calls) or self.eval_calls)
            self.branching = float(data.get("branching_factor_estimate", self.branching) or self.branching)
            self.asp_window = int(data.get("aspiration_window", self.asp_window) or self.asp_window)
            self.asp_retries = int(data.get("aspiration_retries", self.asp_retries) or self.asp_retries)
            self.max_depth = max(self.max_depth, int(data.get("max_depth", self.max_depth) or self.max_depth))
            elapsed_ms = int(data.get("elapsed_ms", self.last_elapsed_ms) or self.last_elapsed_ms)
            self.last_elapsed_ms = max(self.last_elapsed_ms, elapsed_ms)
            if self.tt_probes > 0:
                self.tt_hit_rate = self.tt_hits / self.tt_probes
            self.nodes_series.append((elapsed_ms / 1000.0, float(self.nodes_total)))
            self.nps_series.append((elapsed_ms / 1000.0, float(self.nps)))
            self.hit_rate_series.append((elapsed_ms / 1000.0, float(self.tt_hit_rate * 100.0)))

            depth_hist = data.get("depth_histogram")
            if isinstance(depth_hist, dict):
                self.depth_hist = {
                    int(k): int(v)
                    for k, v in depth_hist.items()
                    if str(k).isdigit() or (isinstance(k, int) and k >= 0)
                }
            cutoff_hist = data.get("cutoff_depth_histogram")
            if isinstance(cutoff_hist, dict):
                self.cutoff_hist = {
                    int(k): int(v)
                    for k, v in cutoff_hist.items()
                    if str(k).isdigit() or (isinstance(k, int) and k >= 0)
                }
            self.heartbeat_idx = (self.heartbeat_idx + 1) % len(SPINNER)
            self.last_batch_wall = time.perf_counter()
            return

        if event == "search_end":
            self.search_reason = str(data.get("reason", "-"))
            self.best_complete = bool(data.get("complete", self.best_complete))
            return

    def _draw_series(
        self,
        canvas: tk.Canvas,
        samples: Iterable[Tuple[float, float]],
        title: str,
        color: str,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        step: bool = False,
    ) -> None:
        canvas.delete("all")
        width = max(60, int(canvas.winfo_width() or canvas["width"]))
        height = max(60, int(canvas.winfo_height() or canvas["height"]))
        margin = 26
        points = list(samples)
        canvas.create_rectangle(margin, margin, width - margin, height - margin, outline="#3a4657")
        canvas.create_text(8, 8, anchor="nw", fill="#9fb3cd", text=title, font=("TkDefaultFont", 9, "bold"))
        if len(points) < 2:
            canvas.create_text(width / 2, height / 2, fill="#7d8ea8", text="waiting for data")
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, x1 = min(xs), max(xs)
        if x1 <= x0:
            x1 = x0 + 1.0
        if y_min is None:
            y_min = min(ys)
        if y_max is None:
            y_max = max(ys)
        if y_max <= y_min:
            y_max = y_min + 1.0

        plot: List[float] = []
        last_x = margin
        last_y = height - margin
        for x, y in points:
            px = margin + ((x - x0) / (x1 - x0)) * (width - 2 * margin)
            py = height - margin - ((y - y_min) / (y_max - y_min)) * (height - 2 * margin)
            if step and plot:
                plot.extend([px, last_y])
            plot.extend([px, py])
            last_x, last_y = px, py
        if len(plot) >= 4:
            canvas.create_line(*plot, fill=color, width=2)

        canvas.create_text(width - margin, height - 10, anchor="se", fill="#8ea0bb", text=f"{x1:.1f}s")
        canvas.create_text(margin + 2, height - 10, anchor="sw", fill="#8ea0bb", text=f"{x0:.1f}s")
        canvas.create_text(width - margin, margin + 2, anchor="ne", fill="#8ea0bb", text=f"{y_max:.1f}")
        canvas.create_text(width - margin, height - margin - 2, anchor="se", fill="#8ea0bb", text=f"{y_min:.1f}")

    def _draw_histogram(self, canvas: tk.Canvas, title: str, hist: Dict[int, int], color: str) -> None:
        canvas.delete("all")
        width = max(60, int(canvas.winfo_width() or canvas["width"]))
        height = max(60, int(canvas.winfo_height() or canvas["height"]))
        margin = 24
        canvas.create_rectangle(margin, margin, width - margin, height - margin, outline="#3a4657")
        canvas.create_text(8, 8, anchor="nw", fill="#9fb3cd", text=title, font=("TkDefaultFont", 9, "bold"))
        if not hist:
            canvas.create_text(width / 2, height / 2, fill="#7d8ea8", text="no samples")
            return
        items = sorted(hist.items())[:16]
        max_v = max(v for _, v in items) or 1
        span_w = max(1, (width - 2 * margin) / max(1, len(items)))
        for idx, (depth, count) in enumerate(items):
            x_left = margin + idx * span_w
            x_right = x_left + span_w * 0.8
            bar_h = ((height - 2 * margin) * (count / max_v))
            y_top = height - margin - bar_h
            canvas.create_rectangle(x_left, y_top, x_right, height - margin, fill=color, outline="")
            canvas.create_text((x_left + x_right) / 2, height - margin + 10, text=str(depth), fill="#8ea0bb")

    def _render(self) -> None:
        self.connection_label.configure(text="Connected" if self.connected else "Disconnected")
        self.peer_label.configure(text=f"peer: {self.peer_label_text}")
        self.listen_label.configure(text=f"listen: {self.listen_label_text}")
        self.slice_label.configure(
            text=f"slice {self._current_slice_ms()} ms" + (" (slow)" if self.slow_mode_var.get() else "")
        )
        self.board_label.configure(text=self.board_text)

        age = "-"
        if self.best_changed_at is not None:
            age = f"{max(0.0, time.perf_counter() - self.best_changed_at):.1f}s"
        status = (
            f"State key: {self.state_key} | depth done {self.depth_complete} | probing {self.depth_probing} | "
            f"max depth {self.max_depth} | reason {self.search_reason}"
        )
        self.status_line.configure(text=status)
        self.metrics_line.configure(
            text=(
                f"Nodes {self.nodes_total} | NPS {self.nps}/s | TT hit {self.tt_hit_rate * 100:.1f}% "
                f"({self.tt_hits}/{self.tt_probes}) | cutoffs {self.cutoffs} | eval {self.eval_calls} | "
                f"branching ~{self.branching:.2f}"
            )
        )
        best_text = (
            f"Best: pit {self.best_move} ({self.best_score:+d})"
            if self.best_move is not None
            else "Best: -"
        )
        best_text += f" | {'perfect' if self.best_complete else 'best so far'} | last changed {age}"
        best_text += f" | aspiration window {self.asp_window} retries {self.asp_retries}"
        self.best_line.configure(text=best_text)
        self.pv_line.configure(text="PV: " + (" -> ".join(str(m) for m in self.pv_moves) if self.pv_moves else "-"))

        heartbeat = "idle"
        if self.last_batch_wall > 0 and (time.perf_counter() - self.last_batch_wall) < 2.5:
            heartbeat = SPINNER[self.heartbeat_idx]
        self.heartbeat_label.configure(
            text=f"heartbeat: {heartbeat} | TT exact {self.tt_exact_reuse} | TT bound {self.tt_bound_reuse}"
        )

        root_lines = ["Root children (score):"]
        for pit in range(1, 7):
            score = self.root_scores.get(pit)
            marker = "*> " if self.best_move == pit else " - "
            if score is None:
                root_lines.append(f"{marker}pit {pit}: --")
            else:
                root_lines.append(f"{marker}pit {pit}: {score:+d}")
        self.root_scores_label.configure(text="\n".join(root_lines))

        pv_tree_lines = ["PV tree (top 3 plies):"]
        pv_entries = self.pv_scored[:3]
        if not pv_entries:
            for idx, move in enumerate(self.pv_moves[:3]):
                pv_tree_lines.append(f"{'  ' * idx}-> pit {move} (score --)")
        else:
            for idx, (move, score, bound) in enumerate(pv_entries):
                score_text = "--"
                if score is not None:
                    if bound == "lower":
                        score_text = f">={score:+d}"
                    elif bound == "upper":
                        score_text = f"<={score:+d}"
                    else:
                        score_text = f"{score:+d}"
                pv_tree_lines.append(f"{'  ' * idx}-> pit {move} (score {score_text})")
        self.pv_tree_label.configure(text="\n".join(pv_tree_lines))

        self._draw_series(self.nodes_canvas, self.nodes_series, "Nodes vs Time", "#4fc3f7")
        self._draw_series(self.nps_canvas, self.nps_series, "NPS vs Time", "#81c784")
        self._draw_series(self.depth_canvas, self.depth_series, "Depth vs Time (step)", "#ffb74d", step=True)
        self._draw_series(self.hit_rate_canvas, self.hit_rate_series, "TT Hit Rate % vs Time", "#ba68c8", y_min=0, y_max=100)
        self._draw_histogram(self.depth_hist_canvas, "Visited Depth Histogram (sampled)", self.depth_hist, "#64b5f6")
        self._draw_histogram(self.cutoff_hist_canvas, "Cutoff Depth Histogram", self.cutoff_hist, "#e57373")

    def _current_slice_ms(self) -> int:
        current = max(20, int(self.slice_ms_var.get()))
        if self.slow_mode_var.get():
            return min(current, 80)
        return current

    def start_listener(self) -> None:
        self.stop_listener()
        endpoint = parse_host_port(self.args.listen)
        if endpoint is None:
            self.listen_label_text = "invalid listen endpoint"
            return
        self.server = TelemetryTCPServer(endpoint[0], endpoint[1], self.event_queue)
        self.server.start()

    def stop_listener(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None

    def start_demo(self) -> None:
        self.stop_demo()
        self.demo_runner = DemoRunner(
            self.event_queue,
            seeds=self.args.seeds,
            topn=self.args.topn,
            slice_ms_getter=self._current_slice_ms,
        )
        self.demo_runner.start()

    def stop_demo(self) -> None:
        if self.demo_runner is not None:
            self.demo_runner.stop()
            self.demo_runner = None

    def shutdown(self) -> None:
        self.stop_demo()
        self.stop_listener()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live visualizer for Mancala solver telemetry")
    parser.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1:8765",
        help="listen endpoint for sidecar telemetry (host:port)",
    )
    parser.add_argument("--demo", action="store_true", help="run demo mode (visualizer drives the solver)")
    parser.add_argument("--seeds", type=int, default=4, help="demo-mode seeds per pit")
    parser.add_argument("--topn", type=int, default=6, help="demo-mode root move list size")
    parser.add_argument("--slice-ms", type=int, default=300, help="demo-mode solve slice duration")
    parser.add_argument("--title", type=str, default="Mancala Search Visualizer", help="window title")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if tk is None or ttk is None:
        print("tkinter is required for mancala_visualizer.py")
        print("Install python3-tk (Debian/Ubuntu) or run in an environment with Tk support.")
        return 2
    root = tk.Tk()
    app = VisualizerApp(root, args)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.shutdown(), root.destroy()))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
