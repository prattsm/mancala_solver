"""Telemetry schema and sinks for Mancala solver instrumentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple
import json
import socket
import threading
import time


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class TelemetryEnvelope:
    event: str
    ts_ms: int
    data: Dict[str, Any]


@dataclass(frozen=True)
class SearchStartEvent:
    state_key: str
    time_limit_ms: Optional[int]
    start_depth: int


@dataclass(frozen=True)
class IterationStartEvent:
    depth: int
    aspiration_alpha: int
    aspiration_beta: int
    guess_score: Optional[int]


@dataclass(frozen=True)
class IterationDoneEvent:
    depth: int
    score: int
    best_move: Optional[int]
    complete: bool
    nodes: int
    elapsed_ms: int
    max_depth: int
    aspiration_window: int
    aspiration_retries: int
    root_scores: list[tuple[int, int]]


@dataclass(frozen=True)
class PVUpdateEvent:
    depth: int
    pv_moves: list[int]
    pv_scored: list[tuple[int, Optional[int], str]]
    score: int


@dataclass(frozen=True)
class NodeBatchEvent:
    nodes_total: int
    nps_estimate: int
    tt_hits: int
    tt_probes: int
    tt_stores: int
    tt_exact_reuse: int
    tt_bound_reuse: int
    cutoffs: int
    cutoff_alpha_beta: int
    cutoff_tt_bound: int
    eval_calls: int
    max_depth: int
    branching_factor_estimate: float
    aspiration_window: int
    aspiration_retries: int
    depth_histogram: Dict[str, int]
    cutoff_depth_histogram: Dict[str, int]
    elapsed_ms: int


@dataclass(frozen=True)
class SearchEndEvent:
    best_move: Optional[int]
    score: int
    depth: int
    complete: bool
    nodes: int
    elapsed_ms: int
    reason: str


class TelemetrySink(Protocol):
    def emit(self, envelope: TelemetryEnvelope) -> None:
        ...

    def close(self) -> None:
        ...


class NullTelemetrySink:
    def emit(self, envelope: TelemetryEnvelope) -> None:
        _ = envelope

    def close(self) -> None:
        return


class CallbackTelemetrySink:
    def __init__(self, callback: Callable[[TelemetryEnvelope], None]) -> None:
        self._callback = callback

    def emit(self, envelope: TelemetryEnvelope) -> None:
        self._callback(envelope)

    def close(self) -> None:
        return


class QueueTelemetrySink:
    def __init__(self, queue: "Queue[TelemetryEnvelope]") -> None:
        self._queue = queue

    def emit(self, envelope: TelemetryEnvelope) -> None:
        self._queue.put(envelope)

    def close(self) -> None:
        return


class ThreadedTCPSink:
    """Non-blocking JSONL sink that writes in a background thread."""

    def __init__(
        self,
        host: str,
        port: int,
        queue_size: int = 1024,
        reconnect_delay_ms: int = 250,
    ) -> None:
        self._host = host
        self._port = port
        self._queue: Queue[TelemetryEnvelope] = Queue(maxsize=max(8, queue_size))
        self._reconnect_delay_s = max(0.05, reconnect_delay_ms / 1000.0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mancala-telemetry-sender", daemon=True)
        self._thread.start()

    def emit(self, envelope: TelemetryEnvelope) -> None:
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(envelope)
            return
        except Full:
            pass

        # Drop oldest when saturated to keep recent telemetry flowing.
        try:
            _ = self._queue.get_nowait()
        except Empty:
            pass
        try:
            self._queue.put_nowait(envelope)
        except Full:
            return

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)

    def _run(self) -> None:
        conn: Optional[socket.socket] = None
        while not self._stop.is_set():
            if conn is None:
                conn = self._try_connect()
                if conn is None:
                    time.sleep(self._reconnect_delay_s)
                    continue
            try:
                envelope = self._queue.get(timeout=0.1)
            except Empty:
                continue
            payload = {
                "event": envelope.event,
                "ts_ms": envelope.ts_ms,
                "data": envelope.data,
            }
            try:
                line = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
                conn.sendall(line)
            except OSError:
                try:
                    conn.close()
                except OSError:
                    pass
                conn = None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    def _try_connect(self) -> Optional[socket.socket]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.3)
        try:
            sock.connect((self._host, self._port))
        except OSError:
            sock.close()
            return None
        sock.settimeout(None)
        return sock


def emit_event(sink: Optional[TelemetrySink], event: str, payload: Mapping[str, Any]) -> None:
    if sink is None:
        return
    envelope = TelemetryEnvelope(event=event, ts_ms=now_ms(), data=dict(payload))
    try:
        sink.emit(envelope)
    except Exception:
        return


def emit_dataclass_event(sink: Optional[TelemetrySink], event: str, payload_obj: object) -> None:
    emit_event(sink, event, asdict(payload_obj))


def parse_host_port(value: str) -> Optional[Tuple[str, int]]:
    raw = value.strip()
    if not raw:
        return None
    if ":" not in raw:
        return None
    host, port_raw = raw.rsplit(":", 1)
    host = host.strip()
    if not host:
        return None
    try:
        port = int(port_raw)
    except ValueError:
        return None
    if port <= 0 or port > 65535:
        return None
    return host, port
