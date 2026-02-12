"""Minimax solver with depth-aware TT and iterative deepening."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import gzip
import os
import pickle
import time

from mancala_engine import (
    OPP,
    YOU,
    State,
    apply_move_fast_with_info,
    is_terminal,
    legal_moves,
)
from mancala_telemetry import (
    IterationDoneEvent,
    IterationStartEvent,
    NodeBatchEvent,
    PVUpdateEvent,
    SearchEndEvent,
    SearchStartEvent,
    TelemetrySink,
    emit_dataclass_event,
)

INF = 10**9
FULL_DEPTH = 1_000_000
CACHE_VERSION = 6
TT_MAX_ENTRIES = 1_000_000
TT_PRUNE_TO = 800_000
ASPIRATION_WINDOW_INIT = 4
ASPIRATION_MAX_RETRIES = 5
INTERRUPT_POLL_MASK = 0xFF
LIVE_POLL_MASK = 0x3FF
LIVE_EMIT_INTERVAL_MS = 120
TELEMETRY_NODE_MASK = 0x3FF
TELEMETRY_DEPTH_SAMPLE_MASK = 0x7
TELEMETRY_EMIT_INTERVAL_MS = 120
CACHE_GZIP_LEVEL = 1
CHILDGEN_POLL_MASK = 0x1
SOW_POLL_MASK = 0x3F
SLICE_DEADLINE_SLACK_MS = 120

EXACT = 0
LOWER = 1
UPPER = 2


@dataclass(frozen=True)
class TTEntry:
    value: int
    flag: int
    best_move: Optional[int]
    depth: int
    proven: bool = False


@dataclass(frozen=True)
class SearchResult:
    best_move: Optional[int]
    score: int
    top_moves: List[Tuple[int, int]]
    depth: int
    complete: bool
    elapsed_ms: int
    nodes: int


@dataclass
class _SearchContext:
    tt: Dict[State, TTEntry]
    deadline: Optional[float]
    interrupt_check: Optional[Callable[[], bool]] = None
    tt_mutation_callback: Optional[Callable[[], None]] = None
    live_nodes_callback: Optional[Callable[[int], None]] = None
    telemetry: Optional["_TelemetryStats"] = None
    iter_depth: int = 0
    nodes: int = 0
    hit_depth_limit: bool = False
    used_unproven_tt: bool = False
    interrupted: bool = False


@dataclass
class _TelemetryStats:
    sink: TelemetrySink
    solve_start: float
    last_emit: float
    state_key: str
    nodes_total: int = 0
    tt_probes: int = 0
    tt_hits: int = 0
    tt_stores: int = 0
    tt_exact_reuse: int = 0
    tt_bound_reuse: int = 0
    cutoffs: int = 0
    cutoff_alpha_beta: int = 0
    cutoff_tt_bound: int = 0
    eval_calls: int = 0
    max_depth: int = 0
    branching_sum: int = 0
    branching_samples: int = 0
    aspiration_window: int = 0
    aspiration_retries: int = 0
    depth_hist: Dict[int, int] = field(default_factory=dict)
    cutoff_hist: Dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _DepthSearchResult:
    best_move: Optional[int]
    score: int
    top_moves: List[Tuple[int, int]]
    fail_low: bool
    fail_high: bool
    root_scores: Tuple[Tuple[int, int], ...] = ()
    aspiration_window: int = 0
    aspiration_retries: int = 0


class SearchTimeout(Exception):
    """Raised when a timed search iteration exceeds its deadline."""


def state_key(state: State) -> str:
    pits_you = ",".join(str(n) for n in state.pits_you)
    pits_opp = ",".join(str(n) for n in state.pits_opp)
    return f"{state.to_move}|{pits_you}|{pits_opp}|{state.store_you}|{state.store_opp}"


def key_to_state(key: str) -> Optional[State]:
    parts = key.split("|")
    if len(parts) != 5:
        return None
    to_move = parts[0]
    if to_move not in {YOU, OPP}:
        return None
    try:
        pits_you = tuple(int(x) for x in parts[1].split(","))
        pits_opp = tuple(int(x) for x in parts[2].split(","))
        store_you = int(parts[3])
        store_opp = int(parts[4])
    except ValueError:
        return None
    if len(pits_you) != 6 or len(pits_opp) != 6:
        return None
    return State(to_move, pits_you, pits_opp, store_you, store_opp)


def default_cache_path() -> Path:
    return Path.home() / ".mancala_cache.pkl.gz"


def _tt_get(tt: Dict[State, TTEntry], state: State) -> Optional[TTEntry]:
    return tt.get(state)


def _prune_tt(tt: Dict[State, TTEntry]) -> bool:
    if len(tt) <= TT_MAX_ENTRIES:
        return False
    pruned = False
    while len(tt) > TT_PRUNE_TO:
        tt.pop(next(iter(tt)))
        pruned = True
    return pruned


def _tt_store(
    tt: Dict[State, TTEntry],
    state: State,
    entry: TTEntry,
    on_mutation: Optional[Callable[[], None]] = None,
) -> None:
    existing = tt.get(state)
    if existing is not None:
        if existing.depth > entry.depth:
            return
        if existing.depth == entry.depth and existing.flag == EXACT and entry.flag != EXACT:
            return
        if (
            existing.depth == entry.depth
            and existing.flag == EXACT
            and entry.flag == EXACT
            and existing.proven
            and not entry.proven
        ):
            return
    tt[state] = entry
    if on_mutation is not None:
        on_mutation()
    if _prune_tt(tt):
        if on_mutation is not None:
            on_mutation()


def load_tt(path: Path) -> Dict[State, TTEntry]:
    # Security: pickle is unsafe for untrusted files. Only load caches you trust.
    try:
        with gzip.open(path, "rb") as handle:
            raw = pickle.load(handle)
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}
    if raw.get("version") != CACHE_VERSION:
        return {}
    raw_tt = raw.get("tt")
    if not isinstance(raw_tt, dict):
        return {}

    tt: Dict[State, TTEntry] = {}
    for key, entry in raw_tt.items():
        state: Optional[State] = None
        if isinstance(key, State):
            state = key
        elif isinstance(key, str):
            state = key_to_state(key)
        else:
            continue

        if state is None:
            continue

        if isinstance(entry, TTEntry):
            if entry.flag not in {EXACT, LOWER, UPPER}:
                continue
            if entry.best_move is not None and not isinstance(entry.best_move, int):
                continue
            if not isinstance(entry.depth, int) or entry.depth < 0:
                continue
            if not isinstance(entry.proven, bool):
                continue
            tt[state] = entry
            continue

        if not isinstance(entry, tuple) or len(entry) not in {4, 5}:
            continue
        if len(entry) == 4:
            value, flag, best_move, depth = entry
            proven = False
        else:
            value, flag, best_move, depth, proven = entry
        if not isinstance(value, int):
            continue
        if flag not in {EXACT, LOWER, UPPER}:
            continue
        if best_move is not None and not isinstance(best_move, int):
            continue
        if not isinstance(depth, int) or depth < 0:
            continue
        if not isinstance(proven, bool):
            continue
        tt[state] = TTEntry(value, flag, best_move, depth, proven)
    _prune_tt(tt)
    return tt


def save_tt(tt: Dict[State, TTEntry], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": CACHE_VERSION,
            "tt": tt,
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp") if path.suffix else path.with_name(path.name + ".tmp")
        with gzip.open(tmp_path, "wb", compresslevel=CACHE_GZIP_LEVEL) as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            if hasattr(handle, "fileobj") and handle.fileobj is not None:
                os.fsync(handle.fileobj.fileno())
        tmp_path.replace(path)
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError:
        return


def terminal_diff(state: State) -> int:
    pits_you = sum(state.pits_you)
    pits_opp = sum(state.pits_opp)
    store_you = state.store_you
    store_opp = state.store_opp
    if pits_you == 0:
        store_opp += pits_opp
    if pits_opp == 0:
        store_you += pits_you
    return store_you - store_opp


def _heuristic_eval(state: State) -> int:
    store_diff = state.store_you - state.store_opp
    pits_diff = sum(state.pits_you) - sum(state.pits_opp)
    return store_diff * 2 + pits_diff


def normalize_state(state: State) -> Tuple[State, int]:
    if state.to_move == YOU:
        return state, 1
    return State(YOU, state.pits_opp, state.pits_you, state.store_opp, state.store_you), -1


def normalize_value_flag(value: int, flag: int, sign: int) -> Tuple[int, int]:
    if sign == 1:
        return value, flag
    value = -value
    if flag == LOWER:
        flag = UPPER
    elif flag == UPPER:
        flag = LOWER
    return value, flag


def denormalize_value_flag(value: int, flag: int, sign: int) -> Tuple[int, int]:
    return normalize_value_flag(value, flag, sign)


def normalize_move(move: Optional[int], sign: int) -> Optional[int]:
    """Convert a move into the normalized state's coordinates."""
    if move is None:
        return None
    # Pit numbering is always 1..6 for the side-to-move, so swapping sides
    # does not change the numeric pit label.
    return move


def denormalize_move(move: Optional[int], sign: int) -> Optional[int]:
    return normalize_move(move, sign)


def _tt_best_move(state: State, tt: Dict[State, TTEntry]) -> Optional[int]:
    norm_state, sign = normalize_state(state)
    entry = _tt_get(tt, norm_state)
    if entry is None:
        return None
    return denormalize_move(entry.best_move, sign)


def ordered_children(
    state: State,
    tt: Dict[State, TTEntry],
    context: Optional[_SearchContext] = None,
) -> List[Tuple[int, State, bool, bool, int]]:
    moves = legal_moves(state)
    if not moves:
        return []

    def _poll_childgen(index: int) -> None:
        if context is None:
            return
        if (index & CHILDGEN_POLL_MASK) == 0:
            _check_deadline(context, force=True)

    def _poll_sow() -> None:
        if context is None:
            return
        _check_deadline(context, force=True)

    best_first = _tt_best_move(state, tt)
    children: List[Tuple[int, State, bool, bool, int]] = []

    rest_moves = [move for move in moves if move != best_first]
    if best_first in moves:
        _poll_childgen(0)
        child_state, extra_turn, capture = apply_move_fast_with_info(
            state,
            best_first,
            poll_callback=_poll_sow if context is not None else None,
            poll_mask=SOW_POLL_MASK,
        )
        if state.to_move == YOU:
            store_gain = child_state.store_you - state.store_you
        else:
            store_gain = child_state.store_opp - state.store_opp
        children.append((best_first, child_state, extra_turn, capture, store_gain))

    rest: List[Tuple[int, State, bool, bool, int]] = []
    for idx, move in enumerate(rest_moves, start=1):
        _poll_childgen(idx)
        child_state, extra_turn, capture = apply_move_fast_with_info(
            state,
            move,
            poll_callback=_poll_sow if context is not None else None,
            poll_mask=SOW_POLL_MASK,
        )
        if state.to_move == YOU:
            store_gain = child_state.store_you - state.store_you
        else:
            store_gain = child_state.store_opp - state.store_opp
        rest.append((move, child_state, extra_turn, capture, store_gain))

    rest.sort(key=lambda item: (item[2], item[3], item[4]), reverse=True)
    return children + rest if children else rest


def _check_deadline(context: _SearchContext, force: bool = False) -> None:
    if (
        context.interrupt_check is not None
        and (force or (context.nodes & INTERRUPT_POLL_MASK) == 0)
        and context.interrupt_check()
    ):
        context.interrupted = True
        raise SearchTimeout()
    if context.deadline is not None and time.perf_counter() >= context.deadline:
        raise SearchTimeout()


def _telemetry_ply(context: _SearchContext, depth_remaining: int) -> int:
    if context.iter_depth <= 0:
        return 0
    return max(0, context.iter_depth - max(0, depth_remaining))


def _histogram_payload(hist: Dict[int, int], limit: int = 24) -> Dict[str, int]:
    items = sorted(hist.items())
    trimmed = items[:limit]
    return {str(depth): count for depth, count in trimmed}


def _telemetry_record_node(context: _SearchContext, depth_remaining: int) -> None:
    stats = context.telemetry
    if stats is None:
        return
    stats.nodes_total += 1
    if (stats.nodes_total & TELEMETRY_DEPTH_SAMPLE_MASK) != 0:
        return
    ply = _telemetry_ply(context, depth_remaining)
    stats.depth_hist[ply] = stats.depth_hist.get(ply, 0) + 1
    if ply > stats.max_depth:
        stats.max_depth = ply


def _telemetry_record_cutoff(context: _SearchContext, depth_remaining: int, kind: str) -> None:
    stats = context.telemetry
    if stats is None:
        return
    stats.cutoffs += 1
    if kind == "alpha_beta":
        stats.cutoff_alpha_beta += 1
    else:
        stats.cutoff_tt_bound += 1
    ply = _telemetry_ply(context, depth_remaining)
    stats.cutoff_hist[ply] = stats.cutoff_hist.get(ply, 0) + 1


def _telemetry_maybe_emit_batch(context: _SearchContext, force: bool = False) -> None:
    stats = context.telemetry
    if stats is None:
        return
    if not force and (stats.nodes_total & TELEMETRY_NODE_MASK) != 0:
        return
    now = time.perf_counter()
    if not force and (now - stats.last_emit) * 1000 < TELEMETRY_EMIT_INTERVAL_MS:
        return

    elapsed_ms = max(1, int((now - stats.solve_start) * 1000))
    nps = int(stats.nodes_total * 1000 / elapsed_ms)
    branching = 0.0
    if stats.branching_samples > 0:
        branching = stats.branching_sum / stats.branching_samples

    emit_dataclass_event(
        stats.sink,
        "node_batch",
        NodeBatchEvent(
            nodes_total=stats.nodes_total,
            nps_estimate=nps,
            tt_hits=stats.tt_hits,
            tt_probes=stats.tt_probes,
            tt_stores=stats.tt_stores,
            tt_exact_reuse=stats.tt_exact_reuse,
            tt_bound_reuse=stats.tt_bound_reuse,
            cutoffs=stats.cutoffs,
            cutoff_alpha_beta=stats.cutoff_alpha_beta,
            cutoff_tt_bound=stats.cutoff_tt_bound,
            eval_calls=stats.eval_calls,
            max_depth=stats.max_depth,
            branching_factor_estimate=round(branching, 3),
            aspiration_window=stats.aspiration_window,
            aspiration_retries=stats.aspiration_retries,
            depth_histogram=_histogram_payload(stats.depth_hist),
            cutoff_depth_histogram=_histogram_payload(stats.cutoff_hist),
            elapsed_ms=elapsed_ms,
        ),
    )
    stats.last_emit = now


def _extract_pv_scored(
    state: State, tt: Dict[State, TTEntry], max_len: int = 16
) -> List[Tuple[int, Optional[int], str]]:
    pv: List[Tuple[int, Optional[int], str]] = []
    seen: set[State] = set()
    current = state
    for _ in range(max_len):
        if current in seen:
            break
        seen.add(current)
        norm_state, sign = normalize_state(current)
        entry = _tt_get(tt, norm_state)
        if entry is None:
            break
        move = denormalize_move(entry.best_move, sign)
        if move is None:
            break
        if move not in legal_moves(current):
            break
        value, flag = denormalize_value_flag(entry.value, entry.flag, sign)
        bound = "exact"
        if flag == LOWER:
            bound = "lower"
        elif flag == UPPER:
            bound = "upper"
        pv.append((move, value, bound))
        current, _, _ = apply_move_fast_with_info(current, move)
        if is_terminal(current):
            break
    return pv


def _extract_pv_moves(state: State, tt: Dict[State, TTEntry], max_len: int = 16) -> List[int]:
    return [move for move, _, _ in _extract_pv_scored(state, tt, max_len=max_len)]


def _decorate_depth_result(
    outcome: _DepthSearchResult,
    aspiration_window: int,
    aspiration_retries: int,
) -> _DepthSearchResult:
    return _DepthSearchResult(
        best_move=outcome.best_move,
        score=outcome.score,
        top_moves=outcome.top_moves,
        fail_low=outcome.fail_low,
        fail_high=outcome.fail_high,
        root_scores=outcome.root_scores,
        aspiration_window=aspiration_window,
        aspiration_retries=aspiration_retries,
    )


def _search_depth(
    state: State, alpha: int, beta: int, depth: int, context: _SearchContext
) -> Tuple[int, bool]:
    _check_deadline(context)
    context.nodes += 1
    _telemetry_record_node(context, depth)
    _telemetry_maybe_emit_batch(context)
    if context.live_nodes_callback is not None and (context.nodes & LIVE_POLL_MASK) == 0:
        context.live_nodes_callback(context.nodes)

    if is_terminal(state):
        return terminal_diff(state), True

    if depth <= 0:
        context.hit_depth_limit = True
        if context.telemetry is not None:
            context.telemetry.eval_calls += 1
        return _heuristic_eval(state), False

    alpha0, beta0 = alpha, beta
    norm_state, sign = normalize_state(state)
    if context.telemetry is not None:
        context.telemetry.tt_probes += 1
    entry = _tt_get(context.tt, norm_state)
    if entry is not None and entry.depth >= depth:
        if context.telemetry is not None:
            context.telemetry.tt_hits += 1
        value, flag = denormalize_value_flag(entry.value, entry.flag, sign)
        if not entry.proven:
            context.used_unproven_tt = True
        if flag == EXACT:
            if context.telemetry is not None:
                context.telemetry.tt_exact_reuse += 1
            if not entry.proven:
                context.hit_depth_limit = True
            return value, entry.proven
        if context.telemetry is not None:
            context.telemetry.tt_bound_reuse += 1
        if flag == LOWER:
            alpha = max(alpha, value)
        elif flag == UPPER:
            beta = min(beta, value)
        if alpha >= beta:
            _telemetry_record_cutoff(context, depth, "tt_bound")
            return value, False

    children = ordered_children(state, context.tt, context=context)
    if context.telemetry is not None:
        context.telemetry.branching_sum += len(children)
        context.telemetry.branching_samples += 1
    if not children:
        return terminal_diff(state), True

    best_move: Optional[int] = None
    best_proven = False
    if state.to_move == YOU:
        best = -INF
        for move, child_state, _, _, _ in children:
            val, val_proven = _search_depth(child_state, alpha, beta, depth - 1, context)
            if val > best or (val == best and val_proven and not best_proven):
                best = val
                best_move = move
                best_proven = val_proven
            alpha = max(alpha, best)
            if alpha >= beta:
                _telemetry_record_cutoff(context, depth, "alpha_beta")
                break
    else:
        best = INF
        for move, child_state, _, _, _ in children:
            val, val_proven = _search_depth(child_state, alpha, beta, depth - 1, context)
            if val < best or (val == best and val_proven and not best_proven):
                best = val
                best_move = move
                best_proven = val_proven
            beta = min(beta, best)
            if alpha >= beta:
                _telemetry_record_cutoff(context, depth, "alpha_beta")
                break

    if best_move is None:
        best_move = children[0][0]

    if best <= alpha0:
        flag = UPPER
    elif best >= beta0:
        flag = LOWER
    else:
        flag = EXACT

    value_norm, flag_norm = normalize_value_flag(best, flag, sign)
    best_move_norm = normalize_move(best_move, sign)
    exact_proven = best_proven if flag == EXACT else False
    if context.telemetry is not None:
        context.telemetry.tt_stores += 1
    _tt_store(
        context.tt,
        norm_state,
        TTEntry(value_norm, flag_norm, best_move_norm, depth, exact_proven),
        on_mutation=context.tt_mutation_callback,
    )
    return best, exact_proven


def search_depth(
    state: State,
    depth: int,
    tt: Dict[State, TTEntry],
    alpha: int = -INF,
    beta: int = INF,
) -> int:
    context = _SearchContext(tt=tt, deadline=None, interrupt_check=None)
    return _search_depth(state, alpha, beta, max(0, depth), context)[0]


def search(state: State, alpha: int, beta: int, tt: Dict[State, TTEntry]) -> int:
    return search_depth(state, FULL_DEPTH, tt, alpha=alpha, beta=beta)


def _best_move_depth(
    state: State,
    topn: int,
    depth: int,
    context: _SearchContext,
    alpha: int = -INF,
    beta: int = INF,
) -> _DepthSearchResult:
    children = ordered_children(state, context.tt, context=context)
    if not children:
        score = terminal_diff(state)
        return _DepthSearchResult(
            best_move=None,
            score=score,
            top_moves=[],
            fail_low=score <= alpha,
            fail_high=score >= beta,
            root_scores=(),
        )

    scored: List[Tuple[int, int]] = []
    alpha_root = alpha
    beta_root = beta
    if state.to_move == YOU:
        for move, child_state, _, _, _ in children:
            _check_deadline(context, force=True)
            move_alpha = alpha_root
            move_beta = beta_root
            val, _ = _search_depth(child_state, move_alpha, move_beta, depth - 1, context)
            if val <= move_alpha or val >= move_beta:
                # Move result is a bound in this aspiration window; re-search for an exact score.
                _check_deadline(context, force=True)
                val, _ = _search_depth(child_state, -INF, INF, depth - 1, context)
            scored.append((move, val))
            alpha_root = max(alpha_root, val)
        scored.sort(key=lambda item: item[1], reverse=True)
    else:
        for move, child_state, _, _, _ in children:
            _check_deadline(context, force=True)
            move_alpha = alpha_root
            move_beta = beta_root
            val, _ = _search_depth(child_state, move_alpha, move_beta, depth - 1, context)
            if val <= move_alpha or val >= move_beta:
                # Move result is a bound in this aspiration window; re-search for an exact score.
                _check_deadline(context, force=True)
                val, _ = _search_depth(child_state, -INF, INF, depth - 1, context)
            scored.append((move, val))
            beta_root = min(beta_root, val)
        scored.sort(key=lambda item: item[1])

    topn = max(0, topn)
    top_moves = scored[:topn] if topn > 0 else []
    best_move = scored[0][0]
    best_score = scored[0][1]
    return _DepthSearchResult(
        best_move=best_move,
        score=best_score,
        top_moves=top_moves,
        fail_low=best_score <= alpha,
        fail_high=best_score >= beta,
        root_scores=tuple(scored),
    )


def _run_depth_iteration_with_aspiration(
    state: State,
    topn: int,
    depth: int,
    guess_score: Optional[int],
    context: _SearchContext,
) -> _DepthSearchResult:
    if guess_score is None:
        if context.telemetry is not None:
            context.telemetry.aspiration_window = 0
            context.telemetry.aspiration_retries = 0
        outcome = _best_move_depth(state, topn, depth, context, -INF, INF)
        return _decorate_depth_result(outcome, aspiration_window=0, aspiration_retries=0)

    window = max(1, ASPIRATION_WINDOW_INIT)
    retries = 0
    while True:
        alpha = guess_score - window
        beta = guess_score + window
        if context.telemetry is not None:
            context.telemetry.aspiration_window = window
            context.telemetry.aspiration_retries = retries
        outcome = _best_move_depth(state, topn, depth, context, alpha, beta)
        if not outcome.fail_low and not outcome.fail_high:
            return _decorate_depth_result(outcome, aspiration_window=window, aspiration_retries=retries)
        retries += 1
        if retries >= ASPIRATION_MAX_RETRIES:
            if context.telemetry is not None:
                context.telemetry.aspiration_window = 0
                context.telemetry.aspiration_retries = retries
            outcome = _best_move_depth(state, topn, depth, context, -INF, INF)
            return _decorate_depth_result(outcome, aspiration_window=0, aspiration_retries=retries)
        window *= 2


def _fallback_best_move(state: State, topn: int) -> Tuple[Optional[int], int, List[Tuple[int, int]]]:
    moves = legal_moves(state)
    if not moves:
        return None, terminal_diff(state), []
    scored: List[Tuple[int, int]] = []
    for move in moves:
        child_state, _, _ = apply_move_fast_with_info(state, move)
        scored.append((move, _heuristic_eval(child_state)))
    if state.to_move == YOU:
        scored.sort(key=lambda item: item[1], reverse=True)
    else:
        scored.sort(key=lambda item: item[1])
    topn = max(0, topn)
    top_moves = scored[:topn] if topn > 0 else []
    return scored[0][0], scored[0][1], top_moves


def _quick_fallback_best_move(
    state: State,
    topn: int,
    move_hint: Optional[int] = None,
    score_hint: Optional[int] = None,
) -> Tuple[Optional[int], int, List[Tuple[int, int]]]:
    moves = legal_moves(state)
    if not moves:
        return None, terminal_diff(state), []
    move = move_hint if move_hint in moves else moves[0]
    score = _heuristic_eval(state) if score_hint is None else score_hint
    topn = max(0, topn)
    top_moves = [(move, score)] if topn > 0 else []
    return move, score, top_moves


def solve_best_move(
    state: State,
    topn: int = 3,
    tt: Optional[Dict[State, TTEntry]] = None,
    time_limit_ms: Optional[int] = None,
    progress_callback: Optional[Callable[[SearchResult], None]] = None,
    start_depth: int = 1,
    guess_score: Optional[int] = None,
    previous_result: Optional[SearchResult] = None,
    interrupt_check: Optional[Callable[[], bool]] = None,
    tt_mutation_callback: Optional[Callable[[], None]] = None,
    live_callback: Optional[Callable[[int, int], None]] = None,
    telemetry_sink: Optional[TelemetrySink] = None,
) -> SearchResult:
    if tt is None:
        tt = {}

    start = time.perf_counter()
    telemetry: Optional[_TelemetryStats] = None
    if telemetry_sink is not None:
        telemetry = _TelemetryStats(
            sink=telemetry_sink,
            solve_start=start,
            last_emit=start,
            state_key=state_key(state),
        )
        emit_dataclass_event(
            telemetry.sink,
            "search_start",
            SearchStartEvent(
                state_key=telemetry.state_key,
                time_limit_ms=time_limit_ms,
                start_depth=max(1, start_depth),
            ),
        )

    def _emit_search_end(result: SearchResult, reason: str) -> None:
        if telemetry is None:
            return
        # Force one final batch so dashboards don't stall between iterations.
        _telemetry_maybe_emit_batch(
            _SearchContext(tt=tt, deadline=None, telemetry=telemetry),
            force=True,
        )
        emit_dataclass_event(
            telemetry.sink,
            "search_end",
            SearchEndEvent(
                best_move=result.best_move,
                score=result.score,
                depth=result.depth,
                complete=result.complete,
                nodes=result.nodes,
                elapsed_ms=result.elapsed_ms,
                reason=reason,
            ),
        )

    def _emit_iteration_start(depth: int, guess: Optional[int]) -> None:
        if telemetry is None:
            return
        if guess is None:
            alpha = -INF
            beta = INF
        else:
            alpha = guess - max(1, ASPIRATION_WINDOW_INIT)
            beta = guess + max(1, ASPIRATION_WINDOW_INIT)
        emit_dataclass_event(
            telemetry.sink,
            "iteration_start",
            IterationStartEvent(
                depth=depth,
                aspiration_alpha=alpha,
                aspiration_beta=beta,
                guess_score=guess,
            ),
        )

    def _emit_iteration_done(depth: int, depth_result: _DepthSearchResult, result: SearchResult) -> None:
        if telemetry is None:
            return
        pv_scored = _extract_pv_scored(state, tt)
        emit_dataclass_event(
            telemetry.sink,
            "iteration_done",
            IterationDoneEvent(
                depth=depth,
                score=depth_result.score,
                best_move=depth_result.best_move,
                complete=result.complete,
                nodes=result.nodes,
                elapsed_ms=result.elapsed_ms,
                max_depth=telemetry.max_depth,
                aspiration_window=depth_result.aspiration_window,
                aspiration_retries=depth_result.aspiration_retries,
                root_scores=list(depth_result.root_scores),
            ),
        )
        emit_dataclass_event(
            telemetry.sink,
            "pv_update",
            PVUpdateEvent(
                depth=depth,
                pv_moves=[move for move, _, _ in pv_scored],
                pv_scored=pv_scored,
                score=depth_result.score,
            ),
        )

    if is_terminal(state):
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        result = SearchResult(
            best_move=None,
            score=terminal_diff(state),
            top_moves=[],
            depth=0,
            complete=True,
            elapsed_ms=elapsed_ms,
            nodes=0,
        )
        _emit_search_end(result, "complete")
        return result

    last_live_emit = start

    def _emit_live(nodes: int) -> None:
        nonlocal last_live_emit
        if live_callback is None:
            return
        now = time.perf_counter()
        if (now - last_live_emit) * 1000 < LIVE_EMIT_INTERVAL_MS:
            return
        live_callback(nodes, int((now - start) * 1000))
        last_live_emit = now

    if time_limit_ms is None:
        _emit_iteration_start(depth=0, guess=None)
        context = _SearchContext(
            tt=tt,
            deadline=None,
            interrupt_check=interrupt_check,
            tt_mutation_callback=tt_mutation_callback,
            live_nodes_callback=_emit_live,
            telemetry=telemetry,
            iter_depth=FULL_DEPTH,
        )
        try:
            depth_result = _best_move_depth(state, topn, FULL_DEPTH, context, -INF, INF)
        except SearchTimeout:
            move, score, top_moves = _quick_fallback_best_move(
                state,
                topn,
                move_hint=_tt_best_move(state, tt),
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            result = SearchResult(
                best_move=move,
                score=score,
                top_moves=top_moves,
                depth=0,
                complete=False,
                elapsed_ms=elapsed_ms,
                nodes=context.nodes,
            )
            _emit_live(result.nodes)
            if progress_callback is not None:
                progress_callback(result)
            _emit_search_end(result, "interrupted")
            return result
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        result = SearchResult(
            best_move=depth_result.best_move,
            score=depth_result.score,
            top_moves=depth_result.top_moves,
            depth=0,
            complete=(not context.hit_depth_limit and not context.used_unproven_tt),
            elapsed_ms=elapsed_ms,
            nodes=context.nodes,
        )
        _emit_live(result.nodes)
        if progress_callback is not None:
            progress_callback(result)
        _emit_iteration_done(depth=0, depth_result=depth_result, result=result)
        _emit_search_end(result, "complete" if result.complete else "timeout")
        return result

    deadline = start + max(0, time_limit_ms) / 1000.0
    hard_deadline = deadline + (SLICE_DEADLINE_SLACK_MS / 1000.0)
    total_nodes = 0
    best_result: Optional[SearchResult] = None
    depth = max(1, start_depth)
    current_guess = guess_score
    timed_out_due_interrupt = False

    while True:
        if time.perf_counter() >= deadline:
            break

        _emit_iteration_start(depth=depth, guess=current_guess)
        context = _SearchContext(
            tt=tt,
            deadline=hard_deadline,
            interrupt_check=interrupt_check,
            tt_mutation_callback=tt_mutation_callback,
            live_nodes_callback=lambda iter_nodes, base_nodes=total_nodes: _emit_live(base_nodes + iter_nodes),
            telemetry=telemetry,
            iter_depth=depth,
        )
        try:
            depth_result = _run_depth_iteration_with_aspiration(
                state=state,
                topn=topn,
                depth=depth,
                guess_score=current_guess,
                context=context,
            )
        except SearchTimeout:
            _emit_live(total_nodes + context.nodes)
            if context.interrupted:
                timed_out_due_interrupt = True
            break

        total_nodes += context.nodes
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        _emit_live(total_nodes)
        result = SearchResult(
            best_move=depth_result.best_move,
            score=depth_result.score,
            top_moves=depth_result.top_moves,
            depth=depth,
            complete=(not context.hit_depth_limit and not context.used_unproven_tt),
            elapsed_ms=elapsed_ms,
            nodes=total_nodes,
        )
        best_result = result
        _emit_iteration_done(depth=depth, depth_result=depth_result, result=result)
        if progress_callback is not None:
            progress_callback(result)
        if result.complete:
            _emit_search_end(result, "complete")
            return result
        current_guess = result.score
        depth += 1

    if best_result is not None:
        _emit_search_end(best_result, "interrupted" if timed_out_due_interrupt else "timeout")
        return best_result

    # Deadline was hit before completing the first requested depth.
    if start_depth > 1 and guess_score is not None:
        legal = legal_moves(state)
        move_guess = previous_result.best_move if previous_result is not None else None
        score_guess = previous_result.score if previous_result is not None else guess_score
        depth_guess = previous_result.depth if previous_result is not None else (start_depth - 1)
        nodes_guess = previous_result.nodes if previous_result is not None else 0
        top_guess = list(previous_result.top_moves) if previous_result is not None else []

        if move_guess not in legal:
            tt_guess = _tt_best_move(state, tt)
            if tt_guess in legal:
                move_guess = tt_guess
            elif legal:
                # Keep a usable best-so-far move even when the current slice
                # timed out before completing one depth.
                move_guess = legal[0]
            else:
                move_guess = None

        if top_guess:
            top_guess = [(move, score) for move, score in top_guess if move in legal]
        if not top_guess and move_guess is not None:
            top_guess = [(move_guess, score_guess)]

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        result = SearchResult(
            best_move=move_guess,
            score=score_guess,
            top_moves=top_guess,
            depth=max(0, depth_guess),
            complete=False,
            elapsed_ms=elapsed_ms,
            nodes=max(0, nodes_guess),
        )
        _emit_search_end(result, "timeout")
        return result

    move, score, top_moves = _quick_fallback_best_move(
        state,
        topn,
        move_hint=_tt_best_move(state, tt),
        score_hint=guess_score,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    result = SearchResult(
        best_move=move,
        score=score,
        top_moves=top_moves,
        depth=0,
        complete=False,
        elapsed_ms=elapsed_ms,
        nodes=0,
    )
    _emit_search_end(result, "timeout")
    return result


def best_move(
    state: State,
    topn: int = 3,
    tt: Optional[Dict[State, TTEntry]] = None,
) -> Tuple[Optional[int], int, List[Tuple[int, int]]]:
    result = solve_best_move(state, topn=topn, tt=tt, time_limit_ms=None)
    return result.best_move, result.score, result.top_moves
