"""Minimax solver with depth-aware TT and iterative deepening."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import gzip
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

INF = 10**9
FULL_DEPTH = 1_000_000
CACHE_VERSION = 5
TT_MAX_ENTRIES = 1_000_000
TT_PRUNE_TO = 800_000

EXACT = 0
LOWER = 1
UPPER = 2


@dataclass(frozen=True)
class TTEntry:
    value: int
    flag: int
    best_move: Optional[int]
    depth: int


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
    nodes: int = 0
    hit_depth_limit: bool = False


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


def _prune_tt(tt: Dict[State, TTEntry]) -> None:
    if len(tt) <= TT_MAX_ENTRIES:
        return
    while len(tt) > TT_PRUNE_TO:
        tt.pop(next(iter(tt)))


def _tt_store(tt: Dict[State, TTEntry], state: State, entry: TTEntry) -> None:
    existing = tt.get(state)
    if existing is not None:
        if existing.depth > entry.depth:
            return
        if existing.depth == entry.depth and existing.flag == EXACT and entry.flag != EXACT:
            return
    tt[state] = entry
    if len(tt) > TT_MAX_ENTRIES:
        _prune_tt(tt)


def load_tt(path: Path) -> Dict[State, TTEntry]:
    # Security: pickle is unsafe for untrusted files. Only load caches you trust.
    try:
        with gzip.open(path, "rb") as handle:
            raw = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError):
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
        if not isinstance(key, str):
            continue
        if not isinstance(entry, tuple) or len(entry) != 4:
            continue
        value, flag, best_move, depth = entry
        if not isinstance(value, int):
            continue
        if flag not in {EXACT, LOWER, UPPER}:
            continue
        if best_move is not None and not isinstance(best_move, int):
            continue
        if not isinstance(depth, int) or depth < 0:
            continue
        state = key_to_state(key)
        if state is None:
            continue
        tt[state] = TTEntry(value, flag, best_move, depth)
    _prune_tt(tt)
    return tt


def save_tt(tt: Dict[State, TTEntry], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": CACHE_VERSION,
            "tt": {
                state_key(state): (entry.value, entry.flag, entry.best_move, entry.depth)
                for state, entry in tt.items()
            },
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp") if path.suffix else path.with_name(path.name + ".tmp")
        with gzip.open(tmp_path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
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


def ordered_children(state: State, tt: Dict[State, TTEntry]) -> List[Tuple[int, State, bool, bool, int]]:
    moves = legal_moves(state)
    if not moves:
        return []

    best_first = _tt_best_move(state, tt)
    children: List[Tuple[int, State, bool, bool, int]] = []

    rest_moves = [move for move in moves if move != best_first]
    if best_first in moves:
        child_state, extra_turn, capture = apply_move_fast_with_info(state, best_first)
        if state.to_move == YOU:
            store_gain = child_state.store_you - state.store_you
        else:
            store_gain = child_state.store_opp - state.store_opp
        children.append((best_first, child_state, extra_turn, capture, store_gain))

    rest: List[Tuple[int, State, bool, bool, int]] = []
    for move in rest_moves:
        child_state, extra_turn, capture = apply_move_fast_with_info(state, move)
        if state.to_move == YOU:
            store_gain = child_state.store_you - state.store_you
        else:
            store_gain = child_state.store_opp - state.store_opp
        rest.append((move, child_state, extra_turn, capture, store_gain))

    rest.sort(key=lambda item: (item[2], item[3], item[4]), reverse=True)
    return children + rest if children else rest


def _check_deadline(context: _SearchContext) -> None:
    if context.deadline is not None and time.perf_counter() >= context.deadline:
        raise SearchTimeout()


def _search_depth(state: State, alpha: int, beta: int, depth: int, context: _SearchContext) -> int:
    _check_deadline(context)
    context.nodes += 1

    if is_terminal(state):
        return terminal_diff(state)

    if depth <= 0:
        context.hit_depth_limit = True
        return _heuristic_eval(state)

    alpha0, beta0 = alpha, beta
    norm_state, sign = normalize_state(state)
    entry = _tt_get(context.tt, norm_state)
    if entry is not None and entry.depth >= depth:
        value, flag = denormalize_value_flag(entry.value, entry.flag, sign)
        if flag == EXACT:
            return value
        if flag == LOWER:
            alpha = max(alpha, value)
        elif flag == UPPER:
            beta = min(beta, value)
        if alpha >= beta:
            return value

    children = ordered_children(state, context.tt)
    if not children:
        return terminal_diff(state)

    best_move: Optional[int] = None
    if state.to_move == YOU:
        best = -INF
        for move, child_state, _, _, _ in children:
            val = _search_depth(child_state, alpha, beta, depth - 1, context)
            if val > best:
                best = val
                best_move = move
            alpha = max(alpha, best)
            if alpha >= beta:
                break
    else:
        best = INF
        for move, child_state, _, _, _ in children:
            val = _search_depth(child_state, alpha, beta, depth - 1, context)
            if val < best:
                best = val
                best_move = move
            beta = min(beta, best)
            if alpha >= beta:
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
    _tt_store(context.tt, norm_state, TTEntry(value_norm, flag_norm, best_move_norm, depth))
    return best


def search_depth(
    state: State,
    depth: int,
    tt: Dict[State, TTEntry],
    alpha: int = -INF,
    beta: int = INF,
) -> int:
    context = _SearchContext(tt=tt, deadline=None)
    return _search_depth(state, alpha, beta, max(0, depth), context)


def search(state: State, alpha: int, beta: int, tt: Dict[State, TTEntry]) -> int:
    return search_depth(state, FULL_DEPTH, tt, alpha=alpha, beta=beta)


def _best_move_depth(
    state: State,
    topn: int,
    depth: int,
    context: _SearchContext,
) -> Tuple[Optional[int], int, List[Tuple[int, int]]]:
    children = ordered_children(state, context.tt)
    if not children:
        return None, terminal_diff(state), []

    scored: List[Tuple[int, int]] = []
    if state.to_move == YOU:
        alpha = -INF
        beta = INF
        best_val = -INF
        for move, child_state, _, _, _ in children:
            val = _search_depth(child_state, alpha, beta, depth - 1, context)
            scored.append((move, val))
            if val > best_val:
                best_val = val
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break
        scored.sort(key=lambda item: item[1], reverse=True)
    else:
        alpha = -INF
        beta = INF
        best_val = INF
        for move, child_state, _, _, _ in children:
            val = _search_depth(child_state, alpha, beta, depth - 1, context)
            scored.append((move, val))
            if val < best_val:
                best_val = val
            beta = min(beta, best_val)
            if alpha >= beta:
                break
        scored.sort(key=lambda item: item[1])

    topn = max(0, topn)
    top_moves = scored[:topn] if topn > 0 else []
    return scored[0][0], scored[0][1], top_moves


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


def solve_best_move(
    state: State,
    topn: int = 3,
    tt: Optional[Dict[State, TTEntry]] = None,
    time_limit_ms: Optional[int] = None,
    progress_callback: Optional[Callable[[SearchResult], None]] = None,
) -> SearchResult:
    if tt is None:
        tt = {}

    start = time.perf_counter()
    if is_terminal(state):
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return SearchResult(
            best_move=None,
            score=terminal_diff(state),
            top_moves=[],
            depth=0,
            complete=True,
            elapsed_ms=elapsed_ms,
            nodes=0,
        )

    if time_limit_ms is None:
        context = _SearchContext(tt=tt, deadline=None)
        move, score, top_moves = _best_move_depth(state, topn, FULL_DEPTH, context)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        result = SearchResult(
            best_move=move,
            score=score,
            top_moves=top_moves,
            depth=0,
            complete=True,
            elapsed_ms=elapsed_ms,
            nodes=context.nodes,
        )
        if progress_callback is not None:
            progress_callback(result)
        return result

    deadline = start + max(0, time_limit_ms) / 1000.0
    total_nodes = 0
    best_result: Optional[SearchResult] = None
    depth = 1

    while True:
        if time.perf_counter() >= deadline:
            break

        context = _SearchContext(tt=tt, deadline=deadline)
        try:
            move, score, top_moves = _best_move_depth(state, topn, depth, context)
        except SearchTimeout:
            break

        total_nodes += context.nodes
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        result = SearchResult(
            best_move=move,
            score=score,
            top_moves=top_moves,
            depth=depth,
            complete=not context.hit_depth_limit,
            elapsed_ms=elapsed_ms,
            nodes=total_nodes,
        )
        best_result = result
        if progress_callback is not None:
            progress_callback(result)
        if result.complete:
            return result
        depth += 1

    if best_result is not None:
        return best_result

    # Deadline was hit before depth 1 could complete.
    move, score, top_moves = _fallback_best_move(state, topn)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return SearchResult(
        best_move=move,
        score=score,
        top_moves=top_moves,
        depth=0,
        complete=False,
        elapsed_ms=elapsed_ms,
        nodes=0,
    )


def best_move(
    state: State,
    topn: int = 3,
    tt: Optional[Dict[State, TTEntry]] = None,
) -> Tuple[Optional[int], int, List[Tuple[int, int]]]:
    result = solve_best_move(state, topn=topn, tt=tt, time_limit_ms=None)
    return result.best_move, result.score, result.top_moves
