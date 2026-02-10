"""Minimax solver with alpha-beta pruning for Mancala (Capture mode)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gzip
import pickle

from mancala_engine import (
    OPP,
    YOU,
    State,
    apply_move_fast_with_info,
    is_terminal,
    legal_moves,
)

INF = 10**9
CACHE_VERSION = 4

EXACT = 0
LOWER = 1
UPPER = 2


@dataclass(frozen=True)
class TTEntry:
    value: int
    flag: int
    best_move: Optional[int]


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


def load_tt(path: Path) -> Dict[State, TTEntry]:
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
        if not isinstance(entry, tuple) or len(entry) != 3:
            continue
        value, flag, best_move = entry
        if not isinstance(value, int):
            continue
        if flag not in {EXACT, LOWER, UPPER}:
            continue
        if best_move is not None and not isinstance(best_move, int):
            continue
        state = key_to_state(key)
        if state is None:
            continue
        tt[state] = TTEntry(value, flag, best_move)
    return tt


def save_tt(tt: Dict[State, TTEntry], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": CACHE_VERSION,
            "tt": {state_key(state): (entry.value, entry.flag, entry.best_move) for state, entry in tt.items()},
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
    entry = tt.get(norm_state)
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

    if children:
        return children + rest
    return rest


def search(state: State, alpha: int, beta: int, tt: Dict[State, TTEntry]) -> int:
    if is_terminal(state):
        return terminal_diff(state)

    alpha0, beta0 = alpha, beta
    norm_state, sign = normalize_state(state)
    entry = tt.get(norm_state)
    if entry is not None:
        value, flag = denormalize_value_flag(entry.value, entry.flag, sign)
        if flag == EXACT:
            return value
        if flag == LOWER:
            alpha = max(alpha, value)
        elif flag == UPPER:
            beta = min(beta, value)
        if alpha >= beta:
            return value

    best_move: Optional[int] = None
    children = ordered_children(state, tt)
    if not children:
        return terminal_diff(state)

    if state.to_move == YOU:
        best = -INF
        for move, child_state, _, _, _ in children:
            val = search(child_state, alpha, beta, tt)
            if val > best:
                best = val
                best_move = move
            alpha = max(alpha, best)
            if alpha >= beta:
                break
    else:
        best = INF
        for move, child_state, _, _, _ in children:
            val = search(child_state, alpha, beta, tt)
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
    tt[norm_state] = TTEntry(value_norm, flag_norm, best_move_norm)
    return best


def best_move(
    state: State,
    topn: int = 3,
    tt: Optional[Dict[State, TTEntry]] = None,
) -> Tuple[Optional[int], int, List[Tuple[int, int]]]:
    if tt is None:
        tt = {}

    children = ordered_children(state, tt)
    if not children:
        return None, terminal_diff(state), []

    scored = []
    if state.to_move == YOU:
        alpha = -INF
        beta = INF
        best_val = -INF
        for move, child_state, _, _, _ in children:
            val = search(child_state, alpha, beta, tt)
            scored.append((move, val))
            if val > best_val:
                best_val = val
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break
    else:
        alpha = -INF
        beta = INF
        best_val = INF
        for move, child_state, _, _, _ in children:
            val = search(child_state, alpha, beta, tt)
            scored.append((move, val))
            if val < best_val:
                best_val = val
            beta = min(beta, best_val)
            if alpha >= beta:
                break

    if state.to_move == YOU:
        scored.sort(key=lambda x: x[1], reverse=True)
    else:
        scored.sort(key=lambda x: x[1])

    topn = max(0, topn)
    top_moves = scored[:topn] if topn > 0 else []
    return scored[0][0], scored[0][1], top_moves
