"""Minimax solver with alpha-beta pruning for Mancala (Capture mode)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from mancala_engine import (
    OPP,
    YOU,
    State,
    apply_move,
    apply_move_with_info,
    final_diff,
    is_terminal,
    legal_moves,
)

INF = 10**9
CACHE_VERSION = 3


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
    return Path.home() / ".mancala_cache.json"


def load_tt(path: Path) -> Dict[State, int]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(raw, dict):
        return {}
    if raw.get("version") != CACHE_VERSION:
        return {}
    raw_tt = raw.get("tt")
    if not isinstance(raw_tt, dict):
        return {}

    tt: Dict[State, int] = {}
    for key, value in raw_tt.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, int):
            continue
        state = key_to_state(key)
        if state is None:
            continue
        tt[state] = value
    return tt


def save_tt(tt: Dict[State, int], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": CACHE_VERSION,
            "tt": {state_key(state): value for state, value in tt.items()},
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp") if path.suffix else path.with_name(path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, separators=(",", ":"))
        tmp_path.replace(path)
    except OSError:
        return


def ordered_moves(state: State) -> List[int]:
    moves = legal_moves(state)
    scored = []
    for move in moves:
        info = apply_move_with_info(state, move)
        if state.to_move == YOU:
            store_gain = info.state.store_you - state.store_you
        else:
            store_gain = info.state.store_opp - state.store_opp
        scored.append((info.extra_turn, info.capture, store_gain, move))
    scored.sort(reverse=True)
    return [m for _, _, _, m in scored]


def search(state: State, alpha: int, beta: int, tt: Dict[State, int]) -> int:
    if is_terminal(state):
        return final_diff(state)

    if state in tt:
        return tt[state]

    if state.to_move == YOU:
        best = -INF
        for move in ordered_moves(state):
            val = search(apply_move(state, move), alpha, beta, tt)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
    else:
        best = INF
        for move in ordered_moves(state):
            val = search(apply_move(state, move), alpha, beta, tt)
            if val < best:
                best = val
            if best < beta:
                beta = best
            if alpha >= beta:
                break

    tt[state] = best
    return best


def best_move(
    state: State,
    topn: int = 3,
    tt: Optional[Dict[State, int]] = None,
) -> Tuple[Optional[int], int, List[Tuple[int, int]]]:
    if tt is None:
        tt = {}

    moves = ordered_moves(state)
    if not moves:
        return None, final_diff(state), []

    scored = []
    for move in moves:
        val = search(apply_move(state, move), -INF, INF, tt)
        scored.append((move, val))

    if state.to_move == YOU:
        scored.sort(key=lambda x: x[1], reverse=True)
    else:
        scored.sort(key=lambda x: x[1])

    topn = max(0, topn)
    top_moves = scored[:topn] if topn > 0 else []
    return scored[0][0], scored[0][1], top_moves
