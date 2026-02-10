"""Core rules engine for GamePigeon Mancala (Capture mode)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

YOU = "YOU"
OPP = "OPP"


@dataclass(frozen=True)
class State:
    to_move: str
    pits_you: Tuple[int, ...]
    pits_opp: Tuple[int, ...]
    store_you: int
    store_opp: int


@dataclass(frozen=True)
class MoveInfo:
    state: State
    extra_turn: bool
    capture: bool
    trace: "MoveTrace"


@dataclass(frozen=True)
class DropLocation:
    side: str
    index: Optional[int]
    store: Optional[str]


@dataclass(frozen=True)
class CaptureInfo:
    landing_side: str
    landing_index: int
    opposite_side: str
    opposite_index: int
    captured_count: int
    to_store: str


@dataclass(frozen=True)
class MoveTrace:
    mover: str
    picked_pit: int
    picked_index: int
    picked_count: int
    drops: Tuple[DropLocation, ...]
    last_drop: DropLocation
    capture: Optional[CaptureInfo]
    extra_turn: bool
    terminal_after: bool
    sweep_you: int
    sweep_opp: int


RING: Tuple[DropLocation, ...] = (
    DropLocation(YOU, 5, None),
    DropLocation(YOU, 4, None),
    DropLocation(YOU, 3, None),
    DropLocation(YOU, 2, None),
    DropLocation(YOU, 1, None),
    DropLocation(YOU, 0, None),
    DropLocation("STORE", None, YOU),
    DropLocation(OPP, 5, None),
    DropLocation(OPP, 4, None),
    DropLocation(OPP, 3, None),
    DropLocation(OPP, 2, None),
    DropLocation(OPP, 1, None),
    DropLocation(OPP, 0, None),
    DropLocation("STORE", None, OPP),
)

RING_INDEX_YOU = {loc.index: idx for idx, loc in enumerate(RING) if loc.side == YOU}
RING_INDEX_OPP = {loc.index: idx for idx, loc in enumerate(RING) if loc.side == OPP}


def initial_state(seeds: int = 4, you_first: bool = True) -> State:
    if seeds < 0:
        raise ValueError("seeds must be non-negative")
    pits = (seeds,) * 6
    return State(YOU if you_first else OPP, pits, pits, 0, 0)


def legal_moves(state: State) -> List[int]:
    pits = state.pits_you if state.to_move == YOU else state.pits_opp
    return [i + 1 for i, n in enumerate(pits) if n > 0]


def is_terminal(state: State) -> bool:
    return sum(state.pits_you) == 0 or sum(state.pits_opp) == 0


def final_diff(state: State) -> int:
    return state.store_you - state.store_opp


def apply_move(state: State, pit_num: int) -> State:
    return apply_move_fast(state, pit_num)


def apply_move_fast(state: State, pit_num: int) -> State:
    return apply_move_fast_with_info(state, pit_num)[0]


def apply_move_fast_with_info(state: State, pit_num: int) -> tuple[State, bool, bool]:
    new_state, extra_turn, capture = _apply_move_fast(state, pit_num)
    return new_state, extra_turn, capture


def apply_move_with_info(state: State, pit_num: int) -> MoveInfo:
    new_state, extra_turn, capture, trace = _apply_move(state, pit_num)
    return MoveInfo(new_state, extra_turn, capture, trace)


def _sow(
    pits_you: List[int],
    pits_opp: List[int],
    store_you: int,
    store_opp: int,
    mover: str,
    pit_index: int,
    record_drops: bool,
) -> tuple[List[int], List[int], int, int, int, DropLocation, Optional[List[DropLocation]]]:
    if mover == YOU:
        seeds = pits_you[pit_index]
        if seeds == 0:
            raise ValueError("illegal move: empty pit")
        pits_you[pit_index] = 0
        pos = RING_INDEX_YOU[pit_index]
    else:
        seeds = pits_opp[pit_index]
        if seeds == 0:
            raise ValueError("illegal move: empty pit")
        pits_opp[pit_index] = 0
        pos = RING_INDEX_OPP[pit_index]

    picked_count = seeds
    drops = [] if record_drops else None
    last_loc: Optional[DropLocation] = None

    while seeds > 0:
        pos = (pos + 1) % len(RING)
        loc = RING[pos]
        if loc.side == "STORE" and loc.store != mover:
            continue

        if loc.side == YOU and loc.index is not None:
            pits_you[loc.index] += 1
        elif loc.side == OPP and loc.index is not None:
            pits_opp[loc.index] += 1
        else:
            if mover == YOU:
                store_you += 1
            else:
                store_opp += 1

        if record_drops and drops is not None:
            drops.append(loc)
        last_loc = loc
        seeds -= 1

    if last_loc is None:
        raise ValueError("illegal move: no drops recorded")

    return pits_you, pits_opp, store_you, store_opp, picked_count, last_loc, drops


def _apply_move_fast(state: State, pit_num: int) -> tuple[State, bool, bool]:
    if pit_num < 1 or pit_num > 6:
        raise ValueError("pit number must be 1..6")

    pits_you = list(state.pits_you)
    pits_opp = list(state.pits_opp)
    store_you = state.store_you
    store_opp = state.store_opp
    to_move = state.to_move

    i = pit_num - 1
    extra_turn = False
    capture = False

    pits_you, pits_opp, store_you, store_opp, _, last_loc, _ = _sow(
        pits_you, pits_opp, store_you, store_opp, to_move, i, False
    )

    if to_move == YOU and last_loc.side == YOU and last_loc.index is not None:
        if pits_you[last_loc.index] == 1:
            opp_i = 5 - last_loc.index
            captured = pits_opp[opp_i] + pits_you[last_loc.index]
            if captured > 0:
                store_you += captured
                pits_you[last_loc.index] = 0
                pits_opp[opp_i] = 0
                capture = True
        extra_turn = last_loc.side == "STORE" and last_loc.store == YOU
    elif to_move == OPP and last_loc.side == OPP and last_loc.index is not None:
        if pits_opp[last_loc.index] == 1:
            you_i = 5 - last_loc.index
            captured = pits_you[you_i] + pits_opp[last_loc.index]
            if captured > 0:
                store_opp += captured
                pits_you[you_i] = 0
                pits_opp[last_loc.index] = 0
                capture = True
        extra_turn = last_loc.side == "STORE" and last_loc.store == OPP
    else:
        extra_turn = last_loc.side == "STORE" and last_loc.store == to_move

    if sum(pits_you) == 0 or sum(pits_opp) == 0:
        if sum(pits_you) == 0:
            store_opp += sum(pits_opp)
            pits_opp = [0, 0, 0, 0, 0, 0]
        if sum(pits_opp) == 0:
            store_you += sum(pits_you)
            pits_you = [0, 0, 0, 0, 0, 0]

    next_to_move = to_move if extra_turn else (OPP if to_move == YOU else YOU)
    new_state = State(next_to_move, tuple(pits_you), tuple(pits_opp), store_you, store_opp)
    return new_state, extra_turn, capture


def _apply_move(state: State, pit_num: int) -> tuple[State, bool, bool, MoveTrace]:
    if pit_num < 1 or pit_num > 6:
        raise ValueError("pit number must be 1..6")

    pits_you = list(state.pits_you)
    pits_opp = list(state.pits_opp)
    store_you = state.store_you
    store_opp = state.store_opp
    to_move = state.to_move

    i = pit_num - 1
    extra_turn = False
    capture = False
    capture_info: Optional[CaptureInfo] = None

    pits_you, pits_opp, store_you, store_opp, picked_count, last, drops = _sow(
        pits_you, pits_opp, store_you, store_opp, to_move, i, True
    )

    if to_move == YOU and last.side == YOU and last.index is not None:
        if pits_you[last.index] == 1:
            opp_i = 5 - last.index
            captured = pits_opp[opp_i] + pits_you[last.index]
            if captured > 0:
                store_you += captured
                pits_you[last.index] = 0
                pits_opp[opp_i] = 0
                capture = True
                capture_info = CaptureInfo(
                    landing_side=YOU,
                    landing_index=last.index,
                    opposite_side=OPP,
                    opposite_index=opp_i,
                    captured_count=captured,
                    to_store=YOU,
                )
        extra_turn = last.side == "STORE" and last.store == YOU
    elif to_move == OPP and last.side == OPP and last.index is not None:
        if pits_opp[last.index] == 1:
            you_i = 5 - last.index
            captured = pits_you[you_i] + pits_opp[last.index]
            if captured > 0:
                store_opp += captured
                pits_you[you_i] = 0
                pits_opp[last.index] = 0
                capture = True
                capture_info = CaptureInfo(
                    landing_side=OPP,
                    landing_index=last.index,
                    opposite_side=YOU,
                    opposite_index=you_i,
                    captured_count=captured,
                    to_store=OPP,
                )
        extra_turn = last.side == "STORE" and last.store == OPP
    else:
        extra_turn = last.side == "STORE" and last.store == to_move

    if not drops:
        raise ValueError("illegal move: no drops recorded")

    sweep_you = 0
    sweep_opp = 0
    terminal_after = False
    if sum(pits_you) == 0 or sum(pits_opp) == 0:
        terminal_after = True
        if sum(pits_you) == 0:
            sweep_opp = sum(pits_opp)
            store_opp += sweep_opp
            pits_opp = [0, 0, 0, 0, 0, 0]
        if sum(pits_opp) == 0:
            sweep_you = sum(pits_you)
            store_you += sweep_you
            pits_you = [0, 0, 0, 0, 0, 0]

    next_to_move = to_move if extra_turn else (OPP if to_move == YOU else YOU)
    new_state = State(next_to_move, tuple(pits_you), tuple(pits_opp), store_you, store_opp)
    trace = MoveTrace(
        mover=to_move,
        picked_pit=pit_num,
        picked_index=i,
        picked_count=picked_count,
        drops=tuple(drops),
        last_drop=drops[-1],
        capture=capture_info,
        extra_turn=extra_turn,
        terminal_after=terminal_after,
        sweep_you=sweep_you,
        sweep_opp=sweep_opp,
    )
    return new_state, extra_turn, capture, trace


def pretty_print(state: State) -> str:
    """
    Pretty, GamePigeon-like vertical Kalah board.

    Conventions (locked to your earlier decisions):
      - Opponent pits: 1..6 left->right, with pit 1 closest to OS (left store).
      - Your pits: pit 1 closest to YS (right store), so displayed left->right as 6..1.
      - Stores shown as left (OS) and right (YS) columns.
    """
    max_val = max(state.store_you, state.store_opp, *state.pits_you, *state.pits_opp)
    digits = max(2, len(str(max_val)))  # keep 2-digit look, expand if needed

    pit_inner = digits + 2              # " " + digits + " "
    store_inner = max(4, digits + 2)    # room for OS/YS labels + value

    def pit_cell(n: int) -> str:
        return f" {n:>{digits}} "

    def label_cell(s: str, inner: int) -> str:
        return f"{s:^{inner}}"

    def store_val_cell(n: int) -> str:
        # center the value to feel more like a "store" column
        return f"{n:^{store_inner}d}"

    # Data rows
    opp_vals = [pit_cell(n) for n in state.pits_opp]                 # pit 1..6
    you_vals = [pit_cell(n) for n in reversed(state.pits_you)]       # pit 6..1

    # Numbering lines aligned to pit columns
    opp_nums = " ".join(f"{i:^{pit_inner}d}" for i in range(1, 7))
    you_nums = " ".join(f"{i:^{pit_inner}d}" for i in range(6, 0, -1))

    # Borders / rows (8 columns: OS + 6 pits + YS)
    border = (
        "+"
        + "-".join(["-" * store_inner] + ["-" * pit_inner] * 6 + ["-" * store_inner]).replace("-", "+")
        + "+"
    )
    # The replace trick above is a clean way to turn segments into +---+---+ without manual join.
    # Example: "+-----+----+...+-----+"

    row_opp = (
        "|"
        + "|".join([label_cell("OS", store_inner), *opp_vals, label_cell("YS", store_inner)])
        + "|"
    )
    row_you = (
        "|"
        + "|".join([store_val_cell(state.store_opp), *you_vals, store_val_cell(state.store_you)])
        + "|"
    )

    # Headings
    turn = f"Turn: {state.to_move}"
    opp_head = "OPPONENT"
    you_head = "YOU"

    # Left padding so headings/numbers visually center over the pit block
    pit_block_width = len(border)
    opp_head_line = opp_head.center(pit_block_width)
    you_head_line = you_head.center(pit_block_width)

    # Number lines start after left store column plus left border + column separator
    num_indent = " " * (1 + store_inner + 1)  # '|' + store col + '|'

    lines = [
        turn,
        "",
        opp_head_line,
        f"{num_indent}{opp_nums}",
        border,
        row_opp,
        row_you,
        border,
        f"{num_indent}{you_nums}",
        you_head_line,
        "",
        "Legend: Opp pits are 1→6 left-to-right (pit 1 near OS). Your pit 1 is near YS (right), so your row shows 6→1.",
    ]
    return "\n".join(lines)
