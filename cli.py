"""CLI for GamePigeon Mancala (Capture mode) solver."""

from __future__ import annotations

import argparse
import atexit
import sys
from pathlib import Path
from typing import List, Optional

from mancala_engine import (
    OPP,
    YOU,
    State,
    apply_move,
    final_diff,
    initial_state,
    is_terminal,
    legal_moves,
    pretty_print,
)
from mancala_solver import default_cache_path, load_tt, save_tt, solve_best_move


def prompt_yes_no(prompt: str) -> bool:
    while True:
        try:
            raw = input(prompt).strip().lower()
        except EOFError:
            print()
            sys.exit(0)
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


def print_help() -> None:
    print("Controls: 1-6 = play pit, u=undo, q=quit, h=help.")
    print("Numbering: pit 1 is closest to each player's store.")


def read_move(prompt: str, allow_enter: bool, default_move: Optional[int]) -> str:
    while True:
        try:
            raw = input(prompt).strip().lower()
        except EOFError:
            print()
            return "q"
        if raw == "":
            if allow_enter and default_move is not None:
                return ""
            print("Please enter a pit number 1-6, or a command.")
            continue
        return raw


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GamePigeon Mancala (Capture mode) CLI solver")
    parser.add_argument("--seeds", type=int, default=4, help="starting seeds per pit (default: 4)")
    parser.add_argument("--topn", type=int, default=3, help="show top N moves (default: 3)")
    parser.add_argument("--explain", action="store_true", help="print evals for top moves")
    parser.add_argument(
        "--time-ms",
        type=int,
        default=300,
        help="search budget per recommendation in milliseconds (default: 300)",
    )
    parser.add_argument(
        "--perfect",
        action="store_true",
        help="search to terminal (ignores --time-ms)",
    )
    args = parser.parse_args(argv)

    if args.seeds < 0:
        print("--seeds must be non-negative")
        return 2

    you_first = prompt_yes_no("Do you go first? (y/n): ")
    state = initial_state(seeds=args.seeds, you_first=you_first)
    history: list[State] = []
    cache_path: Path = default_cache_path()
    tt = load_tt(cache_path)
    atexit.register(lambda: save_tt(tt, cache_path))

    while True:
        print()
        print(pretty_print(state))

        if is_terminal(state):
            print()
            print(
                f"Game over. You {state.store_you} - Opponent {state.store_opp} "
                f"(diff {final_diff(state):+d})"
            )
            return 0

        recommended_move: Optional[int] = None
        if state.to_move == YOU:
            time_limit_ms: Optional[int]
            if args.perfect or args.time_ms <= 0:
                time_limit_ms = None
            else:
                time_limit_ms = args.time_ms
            result = solve_best_move(state, topn=args.topn, tt=tt, time_limit_ms=time_limit_ms)
            recommended_move = result.best_move
            if recommended_move is None:
                print("No legal moves.")
                return 0

            status = "perfect" if result.complete else "best so far"
            print()
            print(f"Recommended: pit {recommended_move} (score: {result.score:+d}, {status})")
            if args.explain and result.top_moves:
                top_str = ", ".join(f"{m}:{e:+d}" for m, e in result.top_moves)
                print(f"Top: {top_str}")
                print(
                    f"Search: depth={result.depth} complete={result.complete} "
                    f"elapsed_ms={result.elapsed_ms} nodes={result.nodes}"
                )
            elif args.explain:
                print(
                    f"Search: depth={result.depth} complete={result.complete} "
                    f"elapsed_ms={result.elapsed_ms} nodes={result.nodes}"
                )

            print("You: choose pit 1..6 (pit 1 is closest to your store).")
            raw = read_move(
                "Your move (1-6, Enter=best, q=quit, u=undo, h=help): ",
                True,
                recommended_move,
            )
        else:
            print("Opponent: enter pit 1..6 (pit 1 is closest to their store).")
            raw = read_move("Opponent move (1-6, q=quit, u=undo, h=help): ", False, None)

        if raw in {"q", "quit"}:
            return 0
        if raw in {"h", "help"}:
            print_help()
            continue
        if raw in {"u", "undo"}:
            if history:
                state = history.pop()
            else:
                print("Nothing to undo.")
            continue

        if raw == "":
            if recommended_move is None:
                print("Enter is only available on your turn when a recommendation is shown.")
                continue
            chosen = recommended_move
        else:
            if not raw.isdigit():
                print("Please enter a pit number 1-6, or a command.")
                continue
            chosen = int(raw)

        if chosen not in legal_moves(state):
            print("Illegal move: pit is empty or out of range.")
            continue

        history.append(state)
        try:
            state = apply_move(state, chosen)
        except ValueError as exc:
            print(str(exc))
            continue


if __name__ == "__main__":
    raise SystemExit(main())
