"""Deterministic benchmark harness for the Mancala solver."""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List

from mancala_engine import State, apply_move, initial_state, is_terminal, legal_moves
from mancala_solver import TTEntry, solve_best_move


def _generate_positions(
    *,
    positions: int,
    max_plies: int,
    seeds: int,
    seed: int,
) -> List[State]:
    rng = random.Random(seed)
    out: List[State] = []
    while len(out) < positions:
        state = initial_state(seeds=seeds, you_first=bool(rng.getrandbits(1)))
        plies = rng.randint(0, max_plies)
        for _ in range(plies):
            if is_terminal(state):
                break
            state = apply_move(state, rng.choice(legal_moves(state)))
        if not is_terminal(state):
            out.append(state)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic solver benchmark")
    parser.add_argument("--positions", type=int, default=20, help="number of positions (default: 20)")
    parser.add_argument("--max-plies", type=int, default=24, help="max random plies from start (default: 24)")
    parser.add_argument("--seeds", type=int, default=4, help="initial seeds per pit (default: 4)")
    parser.add_argument("--seed", type=int, default=12345, help="random seed for position generation")
    parser.add_argument("--topn", type=int, default=3, help="top-N scoring width (default: 3)")
    parser.add_argument("--time-ms", type=int, default=300, help="slice budget for each solve (default: 300)")
    parser.add_argument("--perfect", action="store_true", help="search to terminal for each position")
    parser.add_argument(
        "--reuse-tt",
        action="store_true",
        help="reuse one TT across all positions (default: fresh TT per position)",
    )
    args = parser.parse_args()

    if args.positions <= 0:
        print("--positions must be > 0")
        return 2
    if args.max_plies < 0:
        print("--max-plies must be >= 0")
        return 2
    if args.seeds < 0:
        print("--seeds must be >= 0")
        return 2

    positions = _generate_positions(
        positions=args.positions,
        max_plies=args.max_plies,
        seeds=args.seeds,
        seed=args.seed,
    )
    time_limit_ms = None if args.perfect or args.time_ms <= 0 else args.time_ms
    shared_tt: Dict[State, TTEntry] = {}

    print(
        "idx depth complete nodes solver_ms wall_ms best score top1 "
        f"(positions={len(positions)} seeds={args.seeds} seed={args.seed})"
    )

    total_nodes = 0
    total_solver_ms = 0
    wall_start = time.perf_counter()
    solved_count = 0
    deepest = 0

    for idx, state in enumerate(positions, start=1):
        tt = shared_tt if args.reuse_tt else {}
        start = time.perf_counter()
        result = solve_best_move(
            state,
            topn=args.topn,
            tt=tt,
            time_limit_ms=time_limit_ms,
        )
        wall_ms = int((time.perf_counter() - start) * 1000)
        total_nodes += result.nodes
        total_solver_ms += result.elapsed_ms
        deepest = max(deepest, result.depth)
        if result.complete:
            solved_count += 1
        top1 = result.top_moves[0][0] if result.top_moves else "-"
        print(
            f"{idx:03d} {result.depth:>5d} {str(result.complete):>8} "
            f"{result.nodes:>9d} {result.elapsed_ms:>8d} {wall_ms:>7d} "
            f"{str(result.best_move):>4} {result.score:>+5d} {top1}"
        )

    total_wall_ms = max(1, int((time.perf_counter() - wall_start) * 1000))
    nps_wall = int(total_nodes * 1000 / total_wall_ms)
    avg_solver_ms = total_solver_ms / len(positions)
    avg_nodes = total_nodes / len(positions)
    print(
        "summary "
        f"positions={len(positions)} solved={solved_count} deepest={deepest} "
        f"total_nodes={total_nodes} total_solver_ms={total_solver_ms} total_wall_ms={total_wall_ms} "
        f"nps_wall={nps_wall} avg_solver_ms={avg_solver_ms:.1f} avg_nodes={avg_nodes:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
