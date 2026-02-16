"""Deterministic benchmark harness for the Mancala solver."""

from __future__ import annotations

import argparse
import gc
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from mancala_engine import State, apply_move, initial_state, is_terminal, legal_moves
from mancala_solver import (
    INF,
    TTEntry,
    _SearchContext,
    _best_move_depth,
    key_to_state,
    solve_best_move,
    state_key,
)


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


def _load_positions(path: Path, limit: int) -> List[State]:
    states: List[State] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            key = raw.strip()
            if not key:
                continue
            state = key_to_state(key)
            if state is None:
                raise ValueError(f"invalid state key at line {line_no}: {key!r}")
            if is_terminal(state):
                continue
            states.append(state)
            if len(states) >= limit:
                break
    return states


def _save_positions(path: Path, positions: Sequence[State]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for state in positions:
            handle.write(state_key(state) + "\n")


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _solve_fixed_depth(state: State, topn: int, tt: Dict[State, TTEntry], depth: int):
    start_ns = time.perf_counter_ns()
    context = _SearchContext(tt=tt, deadline_ns=None, iter_depth=depth)
    depth_result = _best_move_depth(state, topn=topn, depth=depth, context=context, alpha=-INF, beta=INF)
    elapsed_ms = max(0, (time.perf_counter_ns() - start_ns) // 1_000_000)
    complete = not context.hit_horizon and not context.used_unproven_exact_tt
    return {
        "best_move": depth_result.best_move,
        "score": depth_result.score,
        "top_moves": depth_result.top_moves,
        "depth": depth,
        "complete": complete,
        "elapsed_ms": int(elapsed_ms),
        "nodes": context.nodes,
    }


def _run_single_solve(
    *,
    state: State,
    topn: int,
    tt: Dict[State, TTEntry],
    time_limit_ms: Optional[int],
    depth: Optional[int],
):
    if depth is not None:
        return _solve_fixed_depth(state, topn=topn, tt=tt, depth=depth)
    result = solve_best_move(
        state,
        topn=topn,
        tt=tt,
        time_limit_ms=time_limit_ms,
    )
    return {
        "best_move": result.best_move,
        "score": result.score,
        "top_moves": result.top_moves,
        "depth": result.depth,
        "complete": result.complete,
        "elapsed_ms": result.elapsed_ms,
        "nodes": result.nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic solver benchmark")
    parser.add_argument("--positions", type=int, default=20, help="number of positions (default: 20)")
    parser.add_argument("--max-plies", type=int, default=24, help="max random plies from start (default: 24)")
    parser.add_argument("--seeds", type=int, default=4, help="initial seeds per pit (default: 4)")
    parser.add_argument("--seed", type=int, default=12345, help="random seed for position generation")
    parser.add_argument("--topn", type=int, default=3, help="top-N scoring width (default: 3)")
    parser.add_argument("--time-ms", type=int, default=300, help="slice budget for each solve (default: 300)")
    parser.add_argument("--depth", type=int, default=None, help="fixed search depth (disables --time-ms/--perfect)")
    parser.add_argument("--perfect", action="store_true", help="search to terminal for each position")
    parser.add_argument(
        "--reuse-tt",
        action="store_true",
        help="reuse one TT across all positions (default: fresh TT per position)",
    )
    parser.add_argument("--repeat", type=int, default=1, help="benchmark repeats for p50/p95 summaries")
    parser.add_argument("--warmup", type=int, default=2, help="number of warmup solves per repeat (not counted)")
    parser.add_argument("--no-gc", action="store_true", help="disable GC during benchmark loop")
    parser.add_argument(
        "--save-positions",
        type=Path,
        default=None,
        help="write sampled benchmark positions (state keys) to file",
    )
    parser.add_argument(
        "--load-positions",
        type=Path,
        default=None,
        help="load benchmark positions (state keys) from file",
    )
    args = parser.parse_args()

    if args.positions <= 0:
        print("--positions must be > 0")
        return 2
    if args.max_plies < 0:
        print("--max-plies must be >= 0")
        return 2
    if args.seeds <= 0:
        print("--seeds must be > 0")
        return 2
    if args.topn < 0:
        print("--topn must be >= 0")
        return 2
    if args.repeat <= 0:
        print("--repeat must be > 0")
        return 2
    if args.warmup < 0:
        print("--warmup must be >= 0")
        return 2
    if args.depth is not None and args.depth <= 0:
        print("--depth must be > 0")
        return 2
    if args.depth is not None and args.perfect:
        print("--depth and --perfect are mutually exclusive")
        return 2
    if args.load_positions is not None and not args.load_positions.exists():
        print(f"--load-positions not found: {args.load_positions}")
        return 2

    if args.load_positions is not None:
        try:
            positions = _load_positions(args.load_positions, args.positions)
        except ValueError as exc:
            print(f"failed to load positions: {exc}")
            return 2
        if len(positions) < args.positions:
            print(
                f"--load-positions provided only {len(positions)} usable non-terminal states; "
                f"need {args.positions}"
            )
            return 2
    else:
        positions = _generate_positions(
            positions=args.positions,
            max_plies=args.max_plies,
            seeds=args.seeds,
            seed=args.seed,
        )
    if args.save_positions is not None:
        _save_positions(args.save_positions, positions)

    mode = "depth" if args.depth is not None else ("perfect" if args.perfect or args.time_ms <= 0 else "timed")
    time_limit_ms = None if mode != "timed" else args.time_ms

    print(
        f"python={sys.version.split()[0]} platform={platform.platform()} "
        f"mode={mode} repeats={args.repeat} warmup={args.warmup} reuse_tt={args.reuse_tt}"
    )
    print(
        "rep idx depth complete nodes solver_ms wall_ms best score top1 "
        f"(positions={len(positions)} seeds={args.seeds} seed={args.seed})"
    )

    gc_was_enabled = gc.isenabled()
    repeat_summaries: List[Dict[str, float]] = []
    if args.no_gc and gc_was_enabled:
        gc.disable()
    try:
        for rep in range(1, args.repeat + 1):
            warmup_count = min(args.warmup, len(positions))
            if warmup_count > 0:
                warmup_tt: Dict[State, TTEntry] = {}
                for warm_state in positions[:warmup_count]:
                    _run_single_solve(
                        state=warm_state,
                        topn=args.topn,
                        tt=warmup_tt if args.reuse_tt else {},
                        time_limit_ms=time_limit_ms,
                        depth=args.depth,
                    )

            total_nodes = 0
            total_solver_ms = 0
            wall_start_ns = time.perf_counter_ns()
            solved_count = 0
            deepest = 0
            shared_tt: Dict[State, TTEntry] = {}

            for idx, state in enumerate(positions, start=1):
                tt = shared_tt if args.reuse_tt else {}
                start_ns = time.perf_counter_ns()
                result = _run_single_solve(
                    state=state,
                    topn=args.topn,
                    tt=tt,
                    time_limit_ms=time_limit_ms,
                    depth=args.depth,
                )
                wall_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
                total_nodes += int(result["nodes"])
                total_solver_ms += int(result["elapsed_ms"])
                deepest = max(deepest, int(result["depth"]))
                if bool(result["complete"]):
                    solved_count += 1
                top1 = result["top_moves"][0][0] if result["top_moves"] else "-"
                print(
                    f"{rep:>3d} {idx:03d} {int(result['depth']):>5d} {str(result['complete']):>8} "
                    f"{int(result['nodes']):>9d} {int(result['elapsed_ms']):>9d} {int(wall_ms):>7d} "
                    f"{str(result['best_move']):>4} {int(result['score']):>+5d} {top1}"
                )

            total_wall_ms = max(1, (time.perf_counter_ns() - wall_start_ns) // 1_000_000)
            nps_wall = int(total_nodes * 1000 / total_wall_ms)
            avg_solver_ms = total_solver_ms / len(positions)
            avg_nodes = total_nodes / len(positions)
            tt_size = len(shared_tt) if args.reuse_tt else 0
            repeat_summaries.append(
                {
                    "solved": float(solved_count),
                    "deepest": float(deepest),
                    "total_nodes": float(total_nodes),
                    "total_solver_ms": float(total_solver_ms),
                    "total_wall_ms": float(total_wall_ms),
                    "nps_wall": float(nps_wall),
                    "avg_solver_ms": float(avg_solver_ms),
                    "avg_nodes": float(avg_nodes),
                    "tt_size": float(tt_size),
                }
            )
            tt_suffix = f" tt_size={tt_size}" if args.reuse_tt else ""
            print(
                "summary "
                f"rep={rep} positions={len(positions)} solved={solved_count} deepest={deepest} "
                f"total_nodes={total_nodes} total_solver_ms={total_solver_ms} total_wall_ms={total_wall_ms} "
                f"nps_wall={nps_wall} avg_solver_ms={avg_solver_ms:.1f} avg_nodes={avg_nodes:.1f}"
                f"{tt_suffix}"
            )
    finally:
        if args.no_gc and gc_was_enabled:
            gc.enable()

    if args.repeat > 1:
        def _print_dist(name: str, key: str, as_int: bool = False) -> None:
            values = [summary[key] for summary in repeat_summaries]
            p50 = _percentile(values, 0.50)
            p95 = _percentile(values, 0.95)
            mean = statistics.fmean(values)
            if as_int:
                print(
                    f"dist {name} min={int(min(values))} p50={int(round(p50))} "
                    f"p95={int(round(p95))} max={int(max(values))} mean={int(round(mean))}"
                )
            else:
                print(
                    f"dist {name} min={min(values):.2f} p50={p50:.2f} "
                    f"p95={p95:.2f} max={max(values):.2f} mean={mean:.2f}"
                )

        _print_dist("nps_wall", "nps_wall", as_int=True)
        _print_dist("avg_nodes", "avg_nodes")
        _print_dist("deepest", "deepest", as_int=True)
        _print_dist("solved", "solved", as_int=True)
        if args.reuse_tt:
            _print_dist("tt_size", "tt_size", as_int=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
