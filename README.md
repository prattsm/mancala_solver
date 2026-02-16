# GamePigeon Mancala (Capture Mode) Solver

CLI and desktop GUI that recommend the best move for GamePigeon Mancala in Capture mode (Kalah rules). It assumes optimal play from both sides and uses minimax + alpha-beta with iterative deepening.

## Setup
- Python 3.12+ recommended
- CLI: no external dependencies
- GUI (recommended in a virtualenv; avoids PEP 668/system package conflicts):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pyside6
```

## Run (CLI)
```bash
python3 cli.py
```

Optional flags:
- `--seeds N` starting seeds per pit (default 4)
- `--topn K` number of top moves to list (default 3)
- `--explain` show evals for top moves
- `--time-ms N` per-turn search budget in milliseconds (default 300)
- `--perfect` search to terminal (ignores `--time-ms`)
- `--telemetry host:port` stream solver telemetry to sidecar visualizer

## Run (GUI)
```bash
python3 mancala_gui.py
```

Optional sidecar telemetry stream for GUI solves:
```bash
MANCALA_TELEMETRY=127.0.0.1:8765 python3 mancala_gui.py
```

## Run (Visualizer)
```bash
python3 mancala_visualizer.py --listen 127.0.0.1:8765
```

Demo mode (visualizer drives the solver itself):
```bash
python3 mancala_visualizer.py --demo
```

## Run (Benchmark)
```bash
python3 bench_solver.py --positions 20 --time-ms 300
```

Optional benchmark flags:
- `--perfect` run full-depth solves
- `--depth N` fixed-depth search mode (more deterministic than time slicing)
- `--reuse-tt` reuse one TT across positions
- `--seed N` deterministic position generation seed
- `--max-plies N` max random plies from start state
- `--repeat N` repeat benchmark runs and print p50/p95 summaries
- `--warmup N` warmup solves per repeat (not counted in summary)
- `--no-gc` disable garbage collection during measured loop
- `--save-positions FILE` save sampled positions as stable state keys
- `--load-positions FILE` benchmark a saved dataset (reproducible across code changes)

## Tests
```bash
python3 -m unittest discover -s tests
```

GUI integration tests use PySide6 and run headless with:
```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests
```

## User Guide (CLI)
- At startup, choose whether you go first.
- On your turn, the tool prints a recommended move (pit 1 is closest to your store) and optional top moves.
- Enter a pit number `1-6` to play, or press Enter to accept the recommendation.
- On opponent turns, enter the pit number they played. Pressing Enter on opponent turns re-prompts.
- Commands: `u` undo, `h` help, `q` quit.
- With `--explain`, CLI prints search depth, `complete` flag, elapsed milliseconds, and visited nodes for each recommendation.

## User Guide (GUI)
- Pick turn order and seed count, then click `Reset` to start a new game.
- If it is your turn, click a pit on your side or press `Play Best`.
- If it is the opponent’s turn, click the pit they played on their side.
- Use `Undo` to revert one move. Enable auto-play to let the solver play your turns.
- Use the animation controls to adjust speed or replay the last move.
- Window size, animation panel state, animation speed, and pit-number visibility persist across sessions.
- While solving, the status line shows iterative-deepening progress with a provisional best move.
- When the solver fully proves the position, the status line shows `Solved (perfect)`.
- GUI continues timed deepening passes in the background until a perfect result is proven.

## Notes
- Board display shows opponent pits left-to-right as `1..6` and your pits left-to-right as `6..1`.
- Evaluation is `your_store - opponent_store` at terminal.
- CLI and GUI both use timed iterative deepening by default and report best-so-far until a position is proven perfect.
- GUI search uses iterative deepening with a time budget and reports the best completed depth so far.
- The solver cache is stored at `~/.mancala_cache.pkl.gz` and reused across sessions.
