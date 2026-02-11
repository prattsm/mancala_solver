# GamePigeon Mancala (Capture Mode) Solver

CLI and desktop GUI that recommend the best move for GamePigeon Mancala in Capture mode (Kalah rules). It assumes optimal play from both sides and searches to terminal states with minimax + alpha-beta.

## Setup
- Python 3.9+ recommended
- CLI: no external dependencies
- GUI: `pip install pyside6`

## Run (CLI)
```bash
python3 cli.py
```

Optional flags:
- `--seeds N` starting seeds per pit (default 4)
- `--topn K` number of top moves to list (default 3)
- `--explain` show evals for top moves

## Run (GUI)
```bash
python3 mancala_gui.py
```

## Tests
```bash
python3 -m unittest discover -s tests
```

## User Guide (CLI)
- At startup, choose whether you go first.
- On your turn, the tool prints a recommended move (pit 1 is closest to your store) and optional top moves.
- Enter a pit number `1-6` to play, or press Enter to accept the recommendation.
- On opponent turns, enter the pit number they played.
- Commands: `u` undo, `h` help, `q` quit.

## User Guide (GUI)
- Pick turn order and seed count, then click `Reset` to start a new game.
- If it is your turn, click a pit on your side or press `Play Best`.
- If it is the opponent’s turn, click the pit they played on their side.
- Use `Undo` to revert one move. Enable auto-play to let the solver play your turns.
- Use the animation controls to adjust speed or replay the last move.
- While solving, the status line shows iterative-deepening progress with a provisional best move.
- When the solver fully proves the position, the status line shows `Solved (perfect)`.
- GUI continues timed deepening passes in the background until a perfect result is proven.

## Notes
- Board display shows opponent pits left-to-right as `1..6` and your pits left-to-right as `6..1`.
- Evaluation is `your_store - opponent_store` at terminal.
- GUI search uses iterative deepening with a time budget and reports the best completed depth so far.
- The solver cache is stored at `~/.mancala_cache.pkl.gz` and reused across sessions.
