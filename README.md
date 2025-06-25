# Automated Flow Solver

*A modular Python automation tool for solving Flow Free puzzles using computer vision, SAT solving, and GUI automation.*

## Features

- Detects and captures puzzle boards from a specified screen region.
- Converts puzzle states into Boolean SAT representations.
- Solves puzzles using constraint-based SAT solving.
- Simulates user interaction to draw solutions in-game.
- Handles edge cases like ads, menus, and unexpected screens.
- Supports graceful manual termination via the ESC key, with performance metrics.

## Requirements

- Python 3.9+
- macOS (Sequoia or newer; required for iPhone screen mirroring)
- Dependencies:
  - `opencv-python`, `numpy`, `pyautogui`, `pynput`, `pycosat`, `Pillow`

## How It Works

1. **Image Capture**: Captures a region of the screen using `pyautogui`.
2. **Parsing**: Extracts grid lines and circle positions with OpenCV.
3. **Solving**: Encodes the puzzle as SAT clauses and solves via `pycosat`.
4. **Simulation**: Automatically draws the solved paths using GUI control.

## Usage

1. Launch Flow Free and open a level.
2. Adjust puzzle window coordinates if needed (`topleft` and `bottomright`).
3. Run the script:
   ```bash
   python dailypuzzlesolver.py
5. Let it solve continuously — press `ESC` to gracefully exit.

## Notes

- Intended for educational and personal use only.
- May require tweaking if the game UI or color scheme changes.
- Optimized for speed, resilience, and minimal manual oversight.

## Additional Files

| File                   | Description |
|------------------------|-------------|
| `flowsolver.py`        | Solves a single Flow Free puzzle from an image input. |
| `pixel_grid_extraction.py` | Extracts grid and circle coordinates from a screenshot. |
| `solver_functions.py`  | Core SAT-based solver logic and helpers. |
| `autosolver_functions.py` | Automates game interaction and GUI simulation. |

## Inspiration

Created as a modular challenge project to explore constraint solving with SAT, real-time automation, and reverse-engineering of visual puzzle logic.
