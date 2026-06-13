import chess.engine
import ipywidgets as widgets
from IPython.display import display, SVG, HTML, clear_output
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


def get_best_move(board, engine):
    """
    Get Stockfish's recommended best move for the current position.

    Uses depth 10 analysis to find the strongest move according to the engine.

    Parameters:
    -----------
    board : chess.Board
        Current chess board position
    engine : chess.engine.SimpleEngine
        Stockfish engine instance (already running)

    Returns:
    --------
    chess.Move or None
        Best move according to Stockfish, or None if analysis fails

    Example:
    --------
    >>> best = get_best_move(board, engine)
    >>> print(f"Stockfish suggests: {best}")
    """
    try:
        result = engine.analyse(board, chess.engine.Limit(depth=10))
        return result["pv"][0]
    except:
        return None


def eval_bar_html(eval_cp, max_cp=500):
    """
    Generate an HTML progress bar visualization of the evaluation.

    Creates a vertical bar showing advantage: White up (white bar) vs Black up (black bar).

    Parameters:
    -----------
    eval_cp : int
        Evaluation in centipawns (positive = White advantage, negative = Black advantage)
    max_cp : int, default=500
        Maximum centipawn value to display (capped at this value)

    Returns:
    --------
    str
        HTML string representing the evaluation bar

    Example:
    --------
    >>> html = eval_bar_html(150)  # White +1.5 advantage
    >>> display(HTML(html))
    """
    eval_cp = max(-max_cp, min(max_cp, eval_cp))
    white_pct = (eval_cp + max_cp) / (2 * max_cp) * 100
    black_pct = 100 - white_pct
    sign = "+" if eval_cp >= 0 else ""

    return f"""
    <div style="
        height: 320px;
        width: 55px;
        border-radius: 6px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        position: relative;
        box-shadow: 0 0 6px rgba(0,0,0,0.3);
        font-family: Arial;
    ">
        <div style="background: white; height: {white_pct}%;"></div>
        <div style="background: #222; height: {black_pct}%;"></div>

        <div style="
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 2px;
            background: red;
        "></div>

        <div style="
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 2px 6px;
            font-size: 12px;
            border-radius: 4px;
        ">
            {sign}{eval_cp}
        </div>
    </div>
    """


def precompute_boards(game_df):
    """
    Precompute all board states for a game.

    Takes a game's move list and rebuilds the board position after each move.
    Used for interactive visualization to quickly switch between moves.

    Parameters:
    -----------
    game_df : pd.DataFrame
        DataFrame containing moves for a single game, sorted by move_index

    Returns:
    --------
    list
        List of chess.Board objects, one after each move

    Example:
    --------
    >>> boards = precompute_boards(game_moves_df)
    >>> print(f"Game has {len(boards)} positions")
    """
    boards = []
    board = chess.Board()

    for _, row in game_df.sort_values("move_index").iterrows():

        try:
            move = board.parse_san(row["move"])
        except:
            print("op")
            continue

        board.push(move)
        boards.append(board.copy())

    return boards


def interactive_game_viewer(result_df, link_id):
    """
    Create an interactive board viewer for analyzing a specific game.

    Displays a chess board with:
    - Green arrow = Stockfish's best move
    - Blue arrow = Move actually played
    - Evaluation bar = Position advantage
    - Slider to navigate through moves

    Parameters:
    -----------
    result_df : pd.DataFrame
        DataFrame containing analysis results (must have 'uuid', 'move', 'eval_after', 'delta', 'move_type')
    game_id : str
        Unique identifier for the game (uuid column)

    Example:
    --------
    >>> interactive_game_viewer(analysis_df, "abc123-def456")
    # Opens interactive widget - use the slider to move through the game
    """
    game = result_df[result_df["link_id"] == link_id]
    boards = precompute_boards(game)
    print(len(boards))

    def update(i):
        clear_output(wait=True)

        board = boards[i]
        row = game.iloc[i]

        # --- BEST MOVE ---
        best_move = get_best_move(board, engine)

        # played move (from SAN)
        try:
            played_move = board.parse_san(row["move"])
        except:
            played_move = None

        # --- ARROWS ---
        arrows = []

        if best_move:
            arrows.append(
                chess.svg.Arrow(
                    best_move.from_square, best_move.to_square, color="green"
                )
            )

        if played_move:
            arrows.append(
                chess.svg.Arrow(
                    played_move.from_square, played_move.to_square, color="blue"
                )
            )

        board_svg = chess.svg.board(board=board, size=320, arrows=arrows)

        board_html = f"""
        <div style="position: relative; display: inline-block;">
            {board_svg}
            <div style="
                position: absolute;
                bottom: 6px;
                left: 8px;
                background: rgba(0,0,0,0.65);
                color: white;
                padding: 2px 6px;
                font-size: 13px;
                border-radius: 4px;
                font-family: Arial;
            ">
                Move {row['move_index']}
            </div>
        </div>
        """

        layout = widgets.HBox(
            [widgets.HTML(board_html), widgets.HTML(eval_bar_html(row["eval_after"]))]
        )

        display(layout)

        print(f"Move: {row['move']}")
        print(f"Eval: {row['eval_after']}")
        print(f"Delta: {row['delta']}")
        print(f"Type: {row['move_type']}")

    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))
    widgets.interact(
        update, i=widgets.IntSlider(min=0, max=len(boards) - 1, step=1, value=0)
    )
    engine.quit()


import sys
from pathlib import Path
import pandas as pd
import chess
import chess.svg
import chess.engine
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output


def eval_bar_horizontal(eval_cp, label=""):
    eval_cp = max(-2000, min(2000, eval_cp))
    x = np.tanh(eval_cp / 200)

    white_pct = (x + 1) / 2 * 100
    black_pct = 100 - white_pct

    return f"""
    <div style="width:250px;">
        <div style="font-size:12px;margin-bottom:4px;text-align:center;">{label}</div>
        <div style="display:flex;width:100%;height:20px;border:1px solid black;">
            <div style="width:{white_pct}%;background:white;"></div>
            <div style="width:{black_pct}%;background:black;"></div>
        </div>
        <div style="font-size:12px;margin-top:4px;text-align:center;">{eval_cp}</div>
    </div>
    """
