import sys
from pathlib import Path
import pandas as pd
import chess
import chess.svg
import chess.engine
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# -----------------------------
# IMPORTS (FIXED)
# -----------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import *
from src.functions import get_best_move, eval_bar_horizontal
from src.cache import evaluate

# -----------------------------
# ENGINE (GLOBAL, SAFE)
# -----------------------------
engine = None


def get_engine():
    global engine
    if engine is None:
        engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))
    return engine


# -----------------------------
# PUZZLE GENERATION
# -----------------------------
def generate_chess_puzzles(result_df):
    puzzles = []

    for game_id, game in result_df.groupby("uuid"):

        game = game.sort_values("move_index").reset_index(drop=True)

        my_color = (
            "white"
            if game.iloc[0]["player"] == "white" and game.iloc[0]["my_move"]
            else "black"
        )

        for i in range(3, len(game)):

            if game.iloc[i]["blunder_type"] != "my_blunder":
                continue  # If I didnt blunder, move on

            # From here on, I row_curr is when I blundered.
            row_prev2 = game.iloc[i - 2]
            row_prev1 = game.iloc[i - 1]
            row_curr = game.iloc[i]

            blunder_type = (
                "opponent_blunder"
                if row_prev1["blunder_type"]
                == "opponent_blunder"  # CASE B: Opponent blundered and I missed it
                else "my_blunder"  # CASE A: I just blundered
            )

            # Fen AFTER I played 2 moves ago
            fen_prev2 = row_prev2["fen"]

            # The board after I played 2 moves ago, before the opponent played 1 move ago
            board_prev2 = chess.Board(fen_prev2)

            # What the opponent played before
            san_prev1 = row_prev1["move"]

            # The board after the opponent played 1 move ago (just before I blundered now)
            board_prev1 = board_prev2.parse_san(san_prev1)

            # The uci of the opponent's move right before I blundered
            uci_prev1 = board_prev1.uci()

            uci_curr = (chess.Board(row_prev1["fen"])).parse_san(row_curr["move"]).uci()

            clock_before = row_curr["clock_before"]

            clock_after = row_curr["clock_after"]

            thinking_time = row_curr["thinking_time"]

            puzzles.append(
                {
                    "uuid": game_id,
                    "link_id": row_curr["link_id"],
                    "type": blunder_type,
                    "my_color": my_color,
                    "fen": row_prev1["fen"],
                    "move_before": uci_prev1,
                    "move_curr": uci_curr,
                    "played_move": row_curr["move"],
                    "eval_before": row_curr["eval_before"],
                    "eval_after": row_curr["eval_after"],
                    "clock_before": clock_before,
                    "clock_after": clock_after,
                    "thinking_time": thinking_time,
                }
            )

    return pd.DataFrame(puzzles)


# -----------------------------
# RENDERER
# -----------------------------
def render_puzzle(row_curr, depth=10):
    board = chess.Board(row_curr["fen"])
    puzzle_type = row_curr["type"]
    my_color = row_curr["my_color"]

    orientation = chess.WHITE if my_color == "white" else chess.BLACK

    try:
        move_obj = chess.Move.from_uci(row_curr["move_before"])
        new_move_obj = chess.Move.from_uci(row_curr["move_curr"])

        fill = {move_obj.from_square: "#fff176", move_obj.to_square: "#ffd54f"}
    except Exception as e:
        print(e)
        fill = {}

    # -----------------------------
    # TITLES
    # -----------------------------
    if puzzle_type == "my_blunder":
        title = (
            "❌ Your Blunder: "
            + row_curr["link_id"]
            + " in move "
            + row_curr["played_move"]
        )
        subtitle = f"You are {my_color.upper()}. You played {row_curr['played_move']}. Find a better move."

    elif puzzle_type == "opponent_blunder":
        title = (
            "🔥 Opponent Blunder: "
            + row_curr["link_id"]
            + " in move "
            + row_curr["played_move"]
        )
        subtitle = f"You are {my_color.upper()}. Opponent blundered. You played {row_curr['played_move']}. Find a stronger move."

    else:
        title = "Puzzle"
        subtitle = f"You are {my_color.upper()}."

    # -----------------------------
    # TOP BARS
    # -----------------------------

    if puzzle_type == "my_blunder":
        eval_before = row_curr["eval_before"]
        eval_after = row_curr["eval_after"]

        bar1 = eval_bar_horizontal(eval_before, "Now")
        bar2 = eval_bar_horizontal(
            eval_after, "You moved " + row_curr["played_move"] + " (blunder)"
        )

    elif puzzle_type == "opponent_blunder":

        # eval after opponent blunder = current position
        eval_after_blunder = row_curr["eval_before"]

        try:
            board_played = board.copy()
            board_played.push(board.parse_san(row_curr["played_move"]))
            eval_after_played = row_curr["eval_after"]
        except Exception:
            eval_after_played = 0

        bar1 = eval_bar_horizontal(eval_after_blunder, "Now")
        bar2 = eval_bar_horizontal(
            eval_after_played, "You moved " + row_curr["played_move"] + " (blunder)"
        )

    else:
        bar1 = ""
        bar2 = ""

    # -----------------------------
    # UI
    # -----------------------------
    move_input = widgets.Text(placeholder="Enter move in SAN")
    button = widgets.Button(description="Submit")
    output = widgets.Output()

    # -----------------------------
    # DISPLAY
    # -----------------------------
    board_svg = chess.svg.board(
        board=board,
        size=350,
        orientation=orientation,
        fill=fill,
        arrows=[
            chess.svg.Arrow(
                new_move_obj.from_square,
                new_move_obj.to_square,
                color="#ff000050",
            )
        ],
    )

    display(HTML(f"<h3>{title}</h3><p>{subtitle}</p>"))

    display(HTML(f"""
    <div style="display:flex;gap:40px;margin-bottom:15px;">
        {bar1}
        {bar2}
    </div>
    """))

    display(HTML(board_svg))
    display(move_input, button, output)

    # -----------------------------
    # BASELINE
    # -----------------------------
    engine = get_engine()
    best_move = get_best_move(board, engine)

    board_best = board.copy()
    board_best.push(best_move)
    eval_best = evaluate(board_best, engine, depth=depth)

    try:
        board_played = board.copy()
        board_played.push(board.parse_san(row_curr["played_move"]))
        eval_played = row_curr["eval_after"]
    except Exception:
        eval_played = None

    # -----------------------------
    # LOGIC
    # -----------------------------
    def on_submit(b):
        with output:
            clear_output()

            try:
                user_move = board.parse_san(move_input.value)
            except Exception:
                print("Invalid move.")
                return

            board_user = board.copy()
            board_user.push(user_move)

            eval_user = evaluate(board_user, engine, depth=depth)
            loss_vs_best = eval_best - eval_user

            print(f"Your move: {move_input.value}")
            print(f"Best move: {board.san(best_move)}")
            print(f"Loss vs best: {loss_vs_best:.1f} cp")

            if eval_played is not None:
                improvement = eval_user - eval_played
                print(f"Improvement vs your move: {improvement:.1f} cp")

                if improvement < -50:
                    print("❌ Worse than your move")
                elif loss_vs_best < 50:
                    print("✅ Excellent")
                elif improvement > 50:
                    print("⚠️ Improvement but not best")
                else:
                    print("➖ Similar to your move")

    button.on_click(on_submit)


def run_puzzle_viewer(result_df):
    puzzles_df = generate_chess_puzzles(result_df)
    if len(puzzles_df) == 0:
        print("No puzzles available.")
        return

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(puzzles_df) - 1,
        step=1,
        description="Puzzle:",
        continuous_update=False,
    )

    output = widgets.Output()

    def update(change):
        with output:
            clear_output(wait=True)
            row_curr = puzzles_df.iloc[slider.value]
            render_puzzle(row_curr)

    slider.observe(update, names="value")

    display(slider, output)

    # initial render
    with output:
        render_puzzle(puzzles_df.iloc[0])
