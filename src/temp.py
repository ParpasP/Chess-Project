import chess
import pandas as pd


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

            if game.iloc[i]["blunder_type"] == "my_blunder":
                continue  # If I didnt blunder, move on

            row_prev2 = game.iloc[i - 2]
            row_prev1 = game.iloc[i - 1]
            row_curr = game.iloc[i]  # ALWAYS MY move - I blundered

            blunder_type = (
                "opponent_blunder"
                if row_prev1["blunder_type"] == "opponent_blunder"  # CASE B
                else "my_blunder"  # CASE A
            )

            fen_prev2 = row_prev2["fen"]  # fen AFTER I played 2 moves ago
            san_prev1 = row_curr["move"]  # what the opponent played before

            # The board after I played 2 moves ago, before the opponent played 1 move ago
            board_prev2 = chess.Board(fen_prev2)

            # The board after the opponent played 1 move ago (just before I blundered now)
            board_prev1 = board_prev2.parse_san(san_prev1)

            # The uci of the opponent's move right before I blundered
            uci_prev1 = board_prev1.uci()

            puzzles.append(
                {
                    "uuid": game_id,
                    "link_id": row_curr["link_id"],
                    "type": blunder_type,
                    "my_color": my_color,
                    "current_fen": row_curr["fen"],
                    "move_before": uci_prev1,
                    "played_move": row_curr["move"],
                    "eval_before": row_curr["eval_before"],
                    "eval_after": row_curr["eval_after"],
                }
            )
    return pd.DataFrame(puzzles)
