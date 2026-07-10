"""
Tactical / engine-based analysis: blunders, mistakes, accuracy trends,
brilliant-move detection, and "where games are lost".

Functions operate on `result_df` (the per-move output of
`src.cache.run_chess_analysis`) and/or `games_df`. Columns expected on
result_df: uuid, link_id, move_index, move_no, player, move, my_move,
eval_before, eval_after (White POV, centipawns), delta (mover POV), blunder,
blunder_type, move_type, fen, end_time.
"""

import sys
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}

_MOVE_TYPE_ORDER = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]


def _mover_eval(row: pd.Series, col: str) -> float:
    """Convert a White-POV eval column to the mover's own perspective."""
    value = row[col]
    return value if row["player"] == "white" else -value


def move_quality_distribution(result_df: pd.DataFrame, my_moves_only: bool = True) -> pd.Series:
    """Counts of each move_type (Best/Excellent/.../Blunder).

    Parameters
    ----------
    result_df : pd.DataFrame
    my_moves_only : bool, default True
        If True, only consider moves played by the analyzed player.
    """
    df = result_df[result_df["my_move"]] if my_moves_only else result_df
    counts = df["move_type"].value_counts()
    return counts.reindex(_MOVE_TYPE_ORDER, fill_value=0)


def per_game_move_quality(result_df: pd.DataFrame) -> pd.DataFrame:
    """Per-game counts of my blunders/mistakes/inaccuracies.

    Returns one row per uuid with columns: n_moves, blunders, mistakes,
    inaccuracies, blunder_rate (per my move).
    """
    mine = result_df[result_df["my_move"]]

    def _count(series, value):
        return (series == value).sum()

    summary = mine.groupby("uuid").agg(
        n_moves=("move_type", "count"),
        blunders=("move_type", lambda s: _count(s, "Blunder")),
        mistakes=("move_type", lambda s: _count(s, "Mistake")),
        inaccuracies=("move_type", lambda s: _count(s, "Inaccuracy")),
    )
    summary["blunder_rate"] = summary["blunders"] / summary["n_moves"]
    return summary


def accuracy_trend(games_df: pd.DataFrame, rolling_window: int = 20) -> pd.DataFrame:
    """Rolling-average accuracy over time (chess.com's own accuracy metric).

    Only games where chess.com computed an accuracy are included (it isn't
    available for every game, especially fast bullet games). Returns a
    dataframe sorted by Time with columns: Time, my_accuracy,
    rolling_accuracy.
    """
    df = games_df.copy()
    df["my_accuracy"] = np.where(
        df["my_color"].str.lower() == "white", df["white_accuracy"], df["black_accuracy"]
    )
    df = df.dropna(subset=["my_accuracy"]).copy()
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    df = df.sort_values("Time")
    df["rolling_accuracy"] = (
        df["my_accuracy"].rolling(window=rolling_window, min_periods=max(3, rolling_window // 4)).mean()
    )
    return df[["Time", "uuid", "my_accuracy", "rolling_accuracy"]].reset_index(drop=True)


def _phase_of_move(move_no) -> str:
    move_no = int(move_no)
    if move_no <= 10:
        return "Opening"
    elif move_no <= 30:
        return "Middlegame"
    return "Endgame"


def eval_loss_by_phase(result_df: pd.DataFrame, my_moves_only: bool = True) -> pd.DataFrame:
    """Average centipawn loss and blunder rate, bucketed by game phase.

    Phase is defined by move number: Opening (moves 1-10), Middlegame
    (11-30), Endgame (31+). This is a simple heuristic, not a true
    material-based phase detector.
    """
    df = result_df[result_df["my_move"]] if my_moves_only else result_df
    df = df.copy()
    df["phase"] = df["move_no"].apply(_phase_of_move)
    # delta is already mover-perspective; a move that loses ground has delta < 0
    df["cp_loss"] = (-df["delta"]).clip(lower=0)

    summary = df.groupby("phase").agg(
        moves=("cp_loss", "count"),
        avg_cp_loss=("cp_loss", "mean"),
        blunder_rate=("blunder", "mean"),
    )
    return summary.reindex(["Opening", "Middlegame", "Endgame"])


def blunder_severity_distribution(result_df: pd.DataFrame, my_moves_only: bool = True) -> pd.Series:
    """Bucket blunders by how bad they were (centipawn loss magnitude).

    Bins: 'Blunder' (300-500cp), 'Serious' (500-1200cp), 'Game-losing' (1200+cp).
    """
    df = result_df[result_df["my_move"]] if my_moves_only else result_df
    blunders = df[df["blunder"].astype(bool)].copy()
    if blunders.empty:
        return pd.Series(
            {"Blunder (300-500cp)": 0, "Serious (500-1200cp)": 0, "Game-losing (1200cp+)": 0}
        )

    loss = -blunders["delta"]
    bins = [-np.inf, 500, 1200, np.inf]
    labels = ["Blunder (300-500cp)", "Serious (500-1200cp)", "Game-losing (1200cp+)"]
    return pd.cut(loss, bins=bins, labels=labels).value_counts().reindex(labels)


def game_state_when_blundered(result_df: pd.DataFrame) -> pd.Series:
    """Where (in terms of position evaluation) my blunders tend to happen.

    Buckets eval_before (from my own perspective, i.e. before the blunder)
    into: Losing big / Losing / Equal / Winning / Winning big.
    """
    mine = result_df[result_df["my_move"] & result_df["blunder"].astype(bool)].copy()
    if mine.empty:
        return pd.Series(dtype=int)

    mine["my_eval_before"] = mine.apply(lambda r: _mover_eval(r, "eval_before"), axis=1)
    bins = [-np.inf, -500, -100, 100, 500, np.inf]
    labels = ["Losing big", "Losing", "Equal", "Winning", "Winning big"]
    return pd.cut(mine["my_eval_before"], bins=bins, labels=labels).value_counts().reindex(labels)


def _format_eval(value: int) -> str:
    """Human-readable eval: 'Mate' for the +-10000 sentinel, else pawns."""
    if value >= 10000:
        return "Mate (for White)"
    if value <= -10000:
        return "Mate (for Black)"
    return f"{value / 100:+.2f}"


def worst_mistakes(result_df: pd.DataFrame, games_df: pd.DataFrame | None = None, n: int = 10) -> pd.DataFrame:
    """The N single moves with the biggest centipawn loss (my moves only).

    Note: eval_before/eval_after use +-10000 as a "forced mate" sentinel
    (see `src.cache.evaluate`), so a move that throws away a forced mate
    shows as a very large cp_loss — `eval_before_display`/`eval_after_display`
    render these as "Mate" instead of a literal centipawn number.
    """
    mine = result_df[result_df["my_move"]].copy()
    mine["cp_loss"] = -mine["delta"]
    worst = mine.sort_values("cp_loss", ascending=False).head(n).copy()

    worst["eval_before_display"] = worst["eval_before"].apply(_format_eval)
    worst["eval_after_display"] = worst["eval_after"].apply(_format_eval)

    cols = [
        "uuid", "link_id", "move_no", "player", "move",
        "eval_before_display", "eval_after_display", "cp_loss", "move_type",
    ]
    worst = worst[cols]

    if games_df is not None:
        worst = worst.merge(
            games_df[["uuid", "simple_opening", "result", "time_class"]], on="uuid", how="left"
        )
    return worst.reset_index(drop=True)


def conversion_collapse(result_df: pd.DataFrame, games_df: pd.DataFrame, threshold: int = 300) -> dict:
    """Rate at which games with a clear advantage (>= threshold cp for me)
    were nonetheless not won.

    Returns dict with n_games_ahead, n_collapsed, collapse_rate.
    """
    df = result_df.copy()
    df["my_eval"] = df.apply(lambda r: _mover_eval(r, "eval_after"), axis=1)
    # my_move is True only on my own moves; the position eval after my move
    # reflects my advantage regardless of whose "turn" column - use all rows
    # per game and take max of "my-perspective" eval computed consistently.
    my_color_map = games_df.set_index("uuid")["my_color"].str.lower()
    df["my_color"] = df["uuid"].map(my_color_map)
    df["my_perspective_eval"] = np.where(
        df["my_color"] == "white", df["eval_after"], -df["eval_after"]
    )

    max_adv = df.groupby("uuid")["my_perspective_eval"].max()
    ahead_games = max_adv[max_adv >= threshold].index

    results = games_df.loc[games_df["uuid"].isin(ahead_games), "result"]
    n_ahead = len(ahead_games)
    n_collapsed = int((results != "Win").sum())

    return {
        "n_games_ahead": int(n_ahead),
        "n_collapsed": n_collapsed,
        "collapse_rate": n_collapsed / n_ahead if n_ahead else 0.0,
    }


# ---------------------------------------------------------------------------
# Brilliant move detection
# ---------------------------------------------------------------------------


def _piece_value_at(board: chess.Board, square: int) -> int:
    piece = board.piece_at(square)
    return PIECE_VALUES[piece.symbol().upper()] if piece else 0


def _detect_sacrifice_recaptures(
    result_df: pd.DataFrame,
    min_sacrifice: int = 2,
    move_types: tuple[str, ...] | None = None,
    eval_window: tuple[float, float] | None = None,
    require_sound: bool = False,
) -> pd.DataFrame:
    """Shared scan for 'a piece moves to a square and is immediately
    recaptured there, at a net material loss to the mover'.

    This is a cheap proxy for a real sacrifice: it only catches immediate
    (1-ply) same-square recapture patterns, not deeper combinations.

    Parameters
    ----------
    move_types : tuple of str, optional
        Restrict to my_move rows with move_type in this set.
    eval_window : (low, high), optional
        Restrict to rows where the mover's eval_before falls in (low, high].
    require_sound : bool, default False
        If True, also require the mover's advantage to not collapse after
        the sacrifice (used by find_brilliant_moves).
    """
    candidates = []

    for uuid, game in result_df.groupby("uuid"):
        game = game.sort_values("move_index").reset_index(drop=True)

        for i in range(len(game) - 1):
            row = game.iloc[i]
            if not row["my_move"]:
                continue
            if move_types is not None and row["move_type"] not in move_types:
                continue

            eval_before_mover = _mover_eval(row, "eval_before")
            if eval_window is not None:
                low, high = eval_window
                if not (low < eval_before_mover <= high):
                    continue

            prior_fen = game.iloc[i - 1]["fen"] if i > 0 else chess.Board().fen()
            if pd.isna(prior_fen):
                continue

            try:
                board = chess.Board(prior_fen)
                move = board.parse_san(row["move"])
            except Exception:
                continue

            moving_piece = board.piece_at(move.from_square)
            if moving_piece is None:
                continue
            moving_value = PIECE_VALUES[moving_piece.symbol().upper()]
            captured_value = _piece_value_at(board, move.to_square)

            board.push(move)

            next_row = game.iloc[i + 1]
            try:
                opp_move = board.parse_san(next_row["move"])
            except Exception:
                continue

            if opp_move.to_square != move.to_square:
                continue  # not an immediate recapture on the same square

            material_swing = moving_value - captured_value
            if material_swing < min_sacrifice:
                continue

            eval_after_mover = _mover_eval(next_row, "eval_after")
            if require_sound and eval_after_mover < eval_before_mover - 30:
                continue  # advantage collapsed after the "sacrifice" - not sound

            candidates.append(
                {
                    "uuid": uuid,
                    "link_id": row["link_id"],
                    "move_no": row["move_no"],
                    "player": row["player"],
                    "move": row["move"],
                    "material_swing": material_swing,
                    "eval_before": eval_before_mover,
                    "eval_after": eval_after_mover,
                    "fen_before": prior_fen,
                }
            )

    return pd.DataFrame(candidates)


def find_sacrifices(result_df: pd.DataFrame, min_sacrifice: int = 2) -> pd.DataFrame:
    """Detect all apparent material sacrifices (sound or not), regardless of
    move quality or resulting evaluation. See `_detect_sacrifice_recaptures`
    for the detection method and its limitations.
    """
    return _detect_sacrifice_recaptures(result_df, min_sacrifice=min_sacrifice)


def find_brilliant_moves(
    result_df: pd.DataFrame, min_sacrifice: int = 2, eval_window: tuple[float, float] = (0, 700)
) -> pd.DataFrame:
    """Heuristically detect 'brilliant' moves: strong (non-Best, non-blunder)
    moves that voluntarily give up material and are immediately recaptured,
    while the mover's advantage is maintained or grows.

    This mirrors chess.com-style 'brilliant' badges: engine considers the
    move very strong ('Excellent'/'Good', deliberately excluding 'Best'
    since brilliancies are about surprising sacrifices, not just following
    the engine's own top line), the mover isn't already crushing (advantage
    within `eval_window`), a real material sacrifice occurs (net material
    given up >= min_sacrifice pawns via a same-square capture/recapture),
    and the position doesn't get worse for the mover afterwards.

    Returns a dataframe of candidate moves with a `material_swing` column
    (pawns given up, net). This is a heuristic, not an authoritative brilliancy
    detector — it only catches simple 1-ply sacrifice/recapture patterns.
    """
    return _detect_sacrifice_recaptures(
        result_df,
        min_sacrifice=min_sacrifice,
        move_types=("Excellent", "Good"),
        eval_window=eval_window,
        require_sound=True,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_move_quality_distribution(result_df: pd.DataFrame, my_moves_only: bool = True):
    counts = move_quality_distribution(result_df, my_moves_only=my_moves_only)
    colors = ["#2E7D32", "#66BB6A", "#AED581", "#FFD54F", "#FF8A65", "#C62828"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Move Quality Distribution" + (" (my moves)" if my_moves_only else ""))
    ax.set_ylabel("Number of moves")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_accuracy_trend(games_df: pd.DataFrame, rolling_window: int = 20):
    trend = accuracy_trend(games_df, rolling_window=rolling_window)
    if trend.empty:
        print("No accuracy data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(trend["Time"], trend["my_accuracy"], s=8, alpha=0.25, color="grey", label="Per game")
    ax.plot(trend["Time"], trend["rolling_accuracy"], color="#3A6EA5", linewidth=2,
            label=f"{rolling_window}-game rolling average")
    ax.set_title("Accuracy Trend Over Time")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_eval_loss_by_phase(result_df: pd.DataFrame):
    summary = eval_loss_by_phase(result_df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(summary.index, summary["avg_cp_loss"], color="#8A4141")
    axes[0].set_title("Avg. Centipawn Loss by Phase")
    axes[0].set_ylabel("Avg. cp lost per move")

    axes[1].bar(summary.index, summary["blunder_rate"] * 100, color="#C62828")
    axes[1].set_title("Blunder Rate by Phase")
    axes[1].set_ylabel("% of moves that are blunders")

    plt.tight_layout()
    plt.show()


def plot_blunder_severity(result_df: pd.DataFrame):
    dist = blunder_severity_distribution(result_df)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(dist.index, dist.values, color=["#FF8A65", "#E64A19", "#B71C1C"])
    ax.set_title("Blunder Severity Distribution")
    ax.set_ylabel("Count")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()


def plot_game_state_when_blundered(result_df: pd.DataFrame):
    dist = game_state_when_blundered(result_df)
    if dist.empty:
        print("No blunders found to plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(dist.index, dist.values, color="#C62828")
    ax.set_title("Position When I Blundered")
    ax.set_xlabel("My position right before the blunder")
    ax.set_ylabel("Number of blunders")
    plt.tight_layout()
    plt.show()
