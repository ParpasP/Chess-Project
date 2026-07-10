"""
Opening repertoire analysis.

All functions take `games_df` (one row per game, as produced by
`src.download_save_load.process_and_save_chess_data`) and are pure/plotting
helpers — they don't mutate the input dataframe.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

_SCORE_MAP = {"Win": 1.0, "Draw": 0.5, "Loss": 0.0}


def _scored(games_df: pd.DataFrame, color: str | None) -> pd.DataFrame:
    """Return a copy of games_df (optionally filtered by my_color) with a
    numeric 'score' column (Win=1, Draw=0.5, Loss=0)."""
    df = games_df.copy()
    if color is not None:
        df = df[df["my_color"].str.lower() == color.lower()]
    df["score"] = df["result"].map(_SCORE_MAP)
    return df


def opening_repertoire(
    games_df: pd.DataFrame, color: str | None = None, min_games: int = 3
) -> pd.DataFrame:
    """Summarize performance per opening.

    Parameters
    ----------
    games_df : pd.DataFrame
        Games dataframe with 'simple_opening', 'my_color', 'result', 'Time'.
    color : {'White', 'Black'}, optional
        Restrict to games played as this color. If None, uses all games.
    min_games : int, default 3
        Minimum number of games required for an opening to be included.

    Returns
    -------
    pd.DataFrame indexed by simple_opening with columns:
        games, wins, draws, losses, win_pct, draw_pct, loss_pct,
        avg_score, first_played, last_played
    """
    df = _scored(games_df, color)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "games", "wins", "draws", "losses",
                "win_pct", "draw_pct", "loss_pct",
                "avg_score", "first_played", "last_played",
            ]
        )

    grouped = df.groupby("simple_opening")
    summary = grouped.agg(
        games=("score", "count"),
        wins=("result", lambda s: (s == "Win").sum()),
        draws=("result", lambda s: (s == "Draw").sum()),
        losses=("result", lambda s: (s == "Loss").sum()),
        avg_score=("score", "mean"),
        first_played=("Time", "min"),
        last_played=("Time", "max"),
    )

    summary["win_pct"] = summary["wins"] / summary["games"] * 100
    summary["draw_pct"] = summary["draws"] / summary["games"] * 100
    summary["loss_pct"] = summary["losses"] / summary["games"] * 100

    summary = summary[summary["games"] >= min_games]
    return summary.sort_values("games", ascending=False)[
        [
            "games", "wins", "draws", "losses",
            "win_pct", "draw_pct", "loss_pct",
            "avg_score", "first_played", "last_played",
        ]
    ]


def opening_diversity_index(games_df: pd.DataFrame, color: str | None = None) -> dict:
    """Shannon-entropy-based measure of repertoire variety.

    Returns
    -------
    dict with keys:
        entropy : raw Shannon entropy (bits) of the simple_opening distribution
        normalized_entropy : entropy / max possible entropy, in [0, 1].
            0 = always plays the same opening, 1 = perfectly uniform spread.
        effective_repertoire : exp(entropy), i.e. the "effective number" of
            distinct openings played (a single easy-to-read number).
        n_distinct_openings : count of unique simple_opening values.
    """
    df = games_df if color is None else games_df[
        games_df["my_color"].str.lower() == color.lower()
    ]
    counts = df["simple_opening"].value_counts()

    if counts.empty:
        return {
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "effective_repertoire": 0.0,
            "n_distinct_openings": 0,
        }

    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    n = len(counts)
    max_entropy = np.log(n) if n > 1 else 1.0

    return {
        "entropy": entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
        "effective_repertoire": float(np.exp(entropy)),
        "n_distinct_openings": int(n),
    }


def best_worst_openings(
    games_df: pd.DataFrame,
    color: str | None = None,
    min_games: int = 5,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (best, worst) opening dataframes sorted by win_pct.

    Both are subsets of `opening_repertoire(...)`, filtered to openings with
    at least `min_games` games, truncated to `top_n` rows.
    """
    summary = opening_repertoire(games_df, color=color, min_games=min_games)
    best = summary.sort_values("win_pct", ascending=False).head(top_n)
    worst = summary.sort_values("win_pct", ascending=True).head(top_n)
    return best, worst


def opening_trend_over_time(
    games_df: pd.DataFrame, top_n: int = 6, freq: str = "ME"
) -> pd.DataFrame:
    """Usage count of the top_n most-played openings over time.

    Returns
    -------
    pd.DataFrame, index = time period (per `freq`), columns = opening names,
    values = number of games played that period. Missing combinations are 0.
    """
    df = games_df.copy()
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    top_openings = df["simple_opening"].value_counts().head(top_n).index

    df = df[df["simple_opening"].isin(top_openings)]
    df["period"] = (
        df["Time"].dt.tz_localize(None).dt.to_period(freq.replace("ME", "M")).dt.to_timestamp()
    )

    pivot = (
        df.groupby(["period", "simple_opening"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=top_openings, fill_value=0)
    )
    return pivot


def eco_family_distribution(games_df: pd.DataFrame, color: str | None = None) -> pd.DataFrame:
    """Distribution of games across the 5 ECO families (A-E).

    Requires the 'eco_code' column (e.g. 'C30'); games with a missing/invalid
    code are excluded. ECO families broadly correspond to:
        A: Flank openings, B: Semi-open (mostly Sicilian), C: Open games
           & French, D: Closed/Queen's-pawn games, E: Indian defenses.
    """
    df = games_df if color is None else games_df[
        games_df["my_color"].str.lower() == color.lower()
    ]
    codes = df["eco_code"].dropna().astype(str)
    families = codes.str[0].where(codes.str[0].str.match(r"[A-E]"))
    counts = families.value_counts().reindex(list("ABCDE"), fill_value=0)
    result = counts.to_frame("games")
    result["pct"] = result["games"] / result["games"].sum() * 100 if result["games"].sum() else 0
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_opening_repertoire(games_df: pd.DataFrame, color: str | None = None, top_n: int = 12):
    """Horizontal bar chart of the most-played openings (games count)."""
    summary = opening_repertoire(games_df, color=color, min_games=1)
    top = summary.sort_values("games", ascending=False).head(top_n)

    if top.empty:
        print("No opening data to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(top))))
    ax.barh(top.index, top["games"], color="#4B8D6A")
    ax.invert_yaxis()
    title = "Opening Repertoire" + (f" ({color})" if color else "")
    ax.set_title(title)
    ax.set_xlabel("Games played")
    plt.tight_layout()
    plt.show()


def plot_win_rate_by_opening(
    games_df: pd.DataFrame, color: str | None = None, min_games: int = 5, top_n: int = 15
):
    """Stacked bar chart of win/draw/loss % for the most-played openings."""
    summary = opening_repertoire(games_df, color=color, min_games=min_games)
    top = summary.sort_values("games", ascending=False).head(top_n)

    if top.empty:
        print(f"No openings with >= {min_games} games to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(top))))
    y = np.arange(len(top))
    ax.barh(y, top["win_pct"], color="#4B8D6A", label="Win %")
    ax.barh(y, top["draw_pct"], left=top["win_pct"], color="#778346", label="Draw %")
    ax.barh(
        y,
        top["loss_pct"],
        left=top["win_pct"] + top["draw_pct"],
        color="#8A4141",
        label="Loss %",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(top.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of games")
    title = "Win Rate by Opening" + (f" ({color})" if color else "")
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()


def plot_opening_trend(games_df: pd.DataFrame, top_n: int = 6, freq: str = "ME"):
    """Line chart of how often the top openings are played over time."""
    pivot = opening_trend_over_time(games_df, top_n=top_n, freq=freq)

    if pivot.empty:
        print("No opening trend data to plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", markersize=3, label=col)

    ax.set_title("Opening Usage Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Games played")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_eco_family_distribution(games_df: pd.DataFrame, color: str | None = None):
    """Bar chart of games per ECO family (A-E)."""
    dist = eco_family_distribution(games_df, color=color)

    if dist["games"].sum() == 0:
        print("No ECO code data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dist.index, dist["games"], color="#3A6EA5")
    for i, (games, pct) in enumerate(zip(dist["games"], dist["pct"])):
        ax.text(i, games + 0.5, f"{pct:.0f}%", ha="center", fontsize=9)
    title = "Games by ECO Family" + (f" ({color})" if color else "")
    ax.set_title(title)
    ax.set_xlabel("ECO family")
    ax.set_ylabel("Games")
    plt.tight_layout()
    plt.show()
