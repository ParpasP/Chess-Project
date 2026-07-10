"""
General game-level statistics: rating progression, time controls,
playing frequency, performance vs. opponent rating, and win-rate trend
over time. Complements `src/statistics.py` (which already covers
color-based results, accuracy, and daytime breakdowns).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

_SCORE_MAP = {"Win": 1.0, "Draw": 0.5, "Loss": 0.0}


def _with_my_rating(games_df: pd.DataFrame) -> pd.DataFrame:
    df = games_df.copy()
    df["my_rating"] = np.where(
        df["my_color"].str.lower() == "white", df["white_rating"], df["black_rating"]
    )
    df["opponent_rating"] = np.where(
        df["my_color"].str.lower() == "white", df["black_rating"], df["white_rating"]
    )
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    return df


def rating_progression(games_df: pd.DataFrame) -> pd.DataFrame:
    """My rating over time, per time_class, sorted chronologically."""
    df = _with_my_rating(games_df)
    return df.sort_values("Time")[["Time", "time_class", "my_rating", "uuid"]].reset_index(drop=True)


def time_control_distribution(games_df: pd.DataFrame) -> pd.Series:
    """Number of games played per time_class."""
    return games_df["time_class"].value_counts()


def playing_frequency(games_df: pd.DataFrame, freq: str = "W") -> pd.Series:
    """Number of games played per time period (default weekly)."""
    df = games_df.copy()
    df["Time"] = pd.to_datetime(df["Time"], utc=True).dt.tz_localize(None)
    return df.set_index("Time").resample(freq).size()


def performance_by_opponent_rating(games_df: pd.DataFrame, bin_size: int = 100) -> pd.DataFrame:
    """Win/draw/loss rate bucketed by rating_diff (my rating - opponent rating).

    Positive bins mean I was rated higher; negative means the opponent was.
    """
    df = _with_my_rating(games_df).copy()
    df["rating_advantage"] = df["my_rating"] - df["opponent_rating"]
    df["score"] = df["result"].map(_SCORE_MAP)

    max_abs = int(np.ceil(df["rating_advantage"].abs().max() / bin_size) * bin_size)
    edges = list(range(-max_abs, max_abs + bin_size, bin_size))
    df["bucket"] = pd.cut(df["rating_advantage"], bins=edges)

    summary = df.groupby("bucket", observed=True).agg(
        games=("score", "count"),
        win_pct=("score", lambda s: (s == 1.0).mean() * 100),
        avg_score=("score", "mean"),
    )
    return summary[summary["games"] > 0]


def win_rate_by_month(games_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly game count and win rate, for an 'improvement curve' view."""
    df = games_df.copy()
    df["Time"] = pd.to_datetime(df["Time"], utc=True).dt.tz_localize(None)
    df["month"] = df["Time"].dt.to_period("M").dt.to_timestamp()
    df["score"] = df["result"].map(_SCORE_MAP)

    return df.groupby("month").agg(
        games=("score", "count"),
        win_pct=("score", lambda s: (s == 1.0).mean() * 100),
        avg_score=("score", "mean"),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_rating_progression(games_df: pd.DataFrame):
    """Line chart of rating over time, one line per time_class."""
    df = rating_progression(games_df)
    fig, ax = plt.subplots(figsize=(11, 5))

    for tc, group in df.groupby("time_class"):
        ax.plot(group["Time"], group["my_rating"], marker=".", markersize=2, linewidth=1, label=tc)

    ax.set_title("Rating Progression Over Time")
    ax.set_ylabel("Rating")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_time_control_distribution(games_df: pd.DataFrame):
    dist = time_control_distribution(games_df)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(dist.index, dist.values, color="#4B8D6A")
    ax.set_title("Games by Time Control")
    ax.set_ylabel("Games")
    plt.tight_layout()
    plt.show()


def plot_playing_frequency(games_df: pd.DataFrame, freq: str = "W"):
    freq_series = playing_frequency(games_df, freq=freq)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(freq_series.index, freq_series.values, width=5 if freq == "W" else 20, color="#3A6EA5")
    ax.set_title(f"Playing Frequency ({'weekly' if freq=='W' else freq})")
    ax.set_ylabel("Games played")
    plt.tight_layout()
    plt.show()


def plot_performance_by_opponent_rating(games_df: pd.DataFrame, bin_size: int = 100):
    summary = performance_by_opponent_rating(games_df, bin_size=bin_size)
    fig, ax = plt.subplots(figsize=(11, 5))
    x_labels = [str(b) for b in summary.index]
    ax.bar(x_labels, summary["win_pct"], color="#4B8D6A")
    ax.axhline(50, color="red", linestyle="--", alpha=0.6)
    ax.set_title("Win Rate vs. Opponent Rating Difference")
    ax.set_xlabel("My rating - opponent rating")
    ax.set_ylabel("Win %")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_win_rate_trend(games_df: pd.DataFrame):
    """'Improvement curve': monthly win rate over time."""
    summary = win_rate_by_month(games_df)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(summary.index, summary["win_pct"], marker="o", color="#3A6EA5")
    ax.axhline(50, color="red", linestyle="--", alpha=0.6)
    ax.set_title("Win Rate Trend by Month")
    ax.set_ylabel("Win %")
    plt.tight_layout()
    plt.show()
