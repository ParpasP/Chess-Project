from matplotlib.patches import Patch
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from config import *
from IPython.display import display


def plot_game(result_df, link_id):
    """
    Plot the evaluation graph for a single game.

    Creates a line chart showing how the engine evaluation changed throughout the game.
    Positive values = White advantage, Negative values = Black advantage.

    Parameters:
    -----------
    result_df : pd.DataFrame
        DataFrame containing analysis results with 'move_index', 'eval_after', 'link_id' columns
    link_id : str or int
        Unique identifier for the game (from the game's URL)

    Example:
    --------
    >>> plot_game(analysis_df, "12345678")
    """
    link_id = str(link_id)
    game = result_df[result_df["link_id"] == link_id]
    my_color = "white" if game.iloc[0]["my_move"] else "black"
    print(f"You played as {my_color}")
    x = game["move_index"].values
    y = game["eval_after"].values

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x, y, color="white", linewidth=0.01, zorder=3)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-1000, 1000)

    ax.fill_between(x, y, 1000, color="black", alpha=0.8, zorder=1)

    # optional reference
    ax.axhline(0, color="grey", linewidth=1, zorder=4)

    ax.set_title("Game review")
    ax.set_xlabel("Move")
    ax.set_ylabel("Evaluation")

    plt.show()


def plot_game_results(games_df, type=None):
    if type is not None:
        games_df = games_df[games_df["time_class"] == type]

    table = pd.crosstab(games_df["result"], games_df["my_color"])

    table = table.reindex(index=["Win", "Loss", "Draw"], fill_value=0)
    table = table.reindex(columns=["White", "Black"], fill_value=0)

    # =========================================================
    # COLORS
    # =========================================================
    color_map = {"White": "#949393", "Black": "#3A3A3A"}

    base_colors = {"Win": "#4B8D6A", "Loss": "#8A4141", "Draw": "#778346"}

    color_variants = {
        "Win": {"White": "#949393", "Black": "#3A3A3A"},
        "Loss": {"White": "#949393", "Black": "#3A3A3A"},
        "Draw": {"White": "#949393", "Black": "#3A3A3A"},
    }

    # =========================================================
    # CREATE SUBPLOTS
    # =========================================================
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # =========================================================
    # 4. STACKED BAR
    # =========================================================
    table.plot(
        kind="bar", stacked=True, ax=axs[0], color=[color_map[c] for c in table.columns]
    )

    axs[0].set_title("Results Split by Outcome and Color")
    axs[0].set_xlabel("Result Type")
    axs[0].set_ylabel("Number of Games")
    axs[0].tick_params(axis="x", rotation=0)
    axs[0].legend(title="Played as")

    # =========================================================
    # 5. DONUT CHART DATA
    # =========================================================
    outer_sizes = table.sum(axis=1)

    inner_sizes = [
        table.loc[result, color] for result in table.index for color in table.columns
    ]

    outer_colors = [base_colors[r] for r in table.index]

    inner_colors = [
        color_variants[result][color]
        for result in table.index
        for color in table.columns
    ]

    outer_labels = [int(v) for v in outer_sizes]
    inner_labels = [int(table.loc[r, c]) for r in table.index for c in table.columns]

    # =========================================================
    # 6. DONUT CHART
    # =========================================================
    size = 0.3

    wedges_outer, _ = axs[1].pie(
        outer_sizes,
        radius=1,
        labels=outer_labels,
        labeldistance=0.85,
        colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor="k"),
    )

    wedges_inner, _ = axs[1].pie(
        inner_sizes,
        radius=1 - size,
        labels=inner_labels,
        labeldistance=0.75,
        colors=inner_colors,
        wedgeprops=dict(width=size, edgecolor="k"),
    )

    # =========================================================
    # LEGENDS
    # =========================================================
    legend1 = axs[1].legend(
        wedges_outer,
        table.index,
        title="Result",
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )

    legend2 = axs[1].legend(
        handles=[
            Patch(facecolor="#949393", label="White"),
            Patch(facecolor="#3A3A3A", label="Black"),
        ],
        title="Color",
        loc="upper left",
        bbox_to_anchor=(1, 0.75),
    )

    axs[1].add_artist(legend1)

    axs[1].set(aspect="equal")
    axs[1].set_title("Performance Breakdown")

    # =========================================================
    # FINAL
    # =========================================================
    plt.tight_layout()
    plt.show()


def chi_square_analysis(games_df, type=None, alpha=0.05, plot=True):
    if type is not None:
        games_df = games_df[games_df["time_class"] == type]

    table = pd.crosstab(games_df["result"], games_df["my_color"])
    # --- Test ---
    chi2, p, dof, expected = chi2_contingency(table)

    expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)

    # --- Standardized residuals ---
    residuals = (table - expected_df) / np.sqrt(expected_df)

    # --- Compact summary dataframe ---
    summary_df = pd.DataFrame(
        {
            "Statistic": ["Chi-square", "p-value", "Degrees of freedom", "Sample size"],
            "Value": [round(chi2, 4), round(p, 6), dof, table.values.sum()],
        }
    )
    r, c = table.shape
    N = len(games_df)
    cramers_v = np.sqrt(chi2 / (N * min(r - 1, c - 1)))

    # --- Interpretation ---
    if p < alpha:
        if cramers_v < 0.1:
            strength = "negligible"
        elif cramers_v < 0.3:
            strength = "small"
        elif cramers_v < 0.5:
            strength = "moderate"
        else:
            strength = "strong"

        interpretation = (
            f"Statistically significant association (p = {p:.4f}). "
            f"Effect size is {strength} (Cramer's V = {cramers_v:.3f})."
        )
    else:
        interpretation = (
            f"No statistically significant association (p = {p:.4f}). "
            "No evidence of dependence between variables."
        )

    # --- Plot ---
    if plot:

        fig, ax = plt.subplots(figsize=(6, 4))

        im = ax.imshow(residuals, aspect="auto")

        # Labels
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns)

        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)

        # Annotate cells
        for i in range(residuals.shape[0]):
            for j in range(residuals.shape[1]):
                ax.text(j, i, f"{residuals.iloc[i, j]:.2f}", ha="center", va="center")

        plt.colorbar(im, ax=ax, label="Standardized residual")

        plt.title("Chi-square Residuals")
        plt.xlabel(table.columns.name)
        plt.ylabel(table.index.name)

        plt.tight_layout()
        plt.show()

    return summary_df, interpretation


def openings(games_df, type=None, limit=20):  # TODO: I have TODO something
    df = games_df.copy()
    if type is not None:
        df = games_df[games_df["time_class"] == type]  # BUG here

    score_map = {"Win": 1, "Draw": 0.5, "Loss": 0}
    df["result"] = games_df["result"].map(score_map)

    # --- group by opening ---
    opening_stats = (
        df.groupby(["my_color", "simple_opening"])
        .agg(games=("result", "count"), avg_score=("result", "mean"))
        .reset_index()
    )

    # --- filter small samples ---
    opening_stats = opening_stats[opening_stats["games"] >= limit]

    # --- top openings ---
    top_white = (
        opening_stats[opening_stats["my_color"] == "White"]
        .sort_values("avg_score", ascending=False)
        .head(10)
    )

    top_black = (
        opening_stats[opening_stats["my_color"] == "Black"]
        .sort_values("avg_score", ascending=False)
        .head(10)
    )

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(top_white["simple_opening"], top_white["avg_score"])
    axes[0].invert_yaxis()
    axes[0].set_title("Best Openings as White")
    axes[0].set_xlabel("Average Score")

    axes[1].barh(top_black["simple_opening"], top_black["avg_score"])
    axes[1].invert_yaxis()
    axes[1].set_title("Best Openings as Black")
    axes[1].set_xlabel("Average Score")

    plt.tight_layout()
    plt.show()


def accuracy(games_df, type=None):
    df = games_df.copy()
    if type is not None:
        df = games_df[games_df["time_class"] == type]

    df["my_accuracy"] = np.where(
        df["my_color"] == "White", df["white_accuracy"], df["black_accuracy"]
    )

    accuracy_stats = (
        df.groupby(["my_color", "result"])
        .agg(avg_accuracy=("my_accuracy", "mean"))
        .reset_index()
    )

    pivot_df = accuracy_stats.pivot(
        index="my_color", columns="result", values="avg_accuracy"
    )

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap=[[1, 1, 1]],  # white cells
        cbar=False,
        linewidths=1,
        linecolor="black",
    )

    plt.title("Average Accuracy")
    plt.show()


def daytime(games_df, type=None):
    df = games_df.copy()
    if type is not None:
        df = games_df[games_df["time_class"] == type]

    df["hour"] = df["Time"].dt.round("h").dt.hour

    daytime_stats = (
        df.groupby(["hour", "result"]).agg(games=("result", "count")).reset_index()
    )

    sns.lineplot(data=daytime_stats, x="hour", y="games", hue="result", marker="o")

    plt.xticks(range(24))
    plt.title("Games by Hour and Result")
    plt.ylabel("Number of Games")
    plt.show()


def blunders(result_df):
    df = result_df[result_df["blunder_type"] == "my_blunder"]
    x = np.linspace(df["eval_before"].min(), df["eval_before"].max(), 200)
    plt.scatter(df["eval_before"], df["eval_after"], color="r")
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)
    plt.axhline(0, color="black")  # x-axis line
    plt.axvline(0, color="black")  # y-axis line
    plt.plot(x, x - MISTAKE_LL, color="black")
    plt.plot(x, x + MISTAKE_LL, color="black")
    plt.plot(x, x, color="black")
    plt.xlabel("Eval_before")
    plt.ylabel("Eval_after")

    plt.show()


def first_blunder_by_opening(result_df, games_df):

    # -------------------------------------------------
    # Keep only games that were actually analyzed
    # -------------------------------------------------
    valid_games = games_df[games_df["uuid"].isin(result_df["uuid"])].copy()

    # -------------------------------------------------
    # Keep only my blunders
    # -------------------------------------------------
    blunders = result_df[
        result_df["blunder_type"].astype(str).str.strip() == "my_blunder"
    ].copy()

    # Ensure numeric move numbers
    blunders["move_no"] = pd.to_numeric(blunders["move_no"], errors="coerce")

    # -------------------------------------------------
    # First blunder move for each game
    # -------------------------------------------------
    first_blunder = (
        blunders.groupby("uuid")["move_no"].min().reset_index(name="first_blunder_move")
    )

    # -------------------------------------------------
    # Merge with ALL analyzed games
    # -------------------------------------------------
    all_games = valid_games[["uuid", "simple_opening"]].copy()

    merged = all_games.merge(first_blunder, on="uuid", how="left")
    # -------------------------------------------------
    # Opening statistics
    # -------------------------------------------------
    opening_stats = (
        merged.groupby("simple_opening")
        .agg(
            games=("uuid", "count"),
            blunder_games=("first_blunder_move", "count"),
            avg_first_blunder=("first_blunder_move", "mean"),
        )
        .reset_index()
    )
    # -------------------------------------------------
    # Percentage of games without blunders
    # -------------------------------------------------
    opening_stats["no_blunder_pct"] = (
        1 - opening_stats["blunder_games"] / opening_stats["games"]
    ) * 100

    # -------------------------------------------------
    # Keep only openings with enough games
    # -------------------------------------------------
    opening_stats = opening_stats[opening_stats["games"] > 10]

    # -------------------------------------------------
    # Print statistics
    # -------------------------------------------------
    print(opening_stats.sort_values(by="avg_first_blunder", ascending=False))

    # -------------------------------------------------
    # Filter plotting dataframe
    # -------------------------------------------------
    valid_openings = opening_stats["simple_opening"]

    plot_df = merged[merged["simple_opening"].isin(valid_openings)]

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    plt.figure(figsize=(10, 8))

    sns.boxplot(data=plot_df, x="first_blunder_move", y="simple_opening")

    plt.xlabel("Move Number of First Blunder")
    plt.ylabel("Opening")
    plt.title("First Blunder Move by Opening")

    plt.show()
