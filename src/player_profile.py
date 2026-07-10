"""
Player style / profile analysis: castling habits, queen trades, piece
activity, pawn structure, king safety, and a lightweight aggregate "style
profile" with simple game clustering.

These are exploratory, heuristic features computed directly from available
data (SAN move text and FENs) — not a validated psychological model of
playing style. Treat the style profile as a fun summary, not a scientific
instrument.
"""

import sys
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}


# ---------------------------------------------------------------------------
# Move-text-based features (only need moves_df, no engine analysis required)
# ---------------------------------------------------------------------------


def first_move_stats(moves_df: pd.DataFrame) -> pd.Series:
    """Distribution of the player's first move in games played as White."""
    first_moves = moves_df[(moves_df["move_index"] == 0) & (moves_df["my_move"])]
    return first_moves["move"].value_counts()


def castling_stats(moves_df: pd.DataFrame, games_df: pd.DataFrame) -> dict:
    """How and when the player castles.

    Returns dict with counts of kingside/queenside/no castling, percentages,
    and the average move number of castling (when it happens).
    """
    mine = moves_df[moves_df["my_move"]].copy()
    mine["move_no"] = pd.to_numeric(mine["move_no"], errors="coerce")

    castle_moves = mine[mine["move"].isin(["O-O", "O-O-O"])].sort_values("move_index")
    first_castle = castle_moves.groupby("uuid").first()

    n_games = games_df["uuid"].nunique()
    n_kingside = int((first_castle["move"] == "O-O").sum())
    n_queenside = int((first_castle["move"] == "O-O-O").sum())
    n_castled = n_kingside + n_queenside
    n_no_castle = n_games - n_castled

    return {
        "n_games": n_games,
        "n_kingside": n_kingside,
        "n_queenside": n_queenside,
        "n_no_castle": n_no_castle,
        "kingside_pct": n_kingside / n_games * 100 if n_games else 0.0,
        "queenside_pct": n_queenside / n_games * 100 if n_games else 0.0,
        "no_castle_pct": n_no_castle / n_games * 100 if n_games else 0.0,
        "avg_castle_move": float(first_castle["move_no"].mean()) if n_castled else None,
    }


def king_safety_proxy(moves_df: pd.DataFrame, early_ply_cutoff: int = 15) -> dict:
    """Rough king-safety proxy: how often the king is walked manually
    (not castling) in the first `early_ply_cutoff` plies.

    A high rate suggests either castling is skipped in favor of early king
    moves (risky) or frequent king marches under attack.
    """
    mine = moves_df[moves_df["my_move"] & (moves_df["move_index"] < early_ply_cutoff)].copy()
    king_walks = mine[mine["move"].str.match(r"^K[a-h]") & ~mine["move"].isin(["O-O", "O-O-O"])]

    n_games = moves_df["uuid"].nunique()
    games_with_walk = king_walks["uuid"].nunique()

    return {
        "games_with_early_king_move": int(games_with_walk),
        "pct_games_with_early_king_move": games_with_walk / n_games * 100 if n_games else 0.0,
        "total_early_king_moves": int(len(king_walks)),
    }


def _piece_type_of_move(move: str) -> str:
    if move in ("O-O", "O-O-O"):
        return "King (castling)"
    if move[:1] in ("K", "Q", "R", "B", "N"):
        return {"K": "King", "Q": "Queen", "R": "Rook", "B": "Bishop", "N": "Knight"}[move[0]]
    return "Pawn"


def piece_activity_stats(moves_df: pd.DataFrame) -> pd.Series:
    """Distribution of which piece type is moved, for the player's own moves.

    Derived purely from SAN prefixes (no board reconstruction needed).
    """
    mine = moves_df[moves_df["my_move"]].copy()
    mine["piece"] = mine["move"].astype(str).apply(_piece_type_of_move)
    return mine["piece"].value_counts()


def game_length_stats(games_df: pd.DataFrame, moves_df: pd.DataFrame) -> pd.DataFrame:
    """Average number of full moves per game, broken down by result."""
    plies_per_game = moves_df.groupby("uuid").size().rename("plies")
    merged = games_df.merge(plies_per_game, on="uuid", how="left")
    merged["full_moves"] = merged["plies"] / 2

    return merged.groupby("result").agg(
        games=("uuid", "count"),
        avg_full_moves=("full_moves", "mean"),
        median_full_moves=("full_moves", "median"),
    )


def endgame_frequency(games_df: pd.DataFrame, moves_df: pd.DataFrame, endgame_move_no: int = 30) -> dict:
    """% of games that reach an endgame phase (heuristically: past move 30)
    versus games decided earlier in the opening/middlegame.
    """
    plies_per_game = moves_df.groupby("uuid")["move_no"].max().rename("last_move_no")
    merged = games_df.merge(plies_per_game, on="uuid", how="left")
    n_games = len(merged)
    n_endgame = int((merged["last_move_no"] >= endgame_move_no).sum())

    return {
        "n_games": n_games,
        "n_reached_endgame": n_endgame,
        "endgame_pct": n_endgame / n_games * 100 if n_games else 0.0,
    }


# ---------------------------------------------------------------------------
# FEN-based features (need result_df with a 'fen' column from run_chess_analysis)
# ---------------------------------------------------------------------------


def queen_trade_stats(result_df: pd.DataFrame, ply_tolerance: int = 4) -> pd.Series:
    """Classify each game as: queens traded off (both queens leave the board
    within `ply_tolerance` plies of each other), one queen lost without a
    matching trade ("queen_sac_or_imbalance"), or queens stayed on the board.
    """
    df = result_df.dropna(subset=["fen"]).copy()
    board_part = df["fen"].str.split(" ").str[0]
    df["white_queens"] = board_part.str.count("Q")
    df["black_queens"] = board_part.str.count("q")

    statuses = []
    for _, g in df.groupby("uuid"):
        w_zero = g.loc[g["white_queens"] == 0, "move_index"]
        b_zero = g.loc[g["black_queens"] == 0, "move_index"]
        w0 = w_zero.min() if not w_zero.empty else None
        b0 = b_zero.min() if not b_zero.empty else None

        if w0 is not None and b0 is not None and abs(w0 - b0) <= ply_tolerance:
            statuses.append("traded")
        elif w0 is not None or b0 is not None:
            statuses.append("queen_sac_or_imbalance")
        else:
            statuses.append("queens_stayed_on")

    return pd.Series(statuses).value_counts()


def material_imbalance_by_phase(result_df: pd.DataFrame) -> pd.DataFrame:
    """Average absolute material imbalance (White material - Black material,
    in pawns), bucketed by game phase (move number heuristic).
    """
    df = result_df.dropna(subset=["fen"]).copy()
    board_part = df["fen"].str.split(" ").str[0]

    for piece, value in PIECE_VALUES.items():
        if value == 0:
            continue
        df[f"w_{piece}"] = board_part.str.count(piece) * value
        df[f"b_{piece}"] = board_part.str.count(piece.lower()) * value

    white_cols = [f"w_{p}" for p in PIECE_VALUES if PIECE_VALUES[p]]
    black_cols = [f"b_{p}" for p in PIECE_VALUES if PIECE_VALUES[p]]
    df["material_imbalance"] = df[white_cols].sum(axis=1) - df[black_cols].sum(axis=1)

    df["phase"] = pd.cut(
        df["move_no"].astype(int),
        bins=[0, 10, 30, np.inf],
        labels=["Opening", "Middlegame", "Endgame"],
    )

    return df.groupby("phase", observed=True).agg(
        avg_abs_material_imbalance=("material_imbalance", lambda s: s.abs().mean()),
    )


def pawn_structure_summary(result_df: pd.DataFrame) -> dict:
    """Average doubled-pawn and pawn-island count in the player's own
    structure, sampled from each game's final analyzed position.

    A higher average doubled-pawn count suggests a looser structure; more
    pawn islands generally means a more fragmented, harder-to-defend structure.
    """
    last_rows = (
        result_df.dropna(subset=["fen"])
        .sort_values("move_index")
        .groupby("uuid")
        .tail(1)
    )
    # One lookup per game instead of re-scanning all of result_df per row.
    my_color_by_uuid = (
        result_df[result_df["my_move"]].groupby("uuid")["player"].first()
    )

    doubled_counts = []
    island_counts = []

    for _, row in last_rows.iterrows():
        player = my_color_by_uuid.get(row["uuid"])
        if player is None:
            continue
        my_color = chess.WHITE if player == "white" else chess.BLACK

        try:
            board = chess.Board(row["fen"])
        except Exception:
            continue

        pawn_squares = board.pieces(chess.PAWN, my_color)
        files = [chess.square_file(sq) for sq in pawn_squares]
        if not files:
            continue

        file_counts = pd.Series(files).value_counts()
        doubled_counts.append(int((file_counts > 1).sum()))

        occupied_files = sorted(set(files))
        islands = 1
        for a, b in zip(occupied_files, occupied_files[1:]):
            if b - a > 1:
                islands += 1
        island_counts.append(islands)

    return {
        "n_games_sampled": len(doubled_counts),
        "avg_doubled_pawn_files": float(np.mean(doubled_counts)) if doubled_counts else None,
        "avg_pawn_islands": float(np.mean(island_counts)) if island_counts else None,
    }


# ---------------------------------------------------------------------------
# Aggregate style profile
# ---------------------------------------------------------------------------


def compute_style_profile(
    games_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    result_df: pd.DataFrame,
    sacrifices_df: pd.DataFrame | None = None,
) -> dict:
    """Combine several signals into a small set of [0,1]-normalized style
    dimensions. Heuristic and exploratory — not a validated psychometric
    model, just a readable summary of tendencies already visible in the
    other stats.

    Dimensions
    ----------
    aggression : normalized sacrifice frequency (per game)
    tactical_sharpness : share of moves that are decisive (Blunder/Mistake/
        Best), a proxy for how often positions are sharp/complicated
    solidity : inverse of blunder rate (1 - blunder_rate, clipped)
    endgame_reach : % of games that reach an endgame phase
    repertoire_diversity : opening diversity's normalized entropy
    """
    from src.opening_analysis import opening_diversity_index
    from src.tactics_analysis import move_quality_distribution, find_sacrifices

    if sacrifices_df is None:
        sacrifices_df = find_sacrifices(result_df)

    n_games = games_df["uuid"].nunique()
    sac_rate = len(sacrifices_df) / n_games if n_games else 0.0
    aggression = float(np.clip(sac_rate / 2.0, 0, 1))  # 2 sacs/game ~ very aggressive ceiling

    quality = move_quality_distribution(result_df, my_moves_only=True)
    total_moves = quality.sum()
    blunder_rate = quality.get("Blunder", 0) / total_moves if total_moves else 0.0
    decisive_rate = (
        quality.get("Blunder", 0) + quality.get("Mistake", 0) + quality.get("Best", 0)
    ) / total_moves if total_moves else 0.0

    solidity = float(np.clip(1 - blunder_rate / 0.1, 0, 1))  # 10% blunder rate -> 0
    tactical_sharpness = float(np.clip(decisive_rate, 0, 1))

    endgame = endgame_frequency(games_df, moves_df)
    endgame_reach = float(np.clip(endgame["endgame_pct"] / 100, 0, 1))

    diversity = opening_diversity_index(games_df)
    repertoire_diversity = float(np.clip(diversity["normalized_entropy"], 0, 1))

    return {
        "aggression": aggression,
        "tactical_sharpness": tactical_sharpness,
        "solidity": solidity,
        "endgame_reach": endgame_reach,
        "repertoire_diversity": repertoire_diversity,
    }


def plot_style_radar(style_profile: dict):
    """Radar/spider chart of the style profile dimensions."""
    labels = list(style_profile.keys())
    values = list(style_profile.values())
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="#3A6EA5", linewidth=2)
    ax.fill(angles, values, color="#3A6EA5", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label.replace("_", " ").title() for label in labels])
    ax.set_ylim(0, 1)
    ax.set_title("Player Style Profile", pad=20)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Clustering games into styles
# ---------------------------------------------------------------------------


def _per_game_features(games_df: pd.DataFrame, result_df: pd.DataFrame, moves_df: pd.DataFrame) -> pd.DataFrame:
    """Build a small per-game numeric feature matrix for clustering."""
    mine = result_df[result_df["my_move"]].copy()
    mine["cp_loss"] = (-mine["delta"]).clip(lower=0)

    per_game = mine.groupby("uuid").agg(
        avg_cp_loss=("cp_loss", "mean"),
        blunder_rate=("blunder", "mean"),
        n_moves=("move", "count"),
    )

    plies_per_game = moves_df.groupby("uuid").size().rename("plies")
    captures = moves_df[moves_df["my_move"] & moves_df["move"].str.contains("x", na=False)]
    capture_rate = (captures.groupby("uuid").size() / moves_df[moves_df["my_move"]].groupby("uuid").size()).rename(
        "capture_rate"
    )

    features = per_game.join(plies_per_game, how="left").join(capture_rate, how="left")
    features["capture_rate"] = features["capture_rate"].fillna(0)
    features = features.dropna()
    return features


def cluster_games_by_style(
    games_df: pd.DataFrame,
    result_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster games into style groups using KMeans over per-game features
    (avg centipawn loss, blunder rate, game length, capture rate).

    Returns
    -------
    (assignments, cluster_summary) :
        assignments : DataFrame indexed by uuid with a 'cluster' column
        cluster_summary : DataFrame of per-cluster feature means, to help
            interpret what each cluster represents
    """
    features = _per_game_features(games_df, result_df, moves_df)
    if len(features) < n_clusters:
        raise ValueError(f"Not enough analyzed games ({len(features)}) for {n_clusters} clusters.")

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    assignments = features.copy()
    assignments["cluster"] = labels

    cluster_summary = assignments.groupby("cluster").agg(
        games=("n_moves", "count"),
        avg_cp_loss=("avg_cp_loss", "mean"),
        blunder_rate=("blunder_rate", "mean"),
        avg_plies=("plies", "mean"),
        avg_capture_rate=("capture_rate", "mean"),
    )

    return assignments[["cluster"]], cluster_summary


def plot_game_clusters(games_df: pd.DataFrame, result_df: pd.DataFrame, moves_df: pd.DataFrame, n_clusters: int = 3):
    """PCA-projected 2D scatter of games colored by style cluster."""
    features = _per_game_features(games_df, result_df, moves_df)
    if len(features) < n_clusters:
        print("Not enough analyzed games to cluster.")
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", alpha=0.6, s=20)
    ax.set_title(f"Games Clustered by Style (k={n_clusters})")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    plt.tight_layout()
    plt.show()
