# ============================================================================
# IMPORTS & CONFIGURATION
# ============================================================================
import json
import chess.engine
import sys
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def evaluate(
    board: chess.Board, engine: chess.engine.SimpleEngine, depth: int = 10
) -> int:
    """Returns the evaluation given the board, the engine and the depth"""

    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()

        if score.is_mate():
            return 10000 if score.mate() > 0 else -10000

        val = score.score()
        return 0 if val is None else val
    except:
        return 0


def classify_move(delta: int) -> str:
    """Classifies move as Best, Excellent, Good, Inaccuracy, Mistake or Blunder"""
    if delta >= BEST_LL:
        return "Best"
    elif delta >= EXCELLENT_LL:
        return "Excellent"
    elif delta >= GOOD_LL:
        return "Good"
    elif delta >= INACCURACY_LL:
        return "Inaccuracy"
    elif delta >= MISTAKE_LL:
        return "Mistake"
    else:
        return "Blunder"


def is_blunder(delta: int) -> int:
    """Return 1 if delta indicates blunder else 0"""
    return int(delta < MISTAKE_LL)


def save_cache(cache) -> None:
    """Save evaluations cache to JSON file."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    return None


def load_evaluation_cache() -> dict:
    """Load cached evaluations from JSON file."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            return cache
        except:
            print("⚠️ Error opening cache file")
            return {}
    else:
        print("📦 No cache found, starting fresh")
        return {}


def get_cache_key(game_id, move_no, turn) -> str:
    """Create unique key for each evaluation."""
    return f"{game_id}_{move_no}_{turn}"


def should_compute(cache, cache_key, given_depth):
    """Check if we need to compute this position."""
    if cache_key not in cache:
        return True
    return cache[cache_key]["depth"] < given_depth


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def run_chess_analysis(
    games_df, moves_df, time_control=None, n_games=50, depth=10, use_cache=True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run chess analysis with caching to avoid re-evaluating moves.

    Parameters:
    -----------
    use_cache : bool
        Whether to use cached evaluations (default True)
    """

    # -----------------------------
    # TIME FILTER
    # -----------------------------
    filtered_games = filter_games_by_time_control(games_df, time_control)

    # Make sure games are sorted by most recent first
    filtered_games = filtered_games.sort_values("end_time", ascending=False)

    # Then select
    if n_games == -1:
        selected_games = filtered_games["uuid"].drop_duplicates()
    else:
        selected_games = take_last_n_games(filtered_games, n_games)

    moves = moves_df[moves_df["uuid"].isin(selected_games)].copy()

    if moves.empty:
        print("⚠️ No moves found after filtering")
        return pd.DataFrame()

    print(f"✅ Games selected: {len(selected_games)}")
    print(f"✅ Moves selected: {len(moves)}")

    # -----------------------------
    # LOAD CACHE
    # -----------------------------
    if use_cache:
        cache = load_evaluation_cache()
        print(f"Loaded {len(cache)} cached evaluations")
    else:
        cache = {}
    cached_count = 0
    computed_count = 0

    # -----------------------------
    # ENGINE (only if we need it)
    # -----------------------------
    engine = None

    # -----------------------------
    # PROCESS
    # -----------------------------
    rows = []
    with tqdm(
        total=len(moves), desc="Analyzing moves", unit="move", colour="RED"
    ) as pbar:
        for game_id, game_df in moves.groupby("uuid"):
            board = chess.Board()
            game_df = game_df.sort_values("move_index", ascending=True)
            eval_current = 0

            for _, row in game_df.iterrows():
                eval_before = eval_current

                try:
                    move = board.parse_san(row["move"])
                    board.push(move)
                    fen = board.fen()
                except Exception as e:
                    print(f"Error: {e} at game {game_id} in move {row['move_index']}")
                    continue

                cache_key = get_cache_key(game_id, row["move_no"], row["turn"])

                if should_compute(cache, cache_key, depth):
                    if engine is None:
                        engine = chess.engine.SimpleEngine.popen_uci(
                            str(STOCKFISH_PATH)
                        )

                    if board.is_checkmate():
                        eval_after = eval_before
                    else:
                        eval_after = evaluate(board, engine, depth)

                    cache[cache_key] = {"depth": depth, "eval": eval_after, "fen": fen}
                    computed_count += 1
                else:
                    eval_after = cache[cache_key]["eval"]
                    cached_count += 1

                if board.is_checkmate():
                    eval_after = eval_before

                if computed_count > 0 and computed_count % 100 == 0:
                    save_cache(cache)

                eval_current = eval_after

                if row["turn"] == "white":
                    delta = eval_after - eval_before
                else:
                    delta = eval_before - eval_after

                blunder_flag = is_blunder(delta)
                is_my_move = row["my_move"]

                blunder_type = (
                    "my_blunder"
                    if blunder_flag and is_my_move
                    else (
                        "opponent_blunder"
                        if blunder_flag and not is_my_move
                        else "no_blunder"
                    )
                )

                rows.append(
                    {
                        "link_id": row["link_id"],
                        "uuid": game_id,
                        "move_index": row["move_index"],
                        "move_no": row["move_no"],
                        "player": row["turn"],
                        "move": row["move"],
                        "my_move": is_my_move,
                        "eval_before": eval_before,
                        "eval_after": eval_after,
                        "delta": delta,
                        "blunder": blunder_flag,
                        "blunder_type": blunder_type,
                        "move_type": classify_move(delta),
                        "fen": fen,
                        "end_time": row["end_time"],
                    }
                )
                pbar.update(1)

    if use_cache:
        save_cache(cache)
        print(f"💾 Saved {len(cache)} evaluations to cache {' '*30}")

    # Close engine if it was started
    if engine:
        engine.quit()

    result_df = pd.DataFrame(rows)
    moves_df_new = update_moves(moves_df, verbose=False)

    # -----------------------------
    # FINAL REPORT
    # -----------------------------
    print("\n" + "=" * 50)
    print(f"Total moves analyzed: {len(result_df)}")
    print(f"Games analyzed: {result_df['uuid'].nunique()}")
    print(f"Cache hits: {cached_count}")
    print(f"New evaluations: {computed_count}")
    print(f"Cache size: {len(cache)} total entries")

    if not result_df.empty:
        result_df = result_df.sort_values(
            ["end_time", "move_index"], ascending=[False, True]
        )

    return result_df, moves_df_new


def filter_games_by_time_control(
    games_df: pd.DataFrame, time_control: str = None
) -> pd.DataFrame:
    if time_control is None:
        return games_df

    if time_control in games_df["time_class"].unique():
        return games_df[games_df["time_classl"] == time_control]
    else:
        print("❌ time_control not found")
        print("Available time_control values:", games_df["time_control"].unique())
        return pd.DataFrame()


def take_last_n_games(df: pd.DataFrame, n_games: int) -> pd.DataFrame:
    # Drop duplicates based on 'uuid' and keep the row with the most recent end_time
    df_sorted = df.sort_values("end_time", ascending=False)
    return (df_sorted.iloc[:n_games])["uuid"]


def update_moves(moves_df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Add cached evaluations and FEN to moves_df.
    Also calculates eval_before from previous move's eval_after.
    """
    moves_df = moves_df.copy()
    cache = load_evaluation_cache()
    total_rows = len(moves_df)

    if verbose:
        print(f"📊 Updating {total_rows} moves with cached data...")

    # Create temporary cache key column
    moves_df["_cache_key"] = moves_df.apply(
        lambda row: get_cache_key(row["uuid"], row["move_no"], row["turn"]), axis=1
    )

    if verbose:
        print("  ✓ Cache keys generated")

    # Add cached data
    cached_data = moves_df["_cache_key"].apply(lambda key: cache.get(key, {}))

    moves_df["fen"] = cached_data.apply(lambda x: x.get("fen", None))
    moves_df["eval_after"] = cached_data.apply(lambda x: x.get("eval", None))
    moves_df["depth"] = cached_data.apply(lambda x: x.get("depth", 0))

    # Count how many were found in cache
    found_count = moves_df["eval_after"].notna().sum()
    if verbose:
        print(f"  ✓ Found {found_count}/{total_rows} moves in cache")

    # Drop temporary column
    moves_df = moves_df.drop("_cache_key", axis=1)

    # Sort for correct order
    moves_df = moves_df.sort_values(["uuid", "move_index"]).reset_index(drop=True)

    if verbose:
        print("  ✓ Sorting moves by game and move order")

    # Calculate eval_before (0 for first move, otherwise previous eval_after)
    moves_df["eval_before"] = moves_df.groupby("uuid")["eval_after"].shift(1).fillna(0)

    if verbose:
        print(f"✅ Update complete! {found_count} moves have cached evaluations")

    return moves_df
