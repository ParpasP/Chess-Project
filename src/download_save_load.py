# ============================================================================
# IMPORTS & CONFIGURATION
# ============================================================================
import json
import sys
from pathlib import Path
import pandas as pd
from io import StringIO
import chess.pgn
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
from config import *
from src.openings import simplify_opening
from src.cache import get_cache_key, load_evaluation_cache

from chessdotcom import get_player_game_archives, get_player_games_by_month, Client

load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")
TIMEZONE = os.getenv("TIME_ZONE")

# Configure Chess.com API
Client.request_config["headers"]["User-Agent"] = USER_AGENT


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def clock_to_seconds(clock_str) -> None | float:
    if clock_str is None:
        return None
    parts = clock_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return None


def extract_result(game, username):

    my_result_raw = (
        game["white"]["result"]
        if game["white"]["username"] == username
        else game["black"]["result"]
    )

    draw_results = {
        "agreed",
        "stalemate",
        "repetition",
        "timevsinsufficient",
        "50move",
        "insufficient",
        "draw",
    }

    if my_result_raw == "win":
        result = "Win"
    elif my_result_raw in draw_results:
        result = "Draw"
    else:
        result = "Loss"

    return result, my_result_raw


def extract_moves_with_time(pgn) -> dict:
    move_section = pgn.split("\n\n", 1)[1]

    pattern = r"(\d+\.(?:\.\.)?)\s*([^\{]+)\{?\[%clk ([0-9:.\s]+)\]?\}?\s*(?:([^\{]+)\{?\[%clk ([0-9:.\s]+)\]?\})?"

    matches = re.findall(pattern, move_section)

    result = {}

    for match in matches:
        move_no = match[0].replace(".", "").replace("..", "")
        white_move = match[1].strip()
        white_clk = match[2].strip()
        black_move = match[3].strip().split(" ")[-1] if match[3] else None
        black_clk = match[4].strip() if match[4] else None
        result[move_no] = {
            "white": white_move,
            "white_clk": white_clk,
            "black": black_move,
            "black_clk": black_clk,
        }

    return result


def extract_eco_code(pgn) -> str:
    pattern = r'\[ECOUrl ".*?/([^"]+)"\]'
    eco_code_match = re.search(pattern, pgn)
    eco_code = eco_code_match.group(1) if eco_code_match else None

    return eco_code


def construct_move_seq(pgn) -> str:
    game = chess.pgn.read_game(StringIO(pgn))
    board = game.board()
    moves = []

    for move in game.mainline_moves():
        moves.append(board.san(move))  # SAN = human-readable move
        board.push(move)

    result = "|".join(moves)
    return result


def download_chess_games(
    username, output_path=DATA_RAW_DIR, force_download=False
) -> None:
    """
    Download chess games from Chess.com API.

    Parameters:
    -----------
    username : str
        Chess.com username
    output_path : Path or str
        Where to save the JSON file
    force_download : bool
        If False, skips download if file exists
        If True, always downloads fresh data

    Returns:
    --------
    games_list: list of game dictionaries
    """
    output_path = Path(output_path)
    file_path = output_path / f"ChessGames_{username}.json"

    # Check if file exists and we're not forcing download
    if file_path.exists() and not force_download:
        print(f"✓ File already exists: {file_path}")
        print(f"  To download fresh data, use force_download=True")
        return

    # Download fresh data
    print(f"📥 Downloading games for {username}...")

    # Get archives
    game_archives = get_player_game_archives(username).json["archives"]

    # Extract months
    played_months = [archive[-7:] for archive in game_archives]

    # Download all games
    all_games = []
    for month in played_months:
        year, month_num = month.split("/")
        print(f"  Fetching {year}-{month_num}...", end=" ")

        month_games = get_player_games_by_month(username, year=year, month=month_num)
        games_count = len(month_games.json["games"])
        all_games.extend(month_games.json["games"])
        print(f"✓ {games_count} games")

    # Save to file
    file_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if needed
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_games, f, indent=4)

    print(f"\n✅ Downloaded {len(all_games)} total games")
    print(f"   Saved to: {file_path}")
    return


def load_chess_games(username: str = MY_CHESS_USERNAME) -> list:
    """
    Load chess games from /data/raw/f"ChessGames_{username}.json".

    Parameters:
    -----------
    username : str
        Chess.com username.

    Returns:
    --------
    games_list: list of game dictionaries

    """

    file_path = DATA_RAW_DIR / f"ChessGames_{username}.json"

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"No saved games found at: {file_path}\n"
            f"Use download_chess_games() first to download your games."
        )

    # Load and return games
    print(f"📂 Loading games from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        games = json.load(f)

    print(f"✓ Loaded {len(games)} games")
    return games


def process_and_save_chess_data(
    all_games=None, username=MY_CHESS_USERNAME, verbose=True, format="csv"
):
    """
    Process raw chess games into dataframes and save to disk.
    Does NOT load existing data - always processes from raw games.

    Parameters:
    -----------
    all_games : list, optional
        List of game dictionaries from Chess.com API.
        If None, tries to load from JSON file.
    username : str
        Your username
    verbose : bool
        Print progress updates
    format : str
        Format to save ("parquet", "feather", "csv")

    Returns:
    --------
    tuple : (games_df, moves_df)
    """

    # Get raw games if not provided
    if all_games is None:
        json_path = DATA_RAW_DIR / f"ChessGames_{username}.json"
        if json_path.exists():
            if verbose:
                print(f"📂 Loading raw games from: {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                all_games = json.load(f)
            if verbose:
                print(f"✓ Loaded {len(all_games)} raw games")
        else:
            raise FileNotFoundError(
                f"No raw games found at {json_path}\n"
                f"Use download_chess_games() first to download your games."
            )

    # Process the games
    games_rows = []
    moves_rows = []
    total_games = len(all_games)

    if verbose:
        print(f"\n🔄 Processing {total_games} games...")
    cache = load_evaluation_cache()
    for idx, game in enumerate(all_games):
        if verbose and idx % 100 == 0:
            print(f"  Game {idx}/{total_games}...", end="\r")

        try:
            pgn = game["pgn"]
            if not pgn:
                continue

            game_id = game["uuid"]

            ts = game["end_time"]
            dt = datetime.fromtimestamp(ts, tz=ZoneInfo("UTC"))
            dt_cest = dt.astimezone(ZoneInfo("Europe/Amsterdam"))

            my_color = "White" if game["white"]["username"] == username else "Black"

            opening_name = game["eco"].split("/")[-1] if "eco" in game else None

            # Safely convert ratings to int
            try:
                white_rating = int(game["white"]["rating"])
                black_rating = int(game["black"]["rating"])
                rating_diff = white_rating - black_rating
            except (ValueError, TypeError):
                white_rating = game["white"]["rating"]
                black_rating = game["black"]["rating"]
                rating_diff = None

            result, raw_result = extract_result(game, username)

            # Game level data
            games_rows.append(
                {
                    "uuid": game_id,
                    "link_id": game["url"].split("/")[-1],
                    "link": game["url"],
                    "time_class": game["time_class"],
                    "time_control": game["time_control"],
                    "rated": game["rated"],
                    "white": game["white"]["username"],
                    "white_rating": white_rating,
                    "black": game["black"]["username"],
                    "black_rating": black_rating,
                    "rating_diff": rating_diff,
                    "opening": opening_name,
                    "simple_opening": simplify_opening(opening_name),
                    "eco_code": extract_eco_code(pgn),
                    "my_color": my_color,
                    "result": result,
                    "raw_result": raw_result,
                    "white_accuracy": (
                        game["accuracies"]["white"] if "accuracies" in game else None
                    ),
                    "black_accuracy": (
                        game["accuracies"]["black"] if "accuracies" in game else None
                    ),
                    "Time": dt_cest,
                    "end_time": ts,
                    "moves": construct_move_seq(pgn),
                }
            )

            # Move level data
            moves_with_time = extract_moves_with_time(pgn)
            if moves_with_time:
                eval_history = []
                for move_no, move_data in moves_with_time.items():
                    for turn in ["white", "black"]:
                        move_index = (int(move_no) - 1) * 2 + (
                            0 if turn == "white" else 1
                        )
                        clock = move_data.get(f"{turn}_clk")
                        move_text = move_data.get(turn)
                        cache_key = get_cache_key(game_id, move_no, turn)
                        fen = cache.get(cache_key, {}).get("fen")
                        eval_after = cache.get(cache_key, {}).get("eval")

                        if move_text:
                            moves_rows.append(
                                {
                                    "uuid": game["uuid"],
                                    "link_id": game["url"].split("/")[-1],
                                    "time_class": game["time_class"],
                                    "time_control": game["time_control"],
                                    "move_no": int(move_no),
                                    "move_index": move_index,
                                    "turn": turn,
                                    "move": move_text,
                                    "eval_before": (
                                        eval_history[-1] if eval_history else 0
                                    ),
                                    "eval_after": eval_after,
                                    "clock": clock,
                                    "clock_sec": (
                                        clock_to_seconds(clock) if clock else None
                                    ),
                                    "my_move": (turn.lower() == my_color.lower()),
                                    "fen": fen,
                                    "end_time": game["end_time"],
                                }
                            )
                            eval_history.append(eval_after)
        except Exception as e:
            if verbose:
                print(f"Warning: Error processing game {idx}: {e}")
            continue

    # Create dataframes
    games_df = pd.DataFrame(games_rows)
    if not games_df.empty:
        games_df = games_df.sort_values("end_time", ascending=False).reset_index(
            drop=True
        )

    moves_df = pd.DataFrame(moves_rows)
    if not moves_df.empty:
        moves_df = moves_df.sort_values(
            ["end_time", "move_index"], ascending=[False, True]
        ).reset_index(drop=True)

    if verbose:
        print(f"✅ Created {len(games_df)} games and {len(moves_df)} moves")

    # Save the processed dataframes
    save_chess_dataframes(games_df, moves_df, username, format)

    return games_df, moves_df


def save_chess_dataframes(
    games_df, moves_df, username, format="parquet"
) -> tuple[Path, Path]:
    """
    Save games and moves dataframes to disk.

    Parameters:
    -----------
    games_df : pd.DataFrame
        Games dataframe
    moves_df : pd.DataFrame
        Moves dataframe
    username : str
        Username for file naming
    format : str
        Format to save ("parquet", "feather", "csv")

    Returns:
    --------
    tuple : (games_path, moves_path)
    """
    # Save games dataframe
    games_path = DATA_PROCESSED_DIR / f"games_df_{username}.{format}"
    if format == "parquet":
        games_df.to_parquet(games_path, index=False)
    elif format == "feather":
        games_df.to_feather(games_path)
    elif format == "csv":
        games_df.to_csv(games_path, index=False)
    else:
        raise ValueError("Format must be 'parquet', 'feather', or 'csv'")

    # Save moves dataframe
    moves_path = DATA_PROCESSED_DIR / f"moves_df_{username}.{format}"
    if format == "parquet":
        moves_df.to_parquet(moves_path, index=False)
    elif format == "feather":
        moves_df.to_feather(moves_path)
    elif format == "csv":
        moves_df.to_csv(moves_path, index=False)

    print(f"✓ Saved {len(games_df)} games to {games_path}")
    print(f"✓ Saved {len(moves_df)} moves to {moves_path}")
    print(
        f"  File sizes: {games_path.stat().st_size / 1024:.1f} KB, {moves_path.stat().st_size / 1024:.1f} KB"
    )

    return games_path, moves_path


def load_chess_dataframes(username, format=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load games and moves dataframes from disk.

    Parameters:
    -----------
    username : str
        Username for file naming
    format : str, optional
        Format to load ("parquet", "feather", "csv").
        If None, auto-detects from existing files.

    Returns:
    --------
    tuple : (games_df, moves_df)
    """

    # Try formats in order of preference
    formats_to_try = [format] if format else ["parquet", "feather", "csv"]

    for fmt in formats_to_try:
        games_path = DATA_PROCESSED_DIR / f"games_df_{username}.{fmt}"
        moves_path = DATA_PROCESSED_DIR / f"moves_df_{username}.{fmt}"

        if games_path.exists() and moves_path.exists():
            # Load based on format
            if fmt == "parquet":
                games_df = pd.read_parquet(games_path)
                moves_df = pd.read_parquet(moves_path)
            elif fmt == "feather":
                games_df = pd.read_feather(games_path)
                moves_df = pd.read_feather(moves_path)
            elif fmt == "csv":
                games_df = pd.read_csv(games_path)
                moves_df = pd.read_csv(moves_path)
            else:
                continue

            print(
                f"✓ Loaded {len(games_df)} games and {len(moves_df)} moves from {fmt} files"
            )
            return games_df, moves_df

    # If we get here, no files found
    raise FileNotFoundError(
        f"No saved dataframes found for {username} in {DATA_PROCESSED_DIR}\n"
        f"Use process_and_save_chess_data() first to create them."
    )


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def get_chess_dataframes(
    username=MY_CHESS_USERNAME, force_reprocess=False, format="csv", verbose=True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Smart function that loads existing dataframes or processes them if needed.

    Parameters:
    -----------
    username : str
        Your username
    force_reprocess : bool
        If True, reprocesses from raw games even if dataframes exist
    format : str
        Format to save/load
    verbose : bool
        Print progress updates

    Returns:
    --------
    tuple : (games_df, moves_df)
    """

    # Try to load existing dataframes
    if not force_reprocess:
        try:
            if verbose:
                print(f"🔍 Looking for existing dataframes for {username}...")
            return load_chess_dataframes(username, format=format)
        except FileNotFoundError:
            if verbose:
                print(f"📦 No existing dataframes found.")

    # Need to process from raw games
    if verbose:
        print(f"🔄 Processing raw games into dataframes...")

    # First, ensure we have raw games
    json_path = DATA_RAW_DIR / f"ChessGames_{username}.json"
    if not json_path.exists():
        if verbose:
            print(f"📥 No raw games found. Downloading from Chess.com API...")

        download_chess_games(username=username, force_download=True)

    # Process and save
    return process_and_save_chess_data(
        all_games=None, username=username, verbose=verbose, format=format
    )
