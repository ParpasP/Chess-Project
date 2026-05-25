from pathlib import Path

# User settings
MY_CHESS_USERNAME = "ParpasP"

# Paths
STOCKFISH_PATH = Path(r"C:\Program Files\stockfish\stockfish-windows-x86-64-avx2.exe")
PROJECT_ROOT = Path(__file__).parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GAMES_JSON_PATH = DATA_RAW_DIR / f"ChessGames_{MY_CHESS_USERNAME}.json"
OPENING_NAMES_JSON_PATH = PROJECT_ROOT / "data" / "openings.json"


# Cache file path
CACHE_DIR = DATA_PROCESSED_DIR / "evaluation_cache"
CACHE_FILE = CACHE_DIR / f"move_evaluations_fen_{MY_CHESS_USERNAME}.json"

# Create directories
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Moves Classificiation: delta >= lower limit
BEST_LL = -10
EXCELLENT_LL = -30
GOOD_LL = -50
INACCURACY_LL = -100
MISTAKE_LL = -300
