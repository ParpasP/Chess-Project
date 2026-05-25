import json
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

# Load the JSON file
OPENING_NAMES_JSON_PATH = Path(__file__).parent.parent / "data" / "openings.json"

with open(OPENING_NAMES_JSON_PATH, "r", encoding="utf-8") as f:
    OPENING_NAMES = json.load(f)


def simplify_opening(opening_name):
    """Convert full opening name to simple form."""
    if not opening_name or opening_name == "Undefined":
        return "Undefined"

    # Try exact match
    if opening_name in OPENING_NAMES:
        return OPENING_NAMES[opening_name]

    # Try prefix match
    for prefix, simplified in OPENING_NAMES.items():
        if opening_name.startswith(prefix):
            return simplified

    print(f"Opening: {opening_name}")
    custom = input("Simplify as (or press Enter to skip): ")

    if custom.strip():
        key_parts = opening_name.split("-")[:3]
        key = "-".join(key_parts)
        OPENING_NAMES[key] = custom

        with open(OPENING_NAMES_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(OPENING_NAMES, f, indent=4)

        print(f"  ✅ Saved: {key} -> {custom}\n")

    return custom
