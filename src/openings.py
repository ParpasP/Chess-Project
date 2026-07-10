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


def _fallback_name(opening_name: str) -> str:
    """Best-effort human-readable name when no mapping exists.

    Turns a Chess.com ECO slug like 'Some-Opening-2.e4' into 'Some Opening'
    by dropping numeric move tokens and replacing dashes with spaces.
    """
    parts = [p for p in opening_name.split("-") if not re.search(r"\d", p)]
    return " ".join(parts) if parts else opening_name


def simplify_opening(opening_name: str, interactive: bool = False) -> str:
    """Convert a Chess.com ECO opening slug to a simplified display name.

    Parameters
    ----------
    opening_name : str
        Raw opening slug, e.g. "Queens-Gambit-Declined-Baltic-Deferred".
    interactive : bool, default False
        If True and no mapping is found, prompts on stdin for a name and
        persists it to data/openings.json (original behavior). If False
        (default, and required for unattended/notebook batch runs), falls
        back to a best-effort simplified name without blocking.

    Returns
    -------
    str
        Simplified opening name, or "Undefined" if opening_name is empty.
    """
    if not opening_name or opening_name == "Undefined":
        return "Undefined"

    # Try exact match
    if opening_name in OPENING_NAMES:
        return OPENING_NAMES[opening_name]

    # Try prefix match (longest prefix wins, so more specific mappings win)
    matches = [
        simplified
        for prefix, simplified in OPENING_NAMES.items()
        if opening_name.startswith(prefix)
    ]
    if matches:
        return max(matches, key=len)

    if not interactive:
        return _fallback_name(opening_name)

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

    return _fallback_name(opening_name)
