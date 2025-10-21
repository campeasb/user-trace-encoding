### BUILD TOKENS.TXT

import os
import re
import json

# --- existing from before ---
def filter_action(value: str):
    """Remove suffixes like '(...)', '<...>', '$...', '1' from an action string."""
    for delim in ["(", "<", "$", "1"]:
        if delim in value and (low_ind := value.index(delim)):
            value = value[:low_ind]
    return value.strip()

def build_action_tokens(df_train, output_file="tokens.txt"):
    uniques = df_train.iloc[:, 2:].stack().unique()
    filtered_uniques = list(
        set([filter_action(un) for un in uniques if not un.startswith("t")])
    )
    filtered_uniques.sort()
    action_to_token = {action: idx for idx, action in enumerate(filtered_uniques)}

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    output_path = os.path.join(base_dir, output_file)

    payload = {"actions": action_to_token}  # keep actions first
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(action_to_token)} action tokens to {output_path}")
    return action_to_token

# --- new helpers for patterns/templates ---
_PATTERN_REGEXES = [
    re.compile(r"\(([^()]*)\)"),     # ( ... )
    re.compile(r"<([^<>]*)>"),       # < ... >
    re.compile(r"\$([^$]+)\$"),      # $ ... $
]

def _extract_patterns_from_cell(cell: str) -> list[str]:
    """Return all template-like patterns found in a cell."""
    if not isinstance(cell, str) or not cell:
        return []
    hits = []
    for rx in _PATTERN_REGEXES:
        hits.extend(m.group(1).strip() for m in rx.finditer(cell) if m.group(1).strip())
    return hits

def build_template_tokens(df_train, tokens_file="tokens.txt"):
    """
    Parse df_train to extract all templates/patterns found after actions,
    assign ids, and write them under 'delimiters' and 'patterns' in tokens_file.
    Keeps existing 'actions' mapping if tokens_file already exists.
    """
    # Collect unique patterns
    col_vals = df_train.iloc[:, 2:].stack().tolist()
    found = set()
    for v in col_vals:
        for pat in _extract_patterns_from_cell(v):
            # ignore time tokens like t5/t10 (they don't match our regexes anyway)
            found.add(pat)

    # Stable ids
    patterns_sorted = sorted(found)
    pattern_to_id = {p: i for i, p in enumerate(patterns_sorted)}

    delimiters = {"(": 0, "<": 1, "$": 2}

    # Resolve path (script vs notebook)
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    path = os.path.join(base_dir, tokens_file)

    # Load existing tokens if any (to preserve your actions)
    existing = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    # Build final payload with predictable field order: actions -> delimiters -> patterns
    actions = existing.get("actions", {})
    payload = {
        "actions": actions,                 # keep whatever you already built
        "delimiters": delimiters,
        "patterns": pattern_to_id,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Saved delimiters ({len(delimiters)}) and patterns ({len(pattern_to_id)}) "
        f"to {path} (actions preserved: {len(actions)})"
    )
    return payload

