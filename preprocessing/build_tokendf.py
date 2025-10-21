### BUILD NEW TOKENISED DATAFRAME

import os
import re
import json
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

# ----------------------------------------------------
# Load tokens
# ----------------------------------------------------
def load_tokens(tokens_file: str = "tokens.txt") -> Dict[str, Dict[str, int]]:
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    path = os.path.join(base_dir, tokens_file)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------------------------------
# Clean + extract actions/patterns
# ----------------------------------------------------
def filter_action(value: str) -> str:
    for delim in ["(", "<", "$", "1"]:
        if isinstance(value, str) and delim in value and (low_ind := value.index(delim)):
            value = value[:low_ind]
    return value.strip() if isinstance(value, str) else value

_RX_PAREN   = re.compile(r"\(([^()]*)\)")
_RX_ANGLE   = re.compile(r"<([^<>]*)>")
_RX_DOLLAR  = re.compile(r"\$([^$]+)\$")

def extract_patterns(cell: str) -> List[Tuple[str, str]]:
    if not isinstance(cell, str) or not cell:
        return []
    out: List[Tuple[str, str]] = []
    for m in _RX_PAREN.finditer(cell):
        out.append(("(", m.group(1).strip()))
    for m in _RX_ANGLE.finditer(cell):
        out.append(("<", m.group(1).strip()))
    for m in _RX_DOLLAR.finditer(cell):
        out.append(("$", m.group(1).strip()))
    return [(d, p) for d, p in out if p]

# ----------------------------------------------------
# Browser map
# ----------------------------------------------------
DEFAULT_BROWSER_MAP = {
    "Firefox": 0,
    "Google Chrome": 1,
    "Microsoft Edge": 2,
    "Opera": 3,
}

# ----------------------------------------------------
# Cell → token tuples
# ----------------------------------------------------
def cell_to_token_tuples(
    cell: Any,
    action_to_id: Dict[str, int],
    delim_to_id: Dict[str, int],
    pattern_to_id: Dict[str, int],
    unknown_id: int = -1
) -> List[Tuple[int, int, int]]:
    if not isinstance(cell, str) or not cell or cell.startswith("t"):
        return []
    action_str = filter_action(cell)
    if not action_str:
        return []
    action_id = action_to_id.get(action_str, unknown_id)
    pairs = extract_patterns(cell)
    if not pairs:
        return [(action_id, unknown_id, unknown_id)]
    tuples = []
    for delim_char, pattern_str in pairs:
        tuples.append(
            (
                action_id,
                delim_to_id.get(delim_char, unknown_id),
                pattern_to_id.get(pattern_str, unknown_id),
            )
        )
    return tuples

# ----------------------------------------------------
# Main function to tokenise + save CSV
# ----------------------------------------------------
def tokenise_df(
    df_train: pd.DataFrame,
    tokens_file: str = "tokens.txt",
    browser_map: Optional[Dict[str, int]] = None,
    output_csv: str = "tokenised_df.csv",
    unknown_id: int = -1,
) -> pd.DataFrame:

    tokens = load_tokens(tokens_file)
    action_to_id   = tokens.get("actions", {})
    delim_to_id    = tokens.get("delimiters", {})
    pattern_to_id  = tokens.get("patterns", {})
    bmap = browser_map or DEFAULT_BROWSER_MAP

    user_col = "user" if "user" in df_train.columns else df_train.columns[0]
    browser_col = "browser" if "browser" in df_train.columns else df_train.columns[1]

    records = []
    for _, row in df_train.iterrows():
        user_val = row[user_col]
        browser_val = row[browser_col]
        browser_token = bmap.get(browser_val, unknown_id)

        tuples_all: List[Tuple[int, int, int]] = []
        for cell in row.iloc[2:]:
            tuples_all.extend(
                cell_to_token_tuples(cell, action_to_id, delim_to_id, pattern_to_id, unknown_id)
            )

        records.append({
            "user": user_val,
            "browser": browser_token,
            "actions_patterns": tuples_all
        })

    new_df = pd.DataFrame.from_records(records)

    # --- Save CSV next to script ---
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    output_path = os.path.join(base_dir, output_csv)
    new_df.to_csv(output_path, index=False)

    print(f"✅ Tokenised dataframe saved to: {output_path}")
    print(f"   → {len(new_df)} rows, {len(new_df.columns)} columns")
    return new_df
