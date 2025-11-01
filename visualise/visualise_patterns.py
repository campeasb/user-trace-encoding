import pandas as pd
import re
from visualise.visualise_actions import filter_action

def count_actions_per_session(df, filtered_uniques):
    """
    Pour chaque session (ligne du CSV), compte le nombre d'occurrences
    de chaque action (dans filtered_uniques).
    Retourne un DataFrame avec :
    - session_id
    - util
    - navigateur
    - colonnes = actions, valeurs = nombre d'occurrences
    """
    all_rows = []

    for idx, row in df.iterrows():
        user = row["util"]
        navigateur = row["navigateur"]
        actions = row.drop(["util", "navigateur"]).astype(str)
        
        # Initialiser le compteur pour chaque action à 0
        action_counts = {action: 0 for action in filtered_uniques}

        # Parcourir chaque cellule de la ligne
        for val in actions:
            if val.startswith("t"):  # ignorer les avancements
                continue
            clean_val = filter_action(val)
            if clean_val in action_counts:
                action_counts[clean_val] += 1

        # Construire une ligne du tableau
        session_data = {
            "session_id": idx + 1,
            "util": user,
            "navigateur": navigateur
        }
        session_data.update(action_counts)
        all_rows.append(session_data)

    # Créer le DataFrame
    df_actions = pd.DataFrame(all_rows)

    print(f"✅ {len(df_actions)} sessions traitées, {len(filtered_uniques)} actions suivies.")
    return df_actions

import re
from collections import defaultdict

# Définition des regex pour chaque type de pattern
pattern_map = {
    "()": re.compile(r"\((.*?)\)"),
    "<>": re.compile(r"<(.*?)>"),
    "$$": re.compile(r"\$(.*?)\$"),
}

def count_unique_pattern_values(df):
    unique_values = defaultdict(set)  # pattern_type -> {valeurs uniques}
    
    # On parcourt toutes les cellules (hors util et navigateur)
    action_cols = df.columns.difference(["util", "navigateur"])
    
    for val in df[action_cols].astype(str).values.flatten():
        for pat_name, pat in pattern_map.items():
            found = pat.findall(val)
            if found:
                unique_values[pat_name].update(found)
    
    # On crée un tableau récapitulatif
    pattern_counts = {
        "pattern_type": [],
        "n_unique_values": [],
        "example_values": []
    }
    
    for pat_type, vals in unique_values.items():
        pattern_counts["pattern_type"].append(pat_type)
        pattern_counts["n_unique_values"].append(len(vals))
        pattern_counts["example_values"].append(", ".join(list(vals)[:5]))  # quelques exemples
    
    return pd.DataFrame(pattern_counts)