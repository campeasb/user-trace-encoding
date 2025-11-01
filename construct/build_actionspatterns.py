import pandas as pd

def action_pattern_as_columns(encoded_rows, max_patterns=3):
    """
    Chaque combinaison (action + patterns) devient une colonne unique.
    Cellule = fréquence d'apparition dans la session.
    """
    # 1️⃣ collecter toutes les combinaisons uniques
    all_combinations = set()
    for row in encoded_rows:
        for lst in row["actions_patterns"]:
            patterns = lst[1:max_patterns+1] if len(lst) > 1 else []
            patterns += [-1]*(max_patterns - len(patterns))
            all_combinations.add( tuple([lst[0]] + patterns) )
    all_combinations = sorted(all_combinations)

    # 2️⃣ créer le DataFrame
    data = []
    for row in encoded_rows:
        row_dict = {"user": row["user"], "navigateur": row["navigateur"]}
        # initialiser toutes les colonnes à 0
        for comb in all_combinations:
            row_dict[comb] = 0
        # compter les occurrences
        counter = {}
        for lst in row["actions_patterns"]:
            patterns = lst[1:max_patterns+1] if len(lst) > 1 else []
            patterns += [-1]*(max_patterns - len(patterns))
            key = tuple([lst[0]] + patterns)
            row_dict[key] += 1
        data.append(row_dict)

    df = pd.DataFrame(data)
    # supprimer les colonnes où la fréquence = 0 pour toutes les sessions
    cols_to_drop = [c for c in df.columns if c not in ["user","navigateur"] and df[c].sum() == 0]
    df.drop(columns=cols_to_drop, inplace=True)
    return df