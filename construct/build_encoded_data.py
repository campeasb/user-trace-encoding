import re
import json

def load_actions(filepath="actions.json"):
    """
    Recharge un mapping {action: id} depuis un fichier JSON.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        action_to_id = json.load(f)
    return action_to_id

def load_mapping(filepath):
    """
    Recharge un mapping depuis un fichier texte au format : valeur: id
    """
    mapping = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            val, idx = line.strip().split(": ", 1)
            mapping[val] = int(idx)
    return mapping

def encode_sessions_from_files(features_train,
                               actions_path="./artifacts/actions.json",
                               patterns_path="./artifacts/patterns.txt",
                               navigateurs_path="./artifacts/browsers.txt"):
    """
    Encode les sessions en utilisant directement les fichiers de mapping existants.
    Chaque action est associée à une liste de patterns (même si plusieurs patterns se suivent).
    Renvoie une liste de dictionnaires : user, navigateur, actions_patterns.
    """

    actions_to_id = load_actions(actions_path)
    pattern_to_id = load_mapping(patterns_path)
    navigateurs_to_id = load_mapping(navigateurs_path)

    pattern_parentheses = re.compile(r"\((.*?)\)")
    pattern_chevrons = re.compile(r"<(.*?)>")
    pattern_dollar = re.compile(r"\$(.*?)\$")

    encoded_rows = []

    for _, row in features_train.iterrows():
        user_id = row["util"]  # garder tel quel
        navigateur_id = navigateurs_to_id.get(row["navigateur"], 0)
        actions_encoded = []

        for val in row.iloc[2:].dropna().astype(str):
            # Identifier l'action
            action_name = None
            for act in actions_to_id:
                if val.startswith(act):
                    action_name = act
                    break
            if not action_name:
                continue

            # Extraire tous les patterns présents dans la même cellule
            patterns = (
                pattern_parentheses.findall(val) +
                pattern_chevrons.findall(val) +
                pattern_dollar.findall(val)
            )
            # Convertir patterns en leurs IDs et retirer ceux qui n'existent pas
            pattern_ids = [pattern_to_id[p] for p in patterns if p in pattern_to_id]

            if pattern_ids:
                actions_encoded.append([actions_to_id[action_name]] + pattern_ids)
            else:
                actions_encoded.append([actions_to_id[action_name]])

        encoded_rows.append({
            "user": user_id,
            "navigateur": navigateur_id,
            "actions_patterns": actions_encoded
        })

    return encoded_rows