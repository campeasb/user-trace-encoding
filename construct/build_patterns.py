import re

def extract_and_save_patterns(features_train, output_path="./artifacts/patterns.txt"):
    """
    Extrait tous les patterns du DataFrame ((), <>, $$),
    leur associe un identifiant numérique et les enregistre dans un fichier texte.

    Args:
        features_train (pd.DataFrame): jeu de données contenant les actions et les patterns.
        output_path (str): chemin du fichier texte où sauvegarder le mapping.
    """
    
    # --- 1. Définir les regex pour extraire les patterns
    pattern_parentheses = re.compile(r"\((.*?)\)")
    pattern_chevrons = re.compile(r"<(.*?)>")
    pattern_dollar = re.compile(r"\$(.*?)\$")

    # --- 2. Extraire toutes les occurrences
    all_patterns = []

    for col in features_train.columns:
        for val in features_train[col].dropna().astype(str):
            all_patterns += pattern_parentheses.findall(val)
            all_patterns += pattern_chevrons.findall(val)
            all_patterns += pattern_dollar.findall(val)

    # --- 3. Supprimer doublons et trier
    unique_patterns = sorted(set(all_patterns))

    # --- 4. Créer le dictionnaire {pattern: id}
    pattern_to_id = {pat: i + 1 for i, pat in enumerate(unique_patterns)}

    # --- 5. Sauvegarder dans un fichier texte
    with open(output_path, "w", encoding="utf-8") as f:
        for pat, idx in pattern_to_id.items():
            f.write(f"{pat}: {idx}\n")

    print(f"{len(pattern_to_id)} patterns uniques enregistrés dans '{output_path}'")

    # --- 6. Retourner le dictionnaire pour réutilisation éventuelle
    return pattern_to_id