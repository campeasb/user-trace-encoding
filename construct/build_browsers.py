def extract_and_save_browsers(features_train, output_path="./artifacts/browsers.txt"):
    """
    Extrait tous les navigateurs distincts du DataFrame,
    leur associe un identifiant numérique et les enregistre dans un fichier texte.

    Args:
        features_train (pd.DataFrame): jeu de données contenant une colonne 'navigateur'
        output_path (str): chemin du fichier texte où sauvegarder le mapping
    """

    if "navigateur" not in features_train.columns:
        raise ValueError("La colonne 'navigateur' est absente du DataFrame.")

    # --- 1. Extraire les navigateurs uniques
    unique_nav = sorted(features_train["navigateur"].dropna().unique())

    # --- 2. Créer le dictionnaire {navigateur: id}
    navigateur_to_id = {nav: i + 1 for i, nav in enumerate(unique_nav)}

    # --- 3. Sauvegarder dans un fichier texte
    with open(output_path, "w", encoding="utf-8") as f:
        for nav, idx in navigateur_to_id.items():
            f.write(f"{nav}: {idx}\n")

    print(f"{len(navigateur_to_id)} navigateurs enregistrés dans '{output_path}'")

    # --- 4. Retourner le dictionnaire
    return navigateur_to_id

