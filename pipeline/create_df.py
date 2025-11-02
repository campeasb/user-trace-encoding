import pandas as pd

def read_ds(ds_name: str):
    # chemin complet du fichier CSV
    path = f"./data/{ds_name}.csv"
    
    # ouvrir le fichier et lire les lignes
    with open(path, "r", encoding="utf-8") as f:
        lignes = f.readlines()
    
    data = []
    for ligne in lignes:
        # séparer les colonnes CSV
        elements = ligne.strip().split(",")
        # remplacer les cellules vides par ""
        elements = [e if e != "" else "" for e in elements]
        data.append(elements)
    
    # trouver le nombre max de colonnes
    max_cols = max(len(ligne) for ligne in data)
    columns = ["util", "navigateur"] + [f"col{i}" for i in range(3, max_cols + 1)]
    
    # créer le DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df