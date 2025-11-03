import pandas as pd
import os
import csv

def reader(ds_name: str, training=True) -> pd.DataFrame:
    """Lecteur CSV robuste pour des fichiers dont les lignes ont un nombre de champs variable.
    Utilise ``csv.reader`` pour parser les lignes, complète chaque ligne jusqu'au
    maximum de colonnes observé avec des chaînes vides, puis renvoie un
    ``pandas.DataFrame`` où les valeurs manquantes sont remplacées par des chaînes vides.

    - Accepte indifféremment « train » ou « train.csv » (même logique pour tout nom fourni).
    - Le paramètre ``training`` indique la présence d'une colonne identifiant l'utilisateur.

    Arguments:
        ds_name: Nom du fichier (avec ou sans l'extension « .csv »).
        training: Si True, on s'attend à un schéma « utilisateur, navigateur, actions… ».
                  Si False, on s'attend à « navigateur, actions… » (pas d'utilisateur).

    Retourne:
        Un DataFrame avec des colonnes renommées et des séquences d'actions
        alignées sur la même largeur (remplissage par chaînes vides).
    """
    filename = ds_name if ds_name.endswith('.csv') else ds_name + '.csv'
    path = os.path.join('data', filename)
    # Lecture via csv.reader pour éviter les erreurs de tokenisation du moteur C de pandas
    # lorsque certaines lignes sont mal formées (nombre de champs variable)
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    if not rows:
        print(f"No rows read from {path}")
        return pd.DataFrame()

    # Compléter les cellules manquantes pour en faire un tableau régulier et
    # ajouter la longueur de séquence (nombre d'actions)
    max_cols = max(len(r) for r in rows)
    if training:
        # ordre: utilisateur, navigateur, longueur de séquence d'actions, séquence, remplissages
        padded = [[r[0]] + [r[1]] + [len(r) - 2] + r[2:] + [''] * (max_cols - len(r)) for r in rows] 
    else:
        # ordre: navigateur, longueur de séquence d'actions, séquence, remplissages
        padded = [[r[0]] + [len(r) - 1] + r[1:] + [''] * (max_cols - len(r)) for r in rows]
    df = pd.DataFrame(padded)
    df = df.fillna('')

    # Renommer les colonnes selon le mode (entraînement ou test)
    if training:
        rename_map = {col: ('util' if col == 0 else 'browser' if col == 1 else 'sequence_length' if col == 2  else f'action_{col-2}') for col in df.columns}
    else:
        rename_map = {col: ('browser' if col == 0 else 'sequence_length' if col == 1  else f'action_{col-1}') for col in df.columns}
    df.rename(columns=rename_map, inplace=True)
    return df