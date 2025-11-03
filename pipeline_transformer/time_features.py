import pandas as pd
import re

def compute_time_features(actions: pd.DataFrame, sequence_lengths: pd.DataFrame):
    durations = []
    speeds = []

    # Parcourir chaque séquence d'actions pour récupérer la dernière valeur de temps
    for sequence_idx in range(len(actions)):
        action = actions.iloc[sequence_idx]
        length = sequence_lengths.iloc[sequence_idx]
        last_time = 5

        # Parcourir la séquence à rebours pour trouver la dernière valeur de temps
        for i in range(length-1, 0, -1):
            if re.fullmatch(r'^t\d+$', action.iloc[i]):
                last_time_string = action.iloc[i]
                last_time = int(last_time_string[1:])
                break

        # Calculer la vitesse de session
        n_time_cells = last_time/5 
        n_actions = length - n_time_cells
        speed = round(n_actions / last_time, 6)

        # Stocker les valeurs calculées
        durations.append(last_time)
        speeds.append(speed)

    return pd.DataFrame({'duration': durations, 'speed': speeds})


def bucketize_time_features(time_features: pd.DataFrame, n_buckets: int = 8):
    """Discrétiser les variables temporelles (« duration », « speed ») en classes (buckets).

    Cette fonction applique un découpage par quantiles (``pd.qcut``) séparément à la
    durée et à la vitesse, afin d'obtenir des classes de tailles comparables même en
    présence d'asymétrie dans la distribution.

    Arguments:
        time_features: DataFrame contenant les colonnes « duration » et « speed ».
        n_buckets: Nombre de classes à créer pour chaque variable (par défaut: 8).

    Retourne:
        Un DataFrame avec deux colonnes « duration_bucket » et « speed_bucket »
        contenant les indices de classe (entiers). Les valeurs manquantes éventuelles
        sont remplacées par 0, et les classes sont converties en entiers.
    """
    result = pd.DataFrame()
    
    # Discrétiser la durée avec un découpage par quantiles (robuste aux distributions asymétriques)
    result['duration_bucket'] = pd.qcut(
        time_features['duration'], 
        q=n_buckets, 
        labels=False, 
        duplicates='drop'
    )
    
    # Discrétiser la vitesse avec un découpage par quantiles
    result['speed_bucket'] = pd.qcut(
        time_features['speed'], 
        q=n_buckets, 
        labels=False, 
        duplicates='drop'
    )
    
    # Remplir les NaN (ne devrait pas arriver avec qcut, mais par sécurité)
    result = result.fillna(0).astype(int)
    
    return result