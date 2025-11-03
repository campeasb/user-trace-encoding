from .parser import parse_action_string
import re
import pandas as pd

def tokenize_action_sequence(actions: pd.DataFrame, existing_token_to_idx: dict = None, training = True):
    """Tokeniser des séquences d'actions parsées pour l'entrée du transformer.

    Arguments:
        actions: DataFrame où chaque ligne est une session et chaque cellule une action (ou un pas de temps).
        existing_token_to_idx: Dictionnaire existant « token -> id » à réutiliser en mode inférence.
        training: Si True, les nouveaux tokens vus sont ajoutés au vocabulaire; sinon, ils sont ignorés.

    Retourne:
        all_tokens: Liste de listes d'identifiants de tokens par session.
        token_to_idx: Dictionnaire « token -> id » final (créé ou mis à jour selon le mode).
    """
    all_tokens = []
    TIMESTEP_PATTERN = re.compile(r'^t\d+$')

    if training:
        token_to_idx = {}
    else:
        token_to_idx = existing_token_to_idx

    idx_counter = len(token_to_idx) + 1
    
    for session in actions.itertuples(index=False, name=None):
        sequence_tokens = []
        for action in session:
            if isinstance(action, str) and TIMESTEP_PATTERN.match(action.strip()): # ignorer les tokens de pas de temps
                continue

            if isinstance(action, str) and action == "": # action vide
                continue
            
            # Parser l'action en tuple
            parsed = parse_action_string(action)
            
            # Convertir le tuple en chaîne pour l'indexation/vocabulaire
            token_str = str(parsed)
            
            # Ajouter au vocabulaire si nouveau
            if token_str not in token_to_idx:
                if training:
                    token_to_idx[token_str] = idx_counter
                    idx_counter += 1
                else:
                    continue # ne pas ajouter un token inconnu en mode inférence
            
            sequence_tokens.append(token_to_idx[token_str])
        
        all_tokens.append(sequence_tokens)
    
    return all_tokens, token_to_idx

def tokenize_browser_data(browsers: pd.Series, existing_browser_to_idx: dict = None, training = True):
    """Tokeniser les navigateurs pour l'entrée du transformer.

    Arguments:
        browsers: Série pandas contenant les navigateurs par session.
        existing_browser_to_idx: Dictionnaire existant « navigateur -> id » à réutiliser en inférence.
        training: Si True, ajoute les nouveaux navigateurs au vocabulaire; sinon, réutilise uniquement l'existant.

    Retourne:
        browser_tokens: Liste des identifiants de navigateurs par session.
        browser_to_idx: Dictionnaire « navigateur -> id » final (créé ou mis à jour selon le mode).
    """
    browser_tokens = []
    
    if training:
        browser_to_idx = {}
    else:
        browser_to_idx = existing_browser_to_idx

    idx_counter = len(browser_to_idx)
    
    for browser in browsers:
        # Convertir en chaîne et gérer les valeurs manquantes
        browser_str = str(browser) if pd.notna(browser) and browser != '' else 'unknown'
        
        # Ajouter au vocabulaire si nouveau
        if browser_str not in browser_to_idx:
            browser_to_idx[browser_str] = idx_counter
            idx_counter += 1
        
        browser_tokens.append(browser_to_idx[browser_str])
    
    return browser_tokens, browser_to_idx

def tokenize_username_data(usernames: pd.Series):
    """Tokeniser les identifiants utilisateurs pour l'entraînement du transformer.

    Arguments:
        usernames: Série pandas contenant les identifiants utilisateurs.

    Retourne:
        username_tokens: Liste des identifiants de tokens utilisateurs par session.
        username_to_idx: Dictionnaire « utilisateur -> id » construit à partir des données.
    """
    username_tokens = []
    username_to_idx = {}
    idx_counter = 0
    
    for username in usernames:
        # Convertir en chaîne et gérer les valeurs manquantes
        username_str = str(username) if pd.notna(username) and username != '' else 'unknown'
        
        # Ajouter au vocabulaire si nouveau
        if username_str not in username_to_idx:
            username_to_idx[username_str] = idx_counter
            idx_counter += 1
        
        username_tokens.append(username_to_idx[username_str])
    
    return username_tokens, username_to_idx