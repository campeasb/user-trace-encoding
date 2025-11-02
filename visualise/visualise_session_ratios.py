import matplotlib.pyplot as plt
import re
import math

def plot_progress_by_browser(df, n_users_per_browser):
    """
    Affiche l'évolution du nombre d'actions en fonction de l'avancement
    séparément pour chaque navigateur.
    """
    browsers = df["navigateur"].unique()
    n = len(browsers)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 8))
    axes = axes.flatten()

    for i, browser in enumerate(browsers):
        ax = axes[i]
        df_browser = df[df["navigateur"] == browser]
        
        # On limite à n_users_per_browser pour éviter un affichage surchargé
        sampled_df = df_browser.sample(min(n_users_per_browser, len(df_browser)), random_state=0)
        
        for _, row in sampled_df.iterrows():
            user = row["util"]
            actions_and_progress = row.drop(["util", "navigateur"])
            progresses = []
            action_counts = []
            action_count = 0

            for val in actions_and_progress:
                val = str(val)
                if re.match(r"t\d+", val):
                    progresses.append(int(val[1:]))
                    action_counts.append(action_count)
                elif val.strip() != "" and val.lower() != "nan":
                    action_count += 1

            if progresses:
                ax.plot(progresses, action_counts, marker="o", label=f"{user}", linewidth=0.2)

        ax.set_title(browser)
        ax.set_xlabel("Avancement (secondes)")
        ax.set_ylabel("Nombre d'actions")
        ax.grid(True)
        if len(sampled_df) <= 10:
            ax.legend(fontsize="small", loc="upper left")

    # Cache les axes inutilisés (si <4 navigateurs)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Évolution du nombre d'actions par navigateur", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


import pandas as pd
import re

def compute_session_ratios(df):
    """
    Pour chaque ligne (session) du CSV, calcule :
    - numéro de session
    - utilisateur
    - navigateur
    - nombre total d'actions
    - avancement maximal (tmax)
    - ratio actions / tmax
    """
    results = []

    for idx, row in df.iterrows():
        user = row["util"]
        navigateur = row["navigateur"]
        actions_and_progress = row.drop(["util", "navigateur"])
        action_count = 0
        tmax = 0

        for val in actions_and_progress:
            val = str(val)
            if re.match(r"t\d+", val):
                t = int(val[1:])
                tmax = max(tmax, t)
            elif val.strip() != "" and val.lower() != "nan":
                action_count += 1

        ratio = tmax / action_count if action_count > 0 else 0

        results.append({
            "session_id": idx + 1,
            "util": user,
            "navigateur": navigateur,
            "total_actions": action_count,
            "tmax": tmax,
            "ratio": ratio
        })

    df_sessions = pd.DataFrame(results)
    df_sessions = df_sessions.sort_values(by='total_actions')

    # Affichage du tableau complet
    print(df_sessions.head())  # afficher les 20 premières lignes pour contrôle
    return df_sessions    