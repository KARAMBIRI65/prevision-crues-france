"""Mesures de performance adaptées aux événements rares."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def scores(y, p, seuil: float) -> dict:
    """Scores probabilistes + scores de l'alerte (proba >= seuil).

    - PR-AUC : aire sous la courbe précision-rappel, la bonne mesure quand
      l'événement est rare (une prédiction au hasard vaut le taux de crue).
    - POD (probabilité de détection) : part des crues qui ont été annoncées.
    - FAR (taux de fausses alertes) : part des alertes qui n'ont pas été suivies d'une crue.
    - CSI (indice de succès critique) : score de référence des prévisionnistes.
    """
    y = np.asarray(y).astype(int)
    alerte = np.asarray(p) >= seuil
    vp = int((alerte & (y == 1)).sum())
    fp = int((alerte & (y == 0)).sum())
    fn = int((~alerte & (y == 1)).sum())
    return {
        "PR-AUC": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        "Brier": brier_score_loss(y, p),
        "POD": vp / max(vp + fn, 1),
        "FAR": fp / max(vp + fp, 1),
        "CSI": vp / max(vp + fp + fn, 1),
    }


def evaluer_episodes(table: pd.DataFrame, p: np.ndarray, seuil: float, horizon: int = 3) -> pd.DataFrame:
    """Évaluation par épisode, comme le vit un maire ou un service de secours.

    Pour chaque épisode de crue (premier jour au-dessus du seuil), on regarde si
    une alerte a été lancée la veille ou jusqu'à `horizon` jours avant, et avec
    combien de jours d'avance au maximum.
    """
    t = table[["code", "date", "en_crue_aujourdhui"]].copy()
    t["alerte"] = np.asarray(p) >= seuil
    lignes = []
    for code, g in t.groupby("code"):
        g = g.reset_index(drop=True)
        crue = g["en_crue_aujourdhui"].fillna(0).to_numpy() == 1
        debuts = np.where(crue & ~np.r_[False, crue[:-1]])[0]
        for d in debuts:
            if d < horizon:
                continue
            fenetre = g["alerte"].to_numpy()[d - horizon:d]   # alertes de J-3 à J-1
            avance = horizon - int(np.argmax(fenetre)) if fenetre.any() else 0
            lignes.append({"code": code, "debut": g.loc[d, "date"],
                           "anticipe": bool(fenetre.any()), "avance_j": avance})
    return pd.DataFrame(lignes)


def importance_par_famille(modele, X: pd.DataFrame, y, familles: dict, n_rep: int = 3, graine: int = 0) -> pd.Series:
    """Perte de PR-AUC quand on mélange toutes les variables d'une famille.

    Mélanger une famille entière (plutôt qu'une variable à la fois) évite que des
    variables très corrélées, comme les cumuls de pluie, se « cachent » l'une l'autre.
    """
    rng = np.random.default_rng(graine)
    base = average_precision_score(y, modele.predict_proba(X)[:, 1])
    pertes = {}
    for nom, cols in familles.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        valeurs = []
        for _ in range(n_rep):
            Xp = X.copy()
            ordre = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[ordre]
            valeurs.append(base - average_precision_score(y, modele.predict_proba(Xp)[:, 1]))
        pertes[nom] = float(np.mean(valeurs))
    return pd.Series(pertes).sort_values(ascending=False)
