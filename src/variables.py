"""Construction des variables explicatives et de la cible.

Toutes les variables d'une journée J n'utilisent que l'information connue le
soir de J, sauf la famille « pluie prévue » qui simule une prévision météo.
"""
import numpy as np
import pandas as pd

HORIZON = 3  # on veut savoir si la rivière sera en crue dans les 3 prochains jours

PLUIE_PASSEE = ["pluie_1j", "pluie_3j", "pluie_7j", "pluie_14j", "pluie_30j",
                "pluie_max_3j", "indice_pluie_anterieure"]
SOL = ["bilan_30j", "bilan_90j"]
NEIGE = ["stock_neige", "fonte_3j", "temp_c", "temp_3j"]
DEBIT = ["q_rel", "q_rel_hier", "var_q_1j", "var_q_3j", "q_rel_max_7j", "q_rel_moy_30j"]
SAISON = ["mois_sin", "mois_cos"]
BASSIN = ["log_surface", "alt_med", "denivele"]
PREVISION = ["pluie_prevue_3j", "pluie_prevue_max", "temp_prevue_3j"]

FAMILLES = {
    "Pluie passée": PLUIE_PASSEE,
    "Humidité des sols": SOL,
    "Neige et température": NEIGE,
    "Débit actuel": DEBIT,
    "Saison": SAISON,
    "Caractéristiques du bassin": BASSIN,
    "Pluie prévue": PREVISION,
}
SANS_PREVISION = PLUIE_PASSEE + SOL + NEIGE + DEBIT + SAISON + BASSIN
AVEC_PREVISION = SANS_PREVISION + PREVISION


def _modele_neige(pluie: np.ndarray, temp: np.ndarray, coef_fonte: float = 3.0):
    """Modèle degré-jour : la pluie tombe en neige sous 0 °C et fond de 3 mm par °C positif."""
    stock = np.zeros(len(pluie))
    fonte = np.zeros(len(pluie))
    s = 0.0
    for i, (p, t) in enumerate(zip(pluie, temp)):
        if t <= 0:
            s += p
        f = min(s, coef_fonte * max(t, 0.0))
        s -= f
        stock[i], fonte[i] = s, f
    return stock, fonte


def _une_station(g: pd.DataFrame, seuil: float) -> pd.DataFrame:
    g = g.copy()
    p, q = g["pluie_mm"], g["debit_mm"]

    for n in (1, 3, 7, 14, 30):
        g[f"pluie_{n}j"] = p.rolling(n, min_periods=1).sum()
    g["pluie_max_3j"] = p.rolling(3, min_periods=1).max()
    # pluies anciennes pondérées de moins en moins (facteur 0,9 par jour)
    g["indice_pluie_anterieure"] = p.ewm(alpha=0.1, adjust=False).mean() * 10
    g["bilan_30j"] = (p - g["etp_mm"]).rolling(30, min_periods=1).sum()
    g["bilan_90j"] = (p - g["etp_mm"]).rolling(90, min_periods=1).sum()

    stock, fonte = _modele_neige(p.to_numpy(), g["temp_c"].to_numpy())
    g["stock_neige"] = stock
    g["fonte_3j"] = pd.Series(fonte, index=g.index).rolling(3, min_periods=1).sum()
    g["temp_3j"] = g["temp_c"].rolling(3, min_periods=1).mean()

    # le débit est divisé par le seuil : 1 = niveau de crue, quel que soit le bassin
    qr = q / seuil
    g["q_rel"] = qr
    g["q_rel_hier"] = qr.shift(1)
    g["var_q_1j"] = qr - qr.shift(1)
    g["var_q_3j"] = qr - qr.shift(3)
    g["q_rel_max_7j"] = qr.rolling(7, min_periods=1).max()
    g["q_rel_moy_30j"] = qr.rolling(30, min_periods=10).mean()

    # prévision météo « parfaite » : la pluie réellement tombée les 3 jours suivants
    futur_p = pd.concat([p.shift(-k) for k in range(1, HORIZON + 1)], axis=1)
    futur_t = pd.concat([g["temp_c"].shift(-k) for k in range(1, HORIZON + 1)], axis=1)
    g["pluie_prevue_3j"] = futur_p.sum(axis=1, min_count=HORIZON)
    g["pluie_prevue_max"] = futur_p.max(axis=1)
    g["temp_prevue_3j"] = futur_t.mean(axis=1)

    # cible : le débit dépasse-t-il le seuil au moins un jour entre J+1 et J+3 ?
    futur_q = pd.concat([q.shift(-k) for k in range(1, HORIZON + 1)], axis=1)
    maxi = futur_q.max(axis=1, skipna=False)
    g["crue_a_venir"] = (maxi > seuil).astype(float).where(maxi.notna())
    g["en_crue_aujourdhui"] = (q > seuil).astype(float).where(q.notna())
    return g


def construire_table(df: pd.DataFrame, stations: pd.DataFrame, seuils: pd.Series,
                     echauffement_j: int = 90) -> pd.DataFrame:
    """Table d'apprentissage : une ligne par station et par jour."""
    morceaux = []
    for code, g in df.groupby("code"):
        t = _une_station(g, seuils[code])
        morceaux.append(t.iloc[echauffement_j:])   # on laisse le temps aux cumuls de se remplir
    table = pd.concat(morceaux, ignore_index=True)

    mois = table["date"].dt.dayofyear / 365.25 * 2 * np.pi
    table["mois_sin"], table["mois_cos"] = np.sin(mois), np.cos(mois)

    st = stations.set_index("code")
    table["log_surface"] = np.log10(table["code"].map(st["surface_km2"]))
    table["alt_med"] = table["code"].map(st["alt_med"])
    table["denivele"] = table["code"].map(st["alt_max"] - st["alt_min"])
    table["seuil_crue"] = table["code"].map(seuils)
    return table.dropna(subset=["crue_a_venir"]).reset_index(drop=True)


def prevision_bruitee(table: pd.DataFrame, erreur: float = 0.6, graine: int = 0) -> pd.DataFrame:
    """Dégrade la prévision parfaite pour imiter une vraie prévision météo.

    Chaque cumul prévu est multiplié par un facteur aléatoire log-normal
    (erreur = écart-type du log, 0,6 ≈ une prévision souvent fausse de 50 %),
    et on ajoute une petite erreur de 1,5 °C sur la température prévue.
    """
    rng = np.random.default_rng(graine)
    t = table.copy()
    facteur = np.exp(rng.normal(-erreur ** 2 / 2, erreur, len(t)))
    t["pluie_prevue_3j"] = t["pluie_prevue_3j"] * facteur
    t["pluie_prevue_max"] = t["pluie_prevue_max"] * facteur
    t["temp_prevue_3j"] = t["temp_prevue_3j"] + rng.normal(0, 1.5, len(t))
    return t
