"""Chargement des données CAMELS-FR et définition d'une crue."""
from pathlib import Path

import pandas as pd

DOSSIER_DATA = Path(__file__).resolve().parents[1] / "data"

# Découpage chronologique : on apprend sur le passé, on teste sur le futur.
FIN_ENTRAINEMENT = "2010-12-31"
FIN_VALIDATION = "2014-12-31"   # test : 2015-2018


def charger_series(dossier: Path = DOSSIER_DATA) -> pd.DataFrame:
    """Séries journalières : pluie, température, ETP et débit (en mm/jour) par station."""
    df = pd.read_csv(dossier / "camels_fr_extrait.csv", dtype={"code": str}, parse_dates=["date"])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def charger_stations(dossier: Path = DOSSIER_DATA) -> pd.DataFrame:
    """Une ligne par station : nom, coordonnées, surface du bassin, altitudes."""
    return pd.read_csv(dossier / "stations.csv", dtype={"code": str})


def seuils_de_crue(df: pd.DataFrame, fin_entrainement: str = FIN_ENTRAINEMENT) -> pd.Series:
    """Seuil de crue de chaque station = crue journalière biennale.

    On prend le débit journalier maximal de chaque année, puis la médiane de ces
    maxima : c'est le débit dépassé en moyenne une année sur deux, un ordre de
    grandeur classique du débordement d'une rivière. Calculé uniquement sur la
    période d'entraînement pour ne pas tricher avec le futur.
    """
    passe = df[df["date"] <= fin_entrainement]
    maxima = passe.groupby(["code", passe["date"].dt.year])["debit_mm"].max()
    return maxima.groupby(level="code").median().rename("seuil_crue")


def periode(dates: pd.Series) -> pd.Series:
    """Étiquette entrainement / validation / test selon la date."""
    out = pd.Series("test", index=dates.index)
    out[dates <= FIN_VALIDATION] = "validation"
    out[dates <= FIN_ENTRAINEMENT] = "entrainement"
    return out


def episodes_de_crue(df: pd.DataFrame) -> pd.DataFrame:
    """Liste des épisodes : suites de jours consécutifs au-dessus du seuil."""
    lignes = []
    for code, g in df.groupby("code"):
        en_crue = (g["debit_mm"] > g["seuil_crue"]).to_numpy()
        dates = g["date"].to_numpy()
        debut = None
        for i, c in enumerate(en_crue):
            if c and debut is None:
                debut = i
            if (not c or i == len(en_crue) - 1) and debut is not None:
                fin = i if c else i - 1
                pic = g["debit_mm"].iloc[debut:fin + 1]
                lignes.append({"code": code, "debut": dates[debut], "fin": dates[fin],
                               "duree_j": fin - debut + 1,
                               "pic_relatif": pic.max() / g["seuil_crue"].iloc[0]})
                debut = None
    return pd.DataFrame(lignes)
