"""Tests rapides : python -m pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.donnees import charger_series, charger_stations, seuils_de_crue, periode
from src.variables import construire_table, prevision_bruitee, _modele_neige, AVEC_PREVISION, SANS_PREVISION
from src.modeles import CIBLE, RegleNaive, choisir_seuil_alerte
from src.evaluation import scores, evaluer_episodes


def _petit_jeu():
    df = charger_series()
    codes = df["code"].unique()[:2]
    df = df[df["code"].isin(codes)]
    st = charger_stations()
    return df, st, seuils_de_crue(df)


def test_donnees():
    df, st, seuils = _petit_jeu()
    assert set(df.columns) >= {"code", "date", "pluie_mm", "temp_c", "etp_mm", "debit_mm"}
    assert len(charger_stations()) == 19
    assert (seuils > 0).all()


def test_pas_de_fuite_du_futur():
    """Les variables du jour J ne doivent pas changer si on modifie le débit et la pluie après J."""
    df, st, seuils = _petit_jeu()
    t1 = construire_table(df, st, seuils)
    df2 = df.copy()
    coupure = pd.Timestamp("2010-06-01")
    df2.loc[df2["date"] > coupure, ["pluie_mm", "debit_mm"]] *= 5
    t2 = construire_table(df2, st, seuils)
    jour = (t1["date"] == coupure - pd.Timedelta(days=5))
    pd.testing.assert_frame_equal(t1.loc[jour, SANS_PREVISION], t2.loc[jour, SANS_PREVISION])


def test_cible():
    df, st, seuils = _petit_jeu()
    t = construire_table(df, st, seuils)
    g = t[t["code"] == t["code"].iloc[0]].reset_index(drop=True)
    q = g["q_rel"]
    attendu = (pd.concat([q.shift(-k) for k in (1, 2, 3)], axis=1).max(axis=1) > 1).astype(float)
    ok = attendu.notna() & g["date"].diff().shift(-3).notna()
    assert (g.loc[ok, CIBLE].iloc[:-3] == attendu[ok].iloc[:-3]).all()
    assert t[AVEC_PREVISION].shape[1] == len(AVEC_PREVISION)


def test_neige():
    stock, fonte = _modele_neige(np.array([10.0, 0, 0]), np.array([-5.0, 2, 10]))
    assert stock.tolist() == [10.0, 4.0, 0.0]
    assert fonte.tolist() == [0.0, 6.0, 4.0]


def test_prevision_bruitee_conserve_le_passe():
    df, st, seuils = _petit_jeu()
    t = construire_table(df, st, seuils)
    tb = prevision_bruitee(t)
    pd.testing.assert_frame_equal(t[SANS_PREVISION], tb[SANS_PREVISION])
    assert not np.allclose(t["pluie_prevue_3j"].fillna(0), tb["pluie_prevue_3j"].fillna(0))


def test_scores_et_episodes():
    df, st, seuils = _petit_jeu()
    t = construire_table(df, st, seuils)
    t["periode"] = periode(t["date"])
    p = RegleNaive().predict_proba(t)[:, 1]
    s = scores(t[CIBLE], p, 0.8)
    assert 0 <= s["POD"] <= 1 and s["PR-AUC"] > t[CIBLE].mean()
    assert 0.02 <= choisir_seuil_alerte(t[CIBLE], p) <= 0.9
    ep = evaluer_episodes(t, p, 0.8)
    assert set(ep["avance_j"].unique()) <= {0, 1, 2, 3}
