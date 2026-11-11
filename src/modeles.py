"""Les modèles comparés et le choix du seuil d'alerte."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CIBLE = "crue_a_venir"


class RegleNaive:
    """Référence sans apprentissage : plus la rivière est haute aujourd'hui, plus le risque est fort."""

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        score = np.clip(X["q_rel"].fillna(0).to_numpy(), 0, 1)
        return np.column_stack([1 - score, score])


def regression_logistique():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=2000))


def foret_aleatoire(graine: int = 0):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                max_features=0.4, n_jobs=-1, random_state=graine))


def xgboost(n_arbres: int = 1500, graine: int = 0, **kwargs):
    params = dict(n_estimators=n_arbres, learning_rate=0.03, max_depth=5, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, eval_metric="aucpr",
                  tree_method="hist", n_jobs=-1, random_state=graine)
    params.update(kwargs)
    return XGBClassifier(**params)


def entrainer_xgboost(X_train, y_train, X_val, y_val, graine: int = 0):
    """XGBoost avec arrêt anticipé sur la validation (on garde le bon nombre d'arbres)."""
    modele = xgboost(graine=graine, early_stopping_rounds=100)
    modele.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return modele


def proba(modele, X) -> np.ndarray:
    return modele.predict_proba(X)[:, 1]


def choisir_seuil_alerte(y_val, p_val, beta: float = 2.0) -> float:
    """Seuil de probabilité qui maximise le F-beta sur la validation.

    beta = 2 : rater une crue coûte plus cher qu'une fausse alerte,
    on accorde donc deux fois plus d'importance au rappel qu'à la précision.
    """
    grille = np.linspace(0.02, 0.9, 89)
    scores = [fbeta_score(y_val, p_val >= s, beta=beta, zero_division=0) for s in grille]
    return float(grille[int(np.argmax(scores))])
