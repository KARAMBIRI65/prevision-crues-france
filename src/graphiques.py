"""Style commun des figures."""
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
GRIS = "#898781"
GRILLE = "#e1e0d9"
CRUE = "#d03b3b"          # rouge réservé à « crue »
PLUIE = "#86b6ef"

# une couleur fixe par modèle (identité), la règle naïve en gris neutre
COULEURS_MODELES = {
    "Règle naïve": GRIS,
    "Régression logistique": "#2a78d6",
    "Forêt aléatoire": "#eb6834",
    "XGBoost": "#1baf7a",
}
# du moins au plus d'information météo : un seul bleu, du clair au foncé
COULEURS_SCENARIOS = {
    "Sans prévision météo": "#86b6ef",
    "Prévision réaliste": "#2a78d6",
    "Prévision parfaite": "#104281",
}


def appliquer_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": ENCRE_2, "axes.titlecolor": ENCRE,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRILLE, "grid.linewidth": 0.6,
        "xtick.color": GRIS, "ytick.color": GRIS, "text.color": ENCRE,
        "legend.frameon": False, "font.size": 10, "lines.linewidth": 2,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })
