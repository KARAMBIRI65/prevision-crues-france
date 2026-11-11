# Données

| Fichier | Contenu |
|---|---|
| `camels_fr_extrait.csv` | une ligne par station et par jour, 1999-2018 : `pluie_mm`, `temp_c`, `etp_mm` (évapotranspiration potentielle), `debit_mm` (débit en mm/jour = volume écoulé / surface du bassin) |
| `stations.csv` | une ligne par station : codes Hydroportail, nom, coordonnées de l'exutoire, surface (km²), altitudes min / médiane / max du bassin |

**Source :** extrait de la base **CAMELS-FR** (INRAE), diffusé dans le paquet R
[`airGRdatasets`](https://cran.r-project.org/package=airGRdatasets), licence **CC BY 4.0**.

- Pluie, température et ETP : réanalyse **SAFRAN** de Météo-France (maille de 8 km), moyennée sur chaque bassin.
- Débits : **SCHAPI**, via le site Hydroportail.

Référence : Delaigue, O., Guimarães, G. M., Brigode, P., Génot, B., Perrin, C., Andréassian, V. (2024).
*CAMELS-FR dataset*, Recherche Data Gouv, [doi:10.57745/WH7FJR](https://doi.org/10.57745/WH7FJR).

Pour refaire l'extraction : `python scripts/extraire_donnees.py` (voir l'en-tête du script).
