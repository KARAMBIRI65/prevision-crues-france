"""Reconstruit data/camels_fr_extrait.csv et data/stations.csv à partir du paquet airGRdatasets.

Les données viennent de CAMELS-FR (INRAE), diffusées sous licence CC BY 4.0
dans le paquet R airGRdatasets. Ce script n'est utile que pour refaire
l'extraction : les CSV sont déjà fournis dans data/.

    git clone --depth 1 https://github.com/cran/airGRdatasets.git
    pip install rdata
    python scripts/extraire_donnees.py airGRdatasets/data
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

RACINE = Path(__file__).resolve().parents[1]


def main(dossier_rda: str) -> None:
    series, stations = [], []
    for fichier in sorted(Path(dossier_rda).glob("*.rda")):
        bassin = list(rdata.read_rda(str(fichier)).values())[0]
        meta, ts = bassin["Meta"], bassin["TS"].copy()
        code = str(meta["Code"]["H3"][0])
        hypso = np.asarray(bassin["Hypso"])
        stations.append({
            "code": code,
            "code_h2": str(meta["Code"]["H2"][0]),
            "nom": str(meta["Name"][0]),
            "lon": float(meta["Coor"]["X"][0]),
            "lat": float(meta["Coor"]["Y"][0]),
            "surface_km2": float(meta["Area"][0]),
            "alt_min": hypso[0], "alt_med": hypso[50], "alt_max": hypso[100],
        })
        ts["date"] = pd.to_datetime(ts["Date"], unit="s").dt.strftime("%Y-%m-%d")
        ts["code"] = code
        series.append(ts[["code", "date", "Ptot", "Temp", "Evap", "Qmmd"]])

    df = pd.concat(series).rename(columns={
        "Ptot": "pluie_mm", "Temp": "temp_c", "Evap": "etp_mm", "Qmmd": "debit_mm"})
    df.to_csv(RACINE / "data" / "camels_fr_extrait.csv", index=False, float_format="%.3f")
    st = pd.DataFrame(stations)
    st["nom"] = (st["nom"].str.replace(r" \[.*?\]", "", regex=True)
                 .str.replace(r" - .*$", "", regex=True))
    st.round(5).to_csv(RACINE / "data" / "stations.csv", index=False)
    print(f"{len(st)} stations, {len(df)} lignes")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "airGRdatasets/data")
