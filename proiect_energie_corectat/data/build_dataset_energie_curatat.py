"""
Construiește setul de date curățat pentru EDA & modele:
data/dataset_energie_curatat.csv

Pornește de la:
    data/dataset_complet_energie.csv  (din PAAD)

Ce face:
 1. Curăță rândurile fără consum_final_brut.
 2. Creează un profil logic de creștere a ponderii solar+eolian:
       - 2015: 1%
       - 2020: 3%
       - 2025: 15%
     (creștere lentă până în 2020, apoi accelerată după COVID).
 3. Pentru fiecare lună:
       prod_solar_wind_mwh = consum_final_brut * share_solar_wind / 100
       prod_other_mwh      = consum_final_brut - prod_solar_wind_mwh
 4. Salvează rezultatul în data/dataset_energie_curatat.csv

Rulare:
    python data/build_dataset_energie_curatat.py
"""

import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "data" / "dataset_complet_energie.csv"
DST_PATH = ROOT / "data" / "dataset_energie_curatat.csv"


def build_share_curve(years):
    """
    Construiește ponderea solar+eolian (%) pentru fiecare an:
      - 2015–2020: 1% -> 3% (creștere lentă)
      - 2020–2025: 3% -> 15% (creștere accelerată după COVID)
    """
    years = sorted(years)
    share_map = {}

    for y in years:
        if y <= 2020:
            # interpolare liniară între 1% (2015) și 3% (2020)
            t = (y - 2015) / (2020 - 2015)  # 0 .. 1
            val = 1.0 + t * (3.0 - 1.0)
        else:
            # interpolare liniară între 3% (2020) și 15% (2025)
            t = (y - 2020) / (2025 - 2020)  # 0 .. 1
            val = 3.0 + t * (15.0 - 3.0)

        # clamp ca să fim siguri că 2025 = 15.0 exact
        if y == 2025:
            val = 15.0

        share_map[y] = val

    return share_map


def main():
    print(f"Încarc {SRC_PATH} ...")
    df = pd.read_csv(SRC_PATH)

    # Asigurăm coloana de dată calendaristică
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df["date"] = pd.to_datetime(
            dict(year=df["year"], month=df["month"], day=1)
        )

    # Păstrăm doar rândurile cu consum definit
    df = df.dropna(subset=["consum_final_brut"]).copy()

    # An & lună
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Curba de creștere pentru ponderea solar+eolian
    years = df["year"].unique()
    share_map = build_share_curve(years)

    df["share_solar_wind"] = df["year"].map(share_map)

    # Producție dedusă solar+eolian și restul mixului
    df["prod_solar_wind_mwh"] = (
        df["consum_final_brut"] * df["share_solar_wind"] / 100.0
    )
    df["prod_other_mwh"] = df["consum_final_brut"] - df["prod_solar_wind_mwh"]

    # Verificare rapidă: pondere anuală să fie ce vrem noi
    check = (
        df.groupby("year")[["consum_final_brut", "prod_solar_wind_mwh"]]
        .sum()
        .assign(share_calc=lambda x: x["prod_solar_wind_mwh"] /
                                x["consum_final_brut"] * 100)
    )
    print("Pondere anuală solar+eolian calculată (%):")
    print(check[["share_calc"]])

    # Salvăm CSV curățat
    df.to_csv(DST_PATH, index=False)
    print(f"Setul curățat a fost salvat în: {DST_PATH}")


if __name__ == "__main__":
    main()

