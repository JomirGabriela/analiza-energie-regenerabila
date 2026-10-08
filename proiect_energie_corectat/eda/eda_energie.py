"""
EDA pentru proiectul: Predicția acoperirii consumului de energie electrică
din surse regenerabile (Moldova – energie solară + eoliană).

Set de date folosit: data/dataset_energie_curatat.csv

Coloane relevante:
 - consum_final_brut (MWh)              – consum total pe lună
 - prod_solar_wind_mwh (MWh)            – ENERGIE SOLARĂ + EOLIANĂ ESTIMATĂ
   (calculată astfel încât:
       * în 2025 ponderea să fie ~15% din consumul total
       * creștere lentă 2015–2020, apoi accelerată după COVID)
 - prod_other_mwh (MWh)                 – restul mixului (import + alte surse)
 - share_solar_wind (%)                 – pondere lunară solar+eolian în consum
 - year, month, date                    – timp

Rulare:
    python eda/eda_energie.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "dataset_energie_curatat.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def plot_time_series(df: pd.DataFrame) -> None:
    # 1. Consum final brut
    plt.figure(figsize=(9, 4))
    plt.plot(df["date"], df["consum_final_brut"])
    plt.title("Consum final brut de energie electrică (Moldova)")
    plt.xlabel("Data")
    plt.ylabel("MWh")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("eda/fig_consum_final_brut.png", dpi=150)

    # 2. Producție estimată: solar+eolian vs. restul mixului
    plt.figure(figsize=(9, 4))
    plt.plot(df["date"], df["prod_solar_wind_mwh"],
             label="Solar+eolian (estimat)")
    plt.plot(df["date"], df["prod_other_mwh"],
             label="Alte surse (restul mixului)")
    plt.title("Producție estimată: solar+eolian vs. restul mixului")
    plt.xlabel("Data")
    plt.ylabel("MWh")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("eda/fig_prod_solar_wind_vs_rest.png", dpi=150)

    # 3. Pondere anuală solar+eolian în consum
    df_annual = (
        df.groupby("year")[["consum_final_brut", "prod_solar_wind_mwh"]]
        .sum()
        .reset_index()
    )
    df_annual["share"] = (
        df_annual["prod_solar_wind_mwh"] /
        df_annual["consum_final_brut"] * 100
    )

    plt.figure(figsize=(9, 4))
    plt.plot(df_annual["year"], df_annual["share"], marker="o")
    plt.title("Pondere energie solară+eoliană în consumul total (%) – dedusă")
    plt.xlabel("An")
    plt.ylabel("% din consum")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("eda/fig_share_solar_wind.png", dpi=150)


def main():
    df = load_data()
    print("Structura setului de date curățat:")
    print(df.head())
    print(df.describe())

    plot_time_series(df)
    print("Figurile au fost salvate în folderul eda/:")
    print(" - fig_consum_final_brut.png")
    print(" - fig_prod_solar_wind_vs_rest.png")
    print(" - fig_share_solar_wind.png")


if __name__ == "__main__":
    main()
