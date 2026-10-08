"""
Scenarii – în ce an energia solară + eoliană poate acoperi 100% din
consumul final brut de energie electrică în Republica Moldova.

Folosește:
 - data/dataset_energie_curatat.csv
 - modelul ML antrenat în train_consumption_model.py
   (ml/consumption_model.pkl + ml/feature_columns.json)

Rulare:
    python ml/scenarii_energie.py

Output:
 - 3 grafice PNG salvate în ml/:
      fig_scenariu_prudent.png
      fig_scenariu_moderat.png
      fig_scenariu_accelerat.png
 - mesaje în consolă de forma:
      [Prudent] 100% acoperire ~ anul 2037
"""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- CĂI FIȘIERE ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset_energie_curatat.csv"
MODEL_PATH = ROOT / "ml" / "consumption_model.pkl"
FEATURES_PATH = ROOT / "ml" / "feature_columns.json"


# ---------------- CONFIGURAȚIA SCENARIILOR ----------------
@dataclass
class ScenarioConfig:
    name: str          # titlu frumos (ex: "Prudent (20% creștere/an)")
    label: str         # folosit în numele fișierului (ex: "prudent")
    annual_growth_rate: float   # creștere/an pentru solar+eolian (0.2 = 20%)
    start_year: int
    end_year: int


# ---------------- FEATURE ENGINEERING (la fel ca în train) ----------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplică același feature engineering ca în train_consumption_model.py."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # timp / sezonalitate
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["trend_year"] = df["year"] - df["year"].min()

    # echilibru energetic
    df["net_import_mwh"] = df["import"] - df["export"]
    df["total_supply_mwh"] = (
        df["producere"]
        + df["import"]
        + df["procurat_din_alte_surse"]
        - df["export"]
        - df["variatia_stocurilor"]
    )

    # medii mobile pe 3 luni (lag-uite cu 1 lună – fără leakage)
    for col in ["producere", "import", "prod_solar_wind_mwh"]:
        if col in df.columns:
            df[f"{col}_roll3"] = (
                df[col].rolling(window=3, min_periods=1).mean().shift(1)
            )

    return df


# ---------------- HELPERI PENTRU SCENARII ----------------
def load_hist_data():
    df_raw = pd.read_csv(DATA_PATH)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    return df_raw


def build_future_block(df_hist: pd.DataFrame, cfg: ScenarioConfig) -> pd.DataFrame:
    """
    Construiește blocul de date viitoare (fără feature engineering încă),
    pentru un scenariu dat.
    Creșterea solar+eolian se face anual cu rata cfg.annual_growth_rate,
    păstrând sezonalitatea anului de bază.
    """
    last_year = int(df_hist["year"].max())  # ex: 2025
    base_year = last_year

    # profil sezonier pentru solar+eolian din anul de bază
    df_base = df_hist[df_hist["year"] == base_year]
    solar_by_month = df_base.groupby("month")["prod_solar_wind_mwh"].sum()
    solar_annual_base = solar_by_month.sum()
    if solar_annual_base == 0:
        # fallback de siguranță
        solar_by_month = pd.Series(
            np.ones(12), index=range(1, 13), dtype=float
        )
        solar_annual_base = solar_by_month.sum()

    month_weights = (solar_by_month / solar_annual_base).reindex(
        range(1, 13), fill_value=1 / 12
    )

    # medii lunare pentru restul coloanelor (meteo + mix energetic)
    cols_means = [
        "ALLSKY_SFC_SW_DWN",
        "CLOUD_AMT",
        "PRECTOTCORR_SUM",
        "PS",
        "RH2M",
        "T2M",
        "WS50M",
        "producere",
        "import",
        "procurat_din_alte_surse",
        "variatia_stocurilor",
        "export",
        "prod_other_mwh",
        "pv_energy_kwh_day",
        "wind_energy_kwh_day",
        "share_solar_wind",
    ]
    monthly_means = (
        df_hist.groupby("month")[cols_means]
        .mean()
        .reindex(range(1, 13))
        .ffill()
        .bfill()
    )

    # nivel de referință pentru ponderea solar+eolian
    cons_base = df_base["consum_final_brut"].sum()
    share_solar_base = (
        solar_annual_base / cons_base if cons_base > 0 else 0.15
    )  # ~15% default

    rows = []

    for year in range(cfg.start_year, cfg.end_year + 1):
        # câți ani au trecut de la anul de bază
        n = year - base_year
        growth_factor = (1.0 + cfg.annual_growth_rate) ** n

        # ponderea solar+eolian în consum (idealizată, nu exactă)
        share_year = min(share_solar_base * growth_factor, 1.0)  # max 100%

        for month in range(1, 13):
            base_vals = monthly_means.loc[month]

            row = {
                "year": year,
                "month": month,
                "date": pd.Timestamp(year=year, month=month, day=1),
            }

            # copiem mediile meteo & mix energetic
            for col in cols_means:
                row[col] = float(base_vals[col])

            # ajustăm producția solar+eolian (nivel absolut, nu doar pondere)
            row["prod_solar_wind_mwh"] = (
                solar_annual_base * growth_factor * month_weights.loc[month]
            )

            # ponderea în % (informativ – utilă și ca feature)
            row["share_solar_wind"] = share_year * 100.0

            rows.append(row)

    df_future = pd.DataFrame(rows)
    return df_future


def run_scenario(cfg: ScenarioConfig):
    # 1. istoric
    df_hist_raw = load_hist_data()
    df_hist_fe = add_features(df_hist_raw)

    # 2. bloc viitor brut + concatenare pentru a putea calcula roll3 corect
    df_future_raw = build_future_block(df_hist_raw, cfg)
    df_all_raw = pd.concat([df_hist_raw, df_future_raw], ignore_index=True)
    df_all_fe = add_features(df_all_raw)

    # separăm înapoi partea viitoare
    df_future_fe = df_all_fe[df_all_fe["year"] >= cfg.start_year].copy()

    # 3. încărcăm modelul + feature-urile
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["features"]

    # ne asigurăm că toate coloanele de feature există în df_all_fe;
    # dacă lipsesc, le completăm cu 0
    for col in feature_cols:
        if col not in df_all_fe.columns:
            df_all_fe[col] = 0.0
            df_future_fe[col] = 0.0

    # mediana istorică pentru completarea NaN-urilor
    medians = df_all_fe[feature_cols].median(numeric_only=True)

    X_future = df_future_fe[feature_cols].copy()
    X_future = X_future.fillna(medians)

    # 4. predicție consum (în MWh)
    y_future = model.predict(X_future)
    df_future_fe["consum_final_brut_pred"] = y_future

    # 5. producția solar+eolian din scenariu (în MWh)
    df_future_prod = df_all_fe[df_all_fe["year"] >= cfg.start_year].copy()

    df_consum_annual = (
        df_future_fe.groupby("year")["consum_final_brut_pred"].sum().reset_index()
    )
    df_solar_annual = (
        df_future_prod.groupby("year")["prod_solar_wind_mwh"].sum().reset_index()
    )

    df_merge = pd.merge(df_consum_annual, df_solar_annual, on="year", how="inner")

    # Raport acoperire (%) = producție solar+eolian / consum
    df_merge["coverage_pct"] = (
        df_merge["prod_solar_wind_mwh"] / df_merge["consum_final_brut_pred"] * 100.0
    )

    years = df_merge["year"].values
    coverage = df_merge["coverage_pct"].values

    # anul în care solar+eolian >= 100% din consum
    idx_full = np.where(coverage >= 100.0)[0]
    year_full = int(years[idx_full[0]]) if len(idx_full) > 0 else None

    # 6. GRAFIC – acoperire în procente, 0–150%
    coverage_clipped = np.clip(coverage, 0, 150)  # nu lăsăm să meargă la 1200%

    plt.figure(figsize=(9, 4.8))

    # linia principală
    plt.plot(
        years,
        coverage_clipped,
        marker="o",
        label="Acoperire regenerabile (%)",
    )

    # linia de 100%
    plt.axhline(100, linestyle="--", color="gray", linewidth=1, label="100% acoperire")

    # zona >100% colorată
    if year_full is not None:
        mask = coverage >= 100.0
        plt.fill_between(
            years,
            100,
            coverage_clipped,
            where=mask,
            alpha=0.15,
            color="green",
            label="Zonă > 100% acoperire",
        )

        # punct roșu + text pentru primul an cu 100%
        y_point = min(coverage_clipped[idx_full[0]], 140)
        plt.scatter(year_full, y_point, color="red", zorder=5)
        plt.text(
            year_full + 0.2,
            y_point + 5,
            f"100% acoperire în {year_full}",
            color="red",
            fontsize=9,
        )

    plt.title(f"{cfg.name} – Acoperire energie regenerabilă (%)")
    plt.xlabel("An")
    plt.ylabel("Acoperire (%)")
    plt.ylim(0, 150)  # focus pe 0–150%
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_path = ROOT / "ml" / f"fig_scenariu_{cfg.label}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    if year_full is not None:
        print(f"[{cfg.name}] 100% acoperire ~ anul {year_full}")
    else:
        print(
            f"[{cfg.name}] în intervalul {cfg.start_year}-{cfg.end_year} "
            f"nu se ajunge la 100% acoperire."
        )


def main():
    df_hist = load_hist_data()
    last_year = int(df_hist["year"].max())  # de ex. 2025

    scenarios = [
        ScenarioConfig(
            name="Prudent (20% creștere/an)",
            label="prudent",
            annual_growth_rate=0.20,
            start_year=last_year + 1,
            end_year=2050,
        ),
        ScenarioConfig(
            name="Moderat (30% creștere/an)",
            label="moderat",
            annual_growth_rate=0.30,
            start_year=last_year + 1,
            end_year=2050,
        ),
        ScenarioConfig(
            name="Accelerat (40% creștere/an)",
            label="accelerat",
            annual_growth_rate=0.40,
            start_year=last_year + 1,
            end_year=2050,
        ),
    ]

    for cfg in scenarios:
        print("=" * 80)
        run_scenario(cfg)


if __name__ == "__main__":
    main()
