"""
ML – Model de predicție a consumului final brut de energie electrică (Moldova)
cu feature engineering mai avansat.

Set de date: data/dataset_energie_curatat.csv
Țintă (y)   : consum_final_brut (MWh)

Rulare:
    python ml/train_consumption_model.py

Output:
    ml/consumption_model.pkl      – modelul antrenat
    ml/feature_columns.json       – lista de coloane folosite la predicție
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --------- Căi fișiere ---------
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "dataset_energie_curatat.csv"
MODEL_PATH = Path(__file__).resolve().parent / "consumption_model.pkl"
FEATURES_PATH = Path(__file__).resolve().parent / "feature_columns.json"


def load_and_feature_engineer():
    # 1. citește și sortează cronologic
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 2. feature-uri de timp
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["trend_year"] = df["year"] - df["year"].min()

    # 3. feature-uri fizice (echilibru energetic)
    df["net_import_mwh"] = df["import"] - df["export"]
    df["total_supply_mwh"] = (
        df["producere"]
        + df["import"]
        + df["procurat_din_alte_surse"]
        - df["export"]
        - df["variatia_stocurilor"]
    )

    # 4. medii mobile pe 3 luni (lag-uite cu 1 lună ca să nu facem leakage)
    for col in ["producere", "import", "prod_solar_wind_mwh"]:
        if col in df.columns:
            df[f"{col}_roll3"] = (
                df[col].rolling(window=3, min_periods=1).mean().shift(1)
            )

    # scoatem rândurile unde nu avem țintă
    df = df.dropna(subset=["consum_final_brut"]).reset_index(drop=True)

    # 5. selectăm coloanele candidate pentru X
    base_features = [
        # Meteo & potențial regenerabile
        "ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "PRECTOTCORR_SUM",
        "PS", "RH2M", "T2M", "WS50M",
        "pv_energy_kwh_day", "wind_energy_kwh_day",
        # Mix energetic
        "producere", "import", "procurat_din_alte_surse",
        "variatia_stocurilor", "export",
        "share_solar_wind", "prod_solar_wind_mwh", "prod_other_mwh",
        # Feature-uri noi
        "month_sin", "month_cos", "trend_year",
        "net_import_mwh", "total_supply_mwh",
        "producere_roll3", "import_roll3", "prod_solar_wind_mwh_roll3",
    ]

    features = [c for c in base_features if c in df.columns]

    X = df[features].copy()
    y = df["consum_final_brut"].values

    # înlocuim eventualele NaN cu mediana coloanei
    X = X.fillna(X.median(numeric_only=True))

    return X, y, features, df


def time_series_train_test_split(X, y, df, test_months: int = 24):
    """
    Split pentru serie temporală: ultimele `test_months` luni = test,
    restul = train (fără shuffle).
    """
    n = len(df)
    test_size = min(test_months, max(int(n * 0.2), 6))  # minim 6 luni

    X_train = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]

    print(f"Train: {len(X_train)} observații  |  Test: {len(X_test)} observații")
    print(
        f"Interval train: {df['date'].iloc[0].date()}  →  {df['date'].iloc[len(X_train)-1].date()}"
    )
    print(
        f"Interval test : {df['date'].iloc[len(X_train)].date()}  →  {df['date'].iloc[-1].date()}"
    )

    return X_train, X_test, y_train, y_test


def main():
    X, y, features, df = load_and_feature_engineer()

    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y, df)

    # ------------- Model -------------
    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("----- Performanța modelului (cu feature engineering inteligent) -----")
    print(f"MAE : {mae:,.2f} MWh")
    print(f"R2  : {r2:.3f}")

    # salvăm modelul + lista de feature-uri
    joblib.dump(model, MODEL_PATH)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump({"features": features}, f, ensure_ascii=False, indent=2)

    print(f"Model salvat în {MODEL_PATH}")
    print(f"Lista de feature-uri salvată în {FEATURES_PATH}")


if __name__ == "__main__":
    main()
