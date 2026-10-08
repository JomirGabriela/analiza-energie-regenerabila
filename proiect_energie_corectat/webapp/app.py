"""
Aplicație web (Streamlit) pentru proiectul de energie regenerabilă în Republica Moldova.

Funcționalități:
 - Dashboard cu serii istorice (consum, producție, pondere solar+eolian).
 - Zonă de scenarii (estimăm anul în care solar+eolian pot acoperi 100% din consum,
   pentru o rată de creștere a capacității instalate).
 - "Chatbot" simplu pentru persoane fizice – estimează investiția și perioada
   de recuperare pentru panouri solare / turbină eoliană mică.

Rulare din folderul proiectului:
    streamlit run webapp/app.py
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------- CĂI FIȘIERE -------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "dataset_energie_curatat.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "ml" / "consumption_model.pkl"
FEATURES_PATH = Path(__file__).resolve().parents[1] / "ml" / "feature_columns.json"


# ------------------- HELPERI COMUNI (la fel ca în ML) -------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplică același feature engineering ca în train_consumption_model.py.
    (an, lună, sin/cos, trend, net_import, total_supply, roll3 etc.)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Sezonalitate
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Trend liniar
    df["trend_year"] = df["year"] - df["year"].min()

    # Echilibru energetic
    if {"import", "export"}.issubset(df.columns):
        df["net_import_mwh"] = df["import"] - df["export"]
    if {
        "producere",
        "import",
        "procurat_din_alte_surse",
        "export",
        "variatia_stocurilor",
    }.issubset(df.columns):
        df["total_supply_mwh"] = (
            df["producere"]
            + df["import"]
            + df["procurat_din_alte_surse"]
            - df["export"]
            - df["variatia_stocurilor"]
        )

    # Medii mobile pe 3 luni (lag cu 1 lună ca să nu facem leakage)
    for col in ["producere", "import", "prod_solar_wind_mwh"]:
        if col in df.columns:
            df[f"{col}_roll3"] = (
                df[col].rolling(window=3, min_periods=1).mean().shift(1)
            )

    return df


def build_future_block(df_hist_raw: pd.DataFrame,
                       annual_growth_rate: float,
                       start_year: int,
                       end_year: int) -> pd.DataFrame:
    """
    Construiește blocul de date viitoare pentru simulare.
    Creșterea solar+eolian se face cu 'annual_growth_rate' / an,
    păstrând sezonalitatea anului de bază.
    """
    last_year = int(df_hist_raw["year"].max())
    base_year = last_year

    # profil sezonier pentru solar+eolian (an de bază)
    df_base = df_hist_raw[df_hist_raw["year"] == base_year]
    solar_by_month = df_base.groupby("month")["prod_solar_wind_mwh"].sum()

    if solar_by_month.sum() == 0:
        # fallback dacă ceva e greșit – distribuție uniformă
        solar_by_month = pd.Series(
            [1.0] * 12, index=range(1, 13), name="prod_solar_wind_mwh"
        )

    solar_annual_base = solar_by_month.sum()
    month_weights = (solar_by_month / solar_annual_base).reindex(
        range(1, 13), fill_value=1 / 12
    )

    # medii lunare pentru celelalte coloane
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
    cols_means = [c for c in cols_means if c in df_hist_raw.columns]

    monthly_means = (
        df_hist_raw.groupby("month")[cols_means]
        .mean()
        .reindex(range(1, 13))
        .ffill()
        .bfill()
    )

    cons_base = df_base["consum_final_brut"].sum()
    share_solar_base = (
        solar_annual_base / cons_base if cons_base > 0 else 0.15
    )  # cca 15% în 2025

    rows = []
    for year in range(start_year, end_year + 1):
        n = year - base_year
        growth_factor = (1.0 + annual_growth_rate) ** n
        share_year = min(share_solar_base * growth_factor, 1.0)  # max 100%

        for month in range(1, 13):
            base_vals = monthly_means.loc[month]

            row = {
                "year": year,
                "month": month,
                "date": pd.Timestamp(year=year, month=month, day=1),
            }

            # copiem mediile
            for col in cols_means:
                row[col] = float(base_vals[col])

            # ajustăm producția solar+eolian (absolut, nu doar pondere)
            row["prod_solar_wind_mwh"] = (
                solar_annual_base * growth_factor * month_weights.loc[month]
            )
            row["share_solar_wind"] = share_year * 100.0

            rows.append(row)

    return pd.DataFrame(rows)


def run_scenario(df_hist_raw: pd.DataFrame,
                 model,
                 feature_cols,
                 annual_growth_rate: float,
                 start_year: int,
                 end_year: int) -> tuple[pd.DataFrame, int | None]:
    """
    Rulează un singur scenariu și întoarce:
      - df cu ani + consum prezis + producție solar+eolian + acoperire (%)
      - anul în care ajungem prima dată la >=100% (sau None).
    """
    # bloc viitor brut + concatenare pentru roll3
    df_future_raw = build_future_block(
        df_hist_raw, annual_growth_rate, start_year, end_year
    )
    df_all_raw = pd.concat([df_hist_raw, df_future_raw], ignore_index=True)

    df_all_fe = add_features(df_all_raw)
    df_future_fe = df_all_fe[df_all_fe["year"] >= start_year].copy()

    # ne asigurăm că toate coloanele de feature există
    for col in feature_cols:
        if col not in df_all_fe.columns:
            df_all_fe[col] = 0.0
        if col not in df_future_fe.columns:
            df_future_fe[col] = 0.0

    medians = df_all_fe[feature_cols].median(numeric_only=True)

    X_future = df_future_fe[feature_cols].copy()
    X_future = X_future.fillna(medians)

    # predicție consum
    y_future = model.predict(X_future)
    df_future_fe["consum_final_brut_pred"] = y_future

    # producție anuală solar+eolian
    df_prod_annual = (
        df_all_fe[df_all_fe["year"] >= start_year]
        .groupby("year")["prod_solar_wind_mwh"]
        .sum()
        .reset_index()
    )
    df_consum_annual = (
        df_future_fe.groupby("year")["consum_final_brut_pred"].sum().reset_index()
    )

    df_merge = pd.merge(df_consum_annual, df_prod_annual, on="year", how="inner")
    df_merge["coverage_pct"] = (
        df_merge["prod_solar_wind_mwh"] / df_merge["consum_final_brut_pred"] * 100.0
    )

    first_full = df_merge[df_merge["coverage_pct"] >= 100.0]
    year_full = int(first_full.iloc[0]["year"]) if not first_full.empty else None

    return df_merge, year_full


# ------------------- ÎNCĂRCARE DATE & MODEL -------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = df.sort_values("date")
    return df


@st.cache_data
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta["features"]


# ------------------- APLICAȚIE STREAMLIT -------------------
def main():
    st.set_page_config(page_title="Energie regenerabilă – Moldova", layout="wide")

    st.title("Energie regenerabilă solară & eoliană în Republica Moldova")

    st.markdown(
        """
        Proiect de analiză și predicție   

        **Obiectiv:** să estimăm în ce an energia regenerabilă _(panouri solare + eoliană)_
        ar putea acoperi **100% din consumul de energie electrică** al Republicii Moldova.  

        Setul de date este construit astfel încât în **2025** ponderea estimată
        a energiei **solar+eolian** să fie ~**15%** din consumul total, iar toate
        graficele din aplicație arată valori în **MWh** sau **% din consum**, ca să fie ușor
        de interpretat.
        """
    )

    df = load_data()

    tab_dash, tab_scen, tab_chat = st.tabs(
        ["📊 Dashboard", "📈 Scenarii 100% acoperire", "🤖 Chat-bot pentru gospodării"]
    )

    # ---------- 1. DASHBOARD ----------
    with tab_dash:
        st.header("1. Dashboard – serii istorice 2015–2025")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Consum final brut de energie electrică (MWh)")
            st.line_chart(df.set_index("date")["consum_final_brut"])

        with c2:
            st.subheader("Producție estimată: solar+eolian vs. restul mixului (MWh)")
            st.line_chart(
                df.set_index("date")[["prod_solar_wind_mwh", "prod_other_mwh"]]
            )

        # Pondere anuală solar+eolian
        df_annual = (
            df.groupby("year")[["consum_final_brut", "prod_solar_wind_mwh"]]
            .sum()
            .reset_index()
        )
        df_annual["share"] = (
            df_annual["prod_solar_wind_mwh"] / df_annual["consum_final_brut"] * 100
        )

        st.subheader("Pondere energie solară+eoliană în consumul total (%) – dedusă")
        st.bar_chart(df_annual.set_index("year")["share"])

    # ---------- 2. SCENARII ----------
    with tab_scen:
        st.header("2. Scenarii – când ajungem la 100% din consum acoperit?")

        model, feature_cols = load_model()

        rate = st.slider(
            "Rata anuală de creștere a capacității instalate solar+eolian",
            min_value=0.05,
            max_value=0.60,
            value=0.30,
            step=0.05,
            help="0.30 înseamnă +30% capacitate instalată în fiecare an.",
        )
        start_year = int(df["year"].max()) + 1
        end_year = st.slider(
            "An final pentru simulare",
            min_value=start_year + 5,
            max_value=start_year + 30,
            value=start_year + 20,
        )

        df_scen, year_full = run_scenario(
            df_hist_raw=df,
            model=model,
            feature_cols=feature_cols,
            annual_growth_rate=rate,
            start_year=start_year,
            end_year=end_year,
        )

        # Pregătim datele pentru vizualizare clară (procente, nu GWh)
        df_plot = df_scen.copy()
        df_plot["coverage_pct_clip"] = df_plot["coverage_pct"].clip(upper=200)
        y_max = max(120, df_plot["coverage_pct_clip"].max() * 1.1)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(
            df_plot["year"],
            df_plot["coverage_pct_clip"],
            marker="o",
            label="Acoperire regenerabile (%)",
        )

        # linie orizontală la 100%
        ax.axhline(
            100,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Prag 100% acoperire",
        )

        # zonă verde >100%
        ax.fill_between(
            df_plot["year"],
            100,
            df_plot["coverage_pct_clip"],
            where=df_plot["coverage_pct_clip"] >= 100,
            alpha=0.15,
            color="green",
            label="Zonă >100% acoperire",
        )

        if year_full is not None:
            cov_year = float(
                df_plot.loc[df_plot["year"] == year_full, "coverage_pct_clip"].iloc[0]
            )
            ax.scatter(year_full, cov_year, color="red", zorder=5)
            ax.text(
                year_full + 0.3,
                cov_year + 5,
                f"100% acoperire în {year_full}",
                color="red",
                fontsize=9,
            )

        ax.set_title(
            f"Scenariu cu {int(rate*100)}% creștere/an – acoperire energie regenerabilă"
        )
        ax.set_xlabel("An")
        ax.set_ylabel("Acoperire din consum (%)")
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        st.pyplot(fig)

        if year_full is not None:
            st.success(
                f"În scenariul ales (**{int(rate*100)}% creștere/an**), "
                f"regenerabilele (solar+eolian) pot ajunge la **100% din consum** "
                f"în jurul anului **{year_full}**. După acest an, producția depășește "
                "teoretic consumul intern (fără a ține cont de stocare și flexibilitatea rețelei)."
            )
        else:
            st.warning(
                "În intervalul de ani selectat nu se ajunge la 100% acoperire. "
                "Poți crește rata de creștere sau extinde perioada de simulare."
            )

    # ---------- 3. CHATBOT ----------
    with tab_chat:
        st.header("3. Chat-bot – estimare pentru gospodării")

        st.markdown(
            """
            Acest *chat-bot* pune câteva întrebări despre locuința ta și îți
            calculează orientativ:

            - ce putere de **panouri solare** / **turbina eoliană mică** ți-ar trebui,
            - cât ar costa investiția,
            - în câți ani s-ar putea **recupera investiția** prin facturi mai mici la lumină.  
            """
        )

        with st.form("chatbot_form"):
            st.write("👋 Salut! Răspunde la întrebările de mai jos:")

            tip = st.selectbox(
                "1) Ce tip de energie regenerabilă te interesează mai mult?",
                ["Nu sunt sigur(ă)", "Panouri solare", "Turbina eoliană mică"],
            )
            zona = st.selectbox("2) Zona geografică", ["Nord", "Centru", "Sud"])
            locuinta = st.selectbox(
                "3) Tip locuință",
                [
                    "Apartament",
                    "Casă la sol (curte)",
                    "Casă la bloc cu acoperiș comun",
                ],
            )
            consum = st.number_input(
                "4) Consum mediu lunar de energie electrică (kWh/lună)",
                min_value=50.0,
                max_value=2000.0,
                value=250.0,
                step=10.0,
            )
            pret_kwh = st.number_input(
                "5) Preț actual energie electrică (lei/kWh)",
                min_value=1.0,
                max_value=10.0,
                value=3.56,
                step=0.1,
            )

            submitted = st.form_submit_button("Calculează scenariile mele")

        if submitted:
            lunar_bill = consum * pret_kwh
            annual_bill = lunar_bill * 12

            st.write(f"📄 Factura ta anuală estimată este **{annual_bill:,.0f} lei/an**.")

            # --- Panouri solare ---
            if tip in ["Nu sunt sigur(ă)", "Panouri solare"]:
                coverage = 0.8  # acoperim ~80% din consum
                invest_cost_per_kw = 1200  # lei per kW – EXEMPU
                needed_kw = consum / 110.0  # 1 kW ~ 110 kWh/lună
                invest_total = needed_kw * invest_cost_per_kw
                annual_savings = annual_bill * coverage
                payback = invest_total / annual_savings if annual_savings > 0 else None

                st.subheader("Scenariu panouri solare")
                st.write(f"- Putere instalată recomandată: **{needed_kw:.1f} kW**")
                st.write(f"- Cost estimativ: **{invest_total:,.0f} lei**")
                if payback is not None:
                    st.write(
                        f"- Perioadă de recuperare aproximativă: **{payback:.1f} ani**"
                    )
                if locuinta == "Apartament":
                    st.info(
                        "La apartamente este nevoie de acces la acoperiș și acordul asociației de locatari."
                    )

            # --- Turbină eoliană mică ---
            if tip in ["Nu sunt sigur(ă)", "Turbina eoliană mică"]:
                invest_cost_per_kw = 1500  # lei per kW – EXEMPU
                needed_kw = min(5.0, consum / 150.0)  # limităm la 5 kW
                invest_total = needed_kw * invest_cost_per_kw
                annual_savings = annual_bill * 0.5  # acoperim ~50% din consum
                payback = invest_total / annual_savings if annual_savings > 0 else None

                st.subheader("Scenariu turbină eoliană mică")
                st.write(f"- Putere instalată recomandată: **{needed_kw:.1f} kW**")
                st.write(f"- Cost estimativ: **{invest_total:,.0f} lei**")
                if payback is not None:
                    st.write(
                        f"- Perioadă de recuperare aproximativă: **{payback:.1f} ani**"
                    )
                st.info(
                    "Turbinele eoliene mici sunt potrivite mai ales în zone rurale deschise, "
                    "cu vânt constant, nu în orașe aglomerate."
                )

            st.success(
                "Valorile sunt orientative – în proiect puteți explica ipotezele și cum s-ar ajusta "
                "în funcție de prețurile reale ale echipamentelor și de schema de sprijin din Moldova."
            )


if __name__ == "__main__":
    main()
