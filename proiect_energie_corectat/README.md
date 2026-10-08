# Proiect energie regenerabilă – versiune corectată

Structură:

- `data/dataset_energie_curatat.csv` – set de date curățat și *dedus* astfel încât:
  - consumul final brut 2015–2025 este luat din statistica.md;
  - producția solară+eoliană este estimată astfel încât în 2025 să reprezinte ~15% din
    consumul total, iar în anii anteriori procentul crește aproximativ liniar;
  - distribuția lunară a producției solar+eolian folosește potențialul NASA
    (pv_energy_kwh_day + wind_energy_kwh_day) ca factor de ponderare.
- `eda/eda_energie.py` – EDA cu:
  - consum final brut;
  - producție solar+eolian vs. restul mixului;
  - pondere anuală solar+eolian (% din consum).
- `ml/train_consumption_model.py` – antrenare model ML (Random Forest) pentru
  consumul lunar de energie electrică.
- `ml/scenarii_energie.py` – scenarii de creștere a capacității solar+eolian și
  estimarea anului când se poate acoperi 100% din consum.
- `webapp/app.py` – aplicație Streamlit (dashboard + scenarii + chatbot pentru gospodării).

## Instalare pachete

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate
# sau Command Prompt:
# .venv\Scripts\activate.bat

pip install --upgrade pip
pip install pandas numpy matplotlib scikit-learn joblib streamlit
```

## Rulare EDA

```bash
python eda/eda_energie.py
```

## Antrenare model ML

```bash
python ml/train_consumption_model.py
```

## Rulare scenarii în consolă

```bash
python ml/scenarii_energie.py
```

## Pornire aplicație web

```bash
streamlit run webapp/app.py
```

Puteți modifica în cod:
- procentul de 15% pentru 2025 (dacă apar date oficiale noi);
- costurile pe kW pentru panouri și turbine;
- ratele de creștere a capacității instalate în scenarii.
