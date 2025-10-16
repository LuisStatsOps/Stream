# S3_C4_streamlit_series.py
# Ejecuta con: streamlit run S3_C4_streamlit_series.py
# Requiere: streamlit, pandas, numpy, matplotlib

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Serie de tiempo interactiva", layout="wide")

st.title("📈 Serie de tiempo interactiva")
st.write("Carga un CSV con columnas **date** y **value**, o usa el dataset sintético incluido.")

# Opción de carga de archivo
uploaded = st.file_uploader("Sube un CSV (opcional)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    # Dataset sintético por defecto (si no suben archivo)
    rng = pd.date_range("2023-01-01", periods=365, freq="D")
    trend = np.linspace(100, 150, len(rng))
    seasonal = 10*np.sin(2*np.pi*rng.dayofweek/7)
    noise = np.random.normal(0, 3, len(rng))
    values = trend + seasonal + noise
    df = pd.DataFrame({"date": rng, "value": values.round(2)})

# Preparación
# Parseo tolerante de fechas y valores
df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(),
                            errors="coerce",      # no rompas: convierte inválidos a NaT
                            infer_datetime_format=True,
                            dayfirst=True)        # útil si tus fechas son dd/mm/yyyy

# Si quedaron NaT (por ejemplo, formato yyyy-mm-dd con dayfirst=True), reintenta al revés
if df["date"].isna().mean() > 0.5:
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(),
                                errors="coerce",
                                infer_datetime_format=True,
                                dayfirst=False)

# Normaliza 'value' (maneja comas decimales, strings, etc.)
df["value"] = (
    df["value"]
      .astype(str)
      .str.replace(",", ".", regex=False)   # por si vienen 12,34
      .str.replace(" ", "", regex=False)
)
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Limpieza final + orden
df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)


# Controles
st.sidebar.header("Controles")
min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input("Rango de fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)

window = st.sidebar.slider("Ventana de media móvil (días)", 1, 60, 7, 1)

# Filtrado por rango
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    dff = df[(df["date"] >= start) & (df["date"] <= end)].copy()
else:
    dff = df.copy()

# Visualizaciones
st.subheader("Serie original")
st.line_chart(dff.set_index("date")["value"])

st.subheader(f"Media móvil {window} días")
dff[f"rolling_{window}"] = dff["value"].rolling(window).mean()
st.line_chart(dff.set_index("date")[[f"rolling_{window}"]])

# Estadísticas básicas
st.subheader("Estadísticas")
col1, col2, col3 = st.columns(3)
col1.metric("Promedio", f"{dff['value'].mean():.2f}")
col2.metric("Mínimo", f"{dff['value'].min():.2f}")
col3.metric("Máximo", f"{dff['value'].max():.2f}")

st.caption("Luis Alberto Baca")
