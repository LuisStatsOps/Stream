import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Series de tiempo (multi-series)", layout="wide")
st.title("📈 Series de tiempo – múltiples columnas")

# --- Carga ---
uploaded = st.file_uploader("Sube un CSV (opcional)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    # fallback sintético
    rng = pd.date_range("2023-01-01", periods=365, freq="D")
    trend = np.linspace(100, 150, len(rng))
    seasonal = 10*np.sin(2*np.pi*rng.dayofweek/7)
    noise = np.random.normal(0, 3, len(rng))
    values = trend + seasonal + noise
    df = pd.DataFrame({"date": rng, "value": values.round(2)})

# --- Normaliza nombres y mapea alias a estándar ---
orig_cols = df.columns.tolist()
df.columns = [c.strip() for c in df.columns]
alias = {"ds": "date", "fecha": "date", "fechas": "date",
         "y": "value", "valor": "value", "valores": "value"}
df = df.rename(columns={c: alias.get(c.lower(), c) for c in df.columns})

# --- Selección de columnas ---
st.sidebar.header("Columnas")
date_candidates = [c for c in df.columns if c.lower() in ["date"] or "fecha" in c.lower()]
num_candidates  = [c for c in df.columns if c != (date_candidates[0] if date_candidates else "") 
                   and pd.api.types.is_numeric_dtype(df[c])]

date_col = st.sidebar.selectbox("Columna de fecha", options=df.columns.tolist(),
                                index=(df.columns.get_loc(date_candidates[0]) if date_candidates else 0))
value_cols = st.sidebar.multiselect("Columna(s) de valor a graficar",
                                    options=[c for c in df.columns if c != date_col],
                                    default=(num_candidates[:1] if num_candidates else []))
if not value_cols:
    st.error("Elige al menos una columna numérica para graficar.")
    st.stop()

# --- Parseo determinista de fechas y valores ---
# (ajusta el formato si tu CSV usa otro, p. ej. '%d/%m/%Y')
DATE_FORMAT = "%Y-%m-%d"   # cámbialo si corresponde

# Primero limpia strings
df[date_col] = df[date_col].astype(str).str.strip()
for c in value_cols:
    df[c] = (df[c].astype(str)
                  .str.replace(",", ".", regex=False)
                  .str.replace(" ", "", regex=False))

# Intenta formato fijo; si falla mucho, cae a inferencia robusta
d_parsed = pd.to_datetime(df[date_col], format=DATE_FORMAT, errors="coerce")
if d_parsed.isna().mean() > 0.1:
    d_parsed = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
df[date_col] = d_parsed

for c in value_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --- Diagnóstico (útil para entender diferencias) ---
rows_in = len(df)
rows_bad_date = df[date_col].isna().sum()
rows_bad_vals = sum(df[c].isna().sum() for c in value_cols)
st.caption(f"Filas originales: {rows_in} • Fechas inválidas: {rows_bad_date} • Valores inválidos (suma sobre columnas): {rows_bad_vals}")

# --- Limpieza + orden ---
df = df.dropna(subset=[date_col] + value_cols).sort_values(date_col).reset_index(drop=True)

# --- Filtro de fechas ---
min_d, max_d = df[date_col].min().date(), df[date_col].max().date()
rango = st.sidebar.date_input("Rango de fechas", (min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(rango, tuple) and len(rango) == 2:
    df = df[(df[date_col] >= pd.to_datetime(rango[0])) & (df[date_col] <= pd.to_datetime(rango[1]))]

# --- Gráfico con matplotlib (sin agregaciones implícitas) ---
st.subheader("Serie(s) seleccionada(s)")
fig, ax = plt.subplots()
df.set_index(date_col)[value_cols].plot(ax=ax)
ax.set_xlabel(str(date_col))
ax.set_ylabel("valor")
ax.grid(True)
st.pyplot(fig)

# --- Stats de la primera serie ---
v = value_cols[0]
c1, c2, c3 = st.columns(3)
c1.metric("Promedio", f"{df[v].mean():.2f}")
c2.metric("Mínimo", f"{df[v].min():.2f}")
c3.metric("Máximo", f"{df[v].max():.2f}")
