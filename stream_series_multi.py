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

# ---- Parámetros deterministas ----
fmt = st.sidebar.text_input(
    "Formato de fecha (opcional, ej. %Y-%m-%d o %d/%m/%Y)",
    value="%Y-%m-%d"  # cambia si tu CSV usa dd/mm/yyyy
)
agg = st.sidebar.selectbox(
    "Si hay fechas duplicadas, ¿cómo agrego?",
    ["promedio", "suma", "mediana", "mantener todas"]
)

# ---- Parseo y normalización robusta ----
raw_dates = df[date_col].astype(str).str.strip()

# Intento 1: usar formato fijo
dates_parsed = pd.to_datetime(raw_dates, format=fmt, errors="coerce")
# Si muchas fallan, intento 2: inferencia
if dates_parsed.isna().mean() > 0.1:
    dates_parsed = pd.to_datetime(raw_dates, errors="coerce", infer_datetime_format=True)

df["_date_parsed"] = dates_parsed.dt.tz_localize(None)  # quita tz si apareciera

# Valores numéricos (coma decimal, espacios)
for c in value_cols:
    df[c] = (df[c].astype(str)
                  .str.replace(",", ".", regex=False)
                  .str.replace(" ", "", regex=False))
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Diagnóstico antes de limpiar
n0 = len(df)
bad_date = df["_date_parsed"].isna().sum()
bad_vals = {c: df[c].isna().sum() for c in value_cols}
st.caption(f"Filas totales: {n0} • Fechas inválidas: {bad_date} • "
           f"Valores inválidos: {sum(bad_vals.values())} (por col: {bad_vals})")

# Limpieza y orden
df = df.dropna(subset=["_date_parsed"] + value_cols).copy()
df = df.sort_values("_date_parsed")

# Duplicados por fecha: política explícita
if agg != "mantener todas":
    how = {"promedio": "mean", "suma": "sum", "mediana": "median"}[agg]
    df = df.groupby("_date_parsed", as_index=True)[value_cols].agg(how).sort_index()

# ---- Gráfico (matplotlib evita agregaciones implícitas) ----
import matplotlib.pyplot as plt
st.subheader("Serie(s) seleccionada(s)")

if "_date_parsed" in df.columns:
    df_plot = df.set_index("_date_parsed")[value_cols]
else:
    # ya está agrupado por fecha
    df_plot = df[value_cols]

fig, ax = plt.subplots()
df_plot.plot(ax=ax)
ax.set_xlabel(date_col)
ax.set_ylabel("valor")
ax.grid(True)
st.pyplot(fig)

# --- Stats de la primera serie ---
v = value_cols[0]
c1, c2, c3 = st.columns(3)
c1.metric("Promedio", f"{df[v].mean():.2f}")
c2.metric("Mínimo", f"{df[v].min():.2f}")
c3.metric("Máximo", f"{df[v].max():.2f}")
