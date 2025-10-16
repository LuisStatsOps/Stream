import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Series de tiempo (multi-series)", layout="wide")
st.title("📈 Series de tiempo – múltiples columnas")


# --- Carga ---
uploaded = st.file_uploader("Sube un CSV (opcional)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    rng = pd.date_range("2023-01-01", periods=365, freq="D")
    trend = np.linspace(100, 150, len(rng))
    seasonal = 10*np.sin(2*np.pi*rng.dayofweek/7)
    noise = np.random.normal(0, 3, len(rng))
    values = trend + seasonal + noise
    df = pd.DataFrame({"date": rng, "value": values.round(2)})

# --- Normaliza nombres (no cambia datos) ---
df.columns = [c.strip() for c in df.columns]

# --- Detecta candidatos ---
date_guess = [c for c in df.columns if "date" in c.lower() or "fecha" in c.lower() or c.lower()=="ds"]
num_cols   = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# --- Controles en sidebar ---
st.sidebar.header("Columnas")
date_col = st.sidebar.selectbox(
    "Columna de fecha",
    options=df.columns.tolist(),
    index=(df.columns.get_loc(date_guess[0]) if date_guess else 0),
)

value_cols = st.sidebar.multiselect(
    "Columna(s) de valor a graficar",
    options=[c for c in df.columns if c != date_col],
    default=(num_cols[:1] if num_cols else []),
)

# Validación mínima
if not value_cols:
    st.error("Elige al menos una columna numérica para graficar.")
    st.stop()

# --- Parseo robusto ---
df[date_col] = pd.to_datetime(df[date_col].astype(str).str.strip(),
                              errors="coerce", infer_datetime_format=True, dayfirst=True)
if df[date_col].isna().mean() > 0.5:
    df[date_col] = pd.to_datetime(df[date_col].astype(str).str.strip(),
                                  errors="coerce", infer_datetime_format=True, dayfirst=False)

# Forza numérico en las columnas de valor
for c in value_cols:
    df[c] = pd.to_numeric(
        df[c].astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False),
        errors="coerce"
    )

# --- Limpieza y orden ---
df = df.dropna(subset=[date_col] + value_cols).sort_values(date_col).reset_index(drop=True)

# --- Graficar ---
st.subheader("Serie(s) seleccionada(s)")
st.line_chart(df.set_index(date_col)[value_cols])

# Estadísticas básicas de la primera serie seleccionada
st.subheader("Estadísticas")
col1, col2, col3 = st.columns(3)
v = value_cols[0]
col1.metric("Promedio", f"{df[v].mean():.2f}")
col2.metric("Mínimo", f"{df[v].min():.2f}")
col3.metric("Máximo", f"{df[v].max():.2f}")
