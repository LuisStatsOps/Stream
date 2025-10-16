# S3_C4_streamlit_iris.py
# Ejecuta con: streamlit run S3_C4_streamlit_iris.py
# Requiere: streamlit, scikit-learn, pandas, numpy

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Clasificador Iris (RandomForest)", layout="centered")

st.title("🌸 Clasificador de Iris — RandomForest")
st.write("App demo que entrena un modelo **RandomForest** y permite probar predicciones interactivamente.")

# Carga de datos y entrenamiento rápido (dataset pequeño → rápido)
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Accuracy en test: {acc:.3f}")

# Controles para predicción
st.subheader("Prueba tu propia muestra")
vals = []
for i, feat in enumerate(feature_names):
    # Rango aproximado de cada característica
    fmin, fmax = float(np.min(X[:, i])), float(np.max(X[:, i]))
    default = float(np.median(X[:, i]))
    v = st.slider(feat, fmin, fmax, default)
    vals.append(v)

vals = np.array(vals).reshape(1, -1)
pred = clf.predict(vals)[0]
probs = clf.predict_proba(vals)[0]

st.write(f"**Predicción:** {target_names[pred]}")
st.write("**Probabilidades:**")
probs_df = pd.DataFrame([probs], columns=target_names)
st.dataframe(probs_df.style.format("{:.2f}"))

# Mostrar matriz de confusión (simple)
st.subheader("Matriz de confusión (test)")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
st.dataframe(cm_df)

st.caption("Luis Alberto Baca")
