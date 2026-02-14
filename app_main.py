import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="ATP AI 2026",
    page_icon="🎾",
    layout="centered"
)

# Título Estilizado
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>ATP Match Predictor AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #64748B;'>Inteligencia Artificial aplicada al Tenis Profesional</h3>", unsafe_allow_html=True)

st.write("---")

# Métricas de Orgullo
col1, col2, col3 = st.columns(3)
col1.metric("Precisión del Modelo", "72.36%", "+1.2%")
col2.metric("Partidos Analizados", "+60,000", "2000-2026")
col3.metric("Ranking Actualizado", "2026", "Live")

st.write("---")

# Imagen o Banner (Opcional)
st.markdown("""
### 🚀 ¿Qué puede hacer esta App?

Esta herramienta utiliza algoritmos de **Machine Learning** para predecir el resultado de partidos de tenis ATP. 

Analiza variables complejas como:
* 🧠 **Psicología:** Historial entre jugadores (H2H) y rachas mentales.
* 🔋 **Físico:** Fatiga acumulada y edad.
* 📊 **Jerarquía:** Diferencia real de puntos ATP (no solo ranking).

### 👈 Usa el menú de la izquierda para navegar
* **🏆 Torneos:** Ve los partidos reales programados para hoy (Scraping en vivo).
* **🔮 Predictor:** Simula cualquier partido hipotético (ej: Sinner vs Alcaraz).
""")

# Botón decorativo
st.info("💡 Tip: El modelo tiene mayor precisión en superficies duras (Hard Court).")