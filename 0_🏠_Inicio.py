"""
Página de inicio de la aplicación ATP Predictor.
"""
import streamlit as st
from PIL import Image
import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

st.set_page_config(
    page_title="ATP Predictor Pro",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ocultar elementos de Streamlit
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Título Estilizado
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>ATP Match Predictor AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #64748B;'>Inteligencia Artificial aplicada al Tenis Profesional</h3>", unsafe_allow_html=True)

st.write("---")

# Métricas
col1, col2, col3 = st.columns(3)
col1.metric("Precisión del Modelo", "70.41%", "Corrección Anti-Leakage")
col2.metric("Partidos Analizados", "+30,000", "2000-2026")
col3.metric("Ranking Actualizado", "2026", "Live")

st.write("---")

# Descripción
st.markdown("""
### 🚀 ¿Qué puede hacer esta App?

Esta herramienta utiliza algoritmos de **Machine Learning** para predecir el resultado de partidos de tenis ATP. 

Analiza variables complejas como:
* 🧠 **Psicología:** Historial entre jugadores (H2H) y rachas mentales.
* 🔋 **Físico:** Fatiga acumulada y edad.
* 📊 **Jerarquía:** Diferencia real de puntos ATP (no solo ranking).

### 👈 Usa el menú de la izquierda para navegar
* **🏆 Torneos:** Ve los partidos reales programados para hoy (Scraping en vivo) (Próximamente).
* **🔮 Predictor:** Simula cualquier partido hipotético (ej: Sinner vs Alcaraz).
""")

st.info("💡 Tip: El modelo tiene mayor precisión en superficies duras (Hard Court).")

st.markdown("---")

st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
        Tennis databases, files, and algorithms by 
        <a href='http://www.tennisabstract.com/' target='_blank'>Jeff Sackmann / Tennis Abstract</a> 
        is licensed under a 
        <a href='https://creativecommons.org/licenses/by-nc-sa/4.0/' target='_blank'>CC BY-NC-SA 4.0 License</a>.<br>
        Based on a work at <a href='https://github.com/JeffSackmann' target='_blank'>github.com/JeffSackmann</a>.
    </div>
""", unsafe_allow_html=True)
