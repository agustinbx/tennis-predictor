"""
Página de ranking y perfiles de jugadores.
Muestra el ranking ATP y permite ver perfiles detallados.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from streamlit_app.utils import apply_style, load_perfiles, load_ranking
except ImportError:
    import os
    import joblib
    
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    SCRAPING_DIR = PROJECT_ROOT / "scraping"
    
    def apply_style():
        st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """, unsafe_allow_html=True)
    
    def load_perfiles():
        path = SCRAPING_DIR / "perfiles_jugadores.pkl"
        if path.exists():
            return joblib.load(path)
        return None
    
    def load_ranking():
        path = SCRAPING_DIR / "ranking_2026.csv"
        if path.exists():
            return pd.read_csv(path)
        return None


st.set_page_config(page_title="Ranking ATP", page_icon="🏆", layout="wide")

st.title("🏆 Ranking ATP en Vivo & Perfiles")

apply_style()

# --- CARGAR DATOS ---
df_ranking = load_ranking()
perfiles = load_perfiles()

if df_ranking is None or perfiles is None:
    st.warning("No se encontraron los datos. ¿Ya corriste el pipeline ETL?")
    st.info("Ejecuta: `python scripts/run_etl_pipeline.py`")
    st.stop()

# Función para extraer nombre real
def extraer_nombre_real(url):
    try:
        slug = str(url).split('/')[5]
        return slug.replace('-', ' ').title()
    except:
        return ""

df_ranking['Nombre Completo'] = df_ranking['url_perfil'].apply(extraer_nombre_real)

col1, col2 = st.columns([1, 1])

# --- COLUMNA 1: TABLA DE RANKING ---
with col1:
    st.subheader("Clasificación Mundial")
    
    df_mostrar = df_ranking[['rank', 'Nombre Completo', 'points']].copy()
    df_mostrar.columns = ['Rank', 'Jugador', 'Puntos']
    st.dataframe(df_mostrar.set_index('Rank'), height=600, use_container_width=True)

# --- COLUMNA 2: PERFIL DEL JUGADOR ---
with col2:
    st.subheader("🔍 Analizador de Perfil")
    
    lista_jugadores = [j for j in df_ranking['Nombre Completo'].tolist() if j != ""]
    
    jugador_seleccionado = st.selectbox("Buscar jugador:", lista_jugadores)
    
    if jugador_seleccionado in perfiles:
        p = perfiles[jugador_seleccionado]
        
        # Panel de métricas principales
        c1, c2, c3 = st.columns(3)
        c1.metric("Ranking", int(p.get('rank', 0)))
        c2.metric("Puntos", f"{int(p.get('points', 0)):,}".replace(",", "."))
        c3.metric("País", p.get('ioc', 'UNK'))
        
        st.markdown("#### 🧬 Atributos Biométricos")
        c4, c5 = st.columns(2)
        c4.info(f"**Edad:** {p.get('age', 'N/A')} años")
        c5.info(f"**Altura:** {p.get('ht', 'N/A')} cm")
        
        st.markdown("#### 🎾 Estadísticas Avanzadas")
        st.write(f"**1er Saque Ganado:** {p.get('serve_win', 0)}%")
        st.write(f"**Break Points Salvados:** {p.get('bp_saved', 0)}%")
        st.write(f"**Aces por partido:** {p.get('aces', 0):.1f}")
        st.write(f"**Momentum (Últimos 5):** {int(p.get('momentum', 0) * 100)}%")
        
    else:
        st.info("No hay estadísticas avanzadas para este jugador todavía.")
